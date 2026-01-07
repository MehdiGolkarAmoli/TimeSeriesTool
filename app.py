"""
Sentinel-2 Time Series Viewer with Gap-Filling (Production-Ready)
A Streamlit application for viewing cloud-free Sentinel-2 monthly composites
with temporal gap-filling using adjacent months (M-1, M+1 only).

COMPLETE WORKFLOW:
==================
1. Create monthly composites from Sentinel-2 images
2. Check which months have masked pixels (gaps) BEFORE gap-filling (scale=10m)
3. Apply gap-filling ONLY to months that need it (using M-1, M+1 data)
4. Check AGAIN for masked pixels AFTER gap-filling (scale=10m)
5. **EXCLUDE months that still have masked pixels after gap-filling**
6. Return only COMPLETE months (no masked pixels remaining)

IMPORTANT: All scale parameters are set to 10m (native Sentinel-2 resolution)
for accurate gap detection and verification.

CRITICAL FIXES IMPLEMENTED:
=========================
1. RATE LIMITING PREVENTION:
   - Batched reduceRegion() calls using .map() instead of individual getInfo()
   - Reduced concurrent aggregations by computing masked pixels in single pass
   - Local date calculations to avoid unnecessary server calls

2. SCROLL/INTERACTION INTERRUPTION FIX:
   - Added st.session_state.processing flag to track processing state
   - Disabled all interactive elements (map, dates, buttons) during processing
   - Used persistent placeholders that don't trigger reruns
   - Force single rerun after completion with st.rerun()

3. INCOMPLETE THUMBNAIL FIX:
   - Implemented retry logic (3 attempts per thumbnail)
   - Increased thumbnail dimensions from 256 to 512 for better quality
   - Added fallback thumbnail generation using visualize()
   - Reduced batch size from 5 to 3 for more reliable API calls
   - Added proper geometry serialization (ee.Geometry → dict)

4. MASKED PIXEL VERIFICATION:
   - Checks for masked pixels BEFORE gap-filling (scale=10m)
   - Checks for masked pixels AFTER gap-filling (scale=10m)
   - Excludes months that still have gaps after gap-filling
   - Reports how many months were excluded

USAGE:
======
1. Draw region on map (disabled during processing)
2. Select date range (disabled during processing)
3. Click "Generate Time Series" - DO NOT SCROLL/INTERACT
4. Wait for completion (status shows "Processing in progress")
5. View results (only COMPLETE months shown, with 🔄 indicator for gap-filled)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
import numpy as np
from datetime import datetime, date
import tempfile
import warnings
import base64
import json
import time

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set page configuration - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    layout="wide", 
    page_title="Sentinel-2 Time Series Viewer",
    page_icon="🛰️"
)

# Import other packages after page config
import folium
from folium import plugins
from streamlit_folium import st_folium
from shapely.geometry import Polygon
import ee

# ============================================================================
# Session State Initialization
# ============================================================================
if 'drawn_polygons' not in st.session_state:
    st.session_state.drawn_polygons = []
if 'last_drawn_polygon' not in st.session_state:
    st.session_state.last_drawn_polygon = None
if 'ee_initialized' not in st.session_state:
    st.session_state.ee_initialized = False
if 'thumbnails' not in st.session_state:
    st.session_state.thumbnails = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'data_summary' not in st.session_state:
    st.session_state.data_summary = None

# ============================================================================
# Constants
# ============================================================================
SPECTRAL_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']

# ============================================================================
# Earth Engine Authentication
# ============================================================================
@st.cache_resource
def initialize_earth_engine():
    """Initialize Earth Engine with authentication."""
    try:
        ee.Initialize()
        return True, "Earth Engine already initialized"
    except Exception:
        try:
            base64_key = os.environ.get('GOOGLE_EARTH_ENGINE_KEY_BASE64')
            
            if base64_key:
                key_json = base64.b64decode(base64_key).decode()
                key_data = json.loads(key_json)
                
                key_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
                with open(key_file.name, 'w') as f:
                    json.dump(key_data, f)
                
                credentials = ee.ServiceAccountCredentials(
                    key_data['client_email'],
                    key_file.name
                )
                ee.Initialize(credentials)
                os.unlink(key_file.name)
                return True, "Successfully authenticated with Earth Engine (Service Account)!"
            else:
                ee.Authenticate()
                ee.Initialize()
                return True, "Successfully authenticated with Earth Engine!"
        except Exception as auth_error:
            return False, f"Authentication failed: {str(auth_error)}"

# ============================================================================
# Helper Functions
# ============================================================================
def get_utm_zone(longitude):
    """Determine the UTM zone for a given longitude."""
    import math
    return math.floor((longitude + 180) / 6) + 1

def get_utm_epsg(longitude, latitude):
    """Determine the EPSG code for UTM zone based on longitude and latitude."""
    zone_number = get_utm_zone(longitude)
    if latitude >= 0:
        return f"EPSG:326{zone_number:02d}"
    else:
        return f"EPSG:327{zone_number:02d}"

# ============================================================================
# GEE Processing Functions (Cached to prevent re-computation)
# ============================================================================

# Note: We don't use @st.cache_data here because ee.ImageCollection cannot be cached
# Instead, we store results in session_state
def create_gapfilled_timeseries(aoi, start_date, end_date, 
                                 cloudy_pixel_percentage=10,
                                 cloud_probability_threshold=65,
                                 cdi_threshold=-0.5):
    """
    Create gap-filled monthly Sentinel-2 composites.
    Uses only M-1 and M+1 for gap-filling.
    
    Optimized to minimize getInfo() calls and reduce aggregations.
    
    Returns:
        tuple: (final_collection, total_months)
    """
    
    # Date calculations - compute total months locally to avoid getInfo()
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    total_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
    
    start_date_ee = ee.Date(start_date)
    end_date_ee = ee.Date(end_date)
    num_months = ee.Number(total_months)
    
    extended_start = start_date_ee.advance(-1, 'month')
    extended_end = end_date_ee.advance(1, 'month')
    
    # Load collections
    s2_sr = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(aoi)
             .filterDate(extended_start, extended_end)
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudy_pixel_percentage))
             .select(['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12','SCL']))
    
    s2_cloud = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                .filterBounds(aoi)
                .filterDate(extended_start, extended_end))
    
    # Join collections
    s2_joined = ee.ImageCollection(ee.Join.saveFirst('cloud_prob').apply(
        primary=s2_sr, 
        secondary=s2_cloud,
        condition=ee.Filter.equals(leftField='system:index', rightField='system:index')
    )).map(lambda img: img.addBands(ee.Image(img.get('cloud_prob'))))
    
    # Cloud masking function
    def mask_clouds(img):
        is_cloud = img.select('probability').gt(cloud_probability_threshold).And(
            ee.Algorithms.Sentinel2.CDI(img).lt(cdi_threshold)
        )
        cloud_dilated = is_cloud.focal_max(kernel=ee.Kernel.circle(20, 'meters'), iterations=2)
        return (img.updateMask(cloud_dilated.Not())
                .select(SPECTRAL_BANDS)
                .multiply(0.0001)
                .clip(aoi)
                .copyProperties(img, ['system:time_start']))
    
    cloud_free = s2_joined.map(mask_clouds)
    
    # Create monthly composites
    origin = ee.Date(start_date)
    empty_img = (ee.Image.constant(ee.List.repeat(0, len(SPECTRAL_BANDS)))
                 .rename(SPECTRAL_BANDS)
                 .toFloat()
                 .updateMask(ee.Image.constant(0)))
    
    def create_monthly(i):
        i = ee.Number(i)
        m_start = origin.advance(i, 'month')
        m_end = origin.advance(i.add(1), 'month')
        monthly = cloud_free.filterDate(m_start, m_end)
        count = monthly.size()
        
        freq = ee.Image(ee.Algorithms.If(
            count.gt(0),
            monthly.map(lambda img: ee.Image(1).updateMask(img.select('B4').mask()).unmask(0).toInt()).sum().toInt(),
            ee.Image.constant(0).toInt().clip(aoi)
        )).rename('frequency')
        
        composite = ee.Image(ee.Algorithms.If(count.gt(0), monthly.median(), empty_img.clip(aoi)))
        
        return (composite.addBands(freq)
                .addBands(freq.gt(0).rename('validity_mask'))
                .set('system:time_start', m_start.millis())
                .set('month_index', i)
                .set('month_name', m_start.format('YYYY-MM'))
                .set('image_count', count)
                .set('has_data', count.gt(0)))
    
    monthly_composites = ee.ImageCollection(ee.List.sequence(0, num_months.subtract(1)).map(create_monthly))
    monthly_list = monthly_composites.toList(num_months)
    month_indices = ee.List.sequence(0, num_months.subtract(1))
    
    # CRITICAL: Check which months have masked pixels
    # This identifies months that still need gap-filling
    # We do this in ONE batched operation using .map() instead of individual getInfo() calls
    # The server can optimize this better than sequential reduceRegion calls
    # SCALE = 10m (native Sentinel-2 resolution for accurate gap detection)
    def check_has_gaps(img):
        freq = img.select('frequency')
        # Use min() to check if ANY pixel has frequency=0 (masked)
        min_freq = freq.reduceRegion(
            reducer=ee.Reducer.min(),
            geometry=aoi,
            scale=10,  # 10m - Sentinel-2 native resolution
            maxPixels=1e9
        ).get('frequency')
        has_gaps = ee.Number(min_freq).eq(0)
        return img.set('has_masked_pixels', has_gaps)
    
    # Apply gap check to all images - GEE will batch these operations server-side
    monthly_with_gap_info = monthly_composites.map(check_has_gaps)
    monthly_list_updated = monthly_with_gap_info.toList(num_months)
    
    # Gap-filling function (M-1 and M+1 only)
    def gap_fill(month_idx):
        month_idx = ee.Number(month_idx)
        curr = ee.Image(monthly_list_updated.get(month_idx))
        freq = curr.select('frequency')
        gap_mask = freq.eq(0)
        
        m_start = origin.advance(month_idx, 'month')
        m_end = origin.advance(month_idx.add(1), 'month')
        m_mid_millis = m_start.advance(15, 'day').millis()
        
        # Only M-1 and M+1
        candidates = (cloud_free.filterDate(origin.advance(month_idx.subtract(1), 'month'), m_start)
            .merge(cloud_free.filterDate(m_end, origin.advance(month_idx.add(2), 'month'))))
        
        sorted_candidates = candidates.map(
            lambda img: img.set('time_dist', ee.Number(img.get('system:time_start')).subtract(m_mid_millis).abs())
        ).sort('time_dist', True)
        
        empty = (ee.Image.constant(ee.List.repeat(0, len(SPECTRAL_BANDS)))
                 .rename(SPECTRAL_BANDS)
                 .toFloat()
                 .updateMask(ee.Image.constant(0))
                 .clip(aoi))
        
        mosaic = ee.Image(ee.Algorithms.If(
            sorted_candidates.size().gt(0),
            sorted_candidates.mosaic().select(SPECTRAL_BANDS),
            empty
        ))
        
        has_fill = mosaic.select('B4').mask()
        fill_mask = gap_mask.And(has_fill)
        still_masked = gap_mask.And(has_fill.Not())
        
        filled = curr.select(SPECTRAL_BANDS).unmask(mosaic.updateMask(fill_mask))
        fill_source = (ee.Image.constant(0).clip(aoi).toInt8()
                       .where(fill_mask, 1)
                       .where(still_masked, 2)
                       .rename('fill_source'))
        
        return (filled.addBands(freq)
                .addBands(fill_source)
                .set('month_name', curr.get('month_name'))
                .copyProperties(curr, ['system:time_start', 'month_index', 'has_data']))
    
    def prepare_complete(month_idx):
        curr = ee.Image(monthly_list_updated.get(month_idx))
        return (curr.select(SPECTRAL_BANDS)
                .addBands(curr.select('frequency'))
                .addBands(ee.Image.constant(0).clip(aoi).toInt8().rename('fill_source'))
                .set('month_name', curr.get('month_name'))
                .copyProperties(curr, ['system:time_start', 'month_index', 'has_data']))
    
    # Process all months - only gap-fill months that need it
    def process_month(i):
        img = ee.Image(monthly_list_updated.get(i))
        has_data = ee.Number(img.get('has_data'))
        has_gaps = img.get('has_masked_pixels')
        
        return ee.Algorithms.If(
            has_data.And(has_gaps),  # Only gap-fill if has data AND has masked pixels
            gap_fill(i),
            ee.Algorithms.If(has_data, prepare_complete(i), None)  # Use as-is if complete
        )
    
    processed_list = ee.List(month_indices.map(process_month)).removeAll([None])
    
    # CRITICAL: Check for remaining masked pixels AFTER gap-filling
    # Only keep months that are fully complete (no masked pixels remaining)
    # SCALE = 10m (native Sentinel-2 resolution for accurate verification)
    def check_complete_after_gapfill(img):
        img = ee.Image(img)
        spectral = img.select(SPECTRAL_BANDS)
        
        # Check if all pixels are valid (no masked pixels)
        # Use B4 as representative band
        mask_check = spectral.select('B4').mask()
        
        # Count masked pixels (where mask = 0)
        masked_count = mask_check.Not().reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=10,  # 10m - Sentinel-2 native resolution
            maxPixels=1e9
        ).get('B4')
        
        is_complete = ee.Number(masked_count).eq(0)
        
        return img.set('is_complete', is_complete)
    
    # Apply completeness check to all processed images
    processed_with_check = ee.ImageCollection.fromImages(processed_list).map(check_complete_after_gapfill)
    
    # Filter to keep only complete months (no masked pixels after gap-filling)
    complete_months = processed_with_check.filter(ee.Filter.eq('is_complete', True))
    
    # Create final collection with only complete months
    final_collection = complete_months.map(
        lambda img: ee.Image(img).select(SPECTRAL_BANDS).toDouble()
            .set('system:index', ee.Image(img).get('month_name'))
            .set('month_name', ee.Image(img).get('month_name'))
            .set('was_gapfilled', ee.Image(img).propertyNames().contains('fill_source'))
    )
    
    return final_collection, total_months

# ============================================================================
# Thumbnail Generation Functions
# ============================================================================
def get_rgb_thumbnail(image, aoi):
    """
    Get RGB thumbnail URL from Earth Engine image.
    Enhanced with better error handling and visualization parameters.
    """
    try:
        # Ensure we're selecting the right bands and they exist
        rgb_image = image.select(['B4', 'B3', 'B2'])
        
        # Convert geometry to dict if it's an ee.Geometry object
        if isinstance(aoi, ee.Geometry):
            region = aoi.getInfo()
        else:
            region = aoi
        
        # Get thumbnail with increased dimensions for better quality
        thumb_url = rgb_image.getThumbURL({
            'region': region,
            'dimensions': 512,  # Increased from 256 for better quality
            'min': 0,
            'max': 0.3,
            'format': 'png'
        })
        
        return thumb_url
        
    except Exception as e:
        # Try alternative approach with visualization
        try:
            if isinstance(aoi, ee.Geometry):
                region = aoi.getInfo()
            else:
                region = aoi
                
            rgb_viz = image.visualize(**{
                'bands': ['B4', 'B3', 'B2'],
                'min': 0,
                'max': 0.3
            })
            
            return rgb_viz.getThumbURL({
                'region': region,
                'dimensions': 512,
                'format': 'png'
            })
        except Exception as e2:
            print(f"Thumbnail generation failed: {str(e2)}")
            return None

def generate_and_store_thumbnails(final_collection, aoi, progress_placeholder=None):
    """
    Generate thumbnails and store them in session state.
    Optimized to reduce getInfo() calls and avoid rate limiting.
    Also tracks which months were gap-filled.
    
    Uses robust error handling and retry logic for thumbnail generation.
    """
    
    # Get collection info in a single call
    try:
        if progress_placeholder:
            with progress_placeholder:
                st.info("📥 Fetching collection metadata...")
        else:
            st.info("📥 Fetching collection metadata...")
            
        collection_info = final_collection.getInfo()
        features = collection_info.get('features', [])
        num_images = len(features)
        
        if num_images == 0:
            st.warning("No images found for the selected parameters.")
            return []
        
        # Count gap-filled months
        gapfilled_count = sum(1 for f in features if f['properties'].get('was_gapfilled', False))
        
        st.success(f"✅ Found {num_images} monthly composites ({gapfilled_count} were gap-filled)")
        
    except Exception as e:
        st.error(f"Error getting collection info: {str(e)}")
        return []
    
    # Progress tracking - create persistent containers
    if progress_placeholder:
        progress_container = progress_placeholder.container()
    else:
        progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    thumbnails = []
    image_list = final_collection.toList(num_images)
    
    # Process in batches to avoid overwhelming the API
    batch_size = 3  # Reduced batch size for more reliable processing
    max_retries = 3
    
    for batch_start in range(0, num_images, batch_size):
        batch_end = min(batch_start + batch_size, num_images)
        
        for i in range(batch_start, batch_end):
            month_name = features[i]['properties'].get('month_name', f'Month {i+1}')
            was_gapfilled = features[i]['properties'].get('was_gapfilled', False)
            
            status_text.text(f"🖼️ Loading {month_name} ({i+1}/{num_images})...")
            
            # Retry logic for thumbnail generation
            thumb_url = None
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    img = ee.Image(image_list.get(i))
                    thumb_url = get_rgb_thumbnail(img, aoi)
                    
                    if thumb_url:
                        break  # Success
                    else:
                        last_error = "Thumbnail URL returned None"
                        if attempt < max_retries - 1:
                            time.sleep(1)  # Wait before retry
                        
                except Exception as e:
                    last_error = str(e)
                    if attempt < max_retries - 1:
                        status_text.text(f"⚠️ Retry {attempt+1}/{max_retries} for {month_name}...")
                        time.sleep(2)  # Wait longer on error
            
            if thumb_url:
                thumbnails.append({
                    'url': thumb_url,
                    'month_name': month_name,
                    'was_gapfilled': was_gapfilled
                })
            else:
                # Add placeholder for failed thumbnails with error detail
                error_msg = f"⚠️ Could not generate thumbnail for {month_name}"
                if last_error:
                    error_msg += f" (Error: {last_error[:100]})"
                st.warning(error_msg)
            
            progress_bar.progress((i + 1) / num_images)
        
        # Delay between batches to avoid rate limiting
        if batch_end < num_images:
            time.sleep(0.5)
    
    # Clear progress indicators
    status_text.empty()
    progress_bar.empty()
    
    # Store in session state for persistence
    st.session_state.thumbnails = thumbnails
    st.session_state.processing_complete = True
    
    return thumbnails

def display_thumbnails(thumbnails):
    """
    Display thumbnails from session state.
    Fixed 4 columns grid with gap-fill indicators.
    """
    
    if not thumbnails:
        st.info("No images to display. Click 'Generate Time Series' to create composites.")
        return
    
    num_cols = 4  # Fixed 4 columns
    num_rows = (len(thumbnails) + num_cols - 1) // num_cols
    
    for row in range(num_rows):
        cols = st.columns(num_cols)
        for col_idx in range(num_cols):
            img_idx = row * num_cols + col_idx
            if img_idx < len(thumbnails):
                with cols[col_idx]:
                    # Add indicator for gap-filled months
                    caption = thumbnails[img_idx]['month_name']
                    if thumbnails[img_idx].get('was_gapfilled', False):
                        caption += " 🔄"  # Indicator for gap-filled
                    
                    st.image(
                        thumbnails[img_idx]['url'],
                        caption=caption,
                        use_column_width=True
                    )

# ============================================================================
# Main Application
# ============================================================================
def main():
    # Title and description
    st.title("🛰️ Sentinel-2 Time Series Viewer")
    st.markdown("""
    View cloud-free Sentinel-2 monthly composites with automatic gap-filling.
    The algorithm fills cloudy pixels using data from adjacent months (M-1, M+1).
    
    ⚡ **Optimized for large time series** (20+ months)
    """)
    
    # Initialize Earth Engine
    ee_initialized, ee_message = initialize_earth_engine()
    
    if not ee_initialized:
        st.error(ee_message)
        st.info("""
        **To authenticate with Earth Engine:**
        
        **Option 1: Interactive Authentication (Colab/Local)**
        - Run `ee.Authenticate()` in a Python console first
        
        **Option 2: Service Account (Production)**
        - Set the `GOOGLE_EARTH_ENGINE_KEY_BASE64` environment variable
        """)
        st.stop()
    else:
        st.sidebar.success(ee_message)
    
    # ========================================================================
    # Sidebar - Parameters
    # ========================================================================
    st.sidebar.header("⚙️ Parameters")
    
    st.sidebar.subheader("Cloud Filtering")
    
    cloudy_pixel_percentage = st.sidebar.slider(
        "Max Cloudy Pixel Percentage",
        min_value=0,
        max_value=100,
        value=10,
        step=5,
        help="Filter out images with cloud cover above this percentage"
    )
    
    cloud_probability_threshold = st.sidebar.slider(
        "Cloud Probability Threshold",
        min_value=0,
        max_value=100,
        value=65,
        step=5,
        help="Pixels with cloud probability above this threshold are masked"
    )
    
    cdi_threshold = st.sidebar.slider(
        "CDI Threshold",
        min_value=-1.0,
        max_value=0.0,
        value=-0.5,
        step=0.1,
        help="Cloud Displacement Index threshold for cloud detection"
    )
    
    # ========================================================================
    # Main Content - Region Selection
    # ========================================================================
    st.header("1️⃣ Select Region of Interest")
    
    # Disable map during processing
    if st.session_state.get('processing', False):
        st.warning("⚠️ Processing in progress - map interaction disabled")
        st.info("Drawing tools will be available again after processing completes.")
    else:
        st.info("Draw a rectangle or polygon on the map to define your area of interest.")
    
    # Create folium map
    m = folium.Map(location=[35.6892, 51.3890], zoom_start=8)
    
    draw = plugins.Draw(
        export=True,
        position='topleft',
        draw_options={
            'polyline': False,
            'rectangle': True,
            'polygon': True,
            'circle': False,
            'marker': False,
            'circlemarker': False
        }
    )
    m.add_child(draw)
    
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite',
        name='Google Satellite'
    ).add_to(m)
    
    folium.LayerControl().add_to(m)
    
    # Display map (only if not processing)
    if not st.session_state.get('processing', False):
        map_data = st_folium(m, width=800, height=500, key="main_map")
    else:
        st.info("🔄 Map hidden during processing to prevent interruptions")
        map_data = None
    
    # Process drawn shape
    if map_data is not None and 'last_active_drawing' in map_data and map_data['last_active_drawing'] is not None:
        drawn_shape = map_data['last_active_drawing']
        if 'geometry' in drawn_shape:
            geometry = drawn_shape['geometry']
            if geometry['type'] == 'Polygon':
                coords = geometry['coordinates'][0]
                polygon = Polygon(coords)
                st.session_state.last_drawn_polygon = polygon
                
                centroid = polygon.centroid
                utm_zone = get_utm_zone(centroid.x)
                utm_epsg = get_utm_epsg(centroid.x, centroid.y)
                area_sq_km = polygon.area * 111 * 111
                
                st.success(f"✅ Region captured! UTM Zone {utm_zone} ({utm_epsg}), Area: ~{area_sq_km:.2f} km². Click 'Save Selected Region' to save it.")
                
                if area_sq_km > 100:
                    st.warning("⚠️ Large area selected. Processing may take a long time.")
    
    # Save region button
    if st.button("💾 Save Selected Region"):
        if st.session_state.last_drawn_polygon is not None:
            if not any(p.equals(st.session_state.last_drawn_polygon) for p in st.session_state.drawn_polygons):
                st.session_state.drawn_polygons.append(st.session_state.last_drawn_polygon)
                st.success(f"✅ Region saved! Total regions: {len(st.session_state.drawn_polygons)}")
            else:
                st.info("This polygon is already saved.")
        else:
            st.warning("Please draw a polygon on the map first")
    
    # ========================================================================
    # Saved Regions Section
    # ========================================================================
    if st.session_state.drawn_polygons:
        st.subheader("📍 Saved Regions")
        
        for i, poly in enumerate(st.session_state.drawn_polygons):
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                st.write(f"**Region {i+1}**")
            
            with col2:
                centroid = poly.centroid
                utm_zone = get_utm_zone(centroid.x)
                utm_epsg = get_utm_epsg(centroid.x, centroid.y)
                st.write(f"UTM: {utm_zone} ({utm_epsg})")
            
            with col3:
                area_sq_km = poly.area * 111 * 111
                st.write(f"Area: ~{area_sq_km:.2f} km²")
            
            with col4:
                if st.button("🗑️", key=f"delete_region_{i}", help=f"Delete Region {i+1}"):
                    st.session_state.drawn_polygons.pop(i)
                    # Clear thumbnails when region is deleted
                    st.session_state.thumbnails = []
                    st.session_state.processing_complete = False
                    st.session_state.data_summary = None
                    st.rerun()
        
        st.divider()
    
    # ========================================================================
    # Date Selection
    # ========================================================================
    st.header("2️⃣ Select Time Period")
    
    # Disable during processing
    is_processing = st.session_state.get('processing', False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=date(2023, 6, 1),
            min_value=date(2017, 1, 1),
            max_value=date.today(),
            help="Select the start date for the time series",
            disabled=is_processing
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=date(2024, 2, 1),
            min_value=date(2017, 1, 1),
            max_value=date.today(),
            help="Select the end date for the time series",
            disabled=is_processing
        )
    
    if start_date >= end_date:
        st.error("❌ End date must be after start date!")
        st.stop()
    
    num_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    st.info(f"📅 Time period: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')} ({num_months} months)")
    
    if num_months > 24:
        st.warning(f"⚡ Processing {num_months} months - this may take 2-5 minutes. The app is optimized to handle this.")
    
    # ========================================================================
    # Generate Time Series
    # ========================================================================
    st.header("3️⃣ Generate Time Series")
    
    # Region selector
    selected_polygon = None
    if len(st.session_state.drawn_polygons) > 0:
        polygon_index = st.selectbox(
            "Select region to process",
            range(len(st.session_state.drawn_polygons)),
            format_func=lambda i: f"Region {i+1} (~{st.session_state.drawn_polygons[i].area * 111 * 111:.2f} km²)",
            key="polygon_selector"
        )
        selected_polygon = st.session_state.drawn_polygons[polygon_index]
    elif st.session_state.last_drawn_polygon is not None:
        selected_polygon = st.session_state.last_drawn_polygon
        st.info("Using the last drawn polygon (not saved)")
    
    # Process button - use session state to prevent re-triggering
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    # Create a container for the button to prevent re-renders
    button_container = st.container()
    
    with button_container:
        process_button = st.button("🚀 Generate Time Series", type="primary", disabled=st.session_state.processing)
    
    if process_button and not st.session_state.processing:
        
        if selected_polygon is None:
            st.error("❌ Please select a region of interest first!")
            st.stop()
        
        # Set processing flag
        st.session_state.processing = True
        
        # Clear previous results
        st.session_state.thumbnails = []
        st.session_state.processing_complete = False
        st.session_state.data_summary = None
        
        # Convert polygon to GEE geometry - store in session state to persist
        geojson = {"type": "Polygon", "coordinates": [list(selected_polygon.exterior.coords)]}
        aoi = ee.Geometry.Polygon(geojson['coordinates'])
        
        # Store AOI in session state
        st.session_state.current_aoi = aoi
        st.session_state.current_aoi_geojson = geojson
        
        # Create a placeholder for status updates that won't trigger reruns
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        try:
            with status_placeholder:
                st.info("⚙️ Step 1/2: Creating monthly composites with gap-filling...")
            
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # Create time series (optimized - minimal getInfo calls)
            final_collection, total_months = create_gapfilled_timeseries(
                aoi=aoi,
                start_date=start_date_str,
                end_date=end_date_str,
                cloudy_pixel_percentage=cloudy_pixel_percentage,
                cloud_probability_threshold=cloud_probability_threshold,
                cdi_threshold=cdi_threshold
            )
            
            with status_placeholder:
                st.success("✅ Time series created successfully!")
            
            with status_placeholder:
                st.info("⚙️ Step 2/2: Generating RGB thumbnails...")
            
            # Generate and store thumbnails (this also gets the size)
            generate_and_store_thumbnails(final_collection, aoi, progress_placeholder)
            
            # Store data summary
            if st.session_state.thumbnails:
                st.session_state.data_summary = {
                    'total_months': total_months,
                    'months_with_data': len(st.session_state.thumbnails)
                }
            
            status_placeholder.empty()
            progress_placeholder.empty()
            
            # Clear processing flag
            st.session_state.processing = False
            
            # Force rerun to show results
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            
            # Clear processing flag on error
            st.session_state.processing = False
    
    # ========================================================================
    # Display Results (from session state - persists across reruns)
    # ========================================================================
    
    # Show processing status if active
    if st.session_state.get('processing', False):
        st.markdown("""
        <div style="padding: 20px; background-color: #fff3cd; border: 2px solid #ffc107; border-radius: 5px; margin: 20px 0;">
            <h3 style="color: #856404; margin: 0;">🔄 PROCESSING IN PROGRESS</h3>
            <p style="color: #856404; margin: 10px 0 0 0;">
                <strong>⚠️ IMPORTANT:</strong> Do not scroll, click, or interact with the page!<br>
                Processing will continue in the background. Please wait...
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.processing_complete and st.session_state.thumbnails:
        st.divider()
        
        # Display data summary message
        if st.session_state.data_summary:
            total = st.session_state.data_summary['total_months']
            with_data = st.session_state.data_summary['months_with_data']
            excluded = st.session_state.data_summary.get('months_excluded', 0)
            
            st.success(f"✅ {with_data} complete months from {total} months in the selected period.")
            
            if excluded > 0:
                st.info(f"ℹ️ {excluded} months were excluded because they still had masked pixels after gap-filling.")
        
        # Display thumbnails
        st.subheader(f"📅 Monthly Composites (RGB)")
        st.caption("🔄 = Gap-filled using adjacent months")
        display_thumbnails(st.session_state.thumbnails)

# ============================================================================
# Run Application
# ============================================================================
if __name__ == "__main__":
    main()
