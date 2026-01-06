"""
Sentinel-2 Time Series Downloader with Gap-Filling
A Streamlit application for downloading cloud-free Sentinel-2 monthly composites
with temporal gap-filling using adjacent months.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import tempfile
import warnings
import sys
import base64
import json
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set page configuration - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    layout="wide", 
    page_title="Sentinel-2 Time Series Downloader",
    page_icon="🛰️"
)

# Import other packages after page config
import folium
from folium import plugins
from streamlit_folium import st_folium
from shapely.geometry import Polygon, mapping
import ee
import geemap

# ============================================================================
# Session State Initialization
# ============================================================================
if 'drawn_polygon' not in st.session_state:
    st.session_state.drawn_polygon = None
if 'ee_initialized' not in st.session_state:
    st.session_state.ee_initialized = False
if 'downloaded_images' not in st.session_state:
    st.session_state.downloaded_images = []
if 'monthly_composites_info' not in st.session_state:
    st.session_state.monthly_composites_info = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# ============================================================================
# Constants
# ============================================================================
SPECTRAL_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
BAND_NAMES = ['Aerosols', 'Blue', 'Green', 'Red', 'Red Edge 1', 'Red Edge 2', 
              'Red Edge 3', 'NIR', 'Red Edge 4', 'Water Vapor', 'SWIR1', 'SWIR2']

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
            # Try service account authentication from environment variable
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
                # Try interactive authentication
                ee.Authenticate()
                ee.Initialize()
                return True, "Successfully authenticated with Earth Engine!"
        except Exception as auth_error:
            return False, f"Authentication failed: {str(auth_error)}"

# ============================================================================
# GEE Processing Functions
# ============================================================================
def create_gapfilled_timeseries(aoi, start_date, end_date, 
                                 cloudy_pixel_percentage=10,
                                 cloud_probability_threshold=65,
                                 cdi_threshold=-0.5):
    """
    Create gap-filled monthly Sentinel-2 composites.
    
    Parameters:
    -----------
    aoi : ee.Geometry
        Area of interest
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    cloudy_pixel_percentage : int
        Maximum cloudy pixel percentage filter (default: 10)
    cloud_probability_threshold : int
        Cloud probability threshold (default: 65)
    cdi_threshold : float
        CDI threshold for cloud detection (default: -0.5)
    
    Returns:
    --------
    tuple: (final_collection, processed_list, monthly_info)
    """
    
    # Date calculations
    start_date_ee = ee.Date(start_date)
    end_date_ee = ee.Date(end_date)
    num_months = end_date_ee.get('year').subtract(start_date_ee.get('year')).multiply(12).add(
        end_date_ee.get('month').subtract(start_date_ee.get('month')))
    extended_start = start_date_ee.advance(-2, 'month')
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
    
    # Cloud masking function with configurable thresholds
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
        
        masked_count = ee.Algorithms.If(
            count.gt(0),
            freq.eq(0).reduceRegion(ee.Reducer.sum(), aoi, 10, maxPixels=1e13).get('frequency'),
            0
        )
        
        return (composite.addBands(freq)
                .addBands(freq.gt(0).rename('validity_mask'))
                .set('system:time_start', m_start.millis())
                .set('month_index', i)
                .set('month_name', m_start.format('YYYY-MM'))
                .set('image_count', count)
                .set('has_data', count.gt(0))
                .set('masked_pixel_count', masked_count))
    
    monthly_composites = ee.ImageCollection(ee.List.sequence(0, num_months.subtract(1)).map(create_monthly))
    monthly_list = monthly_composites.toList(num_months)
    month_indices = ee.List.sequence(0, num_months.subtract(1))
    
    # Gap-filling function
    def gap_fill(month_idx):
        month_idx = ee.Number(month_idx)
        curr = ee.Image(monthly_list.get(month_idx))
        freq = curr.select('frequency')
        gap_mask = freq.eq(0)
        
        m_start = origin.advance(month_idx, 'month')
        m_end = origin.advance(month_idx.add(1), 'month')
        m_mid_millis = m_start.advance(15, 'day').millis()
        
        # Collect M-1, M+1, M-2
        candidates = (cloud_free.filterDate(origin.advance(month_idx.subtract(1), 'month'), m_start)
            .merge(cloud_free.filterDate(m_end, origin.advance(month_idx.add(2), 'month')))
            .merge(cloud_free.filterDate(origin.advance(month_idx.subtract(2), 'month'),
                                         origin.advance(month_idx.subtract(1), 'month'))))
        
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
        curr = ee.Image(monthly_list.get(month_idx))
        return (curr.select(SPECTRAL_BANDS)
                .addBands(curr.select('frequency'))
                .addBands(ee.Image.constant(0).clip(aoi).toInt8().rename('fill_source'))
                .set('month_name', curr.get('month_name'))
                .copyProperties(curr, ['system:time_start', 'month_index', 'has_data']))
    
    def process_month(i):
        img = ee.Image(monthly_list.get(i))
        has_data = ee.Number(img.get('has_data'))
        masked_count = ee.Number(img.get('masked_pixel_count'))
        return ee.Algorithms.If(
            has_data.And(masked_count.gt(0)), 
            gap_fill(i),
            ee.Algorithms.If(has_data, prepare_complete(i), None)
        )
    
    processed_list = ee.List(month_indices.map(process_month)).removeAll([None])
    
    # Create final collection
    final_collection = ee.ImageCollection.fromImages(processed_list.map(
        lambda img: ee.Image(img).select(SPECTRAL_BANDS).toDouble()
            .set('system:index', ee.Image(img).get('month_name'))
            .set('month_name', ee.Image(img).get('month_name'))
    ))
    
    return final_collection, processed_list, monthly_composites

# ============================================================================
# Download Functions
# ============================================================================
def download_monthly_composites(final_collection, aoi, output_dir, scale=10, crs='EPSG:4326'):
    """
    Download each monthly composite as a separate GeoTIFF file.
    
    Returns:
    --------
    list: List of downloaded file paths and their month names
    """
    downloaded_files = []
    
    # Get the list of images
    image_list = final_collection.toList(final_collection.size())
    num_images = image_list.size().getInfo()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(num_images):
        try:
            img = ee.Image(image_list.get(i))
            month_name = img.get('month_name').getInfo()
            
            status_text.text(f"Downloading {month_name} ({i+1}/{num_images})...")
            
            filename = f"{output_dir}/S2_{month_name}.tif"
            
            # Use geemap for download (handles tiling automatically)
            geemap.download_ee_image(
                image=img,
                filename=filename,
                region=aoi,
                scale=scale,
                crs=crs
            )
            
            downloaded_files.append({
                'path': filename,
                'month_name': month_name,
                'index': i
            })
            
            progress_bar.progress((i + 1) / num_images)
            
        except Exception as e:
            st.warning(f"Error downloading {month_name}: {str(e)}")
            continue
    
    status_text.text("Download complete!")
    return downloaded_files

def download_stacked_image(final_collection, aoi, output_dir, start_date, end_date, scale=10, crs='EPSG:4326'):
    """
    Download all monthly composites as a single stacked multi-band GeoTIFF.
    """
    status_text = st.empty()
    status_text.text("Creating stacked image...")
    
    # Create stacked image
    stacked_image = final_collection.toBands()
    
    filename = f"{output_dir}/S2_Stacked_{start_date}_to_{end_date}.tif"
    
    status_text.text("Downloading stacked image (this may take a while for large areas)...")
    
    # Use geemap for download
    geemap.download_ee_image(
        image=stacked_image,
        filename=filename,
        region=aoi,
        scale=scale,
        crs=crs
    )
    
    status_text.text("Download complete!")
    
    return filename, stacked_image.bandNames().getInfo()

# ============================================================================
# Visualization Functions
# ============================================================================
def visualize_composites(downloaded_files, num_cols=4):
    """Display RGB previews of downloaded composites."""
    import rasterio
    
    num_files = len(downloaded_files)
    if num_files == 0:
        st.warning("No files to display.")
        return
    
    num_rows = (num_files + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
    
    # Flatten axes for easier indexing
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, file_info in enumerate(downloaded_files):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row, col]
        
        try:
            with rasterio.open(file_info['path']) as src:
                # Read RGB bands (B4=Red, B3=Green, B2=Blue -> indices 3, 2, 1)
                if src.count >= 4:
                    r = src.read(4)  # B4 - Red
                    g = src.read(3)  # B3 - Green
                    b = src.read(2)  # B2 - Blue
                    
                    # Normalize for display
                    rgb = np.dstack([r, g, b])
                    
                    # Percentile stretch for better visualization
                    for i in range(3):
                        band = rgb[:, :, i]
                        valid = band[band > 0]
                        if len(valid) > 0:
                            p2, p98 = np.percentile(valid, [2, 98])
                            rgb[:, :, i] = np.clip((band - p2) / (p98 - p2 + 1e-10), 0, 1)
                    
                    ax.imshow(rgb)
                else:
                    # Single band - show grayscale
                    data = src.read(1)
                    ax.imshow(data, cmap='viridis')
            
            ax.set_title(file_info['month_name'], fontsize=10)
            ax.axis('off')
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(file_info['month_name'], fontsize=10)
            ax.axis('off')
    
    # Hide empty subplots
    for idx in range(num_files, num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================================
# Main Application
# ============================================================================
def main():
    # Title and description
    st.title("🛰️ Sentinel-2 Time Series Downloader")
    st.markdown("""
    Download cloud-free Sentinel-2 monthly composites with automatic gap-filling.
    The algorithm fills cloudy pixels using data from adjacent months (M-1, M+1, M-2).
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
    
    # Cloud filtering parameters
    st.sidebar.subheader("Cloud Filtering")
    
    cloudy_pixel_percentage = st.sidebar.slider(
        "Max Cloudy Pixel Percentage",
        min_value=0,
        max_value=100,
        value=10,
        step=5,
        help="Filter out images with cloud cover above this percentage (CLOUDY_PIXEL_PERCENTAGE metadata)"
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
    
    # Export parameters
    st.sidebar.subheader("Export Settings")
    
    export_scale = st.sidebar.selectbox(
        "Export Scale (meters)",
        options=[10, 20, 30, 60],
        index=0,
        help="Spatial resolution for exported images"
    )
    
    export_crs = st.sidebar.selectbox(
        "Export CRS",
        options=['EPSG:4326', 'EPSG:32639', 'EPSG:32638', 'EPSG:32640'],
        index=0,
        help="Coordinate Reference System for exported images"
    )
    
    download_mode = st.sidebar.radio(
        "Download Mode",
        options=["Individual Monthly Files", "Single Stacked File", "Both"],
        index=0,
        help="Choose how to download the time series"
    )
    
    # ========================================================================
    # Main Content
    # ========================================================================
    
    # Step 1: Region Selection
    st.header("1️⃣ Select Region of Interest")
    st.info("Draw a rectangle or polygon on the map to define your area of interest.")
    
    # Create folium map
    m = folium.Map(location=[35.6892, 51.3890], zoom_start=8)
    
    # Add drawing tools
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
    
    # Add satellite basemap
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite',
        name='Google Satellite'
    ).add_to(m)
    
    folium.LayerControl().add_to(m)
    
    # Display map
    map_data = st_folium(m, width=800, height=500)
    
    # Process drawn shape
    if map_data is not None and 'last_active_drawing' in map_data and map_data['last_active_drawing'] is not None:
        drawn_shape = map_data['last_active_drawing']
        if 'geometry' in drawn_shape:
            geometry = drawn_shape['geometry']
            if geometry['type'] == 'Polygon':
                coords = geometry['coordinates'][0]
                polygon = Polygon(coords)
                st.session_state.drawn_polygon = polygon
                
                # Calculate area
                area_sq_km = polygon.area * 111 * 111
                st.success(f"✅ Region selected! Approximate area: {area_sq_km:.2f} km²")
                
                if area_sq_km > 100:
                    st.warning("⚠️ Large area selected. Download may take a long time.")
    
    # Manual coordinate entry
    with st.expander("📝 Or Enter Coordinates Manually"):
        col1, col2 = st.columns(2)
        with col1:
            min_lon = st.number_input("Min Longitude", value=51.0, format="%.4f")
            max_lon = st.number_input("Max Longitude", value=51.5, format="%.4f")
        with col2:
            min_lat = st.number_input("Min Latitude", value=35.5, format="%.4f")
            max_lat = st.number_input("Max Latitude", value=36.0, format="%.4f")
        
        if st.button("Set Region from Coordinates"):
            coords = [
                (min_lon, min_lat),
                (max_lon, min_lat),
                (max_lon, max_lat),
                (min_lon, max_lat),
                (min_lon, min_lat)
            ]
            st.session_state.drawn_polygon = Polygon(coords)
            st.success("✅ Region set from coordinates!")
    
    # Step 2: Date Selection
    st.header("2️⃣ Select Time Period")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=date(2023, 6, 1),
            min_value=date(2017, 1, 1),
            max_value=date.today(),
            help="Select the start date for the time series"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=date(2024, 2, 1),
            min_value=date(2017, 1, 1),
            max_value=date.today(),
            help="Select the end date for the time series"
        )
    
    # Validate dates
    if start_date >= end_date:
        st.error("❌ End date must be after start date!")
        st.stop()
    
    # Calculate number of months
    num_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    st.info(f"📅 Time period: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')} ({num_months} months)")
    
    # Step 3: Download
    st.header("3️⃣ Download Time Series")
    
    # Display current parameters
    st.markdown(f"""
    **Current Parameters:**
    - 🌥️ Max Cloudy Pixel Percentage: **{cloudy_pixel_percentage}%**
    - ☁️ Cloud Probability Threshold: **{cloud_probability_threshold}**
    - 📊 CDI Threshold: **{cdi_threshold}**
    - 📐 Export Scale: **{export_scale}m**
    - 🗺️ Export CRS: **{export_crs}**
    """)
    
    # Download button
    if st.button("🚀 Download Time Series", type="primary", use_container_width=True):
        
        if st.session_state.drawn_polygon is None:
            st.error("❌ Please select a region of interest first!")
            st.stop()
        
        # Convert polygon to GEE geometry
        polygon = st.session_state.drawn_polygon
        geojson = {"type": "Polygon", "coordinates": [list(polygon.exterior.coords)]}
        aoi = ee.Geometry.Polygon(geojson['coordinates'])
        
        # Create output directory
        output_dir = tempfile.mkdtemp()
        st.info(f"📁 Output directory: {output_dir}")
        
        # Process time series
        with st.spinner("Processing Sentinel-2 time series with gap-filling..."):
            try:
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')
                
                st.text("Creating gap-filled monthly composites...")
                
                final_collection, processed_list, monthly_composites = create_gapfilled_timeseries(
                    aoi=aoi,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    cloudy_pixel_percentage=cloudy_pixel_percentage,
                    cloud_probability_threshold=cloud_probability_threshold,
                    cdi_threshold=cdi_threshold
                )
                
                # Get collection info
                collection_size = final_collection.size().getInfo()
                st.success(f"✅ Created {collection_size} monthly composites")
                
                # Download based on mode
                downloaded_files = []
                stacked_file = None
                
                if download_mode in ["Individual Monthly Files", "Both"]:
                    st.text("Downloading individual monthly files...")
                    downloaded_files = download_monthly_composites(
                        final_collection=final_collection,
                        aoi=aoi,
                        output_dir=output_dir,
                        scale=export_scale,
                        crs=export_crs
                    )
                    st.session_state.downloaded_images = downloaded_files
                
                if download_mode in ["Single Stacked File", "Both"]:
                    st.text("Downloading stacked file...")
                    stacked_file, band_names = download_stacked_image(
                        final_collection=final_collection,
                        aoi=aoi,
                        output_dir=output_dir,
                        start_date=start_date_str,
                        end_date=end_date_str,
                        scale=export_scale,
                        crs=export_crs
                    )
                    st.success(f"✅ Stacked file saved: {stacked_file}")
                    st.info(f"Band names: {', '.join(band_names[:5])}... ({len(band_names)} total bands)")
                
                st.session_state.processing_complete = True
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    # Step 4: Visualization
    if st.session_state.processing_complete and len(st.session_state.downloaded_images) > 0:
        st.header("4️⃣ Preview Downloaded Composites")
        
        st.info(f"Showing RGB composites for {len(st.session_state.downloaded_images)} months")
        
        num_cols = st.slider("Number of columns", min_value=2, max_value=6, value=4)
        
        visualize_composites(st.session_state.downloaded_images, num_cols=num_cols)
        
        # Download buttons for files
        st.subheader("📥 Download Files")
        
        for file_info in st.session_state.downloaded_images:
            if os.path.exists(file_info['path']):
                with open(file_info['path'], 'rb') as f:
                    st.download_button(
                        label=f"Download {file_info['month_name']}",
                        data=f,
                        file_name=f"S2_{file_info['month_name']}.tif",
                        mime="image/tiff",
                        key=f"download_{file_info['month_name']}"
                    )

# ============================================================================
# Run Application
# ============================================================================
if __name__ == "__main__":
    main()
