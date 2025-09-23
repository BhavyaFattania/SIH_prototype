import streamlit as st
import folium
from folium import plugins
import pandas as pd
import json
from streamlit_folium import st_folium
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Groundwater Monitoring Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better UI with proper font colors
st.markdown("""
<style>
    /* Global font styling */
    .main .block-container {
        color: #ffffff !important;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2e86ab 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: #ffffff !important;
        text-align: center;
    }
    
    /* Metric cards */
    .metric-card {
        background: #f8f9fa;
        color: #000000 !important;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #dee2e6;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Alert cards with better contrast */
    .alert-card {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: #000000 !important;
        font-weight: 500;
    }
    .alert-critical { 
        background-color: #ffebee; 
        border: 2px solid #f44336; 
        border-left: 6px solid #d32f2f; 
    }
    .alert-warning { 
        background-color: #fff8e1; 
        border: 2px solid #ff9800; 
        border-left: 6px solid #f57c00; 
    }
    .alert-good { 
        background-color: #f1f8e9; 
        border: 2px solid #4caf50; 
        border-left: 6px solid #388e3c; 
    }
    
    /* Streamlit component overrides */
    .stSelectbox > div > div {
        background-color: black !important;
        color: white !important;
        border: 2px solid #cccccc !important;
    }
    
    .stSelectbox label {
        color: white !important;
        font-weight: 600 !important;
    }
    
    .stRadio > div {
        color: black !important;
    }
    
    .stRadio label {
        color: black !important;
        font-weight: 600 !important;
    }
    
    .stMultiSelect > div > div {
        background-color: #ffffff !important;
        color: black !important;
        border: 2px solid #cccccc !important;
    }
    
    .stMultiSelect label {
        color: black !important;
        font-weight: 600 !important;
    }
    
    /* Headers and text */
    h1, h2, h3, h4, h5  {
        color: white !important;
        font-weight: bold !important;
    }
    
    div, span,label, p {
        color: white !important;
    }
    
</style>
""", unsafe_allow_html=True)

# Load your JSON station metadata
@st.cache_data
def load_station_metadata():
    try:
        with open("metadata[1].json", "r", encoding="utf-8") as f:
            stations = json.load(f)
        df = pd.DataFrame(stations)
        df["latitude"] = df["latitude"].astype(float)
        df["longitude"] = df["longitude"].astype(float)
        return df
    except FileNotFoundError:
        # st.error("metadata[1].json file not found.")
        return pd.DataFrame()
    except Exception as e:
        # st.error(f"Error loading metadata: {str(e)}")
        return pd.DataFrame()

# Load real-time data from CSV with robust error handling
@st.cache_data
def load_realtime_data_from_csv():
    try:
        # Try different encoding options
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv("Combined_cleaned.csv", encoding=encoding)
                #st.info(f"File loaded successfully with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            st.error("Could not read CSV file with any encoding")
            return pd.DataFrame()
        
        # Debug: Show original column info
        # st.write("**Debug Info:**")
        # st.write(f"Original columns count: {len(df.columns)}")
        # st.write(f"DataFrame shape: {df.shape}")
        
        # Clean column names - remove NaN columns first
        original_columns = df.columns.tolist()
        # st.write(f"Original columns: {original_columns[:10]}...")  # Show first 10
        
        # Filter out NaN column names
        valid_columns = []
        column_mapping = {}
        
        for i, col in enumerate(df.columns):
            if pd.notna(col) and str(col).strip() != '':
                clean_col = str(col).strip()
                valid_columns.append(clean_col)
                column_mapping[col] = clean_col
            else:
                st.warning(f"Dropping column {i} (NaN or empty): {col}")
        
        # Keep only valid columns
        df = df.loc[:, [col for col in df.columns if pd.notna(col) and str(col).strip() != '']]
        
        # Rename columns to clean names
        df = df.rename(columns=column_mapping)
        
        # st.success(f"Valid columns after cleaning: {list(df.columns)}")
        
        # Check for required columns
        required_columns = ['stationCode']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.write("Available columns:", list(df.columns))
            return pd.DataFrame()
        
        # Process datetime column if exists
        datetime_columns = ['Date_Time', 'DateTime', 'date_time', 'timestamp']
        datetime_col = None
        for col in datetime_columns:
            if col in df.columns:
                datetime_col = col
                break
        
        if datetime_col:
            df['Date_Time'] = pd.to_datetime(df[datetime_col], errors='coerce')
            # st.info(f"Found datetime column: {datetime_col}")
        
        # Process numeric columns
        numeric_columns = ['Value', 'value', 'water_level', 'level']
        value_col = None
        for col in numeric_columns:
            if col in df.columns:
                value_col = col
                break
        
        if value_col and value_col != 'Value':
            df['Value'] = pd.to_numeric(df[value_col], errors='coerce')
            # st.info(f"Found value column: {value_col}")
        elif 'Value' in df.columns:
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        
        if 'well_depth' in df.columns:
            df['well_depth'] = pd.to_numeric(df['well_depth'], errors='coerce')
        
        # Remove rows where stationCode is NaN
        df = df.dropna(subset=['stationCode'])
        
        # Get latest data for each station with better logic
        if 'Date_Time' in df.columns:
            # Remove rows with invalid dates first
            df_with_dates = df.dropna(subset=['Date_Time'])
            
            if not df_with_dates.empty:
                # Sort by Date_Time and get the latest for each station
                df_with_dates = df_with_dates.sort_values(['stationCode', 'Date_Time'])
                latest_data = df_with_dates.groupby('stationCode').last().reset_index()
                
                # Debug info
                # st.write("**Latest data processing:**")
                # st.write(f"Records with valid dates: {len(df_with_dates)}")
                # st.write(f"Unique stations: {len(latest_data)}")
                
                # Show sample of latest dates
                if len(latest_data) > 0:
                    sample_dates = latest_data[['stationCode', 'Date_Time']].head(5)
                    # st.write("**Sample latest dates:**")
                    # st.dataframe(sample_dates)
            else:
                st.warning("No valid dates found in Date_Time column")
                latest_data = df.drop_duplicates(subset='stationCode', keep='last')
        else:
            latest_data = df.drop_duplicates(subset='stationCode', keep='last')
        
        st.success(f"Successfully processed {len(latest_data)} stations")
        return latest_data
        
    except FileNotFoundError:
        st.error("Combined_cleaned.csv file not found. Please ensure the file is in the same directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading Combined_cleaned.csv: {str(e)}")
        st.write("**Detailed error information:**")
        st.write(f"Error type: {type(e).__name__}")
        
        # Try to read just the first few rows to diagnose
        try:
            sample_df = pd.read_csv("Combined_cleaned.csv", nrows=5)
            # st.write("**Sample of first 5 rows:**")
            # st.dataframe(sample_df)
            # st.write("**Column names in sample:**")
            # st.write(list(sample_df.columns))
        except Exception as sample_error:
            st.error(f"Cannot even read sample data: {sample_error}")
        
        return pd.DataFrame()

def get_status_category(value):
    """Categorize water levels"""
    if pd.isna(value):
        return 'No Data', 'gray'
    elif value > -10:
        return 'Good', 'green'
    elif value > -20:
        return 'Moderate', 'orange'
    else:
        return 'Critical', 'red'

def merge_station_data(metadata_df, realtime_df):
    if realtime_df.empty:
        return metadata_df
    
    # Check if required columns exist in realtime_df
    available_columns = ['stationCode']
    optional_columns = ['Value', 'Date_Time', 'well_depth', 'well_aquifer_type', 'stationStatus']
    
    for col in optional_columns:
        if col in realtime_df.columns:
            available_columns.append(col)
    
    merged_df = metadata_df.merge(
        realtime_df[available_columns],
        on='stationCode',
        how='left'
    )
    return merged_df

def create_quick_stats_cards(df):
    """Create summary statistics cards"""
    if 'Value' in df.columns:
        total_stations = len(df)
        active_stations = len(df[df.get('stationStatus', '') == 'Active'])
        
        # Fix the moderate stations logic (was incorrect)
        good_stations = len(df[df['Value'] > -10])
        moderate_stations = len(df[(df['Value'] <= -10) & (df['Value'] > -20)])  # Fixed condition
        critical_stations = len(df[df['Value'] <= -20])
        no_data_stations = len(df[df['Value'].isna()])
        
        return {
            'total': total_stations,
            'active': active_stations,
            'good': good_stations,
            'moderate': moderate_stations,
            'critical': critical_stations,
            'no_data': no_data_stations
        }
    return {}

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ Government Groundwater Monitoring Dashboard</h1>
        <p>Real-time groundwater level monitoring across India • Quick Access • Minimal Navigation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load and process data first (moved outside sidebar)
    metadata_df = load_station_metadata()
    if metadata_df.empty:
        st.error("Please ensure metadata file is available")
        return
    
    realtime_df = load_realtime_data_from_csv()
    df = merge_station_data(metadata_df, realtime_df)
    
    # Sidebar with streamlined controls
    with st.sidebar:
        st.header("📊 Dashboard Controls")
        
        # Data status indicator with date debugging
        if not realtime_df.empty:
            st.success(f"✅ Data loaded: {len(realtime_df)} stations")
            
            # Add date debugging info
            if 'Date_Time' in realtime_df.columns:
                date_info = realtime_df['Date_Time'].describe()
                st.write("**Date_Time Column Info:**")
                # st.write(f"- Valid dates: {realtime_df['Date_Time'].notna().sum()}")
                st.write(f"- Date range: {realtime_df['Date_Time'].min()} to {realtime_df['Date_Time'].max()}")
                
                # Show sample of actual dates
                #sample_dates = realtime_df[['stationCode', 'Date_Time']].head(3)
                #st.write("**Sample dates:**")
                # st.dataframe(sample_dates)
        else:
            st.error("❌ No real-time data loaded")
        
        # Quick filters
        st.subheader("🔍 Quick Filters")
        view_mode = st.radio(
            "View Mode",
            ["Overview", "Critical Alerts", "State-wise", "Station Details"],
            help="Choose how to view your data"
        )
        
        # Dynamic filters based on data
        if not df.empty:
            states = ['All'] + sorted(df['state'].dropna().unique().tolist())
            selected_state = st.selectbox("🗺️ Select State", states)
            
            if selected_state != 'All':
                filtered_for_district = df[df['state'] == selected_state]
                districts = ['All'] + sorted(filtered_for_district['district'].dropna().unique().tolist())
                selected_district = st.selectbox("🏘️ Select District", districts)
            else:
                selected_district = 'All'
                
            # Status filter
            status_options = ['Active', 'Inactive']
            if 'stationStatus' in df.columns:
                available_statuses = df['stationStatus'].dropna().unique().tolist()
                status_options = [s for s in status_options if s in available_statuses]
            
            status_filter = st.multiselect(
                "📊 Station Status",
                status_options,
                default=status_options if status_options else [],
                help="Filter by station operational status"
            )
        else:
            selected_state = 'All'
            selected_district = 'All'
            status_filter = []
    
    # Apply filters
    filtered_df = df.copy()
    if selected_state != 'All':
        filtered_df = filtered_df[filtered_df['state'] == selected_state]
    if selected_district != 'All':
        filtered_df = filtered_df[filtered_df['district'] == selected_district]
    if status_filter and 'stationStatus' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['stationStatus'].isin(status_filter)]
    
    # Quick stats
    stats = create_quick_stats_cards(filtered_df)
    
    if view_mode == "Overview":
        # Quick Stats Dashboard
        if stats:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("🏢 Total Stations", stats['total'])
            with col2:
                st.metric("✅ Active", stats['active'], delta=f"{stats['active']/stats['total']*100:.0f}%" if stats['total'] > 0 else "0%")
            with col3:
                st.metric("🟢 Good Level", stats['good'])
            with col4:
                st.metric("🟡 Moderate", stats['moderate'])
            with col5:
                st.metric("🔴 Critical", stats['critical'], delta=f"-{stats['critical']}" if stats['critical'] > 0 else None)
        
        # Alert Cards
        if 'Value' in filtered_df.columns:
            critical_stations = filtered_df[filtered_df['Value'] <= -20]
            if not critical_stations.empty:
                st.markdown("### 🚨 Critical Alerts")
                for _, station in critical_stations.head(5).iterrows():
                    st.markdown(f"""
                    <div class="alert-card alert-critical">
                        <strong>{station['stationName']}</strong> - {station['state']}, {station['district']}<br>
                        Water Level: <strong>{station['Value']:.2f} m</strong> | Well Depth: {station.get('well_depth', 'N/A')} m
                    </div>
                    """, unsafe_allow_html=True)
    
    elif view_mode == "Critical Alerts":
        if 'Value' in filtered_df.columns:
            critical_df = filtered_df[filtered_df['Value'] <= -20].sort_values('Value')
            
            if not critical_df.empty:
                st.markdown("### 🚨 Critical Water Level Stations")
                
                # Critical stations table
                display_columns = ['stationCode', 'stationName', 'state', 'district', 'Value', 'well_depth']
                available_display_columns = [col for col in display_columns if col in critical_df.columns]
                
                rename_dict = {
                    'stationCode': 'Station Code',
                    'stationName': 'Station Name', 
                    'state': 'State',
                    'district': 'District',
                    'Value': 'Water Level (m)',
                    'well_depth': 'Well Depth (m)'
                }
                
                display_df = critical_df[available_display_columns].rename(columns=rename_dict)
                st.dataframe(display_df, use_container_width=True, height=400)
            else:
                st.success("✅ No critical alerts! All stations within acceptable levels.")
    
    elif view_mode == "State-wise":
        if 'Value' in filtered_df.columns:
            # State-wise summary
            state_summary = filtered_df.groupby('state').agg({
                'stationCode': 'count',
                'Value': ['mean', 'min', 'max']
            }).round(2)
            
            state_summary.columns = ['Total Stations', 'Avg Water Level', 'Min Level', 'Max Level']
            
            st.markdown("### 📊 State-wise Summary")
            st.dataframe(state_summary, use_container_width=True)
            
            # Chart
            chart_data = filtered_df.groupby('state')['Value'].mean().reset_index()
            fig = px.bar(
                chart_data,
                x='state', y='Value',
                title='Average Water Level by State',
                color='Value',
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif view_mode == "Station Details":
        # Station search and details
        if not filtered_df.empty:
            station_options = filtered_df.apply(lambda x: f"{x['stationCode']} - {x['stationName']} ({x['state']})", axis=1).tolist()
            
            selected_station = st.selectbox(
                "🔍 Search and Select Station",
                options=range(len(station_options)),
                format_func=lambda x: station_options[x],
                help="Type to search for a specific station"
            )
            
            if selected_station is not None:
                station = filtered_df.iloc[selected_station]
                
                # Station details cards
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **📍 Location Information**
                    - **Station Code:** {station['stationCode']}
                    - **Name:** {station['stationName']}
                    - **State:** {station['state']}
                    - **District:** {station['district']}
                    - **Coordinates:** {station['latitude']:.4f}, {station['longitude']:.4f}
                    """)
                
                with col2:
                    if 'Value' in station and pd.notna(station['Value']):
                        status, color = get_status_category(station['Value'])
                        
                        # Handle date formatting with better logic
                        last_updated = station.get('Date_Time')
                        last_updated_str = "No Data"
                        
                        if pd.notna(last_updated):
                            try:
                                # Convert to datetime if it's a string
                                if isinstance(last_updated, str):
                                    # Try to parse the string
                                    if last_updated.strip() != '' and last_updated.lower() != 'nan':
                                        parsed_date = pd.to_datetime(last_updated)
                                        last_updated_str = parsed_date.strftime("%Y-%m-%d %H:%M")
                                    else:
                                        last_updated_str = "Invalid Date"
                                elif hasattr(last_updated, 'strftime'):
                                    # It's already a datetime object
                                    last_updated_str = last_updated.strftime("%Y-%m-%d %H:%M")
                                else:
                                    # Try to convert whatever it is
                                    parsed_date = pd.to_datetime(str(last_updated))
                                    last_updated_str = parsed_date.strftime("%Y-%m-%d %H:%M")
                            except (ValueError, TypeError, AttributeError) as e:
                                # If all else fails, show the raw value
                                last_updated_str = f"Parse Error: {str(last_updated)}"
                                st.warning(f"Date parsing error for station {station['stationCode']}: {e}")
                        
                        st.markdown(f"""
                        **💧 Water Data**
                        - **Current Level:** {station['Value']:.2f} m
                        - **Well Depth:** {station.get('well_depth', 'N/A')} m  
                        - **Status:** <span style="color:{color}">**{status}**</span>
                        - **Aquifer Type:** {station.get('well_aquifer_type', 'N/A')}
                        - **Last Updated:** {last_updated_str}
                        - **Debug - Raw Date:** {station.get('Date_Time')}
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("No real-time data available for this station")
    
    # Compact Map Section
    st.markdown("### 🗺️ Interactive Map")
    
    # Create map with appropriate zoom based on filtered data
    if not filtered_df.empty:
        center_lat = filtered_df['latitude'].mean()
        center_lon = filtered_df['longitude'].mean()
        
        # Calculate zoom level based on data spread
        lat_range = filtered_df['latitude'].max() - filtered_df['latitude'].min()
        lon_range = filtered_df['longitude'].max() - filtered_df['longitude'].min()
        zoom_level = max(5, min(10, int(10 - max(lat_range, lon_range) * 5)))
    else:
        center_lat, center_lon, zoom_level = 20.5937, 78.9629, 5
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_level,
        tiles="OpenStreetMap"
    )
    
    # Add markers with enhanced clustering
    marker_cluster = plugins.MarkerCluster(
        name="stations",
        overlay=True,
        control=True,
        icon_create_function="""
        function(cluster) {
            var childCount = cluster.getChildCount();
            var c = ' marker-cluster-';
            if (childCount < 10) {
                c += 'small';
            } else if (childCount < 100) {
                c += 'medium';
            } else {
                c += 'large';
            }
            return new L.DivIcon({ 
                html: '<div><span>' + childCount + '</span></div>', 
                className: 'marker-cluster' + c, 
                iconSize: new L.Point(40, 40) 
            });
        }"""
    ).add_to(m)
    
    # Add markers
    for idx, row in filtered_df.iterrows():
        water_level = row.get('Value', None)
        status, marker_color = get_status_category(water_level)
        
        # Handle date formatting for popup
        last_updated = row.get('Date_Time', 'N/A')
        if pd.notna(last_updated):
            if isinstance(last_updated, str):
                try:
                    parsed_date = pd.to_datetime(last_updated)
                    last_updated_str = parsed_date.strftime("%Y-%m-%d %H:%M")
                except:
                    last_updated_str = str(last_updated)
            else:
                last_updated_str = last_updated.strftime("%Y-%m-%d %H:%M")
        else:
            last_updated_str = "No Data"
        
        # Compact popup
        if pd.notna(water_level):
            popup_html = f"""
            <div style="font-family: Arial; width: 280px;">
                <h4 style="color: #2E86AB; margin: 0 0 8px 0;">{row['stationName']}</h4>
                <table style="width:100%; font-size:12px;">
                    <tr><td><strong>Code:</strong></td><td>{row['stationCode']}</td></tr>
                    <tr><td><strong>Location:</strong></td><td>{row['district']}, {row['state']}</td></tr>
                    <tr><td><strong>Water Level:</strong></td><td style="color:{get_status_category(water_level)[1]}"><strong>{water_level:.2f} m</strong> ({status})</td></tr>
                    <tr><td><strong>Well Depth:</strong></td><td>{row.get('well_depth', 'N/A')} m</td></tr>
                    <tr><td><strong>Status:</strong></td><td>{row.get('stationStatus', 'Unknown')}</td></tr>
                    <tr><td><strong>Updated:</strong></td><td>{last_updated_str}</td></tr>
                </table>
            </div>
            """
        else:
            popup_html = f"""
            <div style="font-family: Arial; width: 250px;">
                <h4>{row['stationName']}</h4>
                <p><strong>Location:</strong> {row['district']}, {row['state']}</p>
                <p><em>No real-time data available</em></p>
            </div>
            """
        
        # Compact tooltip
        tooltip = f"<b>{row['stationName']}</b><br>{status}: {water_level:.2f}m <br>Click here" if pd.notna(water_level) else f"<b>{row['stationName']}</b><br>No data"
        
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=tooltip,
            icon=folium.Icon(color=marker_color, icon="tint", prefix="fa")
        ).add_to(marker_cluster)
    
    # Compact legend
    legend_html = '''
    <div style="position: fixed; top: 10px; right: 10px; width: 150px; 
                background-color: grey ; border:2px solid grey; z-index:9999; 
                font-size:11px; padding: 8px; border-radius: 5px;">
    <h5 style="margin: 0 0 5px 0;">Water Levels</h5>
    <i class="fa fa-circle" style="color:green"></i> Good (> -10m)<br>
    <i class="fa fa-circle" style="color:orange"></i> Moderate<br>  
    <i class="fa fa-circle" style="color:red"></i> Critical (< -20m)<br>
    <i class="fa fa-circle" style="color:gray"></i> No Data
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Display map with optimal height
    st_folium(m, width=1400, height=600, returned_objects=["last_object_clicked"])
    
    # Quick action footer
    if not realtime_df.empty:
        st.success(f"✅ Dashboard operational - {len(filtered_df)} stations displayed")
        if stats and stats.get('critical', 0) > 0:
            st.error(f"🚨 {stats['critical']} stations require immediate attention!")
    else:
        st.info("⚠️ No real-time data available - Check Combined_cleaned.csv file")

if __name__ == "__main__":
    main()
