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
        background: #ffffff;
        color: #000000 !important;  /* black text on white cards */
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
        font-weight: 500;
        color: #000000 !important;  /* readable text inside alert boxes */
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
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #cccccc !important;
    }
    
    .stSelectbox label,
    .stMultiSelect label,
    .stRadio label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    .stRadio > div {
        color: #ffffff !important;
    }
    
    /* Headers and text */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #ffffff !important;
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
        st.error("metadata[1].json file not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading metadata: {str(e)}")
        return pd.DataFrame()

# Load real-time data from CSV
@st.cache_data
def load_realtime_data_from_csv():
    try:
        df = pd.read_csv("Combined_cleaned.csv")
        df.columns = df.columns.str.strip()
        
        if 'Date_Time' in df.columns:
            df['Date_Time'] = pd.to_datetime(df['Date_Time'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
        
        if 'Value' in df.columns:
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        if 'well_depth' in df.columns:
            df['well_depth'] = pd.to_numeric(df['well_depth'], errors='coerce')
            
        if 'Date_Time' in df.columns:
            latest_data = df.loc[df.groupby('stationCode')['Date_Time'].idxmax()]
        else:
            latest_data = df.drop_duplicates(subset='stationCode', keep='last')
        
        return latest_data
    except FileNotFoundError:
        st.error("Combined_cleaned.csv file not found. Please ensure the file is in the same directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading Combined_cleaned.csv: {str(e)}")
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
        
        # Data status indicator
        if not realtime_df.empty:
            st.success(f"✅ Data loaded: {len(realtime_df)} stations")
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
                        
                        # Handle date formatting safely
                        last_updated = station.get('Date_Time', 'N/A')
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
                        
                        st.markdown(f"""
                        **💧 Water Data**
                        - **Current Level:** {station['Value']:.2f} m
                        - **Well Depth:** {station.get('well_depth', 'N/A')} m  
                        - **Status:** <span style="color:{color}">**{status}**</span>
                        - **Aquifer Type:** {station.get('well_aquifer_type', 'N/A')}
                        - **Last Updated:** {last_updated_str}
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
        tooltip = f"<b>{row['stationName']}</b><br>{status}: {water_level:.2f}m" if pd.notna(water_level) else f"<b>{row['stationName']}</b><br>No data"
        
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=tooltip,
            icon=folium.Icon(color=marker_color, icon="tint", prefix="fa")
        ).add_to(marker_cluster)
    
    # Compact legend
    legend_html = '''
    <div style="position: fixed; top: 10px; right: 10px; width: 150px; 
                background-color: grey; border:2px solid grey; z-index:9999; 
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
