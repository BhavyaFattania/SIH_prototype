import streamlit as st
import folium
from folium import plugins
import pandas as pd
import json
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Set page config
st.set_page_config(page_title="Groundwater Monitoring Dashboard", layout="wide", initial_sidebar_state="expanded")

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'overview'
if 'selected_station' not in st.session_state:
    st.session_state.selected_station = None

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
    
    /* Station header styling */
    .station-header {
        background: linear-gradient(90deg, #2e86ab 0%, #1f4e79 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        color: #ffffff !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
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
        color: #2e7d32 !important; 
        border-left: 6px solid #388e3c; 
    }
    
    /* Navigation buttons */
    .nav-button {
        background: #2e86ab;
        color: white !important;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-decoration: none;
        display: inline-block;
        margin: 0.25rem;
        border: none;
        cursor: pointer;
    }
    
    .nav-button:hover {
        background: #1f4e79;
        color: white !important;
    }
    
    /* Back button specific styling */
    .back-button {
        background: #6c757d;
        color: white !important;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        text-decoration: none;
        display: inline-block;
        margin-bottom: 1rem;
        border: none;
        cursor: pointer;
        font-weight: bold;
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
    h1, h2, h3, h5  {
        color: white !important;
        font-weight: bold !important;
    }
    h4 {
        color: black    !important;
        font-weight: bold !important;
    }

    div, span,label, p {
        color: white !important;
    }
    
    /* Chart containers */
    .chart-container {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
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
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

# Load real-time data from CSV with robust error handling
@st.cache_data
def load_realtime_data_from_csv():
    try:
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv("Combined_cleaned.csv", encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            st.error("Could not read CSV file with any encoding")
            return pd.DataFrame()
        
        # Clean column names - remove NaN columns first
        original_columns = df.columns.tolist()
        
        # Filter out NaN column names
        valid_columns = []
        column_mapping = {}
        
        for i, col in enumerate(df.columns):
            if pd.notna(col) and str(col).strip() != '':
                clean_col = str(col).strip()
                valid_columns.append(clean_col)
                column_mapping[col] = clean_col
        
        # Keep only valid columns
        df = df.loc[:, [col for col in df.columns if pd.notna(col) and str(col).strip() != '']]
        
        # Rename columns to clean names
        df = df.rename(columns=column_mapping)
        
        # Check for required columns
        required_columns = ['stationCode']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
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
        
        # Process numeric columns
        numeric_columns = ['Value', 'value', 'water_level', 'level']
        value_col = None
        for col in numeric_columns:
            if col in df.columns:
                value_col = col
                break
        
        if value_col and value_col != 'Value':
            df['Value'] = pd.to_numeric(df[value_col], errors='coerce')
        elif 'Value' in df.columns:
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        
        if 'well_depth' in df.columns:
            df['well_depth'] = pd.to_numeric(df['well_depth'], errors='coerce')
        
        # Remove rows where stationCode is NaN
        df = df.dropna(subset=['stationCode'])
        
        st.success(f"Successfully processed {len(df)} records from {df['stationCode'].nunique()} stations")
        return df
        
    except FileNotFoundError:
        st.error("Combined_cleaned.csv file not found. Please ensure the file is in the same directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading Combined_cleaned.csv: {str(e)}")
        return pd.DataFrame()

# Load full time series data for a specific station
@st.cache_data
def load_station_timeseries(station_code, full_data):
    """Load complete time series data for a specific station"""
    if full_data.empty:
        return pd.DataFrame()
    
    station_data = full_data[full_data['stationCode'] == station_code].copy()
    if not station_data.empty and 'Date_Time' in station_data.columns:
        station_data = station_data.sort_values('Date_Time')
        # Remove any duplicate timestamps
        station_data = station_data.drop_duplicates(subset=['Date_Time'], keep='last')
    
    return station_data

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

def get_latest_data_per_station(df):
    """Get the latest reading for each station"""
    if df.empty:
        return df
    
    if 'Date_Time' in df.columns:
        # Remove rows with invalid dates first
        df_with_dates = df.dropna(subset=['Date_Time'])
        
        if not df_with_dates.empty:
            # Sort by Date_Time and get the latest for each station
            df_with_dates = df_with_dates.sort_values(['stationCode', 'Date_Time'])
            latest_data = df_with_dates.groupby('stationCode').last().reset_index()
            return latest_data
        else:
            return df.drop_duplicates(subset='stationCode', keep='last')
    else:
        return df.drop_duplicates(subset='stationCode', keep='last')

def merge_station_data(metadata_df, realtime_df):
    if realtime_df.empty:
        return metadata_df
    
    # Get latest data per station
    latest_df = get_latest_data_per_station(realtime_df)
    
    # Check if required columns exist in latest_df
    available_columns = ['stationCode']
    optional_columns = ['Value', 'Date_Time', 'well_depth', 'well_aquifer_type', 'stationStatus']
    
    for col in optional_columns:
        if col in latest_df.columns:
            available_columns.append(col)
    
    merged_df = metadata_df.merge(
        latest_df[available_columns],
        on='stationCode',
        how='left'
    )
    return merged_df

def create_quick_stats_cards(df):
    """Create summary statistics cards"""
    if 'Value' in df.columns:
        total_stations = len(df)
        active_stations = len(df[df.get('stationStatus', '') == 'Active'])
        
        good_stations = len(df[df['Value'] > -10])
        moderate_stations = len(df[(df['Value'] <= -10) & (df['Value'] > -20)])
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

def calculate_water_trends(station_data):
    """Calculate trends and statistics for station data"""
    if station_data.empty or 'Value' not in station_data.columns:
        return {}
    
    # Remove NaN values
    clean_data = station_data.dropna(subset=['Value'])
    
    if len(clean_data) < 2:
        return {'trend': 'Insufficient Data'}
    
    # Calculate basic statistics
    current_level = clean_data['Value'].iloc[-1]
    avg_level = clean_data['Value'].mean()
    min_level = clean_data['Value'].min()
    max_level = clean_data['Value'].max()
    
    # Calculate trend (simple linear regression)
    x = np.arange(len(clean_data))
    y = clean_data['Value'].values
    
    # Linear regression
    if len(x) > 1:
        slope = np.polyfit(x, y, 1)[0]
        
        # Categorize trend
        if slope > 0.1:
            trend = "Rising"
            trend_color = "green"
        elif slope < -0.1:
            trend = "Declining"
            trend_color = "red"
        else:
            trend = "Stable"
            trend_color = "blue"
    else:
        trend = "No Trend"
        trend_color = "gray"
        slope = 0
    
    # Calculate monthly recharge pattern (if data spans multiple months)
    recharge_pattern = {}
    if 'Date_Time' in clean_data.columns and len(clean_data) > 10:
        clean_data['Month'] = pd.to_datetime(clean_data['Date_Time']).dt.month
        monthly_avg = clean_data.groupby('Month')['Value'].mean()
        
        # Find peak recharge months (highest water levels)
        if not monthly_avg.empty:
            peak_month = monthly_avg.idxmax()
            low_month = monthly_avg.idxmin()
            recharge_pattern = {
                'peak_month': peak_month,
                'low_month': low_month,
                'seasonal_variation': monthly_avg.max() - monthly_avg.min()
            }
    
    return {
        'current_level': current_level,
        'avg_level': avg_level,
        'min_level': min_level,
        'max_level': max_level,
        'trend': trend,
        'trend_color': trend_color,
        'trend_slope': slope,
        'data_points': len(clean_data),
        'recharge_pattern': recharge_pattern
    }

def create_station_analysis_plots(station_data, station_info):
    """Create comprehensive plots for station analysis"""
    
    if station_data.empty:
        st.warning("No time series data available for detailed analysis.")
        return
    
    # Clean data
    plot_data = station_data.dropna(subset=['Value']).copy()
    
    if plot_data.empty:
        st.warning("No valid water level data for plotting.")
        return
    
    # Ensure Date_Time is datetime
    if 'Date_Time' in plot_data.columns:
        plot_data['Date_Time'] = pd.to_datetime(plot_data['Date_Time'])
        plot_data = plot_data.sort_values('Date_Time')
    
    # 1. Main Time Series Plot
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 📈 Water Level Time Series")
    
    fig_ts = go.Figure()
    
    # Main water level line
    fig_ts.add_trace(go.Scatter(
        x=plot_data['Date_Time'],
        y=plot_data['Value'],
        mode='lines+markers',
        name='Water Level',
        line=dict(color='#2E86AB', width=2),
        marker=dict(size=4),
        hovertemplate='<b>Date:</b> %{x}<br><b>Water Level:</b> %{y:.2f} m<extra></extra>'
    ))
    
    # Add threshold lines
    y_min, y_max = plot_data['Value'].min(), plot_data['Value'].max()
    fig_ts.add_hline(y=-10, line_dash="dash", line_color="orange", 
                     annotation_text="Moderate Level (-10m)")
    fig_ts.add_hline(y=-20, line_dash="dash", line_color="red", 
                     annotation_text="Critical Level (-20m)")
    
    # Add trend line
    if len(plot_data) > 2:
        x_numeric = np.arange(len(plot_data))
        z = np.polyfit(x_numeric, plot_data['Value'], 1)
        trend_line = np.poly1d(z)(x_numeric)
        
        fig_ts.add_trace(go.Scatter(
            x=plot_data['Date_Time'],
            y=trend_line,
            mode='lines',
            name='Trend Line',
            line=dict(color='red', width=2, dash='dot'),
            hovertemplate='<b>Trend:</b> %{y:.2f} m<extra></extra>'
        ))
    
    fig_ts.update_layout(
        title=f'Water Level Trends - {station_info.get("stationName", "Unknown Station")}',
        xaxis_title='Date',
        yaxis_title='Water Level (meters)',
        hovermode='x unified',
        template='plotly_dark',
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig_ts, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create two columns for additional plots
    col1, col2 = st.columns(2)
    
    with col1:
        # 2. Monthly Pattern Analysis
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 📊 Monthly Water Level Pattern")
        
        if len(plot_data) > 30:  # Only if we have enough data
            plot_data['Month'] = plot_data['Date_Time'].dt.month_name()
            plot_data['MonthNum'] = plot_data['Date_Time'].dt.month
            
            monthly_stats = plot_data.groupby('MonthNum').agg({
                'Value': ['mean', 'min', 'max'],
                'Month': 'first'
            }).reset_index()
            
            monthly_stats.columns = ['MonthNum', 'Avg_Level', 'Min_Level', 'Max_Level', 'Month']
            
            fig_monthly = go.Figure()
            
            fig_monthly.add_trace(go.Scatter(
                x=monthly_stats['Month'],
                y=monthly_stats['Avg_Level'],
                mode='lines+markers',
                name='Average Level',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=8)
            ))
            
            fig_monthly.add_trace(go.Scatter(
                x=monthly_stats['Month'],
                y=monthly_stats['Max_Level'],
                mode='lines',
                name='Max Level',
                line=dict(color='green', width=1, dash='dot'),
                fill=None
            ))
            
            fig_monthly.add_trace(go.Scatter(
                x=monthly_stats['Month'],
                y=monthly_stats['Min_Level'],
                mode='lines',
                name='Min Level',
                line=dict(color='red', width=1, dash='dot'),
                fill='tonexty',
                fillcolor='rgba(46, 134, 171, 0.2)'
            ))
            
            fig_monthly.update_layout(
                title='Seasonal Water Level Variation',
                xaxis_title='Month',
                yaxis_title='Water Level (m)',
                template='plotly_dark',
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_monthly, use_container_width=True)
        else:
            st.info("Insufficient data for monthly pattern analysis (need >30 data points)")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # 3. Water Level Distribution
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 📊 Water Level Distribution")
        
        fig_hist = px.histogram(
            plot_data, 
            x='Value',
            nbins=20,
            title='Water Level Frequency Distribution',
            color_discrete_sequence=['#2E86AB'],
            template='plotly_dark'
        )
        
        fig_hist.add_vline(x=plot_data['Value'].mean(), line_dash="dash", 
                          line_color="yellow", annotation_text="Average")
        fig_hist.add_vline(x=-10, line_dash="dash", line_color="orange", 
                          annotation_text="Moderate")
        fig_hist.add_vline(x=-20, line_dash="dash", line_color="red", 
                          annotation_text="Critical")
        
        fig_hist.update_layout(
            xaxis_title='Water Level (m)',
            yaxis_title='Frequency',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 4. Recent Trends (Last 30 days)
    recent_data = plot_data.tail(30) if len(plot_data) >= 30 else plot_data
    
    if len(recent_data) > 1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 🔍 Recent Water Level Changes (Last 30 Readings)")
        
        fig_recent = go.Figure()
        
        fig_recent.add_trace(go.Scatter(
            x=recent_data['Date_Time'],
            y=recent_data['Value'],
            mode='lines+markers',
            name='Recent Water Level',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=6, color='#FF6B6B'),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.1)'
        ))
        
        fig_recent.update_layout(
            title='Recent Water Level Trend',
            xaxis_title='Date',
            yaxis_title='Water Level (m)',
            template='plotly_dark',
            height=350,
            showlegend=False
        )
        
        st.plotly_chart(fig_recent, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def render_station_detail_page(station_code, full_data, metadata_df):
    """Render the detailed station analysis page"""
    
    # Back button
    if st.button("← Back to Overview", key="back_button"):
        st.session_state.page = 'overview'
        st.session_state.selected_station = None
        st.rerun()
    
    # Load station data
    station_data = load_station_timeseries(station_code, full_data)
    station_info = metadata_df[metadata_df['stationCode'] == station_code].iloc[0] if not metadata_df.empty else {}
    
    # Station Header
    st.markdown(f"""
    <div class="station-header">
        <h1>🏛️ Station Detailed Analysis</h1>
        <h2>{station_info.get('stationName', 'Unknown Station')} ({station_code})</h2>
        <p><strong>Location:</strong> {station_info.get('district', 'N/A')}, {station_info.get('state', 'N/A')} | 
           <strong>Coordinates:</strong> {station_info.get('latitude', 'N/A'):.4f}, {station_info.get('longitude', 'N/A'):.4f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate trends and statistics
    trends = calculate_water_trends(station_data)
    
    if trends:
        # Key Metrics Row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if 'current_level' in trends:
                status, color = get_status_category(trends['current_level'])
                st.metric("Current Level", f"{trends['current_level']:.2f} m", 
                         delta=f"Status: {status}")
        
        with col2:
            if 'avg_level' in trends:
                st.metric("Average Level", f"{trends['avg_level']:.2f} m")
        
        with col3:
            if 'trend' in trends:
                trend_icon = "📈" if trends['trend'] == "Rising" else "📉" if trends['trend'] == "Declining" else "➡️"
                st.metric("Trend", f"{trend_icon} {trends['trend']}")
        
        with col4:
            if 'data_points' in trends:
                st.metric("Data Points", f"{trends['data_points']}")
        
        with col5:
            well_depth = station_info.get('well_depth', 'N/A')
            if pd.notna(well_depth) and well_depth != 'N/A':
                st.metric("Well Depth", f"{well_depth} m")
            else:
                st.metric("Well Depth", "Not Available")
        
        # Trend Analysis Card
        if 'recharge_pattern' in trends and trends['recharge_pattern']:
            pattern = trends['recharge_pattern']
            month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                          7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
            
            peak_month_name = month_names.get(pattern.get('peak_month', 0), 'Unknown')
            low_month_name = month_names.get(pattern.get('low_month', 0), 'Unknown')
            
            st.markdown(f"""
            <div class="alert-card alert-good">
                <h4 style="color: Green;">🔄 Recharge Pattern Analysis</h4>
                <strong>Peak Recharge Month:</strong> {peak_month_name} | 
                <strong>Lowest Level Month:</strong> {low_month_name} | 
                <strong>Seasonal Variation:</strong> {pattern.get('seasonal_variation', 0):.2f} m
            </div>
            """, unsafe_allow_html=True)
    
    # Station Information
    st.markdown("### 📋 Station Information")
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown(f"""
        **Basic Information:**
        - **Station Code:** {station_code}
        - **Station Name:** {station_info.get('stationName', 'N/A')}
        - **Station Type:** {station_info.get('stationType', 'N/A')}
        - **Agency:** {station_info.get('agencyName', 'N/A')}
        - **Status:** {station_info.get('stationStatus', 'N/A')}
        """)
    
    with info_col2:
        st.markdown(f"""
        **Technical Information:**
        - **State:** {station_info.get('state', 'N/A')}
        - **District:** {station_info.get('district', 'N/A')}
        - **Block:** {station_info.get('block', 'N/A')}
        - **Aquifer Type:** {station_info.get('well_aquifer_type', 'N/A')}
        - **Data Mode:** {station_info.get('dataAcquisitionMode', 'N/A')}
        """)
    
    # Create comprehensive plots
    create_station_analysis_plots(station_data, station_info)
    
    # Data Quality Assessment
    if not station_data.empty:
        st.markdown("### 📊 Data Quality & Statistics")
        
        quality_col1, quality_col2 = st.columns(2)
        
        with quality_col1:
            total_records = len(station_data)
            valid_records = len(station_data.dropna(subset=['Value']))
            data_completeness = (valid_records / total_records * 100) if total_records > 0 else 0
            
            st.markdown(f"""
            **Data Quality Metrics:**
            - **Total Records:** {total_records}
            - **Valid Records:** {valid_records}
            - **Data Completeness:** {data_completeness:.1f}%
            - **Date Range:** {station_data['Date_Time'].min().strftime('%Y-%m-%d')} to {station_data['Date_Time'].max().strftime('%Y-%m-%d')}
            """)
        
        with quality_col2:
            if 'Value' in station_data.columns:
                value_stats = station_data['Value'].describe()
                st.markdown(f"""
                **Statistical Summary:**
                - **Mean:** {value_stats['mean']:.2f} m
                - **Std Dev:** {value_stats['std']:.2f} m
                - **Min:** {value_stats['min']:.2f} m
                - **Max:** {value_stats['max']:.2f} m
                - **Range:** {value_stats['max'] - value_stats['min']:.2f} m
                """)
        
        # Recent data table
        st.markdown("### 📋 Recent Readings (Last 10)")
        recent_readings = station_data.tail(10)[['Date_Time', 'Value']].copy()
        if not recent_readings.empty:
            recent_readings['Status'] = recent_readings['Value'].apply(lambda x: get_status_category(x)[0])
            recent_readings['Date_Time'] = pd.to_datetime(recent_readings['Date_Time']).dt.strftime('%Y-%m-%d %H:%M')
            recent_readings.columns = ['Date & Time', 'Water Level (m)', 'Status']
            st.dataframe(recent_readings, use_container_width=True)
    
    else:
        st.warning("⚠️ No time series data available for this station")

def main():
    # Load data once at the beginning
    metadata_df = load_station_metadata()
    full_data = load_realtime_data_from_csv()  # Load all time series data
    
    if metadata_df.empty:
        st.error("Please ensure metadata[1].json file is available")
        return
    
    # Navigation logic
    if st.session_state.page == 'station_detail' and st.session_state.selected_station:
        render_station_detail_page(st.session_state.selected_station, full_data, metadata_df)
        return
    
    # Main Overview Page
    render_overview_page(metadata_df, full_data)

def render_overview_page(metadata_df, full_data):
    """Render the main overview page"""
    
    # Get latest data for overview
    latest_data = get_latest_data_per_station(full_data)
    df = merge_station_data(metadata_df, latest_data)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ Government Groundwater Monitoring Dashboard</h1>
        <p>Real-time groundwater level monitoring across Gujarat • Quick Access • Station Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with streamlined controls
    with st.sidebar:
        st.header("📊 Dashboard Controls")
        
        # Data status indicator
        if not latest_data.empty:
            st.success(f"✅ Data loaded: {len(latest_data)} stations")
            if 'Date_Time' in latest_data.columns:
                st.write(f"**Data Range:** {latest_data['Date_Time'].min()} to {latest_data['Date_Time'].max()}")
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
                
                # Add detailed analysis button
                if st.button(f"📊 View Detailed Analysis for {station['stationCode']}", 
                           key=f"detail_{station['stationCode']}"):
                    st.session_state.page = 'station_detail'
                    st.session_state.selected_station = station['stationCode']
                    st.rerun()
                
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
                        
                        # Handle date formatting
                        last_updated = station.get('Date_Time')
                        last_updated_str = "No Data"
                        
                        if pd.notna(last_updated):
                            try:
                                if isinstance(last_updated, str):
                                    if last_updated.strip() != '' and last_updated.lower() != 'nan':
                                        parsed_date = pd.to_datetime(last_updated)
                                        last_updated_str = parsed_date.strftime("%Y-%m-%d %H:%M")
                                    else:
                                        last_updated_str = "Invalid Date"
                                elif hasattr(last_updated, 'strftime'):
                                    last_updated_str = last_updated.strftime("%Y-%m-%d %H:%M")
                                else:
                                    parsed_date = pd.to_datetime(str(last_updated))
                                    last_updated_str = parsed_date.strftime("%Y-%m-%d %H:%M")
                            except (ValueError, TypeError, AttributeError):
                                last_updated_str = "Date Error"
                        
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
    
    # Interactive Map Section
    st.markdown("### 🗺️ Interactive Station Map")
    
    # Create map with appropriate zoom based on filtered data
    if not filtered_df.empty:
        center_lat = filtered_df['latitude'].mean()
        center_lon = filtered_df['longitude'].mean()
        
        # Calculate zoom level based on data spread
        lat_range = filtered_df['latitude'].max() - filtered_df['latitude'].min()
        lon_range = filtered_df['longitude'].max() - filtered_df['longitude'].min()
        zoom_level = max(5, min(10, int(10 - max(lat_range, lon_range) * 5)))
    else:
        center_lat, center_lon, zoom_level = 23.0225, 72.5714, 7  # Gujarat center
    
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
    
    # Add markers with enhanced popups
    for idx, row in filtered_df.iterrows():
        water_level = row.get('Value', None)
        status, marker_color = get_status_category(water_level)
        
        # Handle date formatting for popup
        last_updated = row.get('Date_Time', 'N/A')
        if pd.notna(last_updated):
            try:
                if isinstance(last_updated, str):
                    parsed_date = pd.to_datetime(last_updated)
                    last_updated_str = parsed_date.strftime("%Y-%m-%d %H:%M")
                else:
                    last_updated_str = last_updated.strftime("%Y-%m-%d %H:%M")
            except:
                last_updated_str = str(last_updated)
        else:
            last_updated_str = "No Data"
        
        # Enhanced popup with detailed analysis button
        if pd.notna(water_level):
            popup_html = f"""
            <div style="font-family: Arial; width: 320px; padding: 10px;">
                <h4 style="color: #2E86AB; margin: 0 0 10px 0; border-bottom: 2px solid #2E86AB; padding-bottom: 5px;">
                    {row['stationName']}
                </h4>
                <table style="width:100%; font-size:12px; margin-bottom: 10px;">
                    <tr><td><strong>Station Code:</strong></td><td>{row['stationCode']}</td></tr>
                    <tr><td><strong>Location:</strong></td><td>{row['district']}, {row['state']}</td></tr>
                    <tr><td><strong>Water Level:</strong></td><td style="color:{get_status_category(water_level)[1]}"><strong>{water_level:.2f} m</strong> ({status})</td></tr>
                    <tr><td><strong>Well Depth:</strong></td><td>{row.get('well_depth', 'N/A')} m</td></tr>
                    <tr><td><strong>Status:</strong></td><td>{row.get('stationStatus', 'Unknown')}</td></tr>
                    <tr><td><strong>Last Updated:</strong></td><td>{last_updated_str}</td></tr>
                </table>
                <div style="text-align: center; margin-top: 10px; color: #666; font-size: 11px;">
                    💡 Use 'Station Details' tab to access detailed analysis
                </div>
            </div>
            """
        else:
            popup_html = f"""
            <div style="font-family: Arial; width: 280px; padding: 10px;">
                <h4 style="color: #2E86AB; margin: 0 0 10px 0;">{row['stationName']}</h4>
                <p><strong>Station Code:</strong> {row['stationCode']}</p>
                <p><strong>Location:</strong> {row['district']}, {row['state']}</p>
                <p style="color: #ff6b6b;"><em>⚠️ No real-time data available</em></p>
            </div>
            """
        
        # Enhanced tooltip
        tooltip = f"<b>{row['stationName']}</b><br>{status}: {water_level:.2f}m<br>🖱️ Click for details" if pd.notna(water_level) else f"<b>{row['stationName']}</b><br>No data available"
        
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_html, max_width=350),
            tooltip=tooltip,
            icon=folium.Icon(color=marker_color, icon="tint", prefix="fa")
        ).add_to(marker_cluster)
    
    # Enhanced legend
    legend_html = '''
    <div style="position: fixed; top: 10px; right: 10px; width: 180px; 
                background-color: rgba(255,255,255,0.9); border:2px solid #2E86AB; z-index:9999; 
                font-size:12px; padding: 10px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.3);">
    <h5 style="margin: 0 0 8px 0; color: #2E86AB; text-align: center;">Water Level Status</h5>
    <div style="color: #333;">
        <div style="margin: 3px 0;"><i class="fa fa-circle" style="color:green"></i> <strong>Good:</strong> > -10m</div>
        <div style="margin: 3px 0;"><i class="fa fa-circle" style="color:orange"></i> <strong>Moderate:</strong> -10 to -20m</div>  
        <div style="margin: 3px 0;"><i class="fa fa-circle" style="color:red"></i> <strong>Critical:</strong> < -20m</div>
        <div style="margin: 3px 0;"><i class="fa fa-circle" style="color:gray"></i> <strong>No Data</strong></div>
        <hr style="margin: 8px 0;">
        <div style="font-size: 10px; text-align: center; color: #666;">
            🖱️ Click markers for station info<br>
            📊 Use sidebar for detailed analysis
        </div>
    </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Display map with optimal height and handle station selection
    map_data = st_folium(m, width=1400, height=650, returned_objects=["last_object_clicked"])
    
    # Quick Station Access Section
    st.markdown("### 🚀 Quick Station Access")
    st.markdown("Select any station below to view detailed analysis with time series plots and trend analysis:")
    
    # Create a more user-friendly station selector
    if not filtered_df.empty:
        # Group stations by district for better organization
        districts = sorted(filtered_df['district'].unique())
        
        selected_district_quick = st.selectbox(
            "📍 Select District for Quick Access",
            ['All Districts'] + districts,
            key="quick_district_selector"
        )
        
        if selected_district_quick == 'All Districts':
            quick_stations = filtered_df
        else:
            quick_stations = filtered_df[filtered_df['district'] == selected_district_quick]
        
        # Create columns for station buttons
        if not quick_stations.empty:
            # Sort by station name for better organization
            quick_stations_sorted = quick_stations.sort_values('stationName')
            
            # Create a grid of station buttons
            cols = st.columns(4)  # 4 columns for better layout
            
            for idx, (_, station) in enumerate(quick_stations_sorted.iterrows()):
                col_idx = idx % 4
                
                with cols[col_idx]:
                    # Determine button color based on water level status
                    water_level = station.get('Value', None)
                    if pd.notna(water_level):
                        status, status_color = get_status_category(water_level)
                        status_emoji = "🟢" if status == "Good" else "🟡" if status == "Moderate" else "🔴"
                    else:
                        status_emoji = "⚪"
                        status = "No Data"
                    
                    # Create button with status indicator
                    button_text = f"{status_emoji} {station['stationName']} ({station['district']})"
                    button_help = f"{station['stationName']} - {status}"
                    
                    if st.button(
                        button_text,
                        help=button_help,
                        key=f"quick_btn_{station['stationCode']}",
                        use_container_width=True
                    ):
                        st.session_state.page = 'station_detail'
                        st.session_state.selected_station = station['stationCode']
                        st.rerun()
            
            st.markdown("---")
            st.markdown("**Legend:** 🟢 Good Level | 🟡 Moderate Level | 🔴 Critical Level | ⚪ No Data")
    
    # Handle map clicks for navigation (if supported)
    if map_data and map_data.get("last_object_clicked"):
        clicked_data = map_data["last_object_clicked"]
        if clicked_data and "tooltip" in str(clicked_data):
            st.info("💡 **Quick Tip:** Use the 'Station Details' tab above or the Quick Station Access section below to access comprehensive station analysis.")
    
    # Quick action footer
    if not latest_data.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success(f"✅ Dashboard operational - {len(filtered_df)} stations displayed")
        
        with col2:
            if stats and stats.get('critical', 0) > 0:
                st.error(f"🚨 {stats['critical']} stations require immediate attention!")
            else:
                st.info("📊 All stations within acceptable parameters")
        
        with col3:
            st.info("💡 Use the Quick Station Access section or Station Details tab for comprehensive analysis")
    else:
        st.warning("⚠️ No real-time data available - Check Combined_cleaned.csv file")

if __name__ == "__main__":
    main()
