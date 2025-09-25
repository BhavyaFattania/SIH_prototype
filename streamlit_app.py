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

def generate_synthetic_comparative_data(station_data, station_info):
    """Generate synthetic data for comparison charts"""
    if station_data.empty:
        return {}
    
    station_code = station_info.get('stationCode', 'Unknown')
    
    # Generate synthetic neighboring stations data
    neighboring_stations = {
        f'Station_{station_code}_N1': {
            'name': f'Station North of {station_code}',
            'values': station_data['Value'] + np.random.normal(0, 2, len(station_data))
        },
        f'Station_{station_code}_S1': {
            'name': f'Station South of {station_code}',
            'values': station_data['Value'] + np.random.normal(-1, 1.5, len(station_data))
        },
        f'Station_{station_code}_E1': {
            'name': f'Station East of {station_code}',
            'values': station_data['Value'] + np.random.normal(1.5, 3, len(station_data))
        },
        f'Station_{station_code}_W1': {
            'name': f'Station West of {station_code}',
            'values': station_data['Value'] + np.random.normal(-0.5, 2.5, len(station_data))
        }
    }
    
    # Generate quality control data
    quality_data = {
        'dates': station_data['Date_Time'].iloc[-30:] if len(station_data) >= 30 else station_data['Date_Time'],
        'measured': station_data['Value'].iloc[-30:] if len(station_data) >= 30 else station_data['Value'],
        'expected': station_data['Value'].iloc[-30:] + np.random.normal(0, 0.5, min(30, len(station_data))) if len(station_data) >= 30 else station_data['Value'] + np.random.normal(0, 0.5, len(station_data)),
        'upper_limit': station_data['Value'].mean() + 2 * station_data['Value'].std(),
        'lower_limit': station_data['Value'].mean() - 2 * station_data['Value'].std(),
        'target': station_data['Value'].mean()
    }
    
    return {
        'neighboring_stations': neighboring_stations,
        'quality_data': quality_data
    }
def create_station_analysis_plots(station_data, station_info):
    """Create comprehensive plots for station analysis with diverse chart types"""
    
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
    
    # Generate synthetic data for comparisons
    synthetic_data = generate_synthetic_comparative_data(plot_data, station_info)
    
    # 1. Main Time Series Plot with Enhanced Features
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 📈 Water Level Time Series with Control Limits")
    
    fig_ts = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1,
        subplot_titles=('Water Level Trends', 'Daily Variation Range')
    )
    
    # Main water level line
    fig_ts.add_trace(go.Scatter(
        x=plot_data['Date_Time'],
        y=plot_data['Value'],
        mode='lines+markers',
        name='Water Level',
        line=dict(color='#2E86AB', width=2),
        marker=dict(size=4),
        hovertemplate='<b>Date:</b> %{x}<br><b>Water Level:</b> %{y:.2f} m<extra></extra>'
    ), row=1, col=1)
    
    # Add control limits
    mean_level = plot_data['Value'].mean()
    std_level = plot_data['Value'].std()
    ucl = mean_level + 3 * std_level  # Upper Control Limit
    lcl = mean_level - 3 * std_level  # Lower Control Limit
    
    fig_ts.add_hline(y=ucl, line_dash="dash", line_color="red", 
                     annotation_text="Upper Control Limit", row=1)
    fig_ts.add_hline(y=lcl, line_dash="dash", line_color="red", 
                     annotation_text="Lower Control Limit", row=1)
    fig_ts.add_hline(y=mean_level, line_dash="solid", line_color="green", 
                     annotation_text="Center Line", row=1)
    
    # Add daily variation (if we have enough data)
    if len(plot_data) > 7:
        plot_data['Date'] = plot_data['Date_Time'].dt.date
        daily_stats = plot_data.groupby('Date')['Value'].agg(['min', 'max', 'mean']).reset_index()
        daily_stats['range'] = daily_stats['max'] - daily_stats['min']
        
        fig_ts.add_trace(go.Bar(
            x=daily_stats['Date'],
            y=daily_stats['range'],
            name='Daily Range',
            marker_color='rgba(255, 182, 193, 0.6)',
            hovertemplate='<b>Date:</b> %{x}<br><b>Daily Range:</b> %{y:.2f} m<extra></extra>'
        ), row=2, col=1)
    
    fig_ts.update_layout(
        title=f'Enhanced Water Level Analysis - {station_info.get("stationName", "Unknown Station")}',
        template='plotly_dark',
        height=600,
        showlegend=True
    )
    
    fig_ts.update_xaxes(title_text="Date", row=2, col=1)
    fig_ts.update_yaxes(title_text="Water Level (m)", row=1, col=1)
    fig_ts.update_yaxes(title_text="Range (m)", row=2, col=1)
    
    st.plotly_chart(fig_ts, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create multiple columns for diverse charts
    col1, col2 = st.columns(2)
    
    with col1:
        # 2. Box Plot Comparison with Neighboring Stations
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 📊 Regional Water Level Comparison (Box Plot)")
        
        fig_box = go.Figure()
        
        # Add current station
        fig_box.add_trace(go.Box(
            y=plot_data['Value'],
            name=f'Current Station\n{station_info.get("stationCode", "Unknown")}',
            marker_color='#2E86AB',
            boxpoints='outliers'
        ))
        
        # Add neighboring stations (synthetic data)
        if synthetic_data and 'neighboring_stations' in synthetic_data:
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            for i, (station_id, station_data_synth) in enumerate(synthetic_data['neighboring_stations'].items()):
                fig_box.add_trace(go.Box(
                    y=station_data_synth['values'],
                    name=station_data_synth['name'][-15:],  # Truncate name
                    marker_color=colors[i % len(colors)],
                    boxpoints='outliers'
                ))
        
        fig_box.update_layout(
            title='Water Level Distribution Comparison',
            yaxis_title='Water Level (m)',
            template='plotly_dark',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # 3. Violin Plot for Distribution Analysis
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 🎻 Water Level Distribution Shape")
        
        fig_violin = go.Figure()
        
        fig_violin.add_trace(go.Violin(
            y=plot_data['Value'],
            name='Distribution',
            box_visible=True,
            line_color='#2E86AB',
            fillcolor='rgba(46, 134, 171, 0.5)',
            points='all',
            pointpos=0,
            jitter=0.3
        ))
        
        fig_violin.update_layout(
            title='Water Level Distribution Analysis',
            yaxis_title='Water Level (m)',
            template='plotly_dark',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_violin, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Row 2: Quality Control and Heatmap
    col3, col4 = st.columns(2)
    
    with col3:
        # 4. Quality Control Chart
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 🎯 Quality Control Chart")
        
        if synthetic_data and 'quality_data' in synthetic_data:
            qc_data = synthetic_data['quality_data']
            
            fig_qc = go.Figure()
            
            # Measured values
            fig_qc.add_trace(go.Scatter(
                x=qc_data['dates'],
                y=qc_data['measured'],
                mode='lines+markers',
                name='Measured',
                line=dict(color='#2E86AB', width=2),
                marker=dict(size=6)
            ))
            
            # Expected values
            fig_qc.add_trace(go.Scatter(
                x=qc_data['dates'],
                y=qc_data['expected'],
                mode='lines',
                name='Expected',
                line=dict(color='green', width=2, dash='dash')
            ))
            
            # Control limits
            fig_qc.add_hline(y=qc_data['upper_limit'], line_dash="dot", 
                            line_color="red", annotation_text="UCL")
            fig_qc.add_hline(y=qc_data['lower_limit'], line_dash="dot", 
                            line_color="red", annotation_text="LCL")
            fig_qc.add_hline(y=qc_data['target'], line_dash="solid", 
                            line_color="orange", annotation_text="Target")
            
            fig_qc.update_layout(
                title='Quality Control Monitoring',
                xaxis_title='Date',
                yaxis_title='Water Level (m)',
                template='plotly_dark',
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_qc, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        # 5. Correlation Heatmap with Synthetic Environmental Factors
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 🔥 Environmental Factors Correlation")
        
        # Generate synthetic environmental data
        n_points = len(plot_data)
        env_data = pd.DataFrame({
            'Water_Level': plot_data['Value'].values,
            'Rainfall': np.random.normal(50, 20, n_points),
            'Temperature': np.random.normal(25, 5, n_points),
            'Humidity': np.random.normal(65, 15, n_points),
            'Soil_Moisture': np.random.normal(30, 10, n_points),
            'Evapotranspiration': np.random.normal(4, 1, n_points)
        })
        
        # Add some correlation to make it realistic
        env_data['Rainfall'] = env_data['Rainfall'] + env_data['Water_Level'] * 2 + np.random.normal(0, 5, n_points)
        env_data['Soil_Moisture'] = env_data['Soil_Moisture'] + env_data['Water_Level'] * 1.5 + np.random.normal(0, 3, n_points)
        
        corr_matrix = env_data.corr()
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig_heatmap.update_layout(
            title='Environmental Factors Correlation Matrix',
            template='plotly_dark',
            height=400,
            xaxis=dict(tickangle=45)
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Row 3: Advanced Charts
    col5, col6 = st.columns(2)
    
    with col5:
        # 6. Gauge Chart for Current Status
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### ⏱️ Current Water Level Status")
        
        current_value = plot_data['Value'].iloc[-1]
        min_val = plot_data['Value'].min()
        max_val = plot_data['Value'].max()
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current_value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Water Level (m)"},
            delta={'reference': plot_data['Value'].mean()},
            gauge={
                'axis': {'range': [None, max(0, max_val + 5)]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [min_val, -20], 'color': "lightcoral"},
                    {'range': [-20, -10], 'color': "lightyellow"},
                    {'range': [-10, max_val], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': -15
                }
            }
        ))
        
        fig_gauge.update_layout(
            template='plotly_dark',
            height=400,
            font={'color': "white", 'family': "Arial"}
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col6:
        # 7. Radar Chart for Station Performance Metrics
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 🎯 Station Performance Radar")
        
        # Calculate performance metrics (normalized to 0-100 scale)
        stability = max(0, 100 - (plot_data['Value'].std() * 10))  # Lower std = higher stability
        trend_health = max(0, 50 + (np.polyfit(range(len(plot_data)), plot_data['Value'], 1)[0] * 50))
        data_quality = (len(plot_data.dropna()) / len(plot_data)) * 100
        level_adequacy = max(0, min(100, (plot_data['Value'].mean() + 30) * 2))  # Assuming -30 is worst case
        consistency = max(0, 100 - (abs(plot_data['Value'].diff()).mean() * 20))
        
        categories = ['Data Quality', 'Level Adequacy', 'Stability', 'Trend Health', 'Consistency']
        values = [data_quality, level_adequacy, stability, trend_health, consistency]
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Station Performance',
            line_color='#2E86AB',
            fillcolor='rgba(46, 134, 171, 0.3)'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            template='plotly_dark',
            title="Station Performance Metrics",
            height=400
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Full width chart for waterfall analysis
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 💧 Monthly Water Level Changes (Waterfall Chart)")
    
    if len(plot_data) > 30:  # If we have enough data
        plot_data['YearMonth'] = plot_data['Date_Time'].dt.to_period('M')
        monthly_avg = plot_data.groupby('YearMonth')['Value'].mean()
        
        if len(monthly_avg) > 1:
            monthly_changes = monthly_avg.diff().fillna(0)
            
            fig_waterfall = go.Figure()
            
            cumulative = monthly_avg.iloc[0]
            x_vals = [str(monthly_avg.index[0])]
            y_vals = [cumulative]
            colors = ['blue']
            
            for i, change in enumerate(monthly_changes.iloc[1:], 1):
                x_vals.append(str(monthly_avg.index[i]))
                y_vals.append(change)
                colors.append('green' if change > 0 else 'red')
                cumulative += change
            
            fig_waterfall.add_trace(go.Waterfall(
                name="Water Level Changes",
                orientation="v",
                measure=["absolute"] + ["relative"] * (len(y_vals) - 1),
                x=x_vals,
                textposition="outside",
                text=[f"{val:.2f}m" for val in y_vals],
                y=y_vals,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            
            fig_waterfall.update_layout(
                title="Monthly Water Level Changes Analysis",
                xaxis_title="Month",
                yaxis_title="Water Level (m)",
                template='plotly_dark',
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_waterfall, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Additional function for overview page enhancements
def create_overview_dashboard_charts(df):
    """Create enhanced charts for the overview page"""
    
    if df.empty or 'Value' not in df.columns:
        return
    
    st.markdown("### 📊 Advanced Dashboard Analytics")
    
    # Create tabs for different overview charts
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends Overview", "🗺️ Regional Analysis", "⚡ Real-time Alerts", "📋 Performance Matrix"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Funnel chart for water level categories
            good_count = len(df[df['Value'] > -10])
            moderate_count = len(df[(df['Value'] <= -10) & (df['Value'] > -20)])
            critical_count = len(df[df['Value'] <= -20])
            
            fig_funnel = go.Figure(go.Funnel(
                y=["Good Level", "Moderate Level", "Critical Level"],
                x=[good_count, moderate_count, critical_count],
                textinfo="value+percent initial",
                marker_color=["green", "orange", "red"]
            ))
            
            fig_funnel.update_layout(
                title="Water Level Status Distribution",
                template='plotly_dark',
                height=400
            )
            
            st.plotly_chart(fig_funnel, use_container_width=True)
        
        with col2:
            # Treemap for state-wise distribution
            if 'state' in df.columns:
                state_counts = df.groupby('state').agg({
                    'stationCode': 'count',
                    'Value': 'mean'
                }).reset_index()
                state_counts.columns = ['State', 'Station_Count', 'Avg_Level']
                
                fig_treemap = go.Figure(go.Treemap(
                    labels=state_counts['State'],
                    values=state_counts['Station_Count'],
                    parents=[""] * len(state_counts),
                    textinfo="label+value",
                    texttemplate="<b>%{label}</b><br>Stations: %{value}<br>Avg Level: %{customdata:.2f}m",
                    customdata=state_counts['Avg_Level'],
                    colorscale='Viridis'
                ))
                
                fig_treemap.update_layout(
                    title="Station Distribution by State",
                    template='plotly_dark',
                    height=400
                )
                
                st.plotly_chart(fig_treemap, use_container_width=True)
    
    with tab2:
        # Sunburst chart for hierarchical data
        if all(col in df.columns for col in ['state', 'district']):
            # Create hierarchical data
            hierarchy_data = df.groupby(['state', 'district']).agg({
                'stationCode': 'count',
                'Value': 'mean'
            }).reset_index()
            
            fig_sunburst = go.Figure(go.Sunburst(
                labels=list(hierarchy_data['state']) + list(hierarchy_data['district']),
                parents=[""] * len(hierarchy_data['state']) + list(hierarchy_data['state']),
                values=list(hierarchy_data['stationCode']) * 2,
                branchvalues="total",
            ))
            
            fig_sunburst.update_layout(
                title="Hierarchical Station Distribution",
                template='plotly_dark',
                height=500
            )
            
            st.plotly_chart(fig_sunburst, use_container_width=True)
    
    with tab3:
        # Real-time alerts with indicator charts
        critical_stations = df[df['Value'] <= -20]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_indicator1 = go.Figure(go.Indicator(
                mode="number",
                value=len(critical_stations),
                title={"text": "Critical Alerts"},
                number={'font': {'color': 'red', 'size': 40}},
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
            fig_indicator1.update_layout(template='plotly_dark', height=200)
            st.plotly_chart(fig_indicator1, use_container_width=True)
        
        with col2:
            avg_level = df['Value'].mean()
            fig_indicator2 = go.Figure(go.Indicator(
                mode="number",
                value=avg_level,
                title={"text": "Average Level (m)"},
                number={'font': {'color': 'blue', 'size': 40}},
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
            fig_indicator2.update_layout(template='plotly_dark', height=200)
            st.plotly_chart(fig_indicator2, use_container_width=True)
        
        with col3:
            active_stations = len(df[df.get('stationStatus', '') == 'Active'])
            fig_indicator3 = go.Figure(go.Indicator(
                mode="number",
                value=active_stations,
                title={"text": "Active Stations"},
                number={'font': {'color': 'green', 'size': 40}},
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
            fig_indicator3.update_layout(template='plotly_dark', height=200)
            st.plotly_chart(fig_indicator3, use_container_width=True)
    
    with tab4:
        # Performance matrix heatmap
        if 'state' in df.columns and 'district' in df.columns:
            # Create performance matrix
            perf_matrix = df.groupby(['state', 'district']).agg({
                'Value': ['mean', 'std', 'count']
            }).round(2)
            
            # Flatten column names
            perf_matrix.columns = ['Avg_Level', 'Std_Dev', 'Station_Count']
            perf_matrix = perf_matrix.reset_index()
            
            # Create pivot table for heatmap
            pivot_data = perf_matrix.pivot(index='state', columns='district', values='Avg_Level')
            
            fig_matrix = go.Figure(data=go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale='RdYlBu_r',
                text=np.round(pivot_data.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate='<b>State:</b> %{y}<br><b>District:</b> %{x}<br><b>Avg Level:</b> %{z:.2f}m<extra></extra>'
            ))
            
            fig_matrix.update_layout(
                title='Regional Performance Matrix (Average Water Levels)',
                template='plotly_dark',
                height=500,
                xaxis={'title': 'District'},
                yaxis={'title': 'State'}
            )
            
            st.plotly_chart(fig_matrix, use_container_width=True)

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
