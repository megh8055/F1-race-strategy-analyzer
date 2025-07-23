import streamlit as st
import pandas as pd
from general_f1_model import GeneralF1StrategyModel
import os

# Initialize the general F1 model
model_path = 'general_f1_model.pkl'
data_path = 'f1_strategy_data_partial.csv'

# Load or train the model with caching
def display_strategy_details(strategy_data, race_laps):
    """Helper function to display strategy details"""
    # Strategy metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Predicted Time", f"{strategy_data.get('predicted_race_time', 0):.0f} sec")
    with col2:
        st.metric("Pit Stops", strategy_data.get('optimal_pit_stops', 'N/A'))
    with col3:
        st.metric("Risk Level", strategy_data.get('strategy_risk', 'Unknown'))
    
    # Tyre strategy
    st.subheader("Tyre Strategy")
    tyre_col1, tyre_col2 = st.columns(2)
    
    with tyre_col1:
        st.info(f"**Primary Tyre:** {strategy_data.get('primary_tire_compound', 'N/A')}")
    with tyre_col2:
        st.info(f"**Secondary Tyre:** {strategy_data.get('secondary_tire_compound', 'N/A')}")
    
    # Strategy explanation
    if 'tire_strategy_explanation' in strategy_data:
        st.success(f"**Strategy Explanation:** {strategy_data['tire_strategy_explanation']}")
    
    # Pit windows
    st.subheader("Pit Windows")
    pit_windows = strategy_data.get('pit_windows', [])
    if pit_windows:
        pit_text = ", ".join([f"Lap {lap}" for lap in pit_windows])
        st.write(f"**Recommended Pit Stops:** {pit_text}")
    
    # Recommendations
    st.subheader("Strategic Recommendations")
    recommendations = strategy_data.get('recommendations', [])
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
    else:
        st.write("No specific recommendations available")

@st.cache_resource
def load_or_train_model():
    model = GeneralF1StrategyModel()
    
    if os.path.exists(model_path):
        model.load_model(model_path)
        if model.is_trained:
            return model, "Model loaded successfully!"
    
    # Train new model if loading failed
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
        model.train_model(data)
        model.save_model(model_path)
        return model, f"Model trained with {len(data)} records!"
    
    return model, "No training data found!"

f1_model, model_status = load_or_train_model()

# UI Configuration
st.set_page_config(
    page_title="F1 Race Strategy Analyzer", 
    page_icon="🏎️",
    layout="wide"
)

# Header
st.title("F1 Race Strategy Analyzer")
st.markdown("**Analyze and predict optimal race strategies for Formula 1 races with dual strategic options based on track, team, weather, and car performance factors.**")
st.divider()

# Display model status
if "successfully" in model_status.lower():
    st.success(model_status)
else:
    st.warning(model_status)

# Create two columns
st.header("Race Configuration")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Race Parameters")
    
    # Grand Prix selection
    track = st.selectbox(
        "Select Grand Prix", 
        [
            "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix", 
            "Japanese Grand Prix", "Chinese Grand Prix", "Miami Grand Prix", 
            "Emilia Romagna Grand Prix", "Monaco Grand Prix", "Canadian Grand Prix", 
            "Spanish Grand Prix", "Austrian Grand Prix", "British Grand Prix", 
            "Hungarian Grand Prix", "Belgian Grand Prix", "Dutch Grand Prix", 
            "Italian Grand Prix", "Azerbaijan Grand Prix", "Singapore Grand Prix", 
            "United States Grand Prix", "Mexico City Grand Prix", "Brazilian Grand Prix", 
            "Las Vegas Grand Prix", "Qatar Grand Prix", "Abu Dhabi Grand Prix"
        ],
        index=7,
        help="Select the Formula 1 Grand Prix circuit for strategy analysis"
    )
    
    # Team selection
    team = st.selectbox(
        "Select Team", 
        [
            "Red Bull Racing", "Ferrari", "Mercedes", "McLaren", 
            "Aston Martin", "Alpine", "Williams", "AlphaTauri", 
            "Alfa Romeo", "Haas"
        ],
        index=0,
        help="Select the Formula 1 team for car performance analysis"
    )
    
    starting_pos = st.number_input(
        "Starting Grid Position", 
        min_value=1, 
        max_value=20, 
        value=5, 
        step=1,
        help="Starting position on the grid (1-20)"
    )
    
    # Track conditions section
    st.subheader("Track Conditions")
    
    col_temp, col_weather = st.columns(2)
    
    with col_temp:
        track_temperature = st.selectbox(
            "Track Temperature", 
            ["cold", "cool", "optimal", "warm", "hot", "very_hot"],
            index=2,  # Default to optimal
            help="Track temperature affects tyre performance and degradation"
        )
    
    with col_weather:
        weather_condition = st.selectbox(
            "Weather Condition", 
            ["sunny", "clear", "overcast", "cloudy", "light_rain", "rain", "heavy_rain", "drizzle"],
            index=0,  # Default to sunny
            help="Weather conditions affect visibility and track grip"
        )
    
    # Rainfall intensity input
    st.subheader("Rainfall Analysis")
    rainfall_mm = st.slider(
        "Rainfall Intensity (mm per hour)", 
        min_value=0.0, 
        max_value=25.0, 
        value=0.0, 
        step=0.1,
        help="Precise rainfall measurement for optimal tyre strategy:\n• 0mm: Dry conditions\n• 0.1-1mm: Light drizzle\n• 1-3mm: Light rain (Intermediates)\n• 3-8mm: Moderate rain\n• 8-15mm: Heavy rain (Wet tyres)\n• 15mm+: Extreme conditions"
    )
    
    # Display rainfall and temperature analysis
    col_rain, col_temp_analysis = st.columns(2)
    
    with col_rain:
        if rainfall_mm == 0:
            st.success("Dry conditions - Slick tyres optimal")
        elif rainfall_mm < 1.0:
            st.info("Light drizzle - Monitor track conditions")
        elif rainfall_mm < 3.0:
            st.info("Light rain - Intermediate tyres recommended")
        elif rainfall_mm < 8.0:
            st.warning("Moderate rain - Intermediate/Wet transition")
        elif rainfall_mm < 15.0:
            st.error("Heavy rain - Wet tyres required")
        else:
            st.error("Extreme conditions - Only wet tyres viable")
    
    with col_temp_analysis:
        temp_effects = {
            "cold": "Cold track - Tyre warm-up critical",
            "cool": "Cool conditions - Good tyre life", 
            "optimal": "Optimal temperature - Balanced performance",
            "warm": "Warm track - Increased degradation",
            "hot": "Hot conditions - High tyre wear",
            "very_hot": "Very hot - Extreme tyre degradation"
        }
        st.info(temp_effects[track_temperature])

with col2:
    st.subheader("Track Information")
    
    # Display track characteristics
    track_category = f1_model.get_track_category(track)
    tire_wear = f1_model.get_tire_wear_level(track)
    team_tier = f1_model.get_team_tier(team)
    
    st.metric("Track Category", track_category.replace('_', ' ').title())
    st.metric("Tire Wear Level", tire_wear.title())
    st.metric("Team Performance Tier", f"Tier {team_tier}")
    
    # Combined weather factor from temperature and conditions
    combined_weather = f"{weather_condition}_{track_temperature}"
    weather_factor = f1_model.calculate_rainfall_factor(weather_condition, rainfall_mm)
    
    # Temperature impact on weather factor
    temp_multipliers = {
        "cold": 0.9, "cool": 0.95, "optimal": 1.0, 
        "warm": 1.1, "hot": 1.2, "very_hot": 1.3
    }
    temp_factor = temp_multipliers.get(track_temperature, 1.0)
    final_weather_factor = weather_factor * temp_factor
    
    st.metric("Weather Impact Factor", f"{final_weather_factor:.2f}")
    st.metric("Temperature Factor", f"{temp_factor:.2f}")

if st.button("🚀 Predict Optimal Strategy", type="primary"):
    input_data = {
        "track": track,
        "team": team,
        "starting_pos": starting_pos,
        "weather": weather_condition,
        "track_temperature": track_temperature,
        "rainfall_mm": rainfall_mm
    }

    try:
        strategy = f1_model.predict_strategy(input_data)

        if 'error' in strategy:
            st.error(f"Error: {strategy['error']}")
        else:
            st.success("**Dual Race Strategy Options Predicted!**")
            
            # Display team performance metrics
            st.subheader("Team Performance Analysis")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Team Performance", f"{strategy.get('team_performance_score', 0):.3f}")
            with col2:
                st.metric("Tyre Management", f"{strategy.get('tyre_management_rating', 0):.3f}")
            with col3:
                st.metric("Pit Efficiency", f"{strategy.get('pit_efficiency_rating', 0):.3f}")
            with col4:
                st.metric("Reliability", f"{strategy.get('reliability_rating', 0):.3f}")
            
            # Calculate race laps based on track
            race_laps = 60  # Default F1 race length
            if track == "Monaco Grand Prix":
                race_laps = 78
            elif track == "Belgian Grand Prix":
                race_laps = 44
            elif track == "Italian Grand Prix":
                race_laps = 53
            
            # Dual Strategy Display
            if 'primary_strategy' in strategy and 'alternative_strategy' in strategy:
                primary = strategy['primary_strategy']
                alternative = strategy['alternative_strategy']
                
                # Strategy Selection Tabs
                tab1, tab2, tab3 = st.tabs(["Primary Strategy", "Alternative Strategy", "Strategy Comparison"])
                
                with tab1:
                    st.subheader(f"{primary.get('strategy_type', 'Primary Strategy')}")
                    display_strategy_details(primary, race_laps)
                
                with tab2:
                    st.subheader(f"{alternative.get('strategy_type', 'Alternative Strategy')}")
                    display_strategy_details(alternative, race_laps)
                
                with tab3:
                    st.subheader("Strategy Comparison")
                    if 'strategy_comparison' in strategy:
                        comparison = strategy['strategy_comparison']
                        for key, value in comparison.items():
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                    else:
                        st.write("Strategy comparison not available")
            else:
                st.error("⚠️ Dual strategy information not available. Please check model output.")
                st.write("**Available strategy data:**")
                for key, value in strategy.items():
                    st.write(f"- {key}: {value}")
            
            # Rainfall information display
            if rainfall_mm > 0:
                st.info(f"🌧️ **Rainfall Intensity:** {rainfall_mm} mm/hour")
            
            # Detailed analysis in expander
            with st.expander("📊 Detailed Model Output"):
                st.json(strategy)

    except Exception as e:
        st.error(f"Prediction Failed: {e}")
