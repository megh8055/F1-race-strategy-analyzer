import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from typing import Dict, List, Any

class GeneralF1StrategyModel:
    """
    General F1 Strategy Model that works for all races with improved generalization
    """
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_feature_names = None
        
        # Simplified but comprehensive track categories
        self.track_categories = {
            'street': ['Monaco Grand Prix', 'Singapore Grand Prix', 'Azerbaijan Grand Prix', 
                      'Saudi Arabian Grand Prix', 'Australian Grand Prix', 'Miami Grand Prix', 'Las Vegas Grand Prix'],
            'permanent': ['British Grand Prix', 'Spanish Grand Prix', 'Hungarian Grand Prix', 'Belgian Grand Prix',
                         'Italian Grand Prix', 'Austrian Grand Prix', 'Dutch Grand Prix', 'Japanese Grand Prix',
                         'United States Grand Prix', 'Mexico City Grand Prix', 'Brazilian Grand Prix', 
                         'Abu Dhabi Grand Prix', 'Qatar Grand Prix', 'Bahrain Grand Prix', 'Chinese Grand Prix'],
            'semi_permanent': ['Canadian Grand Prix', 'Emilia Romagna Grand Prix']
        }
        
        # Generalized tire wear patterns
        self.tire_wear_patterns = {
            'low': ['Monaco Grand Prix', 'Italian Grand Prix', 'Las Vegas Grand Prix', 'Emilia Romagna Grand Prix'],
            'medium': ['Japanese Grand Prix', 'Chinese Grand Prix', 'Canadian Grand Prix', 'Austrian Grand Prix',
                      'Hungarian Grand Prix', 'Belgian Grand Prix', 'Dutch Grand Prix', 'Azerbaijan Grand Prix',
                      'Brazilian Grand Prix', 'Abu Dhabi Grand Prix', 'Australian Grand Prix', 'Saudi Arabian Grand Prix'],
            'high': ['Bahrain Grand Prix', 'Miami Grand Prix', 'Spanish Grand Prix', 'British Grand Prix',
                    'Singapore Grand Prix', 'United States Grand Prix', 'Mexico City Grand Prix', 'Qatar Grand Prix']
        }
        
        # Detailed team car performance characteristics (2024 season)
        self.team_performance = {
            'Red Bull Racing': {
                'tier': 1,
                'engine_power': 0.95,  # Excellent power unit
                'aerodynamic_efficiency': 0.98,  # Best aero package
                'tyre_degradation_factor': 0.85,  # Excellent tyre management
                'fuel_efficiency': 0.90,  # Good fuel consumption
                'straight_line_speed': 0.92,  # Very fast on straights
                'cornering_performance': 0.95,  # Excellent in corners
                'reliability': 0.95,  # Very reliable
                'pit_stop_efficiency': 0.98  # Best pit crew
            },
            'Ferrari': {
                'tier': 1,
                'engine_power': 0.93,
                'aerodynamic_efficiency': 0.90,
                'tyre_degradation_factor': 0.88,
                'fuel_efficiency': 0.85,
                'straight_line_speed': 0.95,  # Fastest on straights
                'cornering_performance': 0.88,
                'reliability': 0.85,  # Some reliability issues
                'pit_stop_efficiency': 0.85
            },
            'Mercedes': {
                'tier': 1,
                'engine_power': 0.90,
                'aerodynamic_efficiency': 0.88,
                'tyre_degradation_factor': 0.90,  # Good tyre management
                'fuel_efficiency': 0.88,
                'straight_line_speed': 0.88,
                'cornering_performance': 0.90,
                'reliability': 0.92,
                'pit_stop_efficiency': 0.92
            },
            'McLaren': {
                'tier': 2,
                'engine_power': 0.88,
                'aerodynamic_efficiency': 0.85,
                'tyre_degradation_factor': 0.82,
                'fuel_efficiency': 0.85,
                'straight_line_speed': 0.85,
                'cornering_performance': 0.88,
                'reliability': 0.88,
                'pit_stop_efficiency': 0.88
            },
            'Aston Martin': {
                'tier': 2,
                'engine_power': 0.85,
                'aerodynamic_efficiency': 0.82,
                'tyre_degradation_factor': 0.80,
                'fuel_efficiency': 0.82,
                'straight_line_speed': 0.82,
                'cornering_performance': 0.85,
                'reliability': 0.85,
                'pit_stop_efficiency': 0.82
            },
            'Alpine': {
                'tier': 2,
                'engine_power': 0.82,
                'aerodynamic_efficiency': 0.80,
                'tyre_degradation_factor': 0.78,
                'fuel_efficiency': 0.80,
                'straight_line_speed': 0.80,
                'cornering_performance': 0.82,
                'reliability': 0.80,
                'pit_stop_efficiency': 0.80
            },
            'Williams': {
                'tier': 3,
                'engine_power': 0.78,
                'aerodynamic_efficiency': 0.75,
                'tyre_degradation_factor': 0.75,
                'fuel_efficiency': 0.78,
                'straight_line_speed': 0.78,
                'cornering_performance': 0.75,
                'reliability': 0.82,
                'pit_stop_efficiency': 0.75
            },
            'AlphaTauri': {
                'tier': 3,
                'engine_power': 0.80,
                'aerodynamic_efficiency': 0.78,
                'tyre_degradation_factor': 0.78,
                'fuel_efficiency': 0.80,
                'straight_line_speed': 0.80,
                'cornering_performance': 0.78,
                'reliability': 0.78,
                'pit_stop_efficiency': 0.78
            },
            'Alfa Romeo': {
                'tier': 3,
                'engine_power': 0.75,
                'aerodynamic_efficiency': 0.72,
                'tyre_degradation_factor': 0.72,
                'fuel_efficiency': 0.75,
                'straight_line_speed': 0.75,
                'cornering_performance': 0.72,
                'reliability': 0.75,
                'pit_stop_efficiency': 0.72
            },
            'Haas': {
                'tier': 3,
                'engine_power': 0.78,
                'aerodynamic_efficiency': 0.70,
                'tyre_degradation_factor': 0.70,
                'fuel_efficiency': 0.72,
                'straight_line_speed': 0.75,
                'cornering_performance': 0.70,
                'reliability': 0.70,
                'pit_stop_efficiency': 0.70
            }
        }
    
    def get_track_category(self, track_name: str) -> str:
        """Get track category for any F1 circuit"""
        for category, tracks in self.track_categories.items():
            if track_name in tracks:
                return category
        return 'permanent'  # Default
    
    def get_tire_wear_level(self, track_name: str) -> str:
        """Get tire wear level for any F1 circuit"""
        for level, tracks in self.tire_wear_patterns.items():
            if track_name in tracks:
                return level
        return 'medium'  # Default
    
    def get_team_tier(self, team_name: str) -> int:
        """Get team performance tier"""
        if team_name in self.team_performance:
            return self.team_performance[team_name]['tier']
        return 3  # Default for unknown teams
    
    def get_team_performance(self, team_name: str) -> Dict[str, float]:
        """Get detailed team car performance characteristics"""
        if team_name in self.team_performance:
            return self.team_performance[team_name]
        # Return default performance for unknown teams
        return {
            'tier': 3,
            'engine_power': 0.75,
            'aerodynamic_efficiency': 0.70,
            'tyre_degradation_factor': 0.70,
            'fuel_efficiency': 0.75,
            'straight_line_speed': 0.75,
            'cornering_performance': 0.70,
            'reliability': 0.75,
            'pit_stop_efficiency': 0.70
        }
    
    def calculate_track_specific_performance(self, team_name: str, track_category: str, tire_wear: str) -> Dict[str, float]:
        """Calculate team performance factors specific to track characteristics"""
        team_perf = self.get_team_performance(team_name)
        
        # Track-specific performance adjustments
        track_factors = {
            'street': {
                'aerodynamic_importance': 0.6,  # Less important on street circuits
                'cornering_importance': 0.9,    # Very important
                'straight_line_importance': 0.7,
                'reliability_importance': 0.95  # Critical on street circuits
            },
            'permanent': {
                'aerodynamic_importance': 0.95, # Very important
                'cornering_importance': 0.8,
                'straight_line_importance': 0.9,
                'reliability_importance': 0.8
            },
            'semi_permanent': {
                'aerodynamic_importance': 0.85,
                'cornering_importance': 0.85,
                'straight_line_importance': 0.85,
                'reliability_importance': 0.85
            }
        }
        
        # Tyre wear impact on strategy
        tire_wear_factors = {
            'high': {'tyre_management_importance': 0.95, 'pit_efficiency_importance': 0.9},
            'medium': {'tyre_management_importance': 0.8, 'pit_efficiency_importance': 0.8},
            'low': {'tyre_management_importance': 0.6, 'pit_efficiency_importance': 0.7}
        }
        
        track_factor = track_factors.get(track_category, track_factors['permanent'])
        tire_factor = tire_wear_factors.get(tire_wear, tire_wear_factors['medium'])
        
        # Calculate weighted performance score
        performance_score = (
            team_perf['aerodynamic_efficiency'] * track_factor['aerodynamic_importance'] +
            team_perf['cornering_performance'] * track_factor['cornering_importance'] +
            team_perf['straight_line_speed'] * track_factor['straight_line_importance'] +
            team_perf['reliability'] * track_factor['reliability_importance'] +
            team_perf['tyre_degradation_factor'] * tire_factor['tyre_management_importance'] +
            team_perf['pit_stop_efficiency'] * tire_factor['pit_efficiency_importance']
        ) / 6.0
        
        return {
            'overall_performance': performance_score,
            'tyre_management': team_perf['tyre_degradation_factor'],
            'pit_efficiency': team_perf['pit_stop_efficiency'],
            'reliability_factor': team_perf['reliability'],
            'speed_factor': (team_perf['straight_line_speed'] + team_perf['cornering_performance']) / 2
        }
    
    def calculate_rainfall_factor(self, weather: str, rainfall_mm: float = 0) -> float:
        """Calculate weather impact factor based on conditions and rainfall"""
        if rainfall_mm > 0:
            if rainfall_mm < 1.0:
                return 0.9
            elif rainfall_mm < 3.0:
                return 0.8
            elif rainfall_mm < 8.0:
                return 0.7
            elif rainfall_mm < 15.0:
                return 0.6
            else:
                return 0.5
        
        weather_factors = {
            'dry': 1.0, 'sunny': 1.0, 'overcast': 0.95,
            'light_rain': 0.8, 'rain': 0.7, 'heavy_rain': 0.6,
            'hot': 1.2, 'cold': 0.9
        }
        return weather_factors.get(weather.lower(), 1.0)
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create generalized features for F1 strategy prediction"""
        df = data.copy()
        
        # Track characteristics
        df['track_category'] = df['track'].apply(self.get_track_category)
        df['tire_wear_level'] = df['track'].apply(self.get_tire_wear_level)
        
        # Team characteristics
        df['team_tier'] = df['team'].apply(self.get_team_tier)
        
        # Weather and rainfall
        df['rainfall_mm'] = df.get('rainfall_mm', 0)
        df['weather_factor'] = df.apply(
            lambda row: self.calculate_rainfall_factor(row['weather'], row.get('rainfall_mm', 0)), 
            axis=1
        )
        
        # Position factors
        df['grid_advantage'] = 21 - df['starting_pos']
        df['top_10_start'] = (df['starting_pos'] <= 10).astype(int)
        
        # Strategy complexity (simplified)
        complexity_map = {'street': 0.8, 'semi_permanent': 0.6, 'permanent': 0.5}
        df['strategy_complexity'] = df['track_category'].map(complexity_map)
        
        # Pit stop efficiency
        if 'num_stops' in df.columns:
            df['stops_efficiency'] = df['num_stops'] / (df['starting_pos'] + 1)
        else:
            df['stops_efficiency'] = 0
        
        return df
    
    def prepare_features(self, data: pd.DataFrame, fit_encoders: bool = True) -> tuple:
        """Prepare features for training/prediction"""
        df = data.copy()
        
        # Categorical encoding
        categorical_cols = ['track_category', 'tire_wear_level', 'weather']
        
        for col in categorical_cols:
            if col in df.columns:
                if fit_encoders:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        df[col] = df[col].astype(str)
                        known_categories = set(self.label_encoders[col].classes_)
                        df[col] = df[col].apply(
                            lambda x: x if x in known_categories else self.label_encoders[col].classes_[0]
                        )
                        df[col] = self.label_encoders[col].transform(df[col])
                    else:
                        df[col] = 0
        
        # Select features
        feature_cols = [
            'starting_pos', 'team_tier', 'weather_factor', 'rainfall_mm',
            'grid_advantage', 'top_10_start', 'strategy_complexity', 'stops_efficiency'
        ]
        
        # Add encoded categorical features
        for col in categorical_cols:
            if col in df.columns:
                feature_cols.append(col)
        
        # Filter existing columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        features = df[feature_cols].fillna(0)
        
        if fit_encoders:
            self.training_feature_names = feature_cols
            features_scaled = self.scaler.fit_transform(features)
        else:
            if self.training_feature_names:
                prediction_features = pd.DataFrame(index=features.index)
                for col in self.training_feature_names:
                    if col in features.columns:
                        prediction_features[col] = features[col]
                    else:
                        prediction_features[col] = 0
                features = prediction_features[self.training_feature_names]
            features_scaled = self.scaler.transform(features)
        
        return features_scaled, feature_cols
    
    def train_model(self, data: pd.DataFrame):
        """Train the general F1 strategy model"""
        df = self.engineer_features(data)
        X, feature_names = self.prepare_features(df, fit_encoders=True)
        
        if 'total_time' in df.columns:
            y = df['total_time'].values
        else:
            y = (df['num_stops'] * 1000 + df['starting_pos'] * 100).values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Use RandomForest for better generalization
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Cross-validation for better accuracy assessment
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
        
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"General F1 Model Training Complete!")
        print(f"Training R² Score: {train_score:.4f}")
        print(f"Testing R² Score: {test_score:.4f}")
        print(f"Cross-Validation R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.is_trained = True
        return self.model
    
    def predict_strategy(self, race_params: Dict[str, Any]) -> Dict[str, Any]:
        """Predict optimal strategy for any F1 race"""
        if not self.is_trained:
            return {"error": "Model not trained yet"}
        
        input_df = pd.DataFrame([race_params])
        input_df = self.engineer_features(input_df)
        X, _ = self.prepare_features(input_df, fit_encoders=False)
        
        predicted_time = self.model.predict(X)[0]
        strategy = self.calculate_optimal_strategy(race_params, predicted_time)
        
        return strategy
    
    def get_race_laps(self, track: str) -> int:
        """Get race lap count for specific tracks"""
        race_laps = {
            'Monaco Grand Prix': 78,
            'Belgian Grand Prix': 44,
            'Italian Grand Prix': 53,
            'British Grand Prix': 52,
            'Spanish Grand Prix': 66,
            'Hungarian Grand Prix': 70,
            'Austrian Grand Prix': 71,
            'Dutch Grand Prix': 72,
            'Singapore Grand Prix': 61,
            'Japanese Grand Prix': 53,
            'United States Grand Prix': 56,
            'Mexico City Grand Prix': 71,
            'Brazilian Grand Prix': 71,
            'Abu Dhabi Grand Prix': 58,
            'Bahrain Grand Prix': 57,
            'Saudi Arabian Grand Prix': 50,
            'Australian Grand Prix': 58,
            'Miami Grand Prix': 57,
            'Emilia Romagna Grand Prix': 63,
            'Canadian Grand Prix': 70,
            'Azerbaijan Grand Prix': 51,
            'Las Vegas Grand Prix': 50,
            'Qatar Grand Prix': 57,
            'Chinese Grand Prix': 56
        }
        return race_laps.get(track, 60)  # Default 60 laps
    
    def calculate_optimal_strategy(self, params: Dict[str, Any], predicted_time: float) -> Dict[str, Any]:
        """Calculate two strategic options for any F1 race using detailed car performance"""
        track = params.get('track', 'Unknown')
        team = params.get('team', 'Unknown')
        weather = params.get('weather', 'dry')
        starting_pos = params.get('starting_pos', 10)
        rainfall_mm = params.get('rainfall_mm', 0)
        
        # Get track characteristics
        track_category = self.get_track_category(track)
        tire_wear = self.get_tire_wear_level(track)
        
        # Get team-specific performance characteristics
        team_perf = self.get_team_performance(team)
        track_specific_perf = self.calculate_track_specific_performance(team, track_category, tire_wear)
        
        # Calculate both primary and alternative strategies
        primary_strategy = self._calculate_strategy_option(params, predicted_time, team_perf, track_specific_perf, strategy_type="primary")
        alternative_strategy = self._calculate_strategy_option(params, predicted_time, team_perf, track_specific_perf, strategy_type="alternative")
        
        return {
            'track': track,
            'team': team,
            'track_category': track_category,
            'tire_wear_level': tire_wear,
            'team_performance_score': round(track_specific_perf['overall_performance'], 3),
            'tyre_management_rating': round(team_perf['tyre_degradation_factor'], 3),
            'pit_efficiency_rating': round(team_perf['pit_stop_efficiency'], 3),
            'reliability_rating': round(team_perf['reliability'], 3),
            'primary_strategy': primary_strategy,
            'alternative_strategy': alternative_strategy,
            'strategy_comparison': self._compare_strategies(primary_strategy, alternative_strategy)
        }
    
    def _calculate_strategy_option(self, params: Dict[str, Any], predicted_time: float, team_perf: Dict[str, float], track_specific_perf: Dict[str, float], strategy_type: str) -> Dict[str, Any]:
        """Calculate a specific strategy option (primary or alternative)"""
        track = params.get('track', 'Unknown')
        team = params.get('team', 'Unknown')
        weather = params.get('weather', 'dry')
        starting_pos = params.get('starting_pos', 10)
        rainfall_mm = params.get('rainfall_mm', 0)
        
        track_category = self.get_track_category(track)
        tire_wear = self.get_tire_wear_level(track)
        
        # Strategy-specific approach
        is_aggressive = (strategy_type == "primary" and team_perf['tier'] <= 2) or (strategy_type == "alternative" and team_perf['tier'] == 3)
        
        # Determine tyre strategy based on rainfall, team characteristics, and strategy type
        if rainfall_mm > 0:
            if rainfall_mm < 1.0:
                if strategy_type == "primary":
                    primary_tire, secondary_tire = 'C3', 'Intermediate'
                    tire_note = f'Primary: Conservative start on slicks, ready for intermediates. {team} tyre management: {team_perf["tyre_degradation_factor"]:.2f}'
                else:
                    primary_tire, secondary_tire = 'Intermediate', 'C3'
                    tire_note = f'Alternative: Aggressive intermediate start to gain positions in changing conditions'
            elif rainfall_mm < 3.0:
                if strategy_type == "primary":
                    primary_tire, secondary_tire = 'Intermediate', 'C3'
                    tire_note = f'Primary: Standard intermediate strategy for light rain conditions'
                else:
                    primary_tire, secondary_tire = 'Intermediate', 'Wet'
                    tire_note = f'Alternative: Prepared for worsening conditions with wet tyre backup'
            elif rainfall_mm < 8.0:
                if strategy_type == "primary":
                    primary_tire, secondary_tire = 'Intermediate', 'Wet'
                    tire_note = f'Primary: Intermediates with wet backup for moderate rain'
                else:
                    primary_tire, secondary_tire = 'Wet', 'Intermediate'
                    tire_note = f'Alternative: Aggressive wet start to exploit grip advantage'
            elif rainfall_mm < 15.0:
                if strategy_type == "primary":
                    primary_tire, secondary_tire = 'Wet', 'Intermediate'
                    tire_note = f'Primary: Wet tyres required for heavy rain conditions'
                else:
                    primary_tire, secondary_tire = 'Wet', 'Wet'
                    tire_note = f'Alternative: Full wet strategy for maximum safety'
            else:
                primary_tire, secondary_tire = 'Wet', 'Wet'
                tire_note = f'Both strategies: Only wet tyres viable in extreme conditions'
        else:
            # Dry conditions - strategy-specific tyre choices
            tyre_management = team_perf['tyre_degradation_factor']
            
            if tire_wear == 'high':
                if strategy_type == "primary":
                    if tyre_management > 0.85:
                        primary_tire = 'C3' if starting_pos <= 10 else 'C2'
                        secondary_tire = 'C1'
                        tire_note = f'Primary: Balanced strategy leveraging {team} tyre management on high-wear track'
                    else:
                        primary_tire = 'C2'
                        secondary_tire = 'C1'
                        tire_note = f'Primary: Conservative approach due to {team} tyre concerns'
                else:  # Alternative strategy
                    if tyre_management > 0.80:
                        primary_tire = 'C4' if starting_pos <= 8 else 'C3'
                        secondary_tire = 'C2'
                        tire_note = f'Alternative: Aggressive compound choice to maximize early pace'
                    else:
                        primary_tire = 'C1'
                        secondary_tire = 'C2'
                        tire_note = f'Alternative: Ultra-conservative to ensure race finish'
            elif tire_wear == 'low':
                if strategy_type == "primary":
                    if tyre_management > 0.80:
                        primary_tire = 'C4' if starting_pos <= 5 else 'C3'
                        secondary_tire = 'C2'
                        tire_note = f'Primary: Exploit soft compounds on low-wear track'
                    else:
                        primary_tire = 'C3'
                        secondary_tire = 'C2'
                        tire_note = f'Primary: Moderate approach on low-wear track'
                else:  # Alternative strategy
                    if tyre_management > 0.75:
                        primary_tire = 'C5' if starting_pos <= 8 else 'C4'
                        secondary_tire = 'C3'
                        tire_note = f'Alternative: Maximum attack with softest compounds'
                    else:
                        primary_tire = 'C2'
                        secondary_tire = 'C1'
                        tire_note = f'Alternative: Conservative to ensure points finish'
            else:  # Medium tyre wear
                if strategy_type == "primary":
                    primary_tire = 'C3'
                    secondary_tire = 'C2'
                    tire_note = f'Primary: Standard balanced strategy for medium-wear track'
                else:
                    if is_aggressive:
                        primary_tire = 'C4'
                        secondary_tire = 'C3'
                        tire_note = f'Alternative: Aggressive soft compound strategy'
                    else:
                        primary_tire = 'C2'
                        secondary_tire = 'C1'
                        tire_note = f'Alternative: Conservative hard compound strategy'
        
        # Calculate pit stops based on team performance and strategy type
        pit_efficiency = team_perf['pit_stop_efficiency']
        reliability = team_perf['reliability']
        tyre_management = team_perf['tyre_degradation_factor']
        
        if track_category == 'street' and track == 'Monaco Grand Prix':
            # Monaco special case - overtaking difficulty
            if strategy_type == "primary":
                optimal_stops = 1 if pit_efficiency > 0.85 else 1  # Conservative on Monaco
            else:
                optimal_stops = 2 if starting_pos > 10 else 1  # Alternative: more stops from back
        elif tire_wear == 'high' or rainfall_mm > 3.0:
            if strategy_type == "primary":
                optimal_stops = 2 if tyre_management < 0.80 else (1 if starting_pos <= 6 else 2)
            else:
                if is_aggressive:
                    optimal_stops = 1 if tyre_management > 0.85 else 3  # Risky single or multiple stops
                else:
                    optimal_stops = 2  # Safe two-stop
        else:
            # Standard strategy based on team performance and strategy type
            if strategy_type == "primary":
                if pit_efficiency > 0.90 and starting_pos <= 5:
                    optimal_stops = 1  # Standard single stop for top teams
                elif tyre_management < 0.75:
                    optimal_stops = 2  # Two stops for poor tyre management
                else:
                    optimal_stops = 1 if starting_pos <= 8 else 2
            else:  # Alternative strategy
                if is_aggressive:
                    optimal_stops = 1 if pit_efficiency > 0.85 else 2  # Aggressive single or safe two
                else:
                    optimal_stops = 2 if starting_pos > 12 else 3  # Conservative multiple stops
        
        # Pit windows based on team pit efficiency and strategy type
        race_laps = self.get_race_laps(track)
        
        if optimal_stops == 1:
            # Single stop window adjusted for strategy type
            if strategy_type == "primary":
                pit_windows = [race_laps // 2] if pit_efficiency > 0.85 else [race_laps // 2 - 3]
            else:
                pit_windows = [race_laps // 2 + 5] if is_aggressive else [race_laps // 2 - 5]
        elif optimal_stops == 2:
            # Two stops adjusted for strategy type
            if strategy_type == "primary":
                pit_windows = [race_laps // 3, 2 * race_laps // 3]
            else:
                if is_aggressive:
                    pit_windows = [race_laps // 4, 3 * race_laps // 4]  # Early and late stops
                else:
                    pit_windows = [race_laps // 3 - 3, 2 * race_laps // 3 - 3]  # Earlier stops
        else:  # 3 stops
            pit_windows = [race_laps // 4, race_laps // 2, 3 * race_laps // 4]
        
        # Enhanced risk assessment including team factors and strategy type
        risk_factors = 0
        if rainfall_mm > 5.0:
            risk_factors += 2
            if reliability < 0.80:
                risk_factors += 1
        if starting_pos > 15:
            risk_factors += 1
        if track_category == 'street':
            risk_factors += 1
            if reliability < 0.85:
                risk_factors += 1
        if team_perf['tier'] == 3:
            risk_factors += 1
        
        # Strategy-specific risk adjustments
        if strategy_type == "alternative":
            if is_aggressive:
                risk_factors += 1  # Aggressive strategies are riskier
            else:
                risk_factors -= 1  # Conservative alternatives are safer
        
        # Ensure risk factors don't go below 0
        risk_factors = max(0, risk_factors)
        
        risk_level = "High" if risk_factors >= 4 else "Medium" if risk_factors >= 2 else "Low"
        
        # Strategy type label
        strategy_label = "Primary Strategy" if strategy_type == "primary" else ("Aggressive Alternative" if is_aggressive else "Conservative Alternative")
        
        return {
            'strategy_type': strategy_label,
            'predicted_race_time': round(predicted_time, 2),
            'optimal_pit_stops': optimal_stops,
            'primary_tire_compound': primary_tire,
            'secondary_tire_compound': secondary_tire,
            'tire_strategy_explanation': tire_note,
            'pit_windows': pit_windows,
            'strategy_risk': risk_level,
            'is_aggressive': is_aggressive,
            'recommendations': self.generate_strategy_specific_recommendations(params, optimal_stops, rainfall_mm, track_category, team_perf, strategy_type, is_aggressive)
        }
    
    def generate_team_specific_recommendations(self, params: Dict[str, Any], stops: int, rainfall_mm: float, track_category: str, team_perf: Dict[str, float]) -> List[str]:
        """Generate team-specific strategic recommendations"""
        recommendations = []
        team = params.get('team', 'Unknown')
        starting_pos = params.get('starting_pos', 10)
        track = params.get('track', 'Unknown')
        
        # Team performance-based recommendations
        if team_perf['tier'] == 1:  # Top tier teams
            if team_perf['tyre_degradation_factor'] > 0.90:
                recommendations.append(f"{team} has excellent tyre management - consider aggressive compound choices for track position")
            if team_perf['pit_stop_efficiency'] > 0.95:
                recommendations.append(f"{team} pit crew excellence allows for later pit windows and undercut opportunities")
            if starting_pos <= 5:
                recommendations.append(f"{team} front-row start - maximize track position with strategic tyre management")
        
        elif team_perf['tier'] == 2:  # Mid-tier teams
            recommendations.append(f"{team} should focus on consistent pace and capitalize on top team mistakes")
            if team_perf['reliability'] < 0.85:
                recommendations.append(f"Reliability concerns for {team} - consider conservative strategy to ensure points finish")
            if team_perf['pit_stop_efficiency'] > 0.85:
                recommendations.append(f"{team} good pit crew - use strategic stops to gain track position")
        
        else:  # Lower tier teams
            recommendations.append(f"{team} should prioritize points finish over aggressive strategy")
            if team_perf['tyre_degradation_factor'] < 0.75:
                recommendations.append(f"{team} tyre management weakness - plan for additional pit stops")
            recommendations.append(f"Alternative strategy recommended for {team} to capitalize on chaos or safety cars")
        
        # Track-specific recommendations based on team strengths
        if track_category == 'street':
            if team_perf['reliability'] > 0.90:
                recommendations.append(f"{team} high reliability advantage on demanding street circuit")
            else:
                recommendations.append(f"Extra caution needed for {team} on unforgiving street circuit")
            
            if team_perf['cornering_performance'] > 0.90:
                recommendations.append(f"{team} excellent cornering performance suits tight street circuit")
        
        elif track_category == 'permanent':
            if team_perf['aerodynamic_efficiency'] > 0.90:
                recommendations.append(f"{team} aerodynamic advantage on high-speed permanent circuit")
            if team_perf['straight_line_speed'] > 0.90:
                recommendations.append(f"{team} straight-line speed advantage - consider DRS zones for overtaking")
        
        # Weather-specific team recommendations
        if rainfall_mm > 0:
            if team_perf['reliability'] > 0.90:
                recommendations.append(f"{team} reliability strength crucial in wet conditions")
            else:
                recommendations.append(f"Wet weather increases risk for {team} - extra caution with electronics")
            
            if rainfall_mm > 5.0 and team_perf['tyre_degradation_factor'] > 0.85:
                recommendations.append(f"{team} tyre management skills valuable for wet-to-dry transitions")
        
        # Fuel efficiency recommendations
        if team_perf['fuel_efficiency'] < 0.80:
            recommendations.append(f"{team} fuel consumption concerns - monitor fuel saving opportunities")
        elif team_perf['fuel_efficiency'] > 0.90:
            recommendations.append(f"{team} fuel efficiency allows for aggressive engine modes longer")
        
        # Pit stop strategy based on team efficiency
        if stops == 1:
            if team_perf['pit_stop_efficiency'] > 0.90:
                recommendations.append(f"{team} single-stop strategy viable with excellent pit crew performance")
            else:
                recommendations.append(f"Single-stop risky for {team} - ensure pit crew preparation")
        else:
            if team_perf['pit_stop_efficiency'] > 0.90:
                recommendations.append(f"{team} can use pit stops tactically for undercuts and overcuts")
            else:
                recommendations.append(f"{team} focus on clean, consistent pit stops over tactical timing")
        
        return recommendations
    
    def generate_recommendations(self, params: Dict[str, Any], stops: int, rainfall_mm: float, track_category: str) -> List[str]:
        """Generate strategic recommendations"""
        recommendations = []
        
        # Rainfall recommendations
        if rainfall_mm > 0:
            if rainfall_mm < 3.0:
                recommendations.append("Monitor track conditions for drying opportunities")
            elif rainfall_mm < 8.0:
                recommendations.append("Be prepared for tire compound changes")
            else:
                recommendations.append("Focus on safety - expect reduced visibility")
        
        # Track-specific recommendations
        if track_category == 'street':
            recommendations.append("Track position crucial - avoid risky overtakes")
        elif track_category == 'permanent':
            recommendations.append("Overtaking opportunities available - consider aggressive strategy")
        
        # Position-specific recommendations
        starting_pos = params.get('starting_pos', 10)
        if starting_pos <= 5:
            recommendations.append("Protect track position and manage tires")
        else:
            recommendations.append("Look for undercut opportunities to gain positions")
        
        # Pit stop recommendations
        if stops == 1:
            recommendations.append("Focus on tire conservation for long stint")
        else:
            recommendations.append("Maintain tire temperature during pit windows")
        
        return recommendations
    
    def _compare_strategies(self, primary: Dict[str, Any], alternative: Dict[str, Any]) -> Dict[str, str]:
        """Compare primary and alternative strategies"""
        comparison = {}
        
        # Compare pit stops
        if primary['optimal_pit_stops'] < alternative['optimal_pit_stops']:
            comparison['pit_stops'] = f"Primary uses fewer stops ({primary['optimal_pit_stops']} vs {alternative['optimal_pit_stops']}) - less time in pits"
        elif primary['optimal_pit_stops'] > alternative['optimal_pit_stops']:
            comparison['pit_stops'] = f"Alternative uses fewer stops ({alternative['optimal_pit_stops']} vs {primary['optimal_pit_stops']}) - less time in pits"
        else:
            comparison['pit_stops'] = f"Both strategies use {primary['optimal_pit_stops']} pit stops"
        
        # Compare risk levels
        risk_order = {'Low': 1, 'Medium': 2, 'High': 3}
        primary_risk = risk_order.get(primary['strategy_risk'], 2)
        alt_risk = risk_order.get(alternative['strategy_risk'], 2)
        
        if primary_risk < alt_risk:
            comparison['risk'] = f"Primary strategy is safer ({primary['strategy_risk']} vs {alternative['strategy_risk']} risk)"
        elif primary_risk > alt_risk:
            comparison['risk'] = f"Alternative strategy is safer ({alternative['strategy_risk']} vs {primary['strategy_risk']} risk)"
        else:
            comparison['risk'] = f"Both strategies have {primary['strategy_risk']} risk level"
        
        # Compare tyre compounds
        if primary['primary_tire_compound'] != alternative['primary_tire_compound']:
            comparison['tyres'] = f"Primary uses {primary['primary_tire_compound']}, Alternative uses {alternative['primary_tire_compound']}"
        else:
            comparison['tyres'] = f"Both strategies start on {primary['primary_tire_compound']} tyres"
        
        # Strategy approach comparison
        if alternative['is_aggressive']:
            comparison['approach'] = "Alternative is more aggressive - higher reward potential but increased risk"
        else:
            comparison['approach'] = "Alternative is more conservative - safer but potentially slower"
        
        return comparison
    
    def generate_strategy_specific_recommendations(self, params: Dict[str, Any], stops: int, rainfall_mm: float, track_category: str, team_perf: Dict[str, float], strategy_type: str, is_aggressive: bool) -> List[str]:
        """Generate recommendations specific to the strategy type"""
        recommendations = []
        team = params.get('team', 'Unknown')
        starting_pos = params.get('starting_pos', 10)
        track = params.get('track', 'Unknown')
        
        if strategy_type == "primary":
            recommendations.append(f"Primary strategy: Balanced approach optimized for {team} strengths")
            if stops == 1:
                recommendations.append("Single-stop strategy: Focus on tyre conservation and consistent pace")
            else:
                recommendations.append(f"{stops}-stop strategy: Use pit stops for track position and fresh tyres")
        else:  # Alternative strategy
            if is_aggressive:
                recommendations.append(f"Aggressive alternative: High-risk, high-reward strategy for {team}")
                recommendations.append("Focus on early pace advantage and track position gains")
                if stops == 1:
                    recommendations.append("Risky single-stop: Requires excellent tyre management")
                else:
                    recommendations.append("Use pit stops tactically for undercuts and overcuts")
            else:
                recommendations.append(f"Conservative alternative: Safety-first approach for {team}")
                recommendations.append("Prioritize race finish and points over aggressive positioning")
                if stops >= 2:
                    recommendations.append("Multiple stops ensure fresh tyres and reduced risk")
        
        # Weather-specific strategy recommendations
        if rainfall_mm > 0:
            if strategy_type == "primary":
                recommendations.append("Weather strategy: Standard wet weather protocols")
            else:
                if is_aggressive:
                    recommendations.append("Aggressive wet strategy: Gamble on changing conditions")
                else:
                    recommendations.append("Conservative wet strategy: Minimize risk in tricky conditions")
        
        # Track-specific strategy advice
        if track_category == 'street':
            if strategy_type == "alternative" and is_aggressive:
                recommendations.append("Street circuit aggression: High reward if executed perfectly")
            else:
                recommendations.append("Street circuit caution: Avoid unnecessary risks")
        
        return recommendations
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.is_trained:
            model_data = {
                'model': self.model,
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'training_feature_names': self.training_feature_names
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"General F1 model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.label_encoders = model_data['label_encoders']
            self.scaler = model_data['scaler']
            self.training_feature_names = model_data['training_feature_names']
            self.is_trained = True
            print(f"General F1 model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_trained = False

def main():
    """Test the general F1 model"""
    model = GeneralF1StrategyModel()
    
    try:
        data = pd.read_csv('f1_strategy_data_partial.csv')
        print(f"Loaded {len(data)} records")
        
        model.train_model(data)
        
        # Test scenarios for different race types
        test_scenarios = [
            {'track': 'Monaco Grand Prix', 'team': 'Ferrari', 'starting_pos': 3, 'weather': 'dry'},
            {'track': 'Silverstone', 'team': 'Mercedes', 'starting_pos': 8, 'weather': 'light_rain', 'rainfall_mm': 2.5},
            {'track': 'Monza', 'team': 'Red Bull Racing', 'starting_pos': 1, 'weather': 'sunny'},
            {'track': 'Singapore Grand Prix', 'team': 'McLaren', 'starting_pos': 12, 'weather': 'rain', 'rainfall_mm': 6.0}
        ]
        
        print("\n=== General F1 Strategy Predictions ===")
        for scenario in test_scenarios:
            strategy = model.predict_strategy(scenario)
            print(f"\n🏁 {scenario['track']} - {scenario['team']} (P{scenario['starting_pos']})")
            print(f"   Strategy: {strategy['optimal_pit_stops']} stops, {strategy['primary_tire_compound']} → {strategy['secondary_tire_compound']}")
            print(f"   Risk: {strategy['strategy_risk']}, Category: {strategy['track_category']}")
            if strategy['rainfall_mm'] > 0:
                print(f"   Rainfall: {strategy['rainfall_mm']}mm - {strategy['tire_strategy_explanation']}")
        
        model.save_model('general_f1_model.pkl')
        
    except FileNotFoundError:
        print("Training data not found")
    
    return model

if __name__ == "__main__":
    model = main()
