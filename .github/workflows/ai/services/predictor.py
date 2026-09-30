"""
Demand Predictor using Machine Learning
Uses Random Forest Regressor for demand forecasting
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime

class DemandPredictor:
    def __init__(self):
        self.model = None
        self.model_path = 'trained_models/demand_model.pkl'
        self.info_path = 'trained_models/model_info.pkl'
        self.feature_columns = [
            'day_of_week', 'day_of_month', 'month', 'week_of_year',
            'demand_lag_1', 'demand_lag_7'
        ]
    
    def train(self, training_data):
        """Train the demand prediction model"""
        if training_data.empty:
            raise ValueError("Training data is empty")
        
        # Prepare features
        X = training_data[self.feature_columns].copy()
        y = training_data['demand'].copy()
        
        # Remove rows with NaN
        valid_indices = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(X) < 10:
            raise ValueError("Insufficient data points for training (minimum 10 required)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Calculate accuracy (inverse of MAPE, capped at 100%)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100
        accuracy = max(0, min(100, 100 - mape))
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'accuracy': float(accuracy),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        # Save model and info
        joblib.dump(self.model, self.model_path)
        
        model_info = {
            'trained': True,
            'last_trained': datetime.now().isoformat(),
            'version': '1.0',
            'metrics': metrics,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_info, self.info_path)
        
        return metrics
    
    def load_model(self):
        """Load trained model from disk"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return True
        return False
    
    def predict(self, product_history, days_ahead=30):
        """Predict demand for a specific product"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        if product_history.empty:
            return {
                'demand': 0,
                'confidence': 0
            }
        
        # Get latest data point features
        latest = product_history.iloc[-1]
        
        # Calculate average daily demand from history
        avg_daily_demand = product_history['demand'].mean()
        
        # Simple prediction: average demand * days ahead
        # For more sophisticated prediction, could use time series features
        predicted_demand = avg_daily_demand * days_ahead
        
        # Calculate confidence based on data consistency
        demand_std = product_history['demand'].std()
        demand_mean = product_history['demand'].mean()
        
        if demand_mean > 0:
            cv = demand_std / demand_mean  # Coefficient of variation
            confidence = max(0, min(100, 100 * (1 - cv)))
        else:
            confidence = 50
        
        # Use ML model for adjustment if enough features available
        if all(col in product_history.columns for col in self.feature_columns):
            try:
                X = product_history[self.feature_columns].iloc[-1:].copy()
                ml_prediction = self.model.predict(X)[0]
                
                # Combine statistical and ML predictions (weighted average)
                predicted_demand = (predicted_demand * 0.5) + (ml_prediction * days_ahead * 0.5)
            except Exception:
                pass  # Fall back to statistical prediction
        
        return {
            'demand': max(0, predicted_demand),
            'confidence': confidence
        }
    
    def get_model_info(self):
        """Get model information and metrics"""
        if os.path.exists(self.info_path):
            return joblib.load(self.info_path)
        
        return {
            'trained': False,
            'last_trained': None,
            'version': '1.0',
            'accuracy': None
        }
