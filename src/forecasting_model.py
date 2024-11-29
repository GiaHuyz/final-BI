import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import joblib
from pathlib import Path

class ForecastingModel:
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = Path(model_path) if model_path else Path("../models/lgbm_model.pkl")
        self.feature_columns = None
        self.store_family_stats = None
        
    def prepare_features(self, df, is_training=False):
        """Prepare features for the model"""
        try:
            print("Starting feature preparation...")
            df = df.copy()
            
            # Ensure required columns exist
            required_columns = ['date', 'store_nbr', 'family', 'onpromotion', 'type_code', 'cluster_code', 'dcoilwtico']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Error: Missing required columns: {missing_columns}")
                return None
            
            if is_training:
                # Calculate statistics for each store-family combination
                self.store_family_stats = df.groupby(['store_nbr', 'family']).agg({
                    'sales': ['mean', 'std', 'count']
                }).reset_index()
                self.store_family_stats.columns = ['store_nbr', 'family', 'mean_sales', 'std_sales', 'count']
                
                # Filter out store-family combinations with no sales or very few sales
                valid_combinations = self.store_family_stats[
                    (self.store_family_stats['mean_sales'] > 0) & 
                    (self.store_family_stats['count'] >= 30)  # At least 30 days of data
                ][['store_nbr', 'family']]
                
                # Merge to keep only valid combinations
                df = pd.merge(df, valid_combinations, on=['store_nbr', 'family'], how='inner')
            
            # Time-based features
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['dayofweek'] = df['date'].dt.dayofweek
            df['day_of_year'] = df['date'].dt.dayofyear
            df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
            df['quarter'] = df['date'].dt.quarter
            df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
            df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
            df['days_in_month'] = df['date'].dt.days_in_month
            
            # Ensure sales column exists for non-training data
            if 'sales' not in df.columns:
                df['sales'] = 0
                
            # Lag features for each store-family combination
            for lag in [1, 7, 14, 30]:
                df[f'sales_lag_{lag}'] = df.groupby(['store_nbr', 'family'])['sales'].shift(lag)
                
            # Rolling statistics with forward fill for missing values
            for window in [7, 14, 30]:
                # Rolling mean
                df[f'sales_rolling_mean_{window}'] = df.groupby(['store_nbr', 'family'])['sales'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                ).fillna(method='ffill')
                
                # Rolling std
                df[f'sales_rolling_std_{window}'] = df.groupby(['store_nbr', 'family'])['sales'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                ).fillna(method='ffill')
                
                # Rolling max
                df[f'sales_rolling_max_{window}'] = df.groupby(['store_nbr', 'family'])['sales'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                ).fillna(method='ffill')
            
            # Promotion features
            df['onpromotion'] = df['onpromotion'].fillna(0)
            # Historical promotion effectiveness
            df['promo_rolling_7'] = df.groupby(['store_nbr', 'family'])['onpromotion'].transform(
                lambda x: x.rolling(window=7, min_periods=1).mean()
            ).fillna(0)
            
            # Calculate promotion effectiveness more safely
            def calculate_promo_ratio(group):
                promo_sales = group[group['onpromotion'] == 1]['sales'].mean()
                non_promo_sales = group[group['onpromotion'] == 0]['sales'].mean()
                if pd.isna(promo_sales) or pd.isna(non_promo_sales) or non_promo_sales == 0:
                    return 1.0
                return promo_sales / non_promo_sales
            
            # Calculate promotion sales ratio for each store-family combination
            promo_ratios = df.groupby(['store_nbr', 'family']).apply(calculate_promo_ratio)
            # Convert to dictionary for faster lookup
            promo_ratio_dict = promo_ratios.to_dict()
            
            # Assign ratios back to dataframe
            df['promo_sales_ratio'] = df.apply(
                lambda row: promo_ratio_dict.get((row['store_nbr'], row['family']), 1.0),
                axis=1
            )
            
            # Add historical sales patterns
            if is_training:
                # Calculate average sales by day of week for each store-family
                dow_avg = df.groupby(['store_nbr', 'family', 'dayofweek'])['sales'].mean().reset_index()
                self.dow_patterns = dow_avg
            
            if hasattr(self, 'dow_patterns'):
                df = pd.merge(
                    df,
                    self.dow_patterns,
                    on=['store_nbr', 'family', 'dayofweek'],
                    how='left',
                    suffixes=('', '_dow_avg')
                )
            else:
                df['sales_dow_avg'] = df.groupby(['store_nbr', 'family', 'dayofweek'])['sales'].transform('mean')
            
            # Fill missing values with 0 for prediction features
            feature_columns = (
                ['year', 'month', 'day', 'dayofweek', 'day_of_year', 'is_weekend', 'quarter', 'is_month_start', 'is_month_end', 'days_in_month'] +
                [f'sales_lag_{lag}' for lag in [1, 7, 14, 30]] +
                [f'sales_rolling_mean_{window}' for window in [7, 14, 30]] +
                [f'sales_rolling_std_{window}' for window in [7, 14, 30]] +
                [f'sales_rolling_max_{window}' for window in [7, 14, 30]] +
                ['onpromotion', 'promo_rolling_7', 'promo_sales_ratio', 'sales_dow_avg'] +
                ['type_code', 'cluster_code'] +
                [col for col in df.columns if col.startswith('holiday_')] +
                ['dcoilwtico']
            )
            
            # Store feature columns if training
            if is_training:
                self.feature_columns = feature_columns
            
            # Fill missing values with 0 for prediction
            for col in feature_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
                else:
                    print(f"Warning: Missing feature column {col}")
                    df[col] = 0
            
            print("Feature preparation completed successfully")
            return df
            
        except Exception as e:
            print(f"Error during feature preparation: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def train(self, df):
        """Train the forecasting model"""
        print("Preparing features...")
        df = self.prepare_features(df, is_training=True)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Split features and target
        X = df[self.feature_columns]
        y = df['sales']
        
        print("Training model...")
        self.model = LGBMRegressor(
            objective='regression',
            boosting_type='gbdt',
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=8,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            min_child_weight=1e-5,
            reg_alpha=0.1,
            reg_lambda=0.1,
        )
        
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        metrics_list = []
        
        print("Performing cross-validation...")
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            print(f"Training fold {fold}/5...")
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='rmse',
                callbacks=[
                    early_stopping(stopping_rounds=50),
                    log_evaluation(period=100)
                ]
            )
            
            # Calculate metrics for this fold
            val_pred = self.model.predict(X_val)
            # Apply minimum sales threshold based on historical data
            store_family_min = df.groupby(['store_nbr', 'family'])['sales'].min().reset_index()
            val_data = df.iloc[val_idx].reset_index(drop=True)
            val_data['predicted'] = val_pred
            val_data = pd.merge(
                val_data,
                store_family_min,
                on=['store_nbr', 'family'],
                suffixes=('', '_min')
            )
            val_pred = np.maximum(val_data['predicted'], val_data['sales_min'])
            
            fold_metrics = {
                'RMSE': np.sqrt(mean_squared_error(y_val, val_pred)),
                'MAE': mean_absolute_error(y_val, val_pred),
                'R2': r2_score(y_val, val_pred)
            }
            metrics_list.append(fold_metrics)
            print(f"Fold {fold} metrics:", fold_metrics)
        
        # Save the model and metadata
        print("Saving model...")
        self.save_model()
        
        # Return average metrics across all folds
        avg_metrics = {
            metric: np.mean([fold[metric] for fold in metrics_list])
            for metric in metrics_list[0].keys()
        }
        print("Average metrics across all folds:", avg_metrics)
        return avg_metrics
    
    def predict(self, df):
        """Make predictions using the trained model"""
        try:
            if self.model is None:
                print("Loading model...")
                self.load_model()
                if self.model is None:
                    print("Error: Failed to load model")
                    return None
            
            print("Preparing features...")
            df = self.prepare_features(df, is_training=False)
            if df is None or len(df) == 0:
                print("Error: No data after feature preparation")
                return None
            
            print(f"Feature columns: {self.feature_columns}")
            print(f"DataFrame columns: {df.columns.tolist()}")
            
            # Ensure all feature columns exist
            missing_cols = [col for col in self.feature_columns if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns: {missing_cols}")
                for col in missing_cols:
                    df[col] = 0
            
            X = df[self.feature_columns]
            
            # Verify input data
            print("\nInput Data Verification:")
            print("Missing values in features:", X.isna().sum().sum())
            print("Infinite values in features:", np.isinf(X.values).sum())
            
            # Make predictions
            print("Making predictions...")
            print(f"Input shape: {X.shape}")
            predictions = self.model.predict(X)
            predictions = np.array(predictions, dtype=np.float64)  # Ensure float64 type
            
            print("\nPrediction Statistics:")
            print(f"Predictions dtype: {predictions.dtype}")
            print(f"Raw predictions shape: {predictions.shape}")
            print(f"Raw predictions - Min: {predictions.min():.2f}, Max: {predictions.max():.2f}, Mean: {predictions.mean():.2f}")
            print(f"NaN values in predictions: {np.isnan(predictions).sum()}")
            print(f"Infinite values in predictions: {np.isinf(predictions).sum()}")
            
            # Apply minimum sales threshold based on historical data
            if hasattr(self, 'store_family_stats'):
                df_pred = df[['store_nbr', 'family']].copy()
                df_pred['predicted'] = predictions
                
                # Merge with historical stats
                df_pred = pd.merge(
                    df_pred,
                    self.store_family_stats,
                    on=['store_nbr', 'family'],
                    how='left'
                )
                
                print("\nHistorical Stats:")
                print("Missing values in stats:")
                print(df_pred[['mean_sales', 'std_sales']].isna().sum())
                print("\nStats summary:")
                print(df_pred[['mean_sales', 'std_sales']].describe())
                
                # Fill missing stats with global averages
                df_pred['mean_sales'] = df_pred['mean_sales'].fillna(df_pred['mean_sales'].mean())
                df_pred['std_sales'] = df_pred['std_sales'].fillna(df_pred['std_sales'].mean())
                
                # Calculate weighted prediction
                weighted_predictions = (
                    0.7 * predictions + 
                    0.3 * df_pred['mean_sales'].values
                )
                
                print("\nWeighted Predictions:")
                print(f"Shape: {weighted_predictions.shape}")
                print(f"Min: {weighted_predictions.min():.2f}, Max: {weighted_predictions.max():.2f}, Mean: {weighted_predictions.mean():.2f}")
                print(f"NaN values: {np.isnan(weighted_predictions).sum()}")
                
                # Apply thresholds
                min_threshold = df_pred['mean_sales'].values * 0.3
                max_threshold = df_pred['mean_sales'].values * 2.0 + 2 * df_pred['std_sales'].values
                
                # Apply thresholds and ensure non-negative
                predictions = np.clip(weighted_predictions, min_threshold, max_threshold)
                predictions = np.maximum(predictions, 0)
                
                print("\nFinal Predictions:")
                print(f"Shape: {predictions.shape}")
                print(f"Min: {predictions.min():.2f}, Max: {predictions.max():.2f}, Mean: {predictions.mean():.2f}")
                print(f"NaN values: {np.isnan(predictions).sum()}")
                print(f"Infinite values: {np.isinf(predictions).sum()}")
            
            # Final verification
            if np.isnan(predictions).any() or np.isinf(predictions).any():
                print("\nWarning: Invalid values in final predictions")
                predictions = np.nan_to_num(predictions, nan=0.0, posinf=None, neginf=0.0)
                print("After fixing:")
                print(f"Min: {predictions.min():.2f}, Max: {predictions.max():.2f}, Mean: {predictions.mean():.2f}")
            
            return predictions.astype(np.float64)  # Ensure float64 type
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_model(self):
        """Save the trained model and metadata"""
        try:
            print("Saving model...")
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'model': self.model,
                'store_family_stats': self.store_family_stats,
                'dow_patterns': self.dow_patterns if hasattr(self, 'dow_patterns') else None,
                'feature_columns': self.feature_columns
            }
            
            joblib.dump(model_data, self.model_path)
            print(f"Model saved successfully to {self.model_path}")
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def load_model(self):
        """Load a trained model and metadata"""
        try:
            print(f"Loading model from {self.model_path}")
            if not self.model_path.exists():
                print("Error: Model file not found")
                return None
                
            model_data = joblib.load(self.model_path)
            
            # Load model
            self.model = model_data.get('model')
            if self.model is None:
                print("Error: Invalid model data - model not found")
                return None
                
            # Load metadata
            self.store_family_stats = model_data.get('store_family_stats')
            if model_data.get('dow_patterns') is not None:
                self.dow_patterns = model_data['dow_patterns']
            self.feature_columns = model_data.get('feature_columns')
            
            if self.feature_columns is None:
                print("Error: Invalid model data - feature columns not found")
                return None
                
            print("Model loaded successfully")
            print(f"Feature columns: {self.feature_columns}")
            return self.model
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            return None