import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataPreparation:
    def __init__(self, data_path=None):
        self.data_path = Path(data_path) if data_path else Path("../data/raw")
        
    def load_data(self):
        """Load all required datasets"""
        try:
            train = pd.read_csv(self.data_path / "train.csv", parse_dates=['date'])
            stores = pd.read_csv(self.data_path / "stores.csv")
            oil = pd.read_csv(self.data_path / "oil.csv", parse_dates=['date'])
            holidays = pd.read_csv(self.data_path / "holidays_events.csv", parse_dates=['date'])
            transactions = pd.read_csv(self.data_path / "transactions.csv", parse_dates=['date'])
            
            return train, stores, oil, holidays, transactions
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None, None, None, None, None
    
    def process_oil_prices(self, oil_data):
        """Process oil price data"""
        try:
            oil = oil_data.copy()
            # Fill missing values using forward and backward fill
            oil['dcoilwtico'] = oil['dcoilwtico'].fillna(method='ffill').fillna(method='bfill')
            return oil[['date', 'dcoilwtico']]
        except Exception as e:
            print(f"Error processing oil prices: {str(e)}")
            return None
    
    def process_transactions(self, transactions_data):
        """Process transaction data"""
        try:
            trans = transactions_data.copy()
            
            # Calculate average transactions by store and day of week
            trans['dayofweek'] = trans['date'].dt.dayofweek
            avg_trans = trans.groupby(['store_nbr', 'dayofweek'])['transactions'].mean().reset_index()
            avg_trans.columns = ['store_nbr', 'dayofweek', 'avg_transactions']
            
            return trans[['date', 'store_nbr', 'transactions']]
        except Exception as e:
            print(f"Error processing transactions: {str(e)}")
            return None
    
    def process_holidays(self, holidays_data):
        """Process holidays data"""
        try:
            holidays = holidays_data.copy()
            
            # Create binary columns for holiday types
            holiday_types = pd.get_dummies(holidays['type'], prefix='holiday')
            holidays = pd.concat([holidays[['date']], holiday_types], axis=1)
            
            # Aggregate multiple holidays on the same date
            holidays = holidays.groupby('date').sum().reset_index()
            
            return holidays
        except Exception as e:
            print(f"Error processing holidays: {str(e)}")
            return None
    
    def process_stores(self, stores_data):
        """Process store data"""
        try:
            stores = stores_data.copy()
            
            # Create categorical encodings
            stores['type_code'] = pd.Categorical(stores['type']).codes
            stores['city_code'] = pd.Categorical(stores['city']).codes
            stores['state_code'] = pd.Categorical(stores['state']).codes
            stores['cluster_code'] = stores['cluster']
            
            return stores
        except Exception as e:
            print(f"Error processing stores: {str(e)}")
            return None
    
    def prepare_training_data(self, train, stores, oil, holidays, transactions):
        """Prepare final training dataset"""
        try:
            df = train.copy()
            
            # Add store features
            df = pd.merge(df, stores, on='store_nbr', how='left')
            
            # Add oil prices
            df = pd.merge(df, oil[['date', 'dcoilwtico']], on='date', how='left')
            df['dcoilwtico'] = df['dcoilwtico'].fillna(method='ffill').fillna(method='bfill')
            
            # Add holiday information
            df = pd.merge(df, holidays, on='date', how='left')
            holiday_cols = [col for col in holidays.columns if col.startswith('holiday_')]
            for col in holiday_cols:
                df[col] = df[col].fillna(0)
            
            # Add transaction information
            df = pd.merge(df, transactions[['date', 'store_nbr', 'transactions']], 
                         on=['date', 'store_nbr'], how='left')
            
            # Fill missing transactions with store average
            df['transactions'] = df.groupby('store_nbr')['transactions'].transform(
                lambda x: x.fillna(x.mean()))
            
            # Add time-based features
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['dayofweek'] = df['date'].dt.dayofweek
            df['day_of_year'] = df['date'].dt.dayofyear
            
            # Add cyclical features
            df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
            df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
            df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
            df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
            df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
            df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
            
            # Add binary time indicators
            df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
            df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
            df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
            df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
            df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
            
            # Sort by date and store
            df = df.sort_values(['store_nbr', 'date'])
            
            return df
        except Exception as e:
            print(f"Error preparing training data: {str(e)}")
            return None
    
    def save_processed_data(self, output_dir, **data_dict):
        """Save processed data to specified directory"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            for name, df in data_dict.items():
                if df is not None:
                    df.to_csv(output_path / f"{name}_processed.csv", index=False)
            return True
        except Exception as e:
            print(f"Error saving processed data: {str(e)}")
            return False

if __name__ == "__main__":
    # Initialize data preparation
    dp = DataPreparation()
    
    # Load raw data
    train, stores, oil, holidays, transactions = dp.load_data()
    
    if all(data is not None for data in [train, stores, oil, holidays, transactions]):
        # Process individual datasets
        processed_stores = dp.process_stores(stores)
        processed_oil = dp.process_oil_prices(oil)
        processed_holidays = dp.process_holidays(holidays)
        processed_transactions = dp.process_transactions(transactions)
        
        # Prepare final training data
        final_data = dp.prepare_training_data(
            train, processed_stores, processed_oil, 
            processed_holidays, processed_transactions
        )
        
        # Save processed data
        dp.save_processed_data(
            "../data/processed",
            train=final_data,
            stores=processed_stores,
            oil=processed_oil,
            holidays=processed_holidays,
            transactions=processed_transactions
        )
