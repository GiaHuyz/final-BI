import pandas as pd
import numpy as np
from pathlib import Path
from forecasting_model import ForecastingModel
from data_preparation import DataPreparation

def main():
    # Initialize data preparation
    data_path = Path("../data/raw")
    dp = DataPreparation(data_path)
    
    # Load and process data
    print("Loading data...")
    train, stores, oil, holidays, transactions = dp.load_data()
    
    if train is None:
        print("Error: Could not load data files. Please check if all required files exist in the data/raw directory.")
        return
        
    # Process individual datasets
    print("Processing data...")
    stores_processed = dp.process_stores(stores)
    oil_processed = dp.process_oil_prices(oil)
    holidays_processed = dp.process_holidays(holidays)
    
    # Prepare final training dataset
    print("Preparing final training dataset...")
    final_data = dp.prepare_training_data(train, stores_processed, oil_processed, 
                                        holidays_processed, transactions)
    
    if final_data is None:
        print("Error: Could not prepare training data.")
        return
    
    # Initialize and train model
    print("\nTraining model...")
    print("This may take several minutes depending on the data size.")
    model = ForecastingModel()
    
    try:
        metrics = model.train(final_data)
        
        # Print evaluation metrics
        print("\nTraining completed! Model evaluation metrics (average across folds):")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"MAE: {metrics['MAE']:.2f}")
        print(f"R2 Score: {metrics['R2']:.3f}")
        
        print("\nModel has been saved and is ready for use in the Streamlit app!")
        
    except Exception as e:
        print(f"\nError during model training: {str(e)}")
        return

if __name__ == "__main__":
    main()
