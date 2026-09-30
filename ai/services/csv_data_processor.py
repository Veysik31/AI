"""
CSV Data Processor for RAM-YUM AI System
Handles training data from CSV files instead of database
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

class CsvDataProcessor:
    def __init__(self, csv_directory='training_data'):
        self.csv_dir = csv_directory
    
    def prepare_training_data(self):
        """
        Prepare training data from CSV files
        Returns DataFrame with features for ML training
        """
        movements_file = os.path.join(self.csv_dir, 'stock_movements.csv')
        
        if not os.path.exists(movements_file):
            print(f"✗ CSV file not found: {movements_file}")
            return pd.DataFrame()
        
        # Load stock movements
        df = pd.read_csv(movements_file)
        
        print(f"✓ Loaded {len(df)} movements from CSV")
        
        if df.empty:
            return pd.DataFrame()
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter only STOCK-OUT movements (demand)
        df = df[df['type'] == 'STOCK-OUT'].copy()
        
        if df.empty:
            print("✗ No STOCK-OUT movements found")
            return pd.DataFrame()
        
        # Group by product and date
        daily_demand = df.groupby([
            'product_id', 
            'product_name',
            'category',
            'date'
        ])['quantity'].sum().reset_index()
        
        # Rename for consistency
        daily_demand.columns = ['product_id', 'product_name', 'category', 'date', 'demand']
        
        # Create time features
        daily_demand['day_of_week'] = daily_demand['date'].dt.dayofweek
        daily_demand['day_of_month'] = daily_demand['date'].dt.day
        daily_demand['month'] = daily_demand['date'].dt.month
        daily_demand['week_of_year'] = daily_demand['date'].dt.isocalendar().week
        
        # Add lag features (previous demand)
        for product_id in daily_demand['product_id'].unique():
            mask = daily_demand['product_id'] == product_id
            daily_demand.loc[mask, 'demand_lag_1'] = daily_demand.loc[mask, 'demand'].shift(1)
            daily_demand.loc[mask, 'demand_lag_7'] = daily_demand.loc[mask, 'demand'].shift(7)
        
        # Fill NaN values
        daily_demand = daily_demand.fillna({'demand_lag_1': 0, 'demand_lag_7': 0})
        
        print(f"✓ Prepared {len(daily_demand)} training samples")
        print(f"✓ Products: {daily_demand['product_id'].nunique()}")
        print(f"✓ Date range: {daily_demand['date'].min()} to {daily_demand['date'].max()}")
        
        return daily_demand
    
    def get_product_history(self, product_id, days=90):
        """Get historical demand for a specific product from CSV"""
        movements_file = os.path.join(self.csv_dir, 'stock_movements.csv')
        
        if not os.path.exists(movements_file):
            return pd.DataFrame()
        
        # Load movements
        df = pd.read_csv(movements_file)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter for this product and STOCK-OUT only
        df = df[(df['product_id'] == product_id) & (df['type'] == 'STOCK-OUT')].copy()
        
        if df.empty:
            return pd.DataFrame()
        
        # Group by date and sum quantities
        history = df.groupby('date')['quantity'].sum().reset_index()
        history.columns = ['date', 'demand']
        
        # Sort by date
        history = history.sort_values('date')
        
        # Add time features
        history['day_of_week'] = history['date'].dt.dayofweek
        history['day_of_month'] = history['date'].dt.day
        history['month'] = history['date'].dt.month
        history['week_of_year'] = history['date'].dt.isocalendar().week
        
        # Add lag features
        history['demand_lag_1'] = history['demand'].shift(1).fillna(0)
        history['demand_lag_7'] = history['demand'].shift(7).fillna(0)
        
        # Limit to specified days
        cutoff_date = history['date'].max() - pd.Timedelta(days=days)
        history = history[history['date'] >= cutoff_date]
        
        return history
    
    def get_daily_inventory_snapshots(self):
        """Load daily inventory snapshots from CSV"""
        inventory_file = os.path.join(self.csv_dir, 'daily_inventory.csv')
        
        if not os.path.exists(inventory_file):
            return pd.DataFrame()
        
        df = pd.read_csv(inventory_file)
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def get_supplier_performance(self):
        """Load supplier performance metrics from CSV"""
        supplier_file = os.path.join(self.csv_dir, 'supplier_performance.csv')
        
        if not os.path.exists(supplier_file):
            return pd.DataFrame()
        
        return pd.read_csv(supplier_file)
    
    def get_purchase_orders(self):
        """Load purchase order history from CSV"""
        po_file = os.path.join(self.csv_dir, 'purchase_orders.csv')
        
        if not os.path.exists(po_file):
            return pd.DataFrame()
        
        df = pd.read_csv(po_file)
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['expected_delivery'] = pd.to_datetime(df['expected_delivery'])
        df['actual_delivery'] = pd.to_datetime(df['actual_delivery'])
        
        return df
