"""
Data Processor for RAM-YUM AI System
Handles data extraction and preparation from MySQL
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataProcessor:
    def __init__(self, db_connection):
        self.conn = db_connection
    
    def prepare_training_data(self):
        """
        Prepare training data from historical inventory movements
        Returns DataFrame with features for ML training
        """
        cursor = self.conn.cursor(dictionary=True)
        
        # Get historical data from inventory_movements (outbound transactions)
        cursor.execute("""
            SELECT 
                im.product_id,
                p.sku as product_ref,
                p.name as product_name,
                p.category_id as category,
                im.quantity,
                im.movement_type,
                im.created_at as transaction_date
            FROM inventory_movements im
            JOIN products p ON im.product_id = p.id
            WHERE im.movement_type IN ('STOCK-OUT', 'ADJUSTMENT')
            AND im.created_at >= DATE_SUB(NOW(), INTERVAL 6 MONTH)
            ORDER BY im.created_at ASC
        """)
        
        movements = cursor.fetchall()
        cursor.close()
        
        if not movements:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(movements)
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        
        # Convert quantity to float (MySQL Decimal compatibility)
        df['quantity'] = df['quantity'].astype(float)
        
        # Group by product and date
        daily_demand = df.groupby([
            'product_id', 
            'product_ref',
            'product_name',
            'category',
            pd.Grouper(key='transaction_date', freq='D')
        ])['quantity'].sum().reset_index()
        
        # Rename columns
        daily_demand.columns = ['product_id', 'product_ref', 'product_name', 
                                'category', 'date', 'demand']
        
        # Create features
        daily_demand['day_of_week'] = daily_demand['date'].dt.dayofweek
        daily_demand['day_of_month'] = daily_demand['date'].dt.day
        daily_demand['month'] = daily_demand['date'].dt.month
        daily_demand['week_of_year'] = daily_demand['date'].dt.isocalendar().week
        
        # Add lag features (previous demand)
        for product_id in daily_demand['product_id'].unique():
            mask = daily_demand['product_id'] == product_id
            daily_demand.loc[mask, 'demand_lag_1'] = daily_demand.loc[mask, 'demand'].shift(1)
            daily_demand.loc[mask, 'demand_lag_7'] = daily_demand.loc[mask, 'demand'].shift(7)
        
        # Fill NaN values (fix pandas warning)
        daily_demand = daily_demand.fillna({'demand_lag_1': 0, 'demand_lag_7': 0})
        
        return daily_demand
    
    def get_product_history(self, product_id, days=90):
        """Get historical demand for a specific product"""
        cursor = self.conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT 
                DATE(im.created_at) as date,
                SUM(im.quantity) as demand
            FROM inventory_movements im
            WHERE im.product_id = %s
            AND im.movement_type IN ('STOCK-OUT', 'ADJUSTMENT')
            AND im.created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
            GROUP BY DATE(im.created_at)
            ORDER BY DATE(im.created_at) ASC
        """, (product_id, days))
        
        history = cursor.fetchall()
        cursor.close()
        
        if not history:
            return pd.DataFrame()
        
        df = pd.DataFrame(history)
        df['date'] = pd.to_datetime(df['date'])
        
        # Convert Decimal to float (MySQL returns Decimal, pandas needs float)
        df['demand'] = df['demand'].astype(float)
        
        # Add time features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Add lag features
        df['demand_lag_1'] = df['demand'].shift(1).fillna(0)
        df['demand_lag_7'] = df['demand'].shift(7).fillna(0)
        
        return df
    
    def get_current_inventory_status(self):
        """Get current inventory status for all products"""
        cursor = self.conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT 
                p.id,
                p.sku as product_ref,
                p.name as product_name,
                p.category_id as category,
                p.current_stock,
                p.reorder_level,
                p.is_active as status
            FROM products p
            WHERE p.is_active = 1
            ORDER BY p.name
        """)
        
        products = cursor.fetchall()
        cursor.close()
        
        return pd.DataFrame(products)
