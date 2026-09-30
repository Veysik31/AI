"""
RAM-YUM AI Stock Replenishment System
Flask API for ML-based demand prediction and stock recommendations
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import mysql.connector
import numpy as np
from datetime import datetime, timedelta
import pickle
import os

# Try importing pandas, but make it optional
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("WARNING: pandas not available. Some features may be limited.")

from services.predictor import DemandPredictor
from services.risk_analyzer import RiskAnalyzer

# Only import DataProcessor if pandas is available
if PANDAS_AVAILABLE:
    from services.data_processor import DataProcessor
    from services.csv_data_processor import CsvDataProcessor

app = Flask(__name__)
CORS(app)

# Database configuration — reads from environment variables (Docker) or falls back to localhost
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'user': os.environ.get('DB_USER', 'root'),
    'password': os.environ.get('DB_PASS', ''),
    'database': os.environ.get('DB_NAME', 'rms')
}

def get_db_connection():
    """Create database connection"""
    return mysql.connector.connect(**DB_CONFIG)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'service': 'RAM-YUM AI Stock Replenishment',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/train', methods=['POST'])
def train_models():
    """Train ML models using historical data (database or CSV)"""
    
    if not PANDAS_AVAILABLE:
        return jsonify({
            'success': False,
            'message': 'Pandas not available. Please install: pip install pandas'
        }), 500
    
    try:
        print("=" * 60)
        print("TRAINING STARTED")
        print("=" * 60)
        
        # Check if request specifies data source
        data = request.get_json() if request.is_json else {}
        use_csv = data.get('use_csv', False)
        
        predictor = DemandPredictor()
        
        if use_csv:
            # Use CSV files
            print("📁 Data source: CSV files")
            csv_processor = CsvDataProcessor()
            print("✓ CSV processor initialized")
            
            training_data = csv_processor.prepare_training_data()
            
        else:
            # Use database
            print("🗄️ Data source: Database")
            conn = get_db_connection()
            print("✓ Database connected")
            
            data_processor = DataProcessor(conn)
            print("✓ Services initialized")
            
            training_data = data_processor.prepare_training_data()
            conn.close()
        
        print(f"✓ Training data shape: {training_data.shape}")
        print(f"✓ Columns: {list(training_data.columns)}")
        
        if training_data.empty:
            print("✗ Training data is EMPTY")
            return jsonify({
                'success': False,
                'message': 'Insufficient historical data for training. Generate training data first.'
            }), 400
        
        # Train model
        print("Training model...")
        metrics = predictor.train(training_data)
        print(f"✓ Training complete! Accuracy: {metrics.get('accuracy', 0):.2f}%")
        
        return jsonify({
            'success': True,
            'message': 'Models trained successfully',
            'metrics': metrics,
            'data_source': 'CSV' if use_csv else 'Database',
            'trained_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        print("=" * 60)
        print("✗ TRAINING FAILED")
        print(f"Error: {str(e)}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'message': f'Training failed: {str(e)}'
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict_demand():
    """Predict demand for specific products"""
    try:
        data = request.json
        product_ids = data.get('product_ids', [])
        days_ahead = data.get('days_ahead', 30)
        
        conn = get_db_connection()
        
        # Initialize services
        data_processor = DataProcessor(conn)
        predictor = DemandPredictor()
        risk_analyzer = RiskAnalyzer()
        
        # Load trained model
        if not predictor.load_model():
            return jsonify({
                'success': False,
                'message': 'Model not trained. Please train first.'
            }), 400
        
        results = []
        
        # Get product data
        if product_ids:
            cursor = conn.cursor(dictionary=True)
            placeholders = ','.join(['%s'] * len(product_ids))
            cursor.execute(f"""
                SELECT p.id, p.sku as product_ref, p.name as product_name, p.current_stock, 
                       p.reorder_level, p.category_id as category, p.product_image
                FROM products p
                WHERE p.id IN ({placeholders})
            """, product_ids)
            products = cursor.fetchall()
            cursor.close()
        else:
            # Get all products (exclude archived)
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT p.id, p.sku as product_ref, p.name as product_name, p.current_stock, 
                       p.reorder_level, p.category_id as category, p.product_image
                FROM products p
                WHERE p.is_active = 1
                AND (p.status = 'active' OR p.status IS NULL)
            """)
            products = cursor.fetchall()
            cursor.close()
        
        for product in products:
            # Get historical data
            history = data_processor.get_product_history(product['id'])
            
            if history.empty:
                # No historical data
                results.append({
                    'product_id': product['id'],
                    'product_ref': product['product_ref'],
                    'product_name': product['product_name'],
                    'product_image': product.get('product_image', None),
                    'current_stock': product['current_stock'],
                    'predicted_demand': 0,
                    'recommended_quantity': 0,
                    'risk_status': 'insufficient_data',
                    'confidence': 0
                })
                continue
            
            # Predict demand
            prediction = predictor.predict(history, days_ahead)
            
            # Calculate recommended reorder quantity
            recommended_qty = max(0, int(prediction['demand'] - product['current_stock']))
            
            # Analyze risk
            risk = risk_analyzer.analyze(
                current_stock=product['current_stock'],
                predicted_demand=prediction['demand'],
                reorder_level=product['reorder_level'],
                confidence=prediction['confidence']
            )
            
            results.append({
                'product_id': product['id'],
                'product_ref': product['product_ref'],
                'product_name': product['product_name'],
                'product_image': product.get('product_image', None),
                'category': product['category'],
                'current_stock': product['current_stock'],
                'reorder_level': product['reorder_level'],
                'predicted_demand': int(prediction['demand']),
                'recommended_quantity': recommended_qty,
                'risk_status': risk['status'],
                'risk_level': risk['level'],
                'confidence': round(prediction['confidence'], 2),
                'prediction_period_days': days_ahead
            })
        
        conn.close()
        
        return jsonify({
            'success': True,
            'predictions': results,
            'generated_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    """Get top priority reorder recommendations"""
    try:
        print("=" * 60)
        print("GENERATING RECOMMENDATIONS")
        print("=" * 60)
        
        limit = request.args.get('limit', 20, type=int)
        
        conn = get_db_connection()
        print("✓ Database connected")
        
        # Initialize services
        data_processor = DataProcessor(conn)
        predictor = DemandPredictor()
        risk_analyzer = RiskAnalyzer()
        print("✓ Services initialized")
        
        # Load model
        if not predictor.load_model():
            print("✗ Model not trained")
            return jsonify({
                'success': False,
                'message': 'Model not trained'
            }), 400
        
        print("✓ Model loaded")
        
        # Get all active products (exclude archived)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT p.id, p.sku as product_ref, p.name as product_name, p.current_stock, 
                   p.reorder_level, p.category_id as category, p.product_image
            FROM products p
            WHERE p.is_active = 1 
            AND (p.status = 'active' OR p.status IS NULL)
        """)
        products = cursor.fetchall()
        cursor.close()
        
        print(f"✓ Found {len(products)} products")
        
        recommendations = []
        
        for product in products:
            history = data_processor.get_product_history(product['id'])
            
            if history.empty:
                continue
            
            prediction = predictor.predict(history, 30)
            recommended_qty = max(0, int(prediction['demand'] - product['current_stock']))
            
            risk = risk_analyzer.analyze(
                current_stock=product['current_stock'],
                predicted_demand=prediction['demand'],
                reorder_level=product['reorder_level'],
                confidence=prediction['confidence']
            )
            
            # Only include high-priority items
            if risk['level'] in ['high', 'critical'] or recommended_qty > 0:
                recommendations.append({
                    'product_id': product['id'],
                    'product_ref': product['product_ref'],
                    'product_name': product['product_name'],
                    'product_image': product.get('product_image', None),
                    'category': product['category'],
                    'current_stock': product['current_stock'],
                    'predicted_demand': int(prediction['demand']),
                    'recommended_quantity': recommended_qty,
                    'risk_status': risk['status'],
                    'risk_level': risk['level'],
                    'priority_score': risk['priority_score'],
                    'confidence': round(prediction['confidence'], 2)
                })
        
        print(f"✓ Generated {len(recommendations)} recommendations")
        
        # Sort by priority score (descending)
        recommendations.sort(key=lambda x: x['priority_score'], reverse=True)
        
        conn.close()
        
        return jsonify({
            'success': True,
            'recommendations': recommendations[:limit],
            'total_items': len(recommendations),
            'generated_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        print("=" * 60)
        print("✗ RECOMMENDATIONS FAILED")
        print(f"Error: {str(e)}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'message': f'Failed to generate recommendations: {str(e)}'
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get AI system statistics"""
    try:
        predictor = DemandPredictor()
        model_info = predictor.get_model_info()
        
        # Extract accuracy from metrics if available
        accuracy = None
        if 'metrics' in model_info and 'accuracy' in model_info['metrics']:
            accuracy = model_info['metrics']['accuracy']
        elif 'accuracy' in model_info:
            accuracy = model_info['accuracy']
        
        return jsonify({
            'success': True,
            'model_trained': model_info.get('trained', False),
            'last_trained': model_info.get('last_trained'),
            'model_version': model_info.get('version', '1.0'),
            'accuracy': accuracy,
            'metrics': model_info.get('metrics', {})
        })
        
    except Exception as e:
        print(f"Stats error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs('trained_models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    print("=" * 60)
    print("RAM-YUM AI Stock Replenishment System")
    print("=" * 60)
    print(f"Starting Flask API server...")
    print(f"Database: {DB_CONFIG['database']}@{DB_CONFIG['host']}")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
