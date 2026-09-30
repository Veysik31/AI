# RAM-YUM AI Stock Replenishment System

## Overview

Machine Learning-powered demand forecasting and automated stock replenishment recommendation system using scikit-learn.

## Features

1. **Demand Prediction** - Forecast future product demand based on historical data
2. **Stock Replenishment Recommendations** - ML-driven reorder quantity suggestions
3. **Inventory Risk Analysis** - Identify stockout risks and priority levels

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- MySQL database with historical inventory data

### Setup

1. Install Python dependencies:
```bash
cd ai
pip install -r requirements.txt
```

2. Configure database connection in `app.py`:
```python
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'rms'
}
```

3. Start the AI service:
```bash
python app.py
```

The service will run on `http://localhost:5000`

## Usage

### 1. Train the Model

First time setup requires training the ML model with historical data:

**Via Dashboard:**
- Navigate to AI Stock Replenishment page
- Click "Train Model" button

**Via API:**
```bash
curl -X POST http://localhost:5000/api/train
```

### 2. Get Recommendations

**Via Dashboard:**
- Recommendations load automatically
- Shows predicted demand and reorder quantities
- Risk levels: Critical, High, Medium, Low

**Via API:**
```bash
curl http://localhost:5000/api/recommendations
```

### 3. Create Procurement Request

Click "Create Request" button on any recommendation to:
1. Pre-fill procurement request form
2. Review AI suggestion
3. Modify if needed
4. Submit for approval

## Data Requirements

Minimum data for accurate predictions:
- At least 30 days of historical sales/movement data
- Daily transaction records in `inventory_movements` table
- Movement types: 'sale', 'adjustment_out', 'transfer_out'

## ML Model

**Algorithm:** Random Forest Regressor

**Features Used:**
- Day of week
- Day of month
- Month
- Week of year
- Previous day demand (lag-1)
- Previous week demand (lag-7)

**Training:**
- 80% training, 20% testing
- 100 decision trees
- Max depth: 10
- Evaluated using MAE, RMSE, R²

## API Endpoints

### Health Check
```
GET /api/health
```

### Train Model
```
POST /api/train
Response: {
    "success": true,
    "metrics": {
        "accuracy": 85.5,
        "mae": 2.3,
        "rmse": 3.1
    }
}
```

### Get Predictions
```
POST /api/predict
Body: {
    "product_ids": [1, 2, 3],
    "days_ahead": 30
}
```

### Get Recommendations
```
GET /api/recommendations?limit=20
Response: {
    "success": true,
    "recommendations": [
        {
            "product_id": 1,
            "product_name": "Product Name",
            "current_stock": 10,
            "predicted_demand": 50,
            "recommended_quantity": 40,
            "risk_level": "high",
            "confidence": 85
        }
    ]
}
```

### Get Statistics
```
GET /api/stats
Response: {
    "success": true,
    "model_trained": true,
    "last_trained": "2026-08-16T10:30:00",
    "accuracy": 85.5
}
```

## Important Notes

### Advisory System

AI recommendations are **advisory only**. The system:
- ✅ Provides intelligent suggestions
- ✅ Calculates risk levels
- ✅ Pre-fills forms
- ❌ Does NOT auto-create POs
- ❌ Does NOT auto-confirm orders

**Workflow:**
1. AI generates recommendation
2. User reviews suggestion
3. User creates procurement request
4. Goes through normal approval process
5. Manual PO creation and confirmation

### Data Privacy

- All processing is done locally
- No external API calls
- No data transmission outside the system
- Historical data remains in MySQL database

## Troubleshooting

**"AI Service Not Available"**
- Ensure Python service is running: `python ai/app.py`
- Check port 5000 is not in use
- Verify MySQL connection in `app.py`

**"Insufficient historical data"**
- Need at least 30 days of transaction history
- Check `inventory_movements` table has data
- Ensure movement_type is correctly set

**Low accuracy**
- More historical data improves accuracy
- Ensure data quality (no missing dates, accurate quantities)
- Re-train model after accumulating more data

## Architecture

```
MySQL Database
    ↓
PHP Backend (AIService.php)
    ↓
Python Flask API (app.py)
    ↓
ML Services:
    - DataProcessor (data extraction)
    - DemandPredictor (ML predictions)
    - RiskAnalyzer (risk assessment)
    ↓
Trained Models (Random Forest)
    ↓
JSON Response
    ↓
PHP Dashboard
```

## Version

1.0.0 - Initial Release

## License

Proprietary - RAM-YUM Korean & Japanese Store
