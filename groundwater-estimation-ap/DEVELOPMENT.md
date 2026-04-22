# Development Guide

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Place your data files in the appropriate directories:

```
data/
├── raw/
│   ├── piezometers.csv          # 1,800 piezometer locations
│   ├── villages.csv              # 18,000 village locations
│   ├── rainfall.csv              # Rainfall data by village
│   ├── soil_permeability.csv    # Soil permeability index
│   └── elevation.csv             # Elevation data
├── processed/                    # Auto-generated during training
└── external/                     # Optional geospatial data
```

**CSV Column Requirements:**

- `piezometers.csv`: `piezometer_id, latitude, longitude, depth, timestamp`
- `villages.csv`: `village_id, latitude, longitude`
- `rainfall.csv`: `village_id, rainfall`
- `soil_permeability.csv`: `village_id, soil_permeability`
- `elevation.csv`: `village_id, elevation`

### 3. Training the Model

```bash
# Run the complete training pipeline
python train.py

# Check logs
tail -f logs/groundwater_system.log
```

### 4. Start the API

```bash
# Development mode (with auto-reload)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 5. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Get prediction for village 1
curl -X POST http://localhost:8000/get_village_data/1

# Interactive API docs
# Visit: http://localhost:8000/docs
```

## Code Examples

### Example 1: Train a Custom Model

```python
from src.preprocessing.spatial_join import load_piezometer_data, load_village_data, merge_datasets
from src.modeling.xgb_regressor import GroundwaterEstimator
import pandas as pd

# Load data
villages = load_village_data('data/raw/villages.csv')
piezometers = load_piezometer_data('data/raw/piezometers.csv')

# Merge (assuming feature CSVs are already loaded)
processed = merge_datasets(villages, piezometers, rainfall_df, soil_df, elevation_df)

# Train
X = processed[['rainfall', 'soil_permeability', 'elevation', 'distance_to_nearest_piezometer']]
y = processed['water_level']

estimator = GroundwaterEstimator()
metrics = estimator.train(X, y)

print(f"MAPE: {metrics['mape']:.4f}")
estimator.save_model('models/custom_model.pkl')
```

### Example 2: Detect Anomalies in Time-Series

```python
from src.inference.anomaly_detection import AnomalyDetector
import pandas as pd

# Load time-series data
df = pd.read_csv('data/processed/timeseries.csv')
df['date'] = pd.to_datetime(df['date'])

# Detect anomalies
detector = AnomalyDetector(critical_std_threshold=2.0)
anomaly_json = detector.generate_anomaly_json(df, months=3)

# Save report
detector.save_anomaly_report(anomaly_json, 'data/processed/anomalies.json')
```

### Example 3: Batch Predictions

```python
import pandas as pd
from src.modeling.xgb_regressor import GroundwaterEstimator

# Load model
estimator = GroundwaterEstimator()
estimator.load_model('models/groundwater_model.pkl')

# Load village features
df = pd.read_csv('data/processed/village_features.csv')

# Make predictions
X = df[['rainfall', 'soil_permeability', 'elevation', 'distance_to_nearest_piezometer']]
predictions = estimator.predict(X)

# Create output
results = pd.DataFrame({
    'village_id': df['village_id'],
    'predicted_water_level': predictions
})

results.to_csv('data/processed/predictions.csv', index=False)
```

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Suite

```bash
# Model accuracy tests
pytest tests/test_model_accuracy.py -v

# API endpoint tests  
pytest tests/test_api_endpoints.py -v
```

### Generate Coverage Report

```bash
pytest tests/ --cov=src --cov=api --cov-report=html --cov-report=term
```

Open `htmlcov/index.html` in browser to view coverage.

### Test with Sample Data

```bash
# Create test data
python tests/create_sample_data.py

# Run tests
pytest tests/ -v
```

## Debugging

### Enable Debug Logging

Edit `config/settings.yaml`:
```yaml
logging:
  level: "DEBUG"
```

### Check Model Performance

```python
from src.modeling.xgb_regressor import GroundwaterEstimator
import pandas as pd

estimator = GroundwaterEstimator()
estimator.load_model('models/groundwater_model.pkl')

# Check feature importance
print(estimator.feature_importance_)

# Check training history
print(estimator.training_history_)
```

### Validate Data Quality

```bash
# Check for missing values
python -c "
import pandas as pd
df = pd.read_csv('data/raw/villages.csv')
print(df.isnull().sum())
print(f'Total records: {len(df)}')
"
```

## Performance Optimization

### 1. Feature Engineering
- Consider polynomial features for non-linear relationships
- Add temporal features if time-series data available
- Normalize/scale features appropriately

### 2. Model Tuning
Edit `config/settings.yaml`:
```yaml
model:
  n_estimators: 150        # Increase trees
  max_depth: 8             # Increase complexity
  learning_rate: 0.05      # Lower learning rate
```

### 3. Data Sampling
For large datasets, use stratified sampling:
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    stratify=pd.qcut(y, q=5),  # Stratify by quantiles
    random_state=42
)
```

## Common Issues

### Issue: MAPE exceeds 5% threshold
**Solution:**
1. Check data quality for outliers
2. Add more relevant features
3. Increase model complexity (n_estimators, max_depth)
4. Collect more training data

### Issue: API returns "Model not loaded"
**Solution:**
1. Verify `models/groundwater_model.pkl` exists
2. Check `MODEL_PATH` environment variable
3. Restart FastAPI server

### Issue: Out of memory during training
**Solution:**
1. Reduce dataset size or use sampling
2. Batch process data
3. Increase virtual memory/swap space

## Environment Variables

```bash
# Set custom paths
export MODEL_PATH="./models/custom_model.pkl"
export DATA_PATH="./data/processed/features.csv"

# Set API configuration
export API_HOST="0.0.0.0"
export API_PORT="8000"

# Set logging level
export LOG_LEVEL="DEBUG"
```

## Deploying to Production

### 1. Install Production Dependencies
```bash
pip install gunicorn python-dotenv
```

### 2. Create `.env` file
```
MODEL_PATH=./models/groundwater_model.pkl
DATA_PATH=./data/processed/village_features.csv
LOG_LEVEL=INFO
```

### 3. Run with Gunicorn
```bash
gunicorn api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### 4. Use Docker (Optional)
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Repository Structure Best Practices

- Keep raw data in `data/raw/` - never modify
- Save all outputs to `data/processed/` or `data/external/`
- Use `.gitkeep` to track empty directories
- Add models to `.gitignore` to avoid large file commits
- Document data schemas in README

## Version Control

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# First commit
git commit -m "Initial groundwater estimation system setup"

# Add remote (replace with your repo)
git remote add origin https://github.com/your-org/groundwater-estimation-ap.git
git branch -M main
git push -u origin main
```

## Further Resources

- **XGBoost Documentation**: https://xgboost.readthedocs.io/
- **GeoPandas Guide**: https://geopandas.org/
- **FastAPI Tutorial**: https://fastapi.tiangolo.com/
- **Scikit-learn**: https://scikit-learn.org/

---

**Last Updated:** April 2026
