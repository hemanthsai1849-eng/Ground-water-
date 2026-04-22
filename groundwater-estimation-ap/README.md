# Groundwater Level Estimation System for Andhra Pradesh

## Overview

This project implements a machine learning-based groundwater level estimation system for Andhra Pradesh. It uses spatial analysis of 1,800 piezometer monitoring stations to predict groundwater levels for 18,000 village locations.

## Project Structure

```
groundwater-estimation-ap/
├── data/                    # Data storage
│   ├── raw/                 # Original piezometer, rainfall, geological data
│   ├── processed/           # Cleaned, normalized datasets for training
│   └── external/            # Geomorphological maps, land-use shapefiles
├── notebooks/               # Experimental Jupyter notebooks (EDA, PoC)
├── src/                     # Source code
│   ├── preprocessing/       # Feature engineering & spatial join logic
│   ├── modeling/            # Training scripts (XGBoost, etc.)
│   ├── inference/           # Production prediction scripts
│   └── utils/               # Helper functions
├── api/                     # FastAPI application
│   ├── main.py              # FastAPI endpoints
│   └── schemas.py           # Pydantic validation models
├── tests/                   # Unit and integration tests
│   ├── test_model_accuracy.py    # Validation against 5% error threshold
│   └── test_api_endpoints.py     # API endpoint tests
├── config/                  # Configuration
│   └── settings.yaml        # Configuration parameters
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── .gitignore             # Git ignore rules
```

## Development Phases

### Phase 1: Data Preprocessing & Feature Engineering
Merges 1,800 piezometer locations with 18,000 village locations using spatial join. Normalizes hydrological and geological features.

**Key Functions:**
- `spatial_join_nearest_piezometers()`: Find k-nearest piezometers for each village
- `normalize_features()`: Standardize input features
- `merge_datasets()`: Combine all data sources

### Phase 2: Core AI Model (XGBoost Estimation)
Trains an XGBoost regressor to predict groundwater depth with <5% error rate (MAPE).

**Key Features:**
- Feature importance analysis
- Custom loss function optimization
- MAPE validation against 5% threshold
- Trained on rainfall, geology, land-use data

### Phase 3: Anomaly Detection
Identifies borewell yield changes and critical groundwater variations using time-series analysis.

**Key Functions:**
- `detect_outliers()`: IsolationForest-based anomaly detection
- `detect_critical_anomalies()`: Flag villages with 2m+ std deviation in 3-month rolling window
- `generate_anomaly_json()`: Export flagged villages with scores

### Phase 4: API & Visualization Logic
FastAPI backend providing real-time predictions and anomaly alerts for dashboard integration.

**Endpoints:**
- `GET /health` - API health check
- `POST /get_village_data/{village_id}` - Single village prediction
- `POST /predict_batch` - Batch predictions for multiple villages
- `POST /anomalies/{village_id}` - Anomaly detection for village

## Installation

### Prerequisites
- Python 3.10 or higher
- pip or conda package manager

### Setup

1. **Clone or navigate to project directory:**
   ```bash
   cd groundwater-estimation-ap
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Phase 1: Data Preprocessing

```python
from src.preprocessing.spatial_join import merge_datasets
import geopandas as gpd

# Load data
villages = gpd.read_file('data/raw/villages.shp')
piezometers = gpd.read_file('data/raw/piezometers.shp')
rainfall = pd.read_csv('data/raw/rainfall.csv')
soil = pd.read_csv('data/raw/soil_permeability.csv')
elevation = pd.read_csv('data/raw/elevation.csv')

# Merge datasets
processed_df = merge_datasets(
    villages, piezometers, rainfall, soil, elevation, k=5
)

processed_df.to_csv('data/processed/village_features.csv', index=False)
```

### Phase 2: Model Training

```python
from src.modeling.xgb_regressor import GroundwaterEstimator
import pandas as pd

# Load processed data
df = pd.read_csv('data/processed/village_features.csv')
X = df[['rainfall', 'soil_permeability', 'elevation', 'distance_to_nearest_piezometer']]
y = df['water_level']

# Train model
estimator = GroundwaterEstimator()
metrics = estimator.train(X, y)

print(f"MAPE: {metrics['mape']:.4f} (Target: < 0.05)")
print(f"Feature Importance:\n{metrics['feature_importance']}")

# Save model
estimator.save_model('models/groundwater_model.pkl')
```

### Phase 3: Anomaly Detection

```python
from src.inference.anomaly_detection import AnomalyDetector
import pandas as pd

# Load time-series data
timeseries_df = pd.read_csv('data/processed/timeseries.csv')
timeseries_df['date'] = pd.to_datetime(timeseries_df['date'])

# Detect anomalies
detector = AnomalyDetector(critical_std_threshold=2.0)
anomalies_json = detector.generate_anomaly_json(timeseries_df, months=3)

detector.save_anomaly_report(anomalies_json, 'data/processed/anomalies.json')
```

### Phase 4: API Deployment

```bash
# Start FastAPI server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Test API
curl http://localhost:8000/health
curl -X POST http://localhost:8000/get_village_data/1
```

Visit API documentation at: `http://localhost:8000/docs`

## Testing

### Run all tests:
```bash
pytest tests/ -v
```

### Run model accuracy tests:
```bash
pytest tests/test_model_accuracy.py -v
```

### Run API tests:
```bash
pytest tests/test_api_endpoints.py -v
```

### Generate coverage report:
```bash
pytest tests/ --cov=src --cov=api --cov-report=html
```

## Configuration

Edit `config/settings.yaml` to customize:
- Data paths and file locations
- Model hyperparameters
- Feature engineering settings
- Validation thresholds (MAPE, MAE, RMSE)
- Anomaly detection sensitivity
- API host/port settings

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| MAPE (Mean Absolute Percentage Error) | < 5% | Validation Required |
| MAE (Mean Absolute Error) | < 2m | Validation Required |
| Model Training Time | < 5 minutes | Dependent on data size |
| API Response Time | < 200ms | Single village prediction |

## Data Requirements

### Input Data
1. **Piezometer Data** (1,800 points)
   - Columns: piezometer_id, latitude, longitude, depth, timestamp

2. **Village Data** (18,000 locations)
   - Columns: village_id, latitude, longitude

3. **Hydrogeological Features**
   - Rainfall data (mm/year or monthly)
   - Soil permeability index
   - Elevation (meters)

### Output Data
1. **Processed Features** (`data/processed/village_features.csv`)
2. **Trained Model** (`models/groundwater_model.pkl`)
3. **Anomaly Report** (`data/processed/anomalies.json`)

## API Response Examples

### Successful Prediction
```json
{
  "village_id": 1,
  "predicted_level": 23.5,
  "trend": "RISING",
  "alert_status": "NORMAL",
  "confidence": 0.87,
  "timestamp": "2024-04-22T10:30:00"
}
```

### Anomaly Detection
```json
{
  "village_id": 1,
  "date": "2024-04-15",
  "water_level": 15.2,
  "anomaly_score": -0.85,
  "status": "CRITICAL_ANOMALY",
  "rolling_std": 3.5
}
```

## Troubleshooting

### Model Not Loading
- Check model file exists at `models/groundwater_model.pkl`
- Verify Python pickle compatibility with training environment

### API Connection Issues
- Ensure FastAPI is running: `uvicorn api.main:app --reload`
- Check firewall settings for port 8000

### Feature Normalization Errors
- Verify all input features are numeric
- Check for missing values in raw data
- Ensure feature column names match configuration

## Contributing

1. Create feature branch: `git checkout -b feature/new-feature`
2. Make changes and test: `pytest tests/`
3. Commit with clear messages: `git commit -m "Add new feature"`
4. Push to branch: `git push origin feature/new-feature`

## License

This project is developed for the Government of Andhra Pradesh.

## Contact & Support

For questions or issues:
- Email: [Contact information]
- Project Lead: [Name and contact]
- Department: Irrigation & CAD, Government of Andhra Pradesh

---

**Last Updated:** April 2026
**Version:** 1.0.0
