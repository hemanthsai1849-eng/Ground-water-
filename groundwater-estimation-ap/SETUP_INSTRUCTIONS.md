# SETUP INSTRUCTIONS
## Groundwater Estimation System for Andhra Pradesh

### System Requirements
- **Python**: 3.10 or higher
- **OS**: Windows, macOS, or Linux
- **RAM**: Minimum 4GB (8GB+ recommended for full dataset)
- **Disk**: 2GB free space

---

## QUICK START (5 minutes)

### Step 1: Install Python Packages
```bash
pip install -r requirements.txt
```

### Step 2: Create Sample Data
```bash
python tests/create_sample_data.py
```

### Step 3: Train the Model
```bash
python train.py
```

### Step 4: Start the API
```bash
uvicorn api.main:app --reload
```

### Step 5: Test
Visit: `http://localhost:8000/docs`

---

## DETAILED SETUP

### 1. Virtual Environment Setup

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Prepare Your Data

Create the following CSV files in `data/raw/`:

**1. Piezometer Data** (`piezometers.csv`)
```
piezometer_id,latitude,longitude,depth,timestamp
1,15.5,79.2,45.3,2024-01-01
2,15.6,79.3,52.1,2024-01-02
...
```

**2. Village Locations** (`villages.csv`)
```
village_id,latitude,longitude,village_name
1,15.51,79.21,Hyderabad
2,15.52,79.22,Secunderabad
...
```

**3. Rainfall Data** (`rainfall.csv`)
```
village_id,rainfall
1,850.5
2,920.3
...
```

**4. Soil Permeability** (`soil_permeability.csv`)
```
village_id,soil_permeability
1,4.5
2,6.2
...
```

**5. Elevation Data** (`elevation.csv`)
```
village_id,elevation
1,505.0
2,498.5
...
```

**Optional: Time-Series Data** (`timeseries.csv`)
```
village_id,date,water_level
1,2024-01-01,28.5
1,2024-01-02,28.3
...
```

### 4. Run the Training Pipeline
```bash
python train.py
```

Monitor output:
```
--- PHASE 1: DATA PREPROCESSING ---
--- PHASE 2: MODEL TRAINING ---
--- PHASE 3: ANOMALY DETECTION ---
```

Check logs:
```bash
tail -f logs/groundwater_system.log
```

### 5. Start the API Server
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Verify Installation

**Check API Health:**
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "operational",
  "model_loaded": true,
  "timestamp": "2024-04-22T10:30:00"
}
```

**Get Prediction:**
```bash
curl -X POST http://localhost:8000/get_village_data/1
```

**Interactive API Documentation:**
Open in browser: `http://localhost:8000/docs`

---

## CONFIGURATION

Edit `config/settings.yaml` to customize:

```yaml
# Model parameters
model:
  n_estimators: 100      # Number of trees
  max_depth: 6           # Tree depth
  learning_rate: 0.1     # Learning rate

# Validation threshold (critical!)
validation:
  mape_threshold: 0.05   # 5% error target

# Spatial configuration
spatial:
  k_nearest_piezometers: 5  # Nearest neighbors
```

---

## TESTING

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Suite
```bash
# Model accuracy
pytest tests/test_model_accuracy.py -v

# API endpoints
pytest tests/test_api_endpoints.py -v
```

### Generate Coverage Report
```bash
pytest tests/ --cov=src --cov=api --cov-report=html
open htmlcov/index.html  # macOS
```

---

## TROUBLESHOOTING

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'geopandas'` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: data/raw/piezometers.csv` | Run `python tests/create_sample_data.py` |
| `MAPE exceeds 5% threshold` | Check data quality, try more features, increase model complexity |
| `API returns port already in use` | Change port: `uvicorn api.main:app --port 8001` |
| `Model not loaded` error | Train model first: `python train.py` |

---

## PROJECT STRUCTURE

```
groundwater-estimation-ap/
├── src/                          # Source code
│   ├── preprocessing/            # Data processing
│   │   └── spatial_join.py       # Merge datasets
│   ├── modeling/                 # ML model
│   │   └── xgb_regressor.py     # XGBoost trainer
│   ├── inference/                # Predictions
│   │   └── anomaly_detection.py # Anomaly detection
│   └── utils/                    # Utilities
│       └── spatial_tools.py     # Geo functions
├── api/
│   ├── main.py                   # FastAPI app
│   └── schemas.py                # Data validation
├── data/
│   ├── raw/                      # Input data
│   ├── processed/                # Training data
│   └── external/                 # Reference data
├── tests/                        # Unit tests
│   ├── test_model_accuracy.py
│   ├── test_api_endpoints.py
│   └── create_sample_data.py
├── config/
│   └── settings.yaml             # Configuration
├── models/                       # Trained models
├── logs/                         # Log files
├── train.py                      # Training script
├── requirements.txt              # Dependencies
├── README.md                     # Documentation
└── DEVELOPMENT.md                # Dev guide
```

---

## KEY COMMANDS REFERENCE

```bash
# Setup
python -m venv venv
source venv/bin/activate          # macOS/Linux
pip install -r requirements.txt

# Create test data
python tests/create_sample_data.py

# Train model
python train.py

# Start API
uvicorn api.main:app --reload

# Test API
curl http://localhost:8000/health
curl -X POST http://localhost:8000/get_village_data/1

# Run tests
pytest tests/ -v

# View logs
tail -f logs/groundwater_system.log
```

---

## NEXT STEPS

1. ✓ Install dependencies
2. ✓ Prepare data files
3. ✓ Run `train.py`
4. ✓ Start API with `uvicorn`
5. → Test endpoints at `/docs`
6. → Deploy to production
7. → Monitor performance

---

## PERFORMANCE TARGETS

| Metric | Target | Notes |
|--------|--------|-------|
| MAPE | < 5% | Primary accuracy metric |
| Prediction Time | < 200ms | Per village |
| Model Training | < 5 min | On 18K villages |
| API Response | < 1s | Per request |

---

## ADDITIONAL RESOURCES

- **Documentation**: See [README.md](README.md)
- **Development Guide**: See [DEVELOPMENT.md](DEVELOPMENT.md)
- **API Docs**: `http://localhost:8000/docs` (when running)
- **FastAPI**: https://fastapi.tiangolo.com/
- **XGBoost**: https://xgboost.readthedocs.io/
- **GeoPandas**: https://geopandas.org/

---

**Version**: 1.0.0  
**Last Updated**: April 2026  
**Contact**: [Your Department]
