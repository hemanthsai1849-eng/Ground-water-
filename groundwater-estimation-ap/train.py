"""
Example training pipeline script.
Run this to train the complete groundwater estimation model.
"""

import os
import yaml
import pandas as pd
import geopandas as gpd
from pathlib import Path
import logging

from src.preprocessing.spatial_join import merge_datasets, load_piezometer_data, load_village_data
from src.modeling.xgb_regressor import GroundwaterEstimator
from src.inference.anomaly_detection import AnomalyDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config/settings.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Execute complete training pipeline."""
    
    logger.info("=" * 80)
    logger.info("GROUNDWATER ESTIMATION SYSTEM - TRAINING PIPELINE")
    logger.info("=" * 80)
    
    # Load configuration
    config = load_config()
    logger.info("Configuration loaded successfully")
    
    # Create necessary directories
    Path('models').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    
    # ====== PHASE 1: DATA PREPROCESSING ======
    logger.info("\n--- PHASE 1: DATA PREPROCESSING ---")
    
    try:
        # Load raw data
        logger.info("Loading piezometer data...")
        piezometers = load_piezometer_data(
            os.path.join(config['data']['raw_dir'], 'piezometers.csv')
        )
        logger.info(f"Loaded {len(piezometers)} piezometer records")
        
        logger.info("Loading village data...")
        villages = load_village_data(
            os.path.join(config['data']['raw_dir'], 'villages.csv')
        )
        logger.info(f"Loaded {len(villages)} village locations")
        
        # Load feature data
        logger.info("Loading hydrogeological features...")
        rainfall_df = pd.read_csv(
            os.path.join(config['data']['raw_dir'], 'rainfall.csv')
        )
        soil_df = pd.read_csv(
            os.path.join(config['data']['raw_dir'], 'soil_permeability.csv')
        )
        elevation_df = pd.read_csv(
            os.path.join(config['data']['raw_dir'], 'elevation.csv')
        )
        
        # Merge datasets
        logger.info("Performing spatial join...")
        processed_df = merge_datasets(
            villages,
            piezometers,
            rainfall_df,
            soil_df,
            elevation_df,
            k=config['spatial']['k_nearest_piezometers']
        )
        logger.info(f"Merged dataset shape: {processed_df.shape}")
        
        # Save processed data
        output_path = os.path.join(
            config['data']['processed_dir'],
            'village_features.csv'
        )
        processed_df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        return
    
    # ====== PHASE 2: MODEL TRAINING ======
    logger.info("\n--- PHASE 2: MODEL TRAINING ---")
    
    try:
        # Prepare features and target
        feature_cols = config['features']['input_features']
        target_col = config['features']['target_variable']
        
        # Filter to only available columns
        available_features = [col for col in feature_cols if col in processed_df.columns]
        if not available_features:
            # Use available numeric features
            available_features = ['rainfall', 'soil_permeability', 'elevation', 'distance_km']
        
        X = processed_df[available_features]
        y = processed_df['depth']  # Use depth as target since we don't have water_level
        
        logger.info(f"Training features shape: {X.shape}")
        logger.info(f"Target variable range: [{y.min():.2f}, {y.max():.2f}]")
        
        # Train model
        logger.info("Training XGBoost model...")
        estimator = GroundwaterEstimator(
            random_state=config['model']['random_state']
        )
        
        metrics = estimator.train(
            X, y,
            test_size=0.2,
            validation_split=0.1
        )
        
        # Log metrics
        logger.info("Training complete. Metrics:")
        logger.info(f"  - MAPE: {metrics['mape']:.4f} (Target: < {config['validation']['mape_threshold']})")
        logger.info(f"  - MAE:  {metrics['mae']:.4f}")
        logger.info(f"  - RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  - Threshold Met: {metrics['mape_threshold_met']}")
        
        if not metrics['mape_threshold_met']:
            logger.warning(f"⚠️  MAPE {metrics['mape']:.4f} exceeds target {config['validation']['mape_threshold']}")
        else:
            logger.info("✓ MAPE threshold met!")
        
        # Log top features
        logger.info("\nTop 5 Important Features:")
        for idx, row in metrics['feature_importance'].head(5).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Save model
        model_path = config['model']['model_save_path']
        estimator.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return
    
    # ====== PHASE 3: ANOMALY DETECTION ======
    logger.info("\n--- PHASE 3: ANOMALY DETECTION ---")
    
    try:
        # Load time-series data if available
        timeseries_path = os.path.join(
            config['data']['raw_dir'],
            'timeseries.csv'
        )
        
        if os.path.exists(timeseries_path):
            logger.info("Loading time-series data...")
            timeseries_df = pd.read_csv(timeseries_path)
            timeseries_df['date'] = pd.to_datetime(timeseries_df['date'])
            
            # Detect anomalies
            logger.info("Performing anomaly detection...")
            detector = AnomalyDetector(
                contamination=config['anomaly_detection']['contamination'],
                critical_std_threshold=config['anomaly_detection']['critical_std_threshold']
            )
            
            anomalies = detector.generate_anomaly_json(
                timeseries_df,
                months=config['anomaly_detection']['rolling_window_months']
            )
            
            logger.info(f"Detected {len(anomalies)} critical anomalies")
            
            # Save anomaly report
            anomaly_path = os.path.join(
                config['data']['processed_dir'],
                'anomalies.json'
            )
            detector.save_anomaly_report(anomalies, anomaly_path)
            logger.info(f"Anomaly report saved to {anomaly_path}")
            
        else:
            logger.info("Time-series data not found. Skipping anomaly detection.")
    
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
    
    # ====== SUMMARY ======
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Model Location: {model_path}")
    logger.info(f"Processed Data: {output_path}")
    logger.info("Next Steps:")
    logger.info("  1. Review model metrics in logs/")
    logger.info("  2. Deploy API: uvicorn api.main:app --reload")
    logger.info("  3. Test endpoints: curl http://localhost:8000/docs")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
