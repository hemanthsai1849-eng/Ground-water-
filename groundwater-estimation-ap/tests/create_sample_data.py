"""
Create sample/test data for development and testing.
Run this to generate synthetic data for testing the pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_piezometer_data(n_piezometers: int = 1800) -> pd.DataFrame:
    """
    Create synthetic piezometer data.
    
    Args:
        n_piezometers: Number of piezometer records
    
    Returns:
        DataFrame with piezometer information
    """
    np.random.seed(42)
    
    df = pd.DataFrame({
        'piezometer_id': range(1, n_piezometers + 1),
        'latitude': np.random.uniform(12.5, 19.0, n_piezometers),
        'longitude': np.random.uniform(76.5, 84.5, n_piezometers),
        'depth': np.random.uniform(5, 100, n_piezometers),  # meters
        'timestamp': pd.date_range('2024-01-01', periods=n_piezometers, freq='D')
    })
    
    return df


def create_village_data(n_villages: int = 18000) -> pd.DataFrame:
    """
    Create synthetic village location data.
    
    Args:
        n_villages: Number of village records
    
    Returns:
        DataFrame with village locations
    """
    np.random.seed(42)
    
    df = pd.DataFrame({
        'village_id': range(1, n_villages + 1),
        'latitude': np.random.uniform(12.5, 19.0, n_villages),
        'longitude': np.random.uniform(76.5, 84.5, n_villages),
        'village_name': [f'Village_{i}' for i in range(1, n_villages + 1)]
    })
    
    return df


def create_rainfall_data(n_villages: int = 18000) -> pd.DataFrame:
    """Create synthetic rainfall data."""
    np.random.seed(42)
    
    df = pd.DataFrame({
        'village_id': range(1, n_villages + 1),
        'rainfall': np.random.uniform(500, 1500, n_villages)  # mm/year
    })
    
    return df


def create_soil_permeability_data(n_villages: int = 18000) -> pd.DataFrame:
    """Create synthetic soil permeability data."""
    np.random.seed(42)
    
    df = pd.DataFrame({
        'village_id': range(1, n_villages + 1),
        'soil_permeability': np.random.uniform(1, 10, n_villages)  # Index 1-10
    })
    
    return df


def create_elevation_data(n_villages: int = 18000) -> pd.DataFrame:
    """Create synthetic elevation data."""
    np.random.seed(42)
    
    df = pd.DataFrame({
        'village_id': range(1, n_villages + 1),
        'elevation': np.random.uniform(100, 1000, n_villages)  # meters
    })
    
    return df


def create_timeseries_data(n_villages: int = 100, days: int = 365) -> pd.DataFrame:
    """
    Create synthetic time-series groundwater data.
    
    Args:
        n_villages: Number of villages
        days: Number of days of data
    
    Returns:
        DataFrame with time-series data
    """
    np.random.seed(42)
    
    records = []
    
    for village_id in range(1, n_villages + 1):
        # Create synthetic time series with seasonal pattern
        base_level = np.random.uniform(20, 50)
        trend = np.random.uniform(-0.1, 0.1)
        
        for day in range(days):
            seasonal = 5 * np.sin(2 * np.pi * day / 365)
            noise = np.random.normal(0, 1)
            water_level = base_level + trend * day + seasonal + noise
            
            records.append({
                'village_id': village_id,
                'date': pd.Timestamp('2023-01-01') + pd.Timedelta(days=day),
                'water_level': max(water_level, 0)  # No negative levels
            })
    
    return pd.DataFrame(records)


def main():
    """Create all sample data files."""
    
    logger.info("Creating sample data for development and testing...")
    
    # Create data directories if they don't exist
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    
    # Create piezometer data
    logger.info("Creating piezometer data (1,800 records)...")
    piezometer_df = create_piezometer_data(1800)
    piezometer_df.to_csv('data/raw/piezometers.csv', index=False)
    logger.info(f"✓ Saved: data/raw/piezometers.csv ({len(piezometer_df)} records)")
    
    # Create village data
    logger.info("Creating village data (18,000 records)...")
    village_df = create_village_data(18000)
    village_df.to_csv('data/raw/villages.csv', index=False)
    logger.info(f"✓ Saved: data/raw/villages.csv ({len(village_df)} records)")
    
    # Create feature data
    logger.info("Creating rainfall data...")
    rainfall_df = create_rainfall_data(18000)
    rainfall_df.to_csv('data/raw/rainfall.csv', index=False)
    logger.info(f"✓ Saved: data/raw/rainfall.csv")
    
    logger.info("Creating soil permeability data...")
    soil_df = create_soil_permeability_data(18000)
    soil_df.to_csv('data/raw/soil_permeability.csv', index=False)
    logger.info(f"✓ Saved: data/raw/soil_permeability.csv")
    
    logger.info("Creating elevation data...")
    elevation_df = create_elevation_data(18000)
    elevation_df.to_csv('data/raw/elevation.csv', index=False)
    logger.info(f"✓ Saved: data/raw/elevation.csv")
    
    # Create time-series data
    logger.info("Creating time-series data (100 villages, 365 days)...")
    timeseries_df = create_timeseries_data(100, 365)
    timeseries_df.to_csv('data/raw/timeseries.csv', index=False)
    logger.info(f"✓ Saved: data/raw/timeseries.csv ({len(timeseries_df)} records)")
    
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLE DATA CREATION COMPLETE")
    logger.info("=" * 80)
    logger.info("Created the following files:")
    logger.info("  ✓ data/raw/piezometers.csv (1,800 records)")
    logger.info("  ✓ data/raw/villages.csv (18,000 records)")
    logger.info("  ✓ data/raw/rainfall.csv (18,000 records)")
    logger.info("  ✓ data/raw/soil_permeability.csv (18,000 records)")
    logger.info("  ✓ data/raw/elevation.csv (18,000 records)")
    logger.info("  ✓ data/raw/timeseries.csv (36,500 records)")
    logger.info("\nNext steps:")
    logger.info("  1. Run: python train.py")
    logger.info("  2. Start API: uvicorn api.main:app --reload")
    logger.info("  3. Test: curl http://localhost:8000/health")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
