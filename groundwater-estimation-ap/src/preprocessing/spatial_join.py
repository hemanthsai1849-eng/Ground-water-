"""
Phase 1: Data Preprocessing & Feature Engineering
Spatial join of piezometer and village datasets with feature normalization.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def load_piezometer_data(filepath: str) -> gpd.GeoDataFrame:
    """
    Load piezometer data and convert to GeoDataFrame.
    
    Args:
        filepath: Path to piezometer CSV file (must contain 'latitude', 'longitude', 'depth', 'timestamp')
    
    Returns:
        GeoDataFrame with geometry column
    """
    df = pd.read_csv(filepath)
    geometry = gpd.points_from_xy(df['longitude'], df['latitude'])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    return gdf


def load_village_data(filepath: str) -> gpd.GeoDataFrame:
    """
    Load village location data and convert to GeoDataFrame.
    
    Args:
        filepath: Path to village CSV file (must contain 'village_id', 'latitude', 'longitude')
    
    Returns:
        GeoDataFrame with geometry column
    """
    df = pd.read_csv(filepath)
    geometry = gpd.points_from_xy(df['longitude'], df['latitude'])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    return gdf


def spatial_join_nearest_piezometers(
    villages_gdf: gpd.GeoDataFrame,
    piezometers_gdf: gpd.GeoDataFrame,
    k: int = 5
) -> pd.DataFrame:
    """
    Perform spatial join to find k-nearest piezometers for each village.
    
    Args:
        villages_gdf: GeoDataFrame of village locations
        piezometers_gdf: GeoDataFrame of piezometer locations
        k: Number of nearest piezometers to find (default: 5)
    
    Returns:
        DataFrame with village_id and distances to k-nearest piezometers
    """
    results = []
    
    # Extract coordinates for faster computation
    village_coords = np.array([[geom.x, geom.y] for geom in villages_gdf.geometry])
    piezometer_coords = np.array([[geom.x, geom.y] for geom in piezometers_gdf.geometry])
    piezometer_ids = piezometers_gdf['piezometer_id'].values
    
    for idx, village_id in enumerate(villages_gdf['village_id']):
        village_point = village_coords[idx]
        
        # Calculate distances using numpy (faster than iterating)
        distances = np.sqrt(np.sum((piezometer_coords - village_point) ** 2, axis=1))
        
        # Get k nearest indices
        nearest_indices = np.argsort(distances)[:k]
        
        for rank, nearest_idx in enumerate(nearest_indices):
            results.append({
                'village_id': village_id,
                'piezometer_id': piezometer_ids[nearest_idx],
                'distance_km': distances[nearest_idx] * 111.32,  # Convert degrees to km
                'rank': rank + 1
            })
    
    return pd.DataFrame(results)


def normalize_features(
    df: pd.DataFrame,
    feature_columns: list
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Normalize input features using StandardScaler.
    
    Args:
        df: DataFrame with features to normalize
        feature_columns: List of column names to normalize
    
    Returns:
        Tuple of (normalized DataFrame, fitted scaler object)
    """
    scaler = StandardScaler()
    df_normalized = df.copy()
    
    df_normalized[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    return df_normalized, scaler


def merge_datasets(
    villages_gdf: gpd.GeoDataFrame,
    piezometers_gdf: gpd.GeoDataFrame,
    rainfall_df: pd.DataFrame,
    soil_permeability_df: pd.DataFrame,
    elevation_df: pd.DataFrame,
    k: int = 5
) -> pd.DataFrame:
    """
    Merge all datasets into a single cleaned DataFrame ready for ML.
    
    Args:
        villages_gdf: Village locations
        piezometers_gdf: Piezometer locations
        rainfall_df: Rainfall data (must have village_id)
        soil_permeability_df: Soil permeability index (must have village_id)
        elevation_df: Elevation data (must have village_id)
        k: Number of nearest piezometers
    
    Returns:
        Merged and cleaned DataFrame
    """
    # Spatial join
    spatial_data = spatial_join_nearest_piezometers(villages_gdf, piezometers_gdf, k)
    
    # Merge with piezometer depth information
    piezometer_info = piezometers_gdf[['piezometer_id', 'depth']].drop_duplicates()
    spatial_data = spatial_data.merge(piezometer_info, on='piezometer_id', how='left')
    
    # Merge with village features
    merged_df = spatial_data.merge(rainfall_df, on='village_id', how='left')
    merged_df = merged_df.merge(soil_permeability_df, on='village_id', how='left')
    merged_df = merged_df.merge(elevation_df, on='village_id', how='left')
    
    # Remove rows with missing critical values
    merged_df = merged_df.dropna(subset=['rainfall', 'soil_permeability', 'elevation', 'depth'])
    
    # Normalize features
    feature_columns = ['rainfall', 'soil_permeability', 'elevation']
    merged_df, _ = normalize_features(merged_df, feature_columns)
    
    return merged_df
