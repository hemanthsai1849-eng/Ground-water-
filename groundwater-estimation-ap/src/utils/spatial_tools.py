"""
Utility functions for spatial operations and data conversion.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from typing import Tuple, Optional


def convert_coordinates_to_utm(
    latitude: float,
    longitude: float
) -> Tuple[int, int]:
    """
    Convert latitude/longitude to UTM zone.
    
    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
    
    Returns:
        Tuple of (utm_zone_number, utm_zone_letter)
    """
    utm_zone = (int((longitude + 180) / 6) % 60) + 1
    
    if 84 >= latitude >= 72:
        utm_letter = 'X'
    elif 72 > latitude >= 64:
        utm_letter = 'W'
    elif 64 > latitude >= 56:
        utm_letter = 'V'
    elif 56 > latitude >= 48:
        utm_letter = 'U'
    elif 48 > latitude >= 40:
        utm_letter = 'T'
    elif 40 > latitude >= 32:
        utm_letter = 'S'
    elif 32 > latitude >= 24:
        utm_letter = 'R'
    elif 24 > latitude >= 16:
        utm_letter = 'Q'
    elif 16 > latitude >= 8:
        utm_letter = 'P'
    elif 8 > latitude >= 0:
        utm_letter = 'N'
    elif 0 > latitude >= -8:
        utm_letter = 'M'
    elif -8 > latitude >= -16:
        utm_letter = 'L'
    elif -16 > latitude >= -24:
        utm_letter = 'K'
    elif -24 > latitude >= -32:
        utm_letter = 'J'
    elif -32 > latitude >= -40:
        utm_letter = 'H'
    elif -40 > latitude >= -48:
        utm_letter = 'G'
    elif -48 > latitude >= -56:
        utm_letter = 'F'
    elif -56 > latitude >= -64:
        utm_letter = 'E'
    elif -64 > latitude >= -72:
        utm_letter = 'D'
    elif -72 > latitude >= -80:
        utm_letter = 'C'
    else:
        utm_letter = 'Z'
    
    return utm_zone, utm_letter


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Calculate haversine distance between two coordinate pairs in kilometers.
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
    
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def geodetic_to_cartesian(latitude: float, longitude: float, height: float = 0) -> Tuple[float, float, float]:
    """
    Convert geodetic coordinates to Cartesian coordinates (WGS84).
    
    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        height: Height above ellipsoid in meters
    
    Returns:
        Tuple of (x, y, z) Cartesian coordinates
    """
    a = 6378137.0  # WGS84 semi-major axis
    f = 1 / 298.257223563  # WGS84 flattening
    e_sq = 2 * f - f ** 2  # First eccentricity squared
    
    lat_rad = np.radians(latitude)
    lon_rad = np.radians(longitude)
    
    N = a / np.sqrt(1 - e_sq * np.sin(lat_rad) ** 2)
    
    x = (N + height) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + height) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e_sq) + height) * np.sin(lat_rad)
    
    return x, y, z


def validate_coordinates(
    latitude: float,
    longitude: float
) -> bool:
    """
    Validate coordinate values (Andhra Pradesh region).
    
    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
    
    Returns:
        True if coordinates are valid for Andhra Pradesh
    """
    # Andhra Pradesh bounds (approximate)
    ap_min_lat, ap_max_lat = 12.5, 19.0
    ap_min_lon, ap_max_lon = 76.5, 84.5
    
    return (ap_min_lat <= latitude <= ap_max_lat and 
            ap_min_lon <= longitude <= ap_max_lon)


def filter_by_region(
    gdf: gpd.GeoDataFrame,
    bounds: Tuple[float, float, float, float]
) -> gpd.GeoDataFrame:
    """
    Filter GeoDataFrame by geographic bounds.
    
    Args:
        gdf: GeoDataFrame to filter
        bounds: Tuple of (minx, miny, maxx, maxy)
    
    Returns:
        Filtered GeoDataFrame
    """
    minx, miny, maxx, maxy = bounds
    mask = (
        (gdf.geometry.x >= minx) & (gdf.geometry.x <= maxx) &
        (gdf.geometry.y >= miny) & (gdf.geometry.y <= maxy)
    )
    return gdf[mask]


def calculate_buffer_zone(
    center_lat: float,
    center_lon: float,
    radius_km: float
) -> Tuple[float, float, float, float]:
    """
    Calculate buffer zone bounds around a center point.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_km: Buffer radius in kilometers
    
    Returns:
        Tuple of (minx, miny, maxx, maxy) bounds
    """
    # Approximate conversion: 1 degree ~ 111 km
    delta_degrees = radius_km / 111.32
    
    minx = center_lon - delta_degrees
    maxx = center_lon + delta_degrees
    miny = center_lat - delta_degrees
    maxy = center_lat + delta_degrees
    
    return minx, miny, maxx, maxy
