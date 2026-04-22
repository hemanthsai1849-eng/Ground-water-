"""
Phase 3: Anomaly Detection
Detect anomalies in groundwater level time-series data.
"""

import pandas as pd
import numpy as np
import json
from sklearn.ensemble import IsolationForest
from typing import Dict, List


class AnomalyDetector:
    """Detects anomalies in groundwater time-series data."""
    
    def __init__(self, contamination: float = 0.1, critical_std_threshold: float = 2.0):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of outliers (default: 0.1)
            critical_std_threshold: Standard deviation threshold in meters for critical anomalies
        """
        self.contamination = contamination
        self.critical_std_threshold = critical_std_threshold
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
    
    def detect_outliers(self, timeseries_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect outliers using IsolationForest algorithm.
        
        Args:
            timeseries_df: DataFrame with columns ['date', 'village_id', 'water_level']
        
        Returns:
            DataFrame with original data and 'anomaly' column (-1 for outlier, 1 for normal)
        """
        # Prepare features for detection
        features = timeseries_df[['water_level']].values
        
        # Detect anomalies
        anomaly_labels = self.model.fit_predict(features)
        anomaly_scores = self.model.score_samples(features)
        
        result_df = timeseries_df.copy()
        result_df['anomaly'] = anomaly_labels
        result_df['anomaly_score'] = anomaly_scores
        
        return result_df
    
    def detect_critical_anomalies(self, timeseries_df: pd.DataFrame, months: int = 3) -> Dict:
        """
        Detect critical anomalies based on rolling standard deviation.
        
        Args:
            timeseries_df: DataFrame with columns ['date', 'village_id', 'water_level']
            months: Number of months to use for rolling std calculation (default: 3)
        
        Returns:
            Dictionary of flagged villages with anomaly details
        """
        timeseries_df = timeseries_df.sort_values('date')
        
        flagged_villages = {}
        
        for village_id in timeseries_df['village_id'].unique():
            village_data = timeseries_df[timeseries_df['village_id'] == village_id].copy()
            
            # Calculate rolling standard deviation (approximate: days ~ months * 30)
            rolling_window = months * 30
            rolling_std = village_data['water_level'].rolling(window=rolling_window).std()
            
            # Check for critical anomalies
            critical_mask = rolling_std > self.critical_std_threshold
            
            if critical_mask.any():
                critical_records = village_data[critical_mask].copy()
                critical_records['rolling_std'] = rolling_std[critical_mask]
                
                flagged_villages[village_id] = critical_records.to_dict('records')
        
        return flagged_villages
    
    def generate_anomaly_json(
        self,
        timeseries_df: pd.DataFrame,
        months: int = 3
    ) -> List[Dict]:
        """
        Generate JSON output for flagged villages.
        
        Args:
            timeseries_df: Time-series DataFrame
            months: Number of months for rolling analysis
        
        Returns:
            List of JSON-serializable dictionaries with anomaly information
        """
        flagged_villages = self.detect_critical_anomalies(timeseries_df, months)
        
        json_output = []
        
        for village_id, records in flagged_villages.items():
            for record in records:
                json_output.append({
                    'village_id': str(village_id),
                    'date': str(record['date']),
                    'water_level': float(record['water_level']),
                    'rolling_std': float(record['rolling_std']),
                    'anomaly_score': float(record.get('anomaly_score', np.nan)),
                    'status': 'CRITICAL_ANOMALY',
                    'threshold_exceeded_by': float(
                        record['rolling_std'] - self.critical_std_threshold
                    )
                })
        
        return json_output
    
    def save_anomaly_report(self, anomaly_json: List[Dict], output_filepath: str) -> None:
        """
        Save anomaly report to JSON file.
        
        Args:
            anomaly_json: List of anomaly dictionaries
            output_filepath: Path to save the JSON file
        """
        with open(output_filepath, 'w') as f:
            json.dump(anomaly_json, f, indent=2, default=str)
    
    def get_summary_statistics(self, timeseries_df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for anomalies detected.
        
        Args:
            timeseries_df: Time-series DataFrame
        
        Returns:
            Dictionary with summary statistics
        """
        anomaly_df = self.detect_outliers(timeseries_df)
        
        return {
            'total_records': len(anomaly_df),
            'total_anomalies': (anomaly_df['anomaly'] == -1).sum(),
            'anomaly_percentage': ((anomaly_df['anomaly'] == -1).sum() / len(anomaly_df)) * 100,
            'mean_water_level': anomaly_df['water_level'].mean(),
            'std_water_level': anomaly_df['water_level'].std(),
            'min_water_level': anomaly_df['water_level'].min(),
            'max_water_level': anomaly_df['water_level'].max()
        }
