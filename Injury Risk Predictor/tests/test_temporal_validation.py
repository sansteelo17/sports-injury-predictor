"""
Unit tests for temporal_validation module

Run with: pytest tests/test_temporal_validation.py -v

Author: Claude Sonnet 4.5
Date: 2026-01-23
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.temporal_validation import (
    temporal_split,
    walk_forward_validation,
    get_temporal_classification_splits,
    get_temporal_severity_splits,
    check_temporal_leakage,
    compare_random_vs_temporal_split
)


@pytest.fixture
def sample_injury_data():
    """Create sample injury data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')

    df = pd.DataFrame({
        'Date of Injury': np.random.choice(dates, size=1000),
        'injury_label': np.random.randint(0, 2, size=1000),
        'severity_days': np.random.exponential(scale=14, size=1000),
        'age': np.random.uniform(18, 35, size=1000),
        'workload': np.random.uniform(0, 100, size=1000),
        'position': np.random.choice(['FW', 'MF', 'DF', 'GK'], size=1000),
    })

    return df


class TestTemporalSplit:
    """Test cases for temporal_split function"""

    def test_basic_split(self, sample_injury_data):
        """Test basic temporal split functionality"""
        train_df, val_df, test_df = temporal_split(
            df=sample_injury_data,
            date_column='Date of Injury',
            train_end='2022-06-30',
            val_end='2023-06-30',
            target_column=None
        )

        # Check that all splits have data
        assert len(train_df) > 0, "Training set should not be empty"
        assert len(val_df) > 0, "Validation set should not be empty"
        assert len(test_df) > 0, "Test set should not be empty"

        # Check that total equals original
        assert len(train_df) + len(val_df) + len(test_df) == len(sample_injury_data)

    def test_split_with_target(self, sample_injury_data):
        """Test temporal split with target variable separation"""
        X_train, X_val, X_test, y_train, y_val, y_test = temporal_split(
            df=sample_injury_data,
            date_column='Date of Injury',
            target_column='injury_label',
            train_end='2022-06-30',
            val_end='2023-06-30'
        )

        # Check shapes match
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_test) == len(y_test)

        # Check target column not in X
        assert 'injury_label' not in X_train.columns
        assert 'injury_label' not in X_val.columns
        assert 'injury_label' not in X_test.columns

    def test_temporal_ordering(self, sample_injury_data):
        """Test that temporal ordering is respected"""
        train_df, val_df, test_df = temporal_split(
            df=sample_injury_data,
            date_column='Date of Injury',
            train_end='2022-06-30',
            val_end='2023-06-30',
            target_column=None
        )

        # Check date ranges don't overlap
        train_end_date = pd.to_datetime('2022-06-30')
        val_end_date = pd.to_datetime('2023-06-30')

        assert (train_df['Date of Injury'] < train_end_date).all()
        assert ((val_df['Date of Injury'] >= train_end_date) &
                (val_df['Date of Injury'] < val_end_date)).all()
        assert (test_df['Date of Injury'] >= val_end_date).all()

    def test_invalid_dates(self, sample_injury_data):
        """Test that invalid date order raises error"""
        with pytest.raises(ValueError):
            temporal_split(
                df=sample_injury_data,
                date_column='Date of Injury',
                train_end='2023-06-30',  # Later than val_end!
                val_end='2022-06-30',
                target_column=None
            )


class TestWalkForwardValidation:
    """Test cases for walk_forward_validation function"""

    def test_basic_walk_forward(self, sample_injury_data):
        """Test basic walk-forward validation"""
        splits = walk_forward_validation(
            df=sample_injury_data,
            date_column='Date of Injury',
            target_column='injury_label',
            n_splits=3,
            test_size_months=6,
            gap_months=0
        )

        # Check we got the right number of splits
        assert len(splits) == 3, f"Expected 3 splits, got {len(splits)}"

        # Check each split has 4 components
        for split in splits:
            assert len(split) == 4, "Each split should have 4 components"
            X_train, X_test, y_train, y_test = split

            # Check shapes match
            assert len(X_train) == len(y_train)
            assert len(X_test) == len(y_test)

            # Check target not in features
            assert 'injury_label' not in X_train.columns
            assert 'injury_label' not in X_test.columns

    def test_expanding_window(self, sample_injury_data):
        """Test that training set expands over time"""
        splits = walk_forward_validation(
            df=sample_injury_data,
            date_column='Date of Injury',
            target_column='injury_label',
            n_splits=3,
            test_size_months=6
        )

        # Check that training set size increases with each split
        train_sizes = [len(X_train) for X_train, _, _, _ in splits]

        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i-1], \
                "Training set should expand (or stay same) over time"

    def test_with_gap(self, sample_injury_data):
        """Test walk-forward validation with prediction gap"""
        splits = walk_forward_validation(
            df=sample_injury_data,
            date_column='Date of Injury',
            target_column='injury_label',
            n_splits=2,
            test_size_months=6,
            gap_months=1
        )

        # Should create splits successfully
        assert len(splits) > 0, "Should create splits even with gap"


class TestConvenienceFunctions:
    """Test cases for convenience wrapper functions"""

    def test_get_temporal_classification_splits(self, sample_injury_data):
        """Test classification convenience function"""
        X_train, X_val, X_test, y_train, y_val, y_test = get_temporal_classification_splits(
            df=sample_injury_data,
            date_column='Date of Injury',
            target_column='injury_label',
            train_end='2022-06-30',
            val_end='2023-06-30'
        )

        # Check shapes
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_test) == len(y_test)

        # Check target values are binary
        assert y_train.nunique() <= 2
        assert y_test.nunique() <= 2

    def test_get_temporal_severity_splits(self, sample_injury_data):
        """Test severity regression convenience function"""
        X_train, X_val, X_test, y_train, y_val, y_test = get_temporal_severity_splits(
            df=sample_injury_data,
            date_column='Date of Injury',
            target_column='severity_days',
            train_end='2022-06-30',
            val_end='2023-06-30'
        )

        # Check shapes
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_test) == len(y_test)

        # Check target values are continuous
        assert y_train.min() >= 0, "Severity should be non-negative"
        assert y_test.min() >= 0, "Severity should be non-negative"


class TestLeakageDetection:
    """Test cases for temporal leakage detection"""

    def test_check_temporal_leakage_clean(self, sample_injury_data):
        """Test leakage detection with clean data"""
        report = check_temporal_leakage(
            df=sample_injury_data,
            date_column='Date of Injury',
            feature_columns=['age', 'workload', 'position']
        )

        assert 'warnings' in report
        assert 'n_warnings' in report
        assert isinstance(report['warnings'], list)

    def test_check_temporal_leakage_with_issues(self):
        """Test leakage detection with problematic features"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100),
            'target': np.random.randint(0, 2, 100),
            'feature': np.random.randn(100),
            'future_feature': np.random.randn(100),  # Suspicious name
            'next_value': np.random.randn(100),  # Suspicious name
        })

        report = check_temporal_leakage(
            df=df,
            date_column='date',
            feature_columns=['feature', 'future_feature', 'next_value']
        )

        # Should detect suspicious feature names
        assert report['n_warnings'] > 0, "Should detect suspicious feature names"


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_split_warning(self, sample_injury_data):
        """Test that empty splits generate appropriate warnings"""
        # Use dates that will create empty test set
        with pytest.warns(UserWarning):
            train_df, val_df, test_df = temporal_split(
                df=sample_injury_data,
                date_column='Date of Injury',
                train_end='2022-06-30',
                val_end='2025-12-31',  # After all data
                target_column=None
            )

    def test_string_date_parsing(self):
        """Test that string dates are parsed correctly"""
        df = pd.DataFrame({
            'date': ['2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01'],
            'target': [0, 1, 0, 1],
            'feature': [1, 2, 3, 4]
        })

        train_df, val_df, test_df = temporal_split(
            df=df,
            date_column='date',
            train_end='2021-06-30',
            val_end='2022-06-30',
            target_column=None
        )

        # Should parse dates and split correctly
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0

    def test_missing_dates_warning(self):
        """Test that missing dates generate warnings"""
        df = pd.DataFrame({
            'date': pd.Series(['2020-01-01', None, '2022-01-01', '2023-01-01']),
            'target': [0, 1, 0, 1],
            'feature': [1, 2, 3, 4]
        })

        with pytest.warns(UserWarning):
            train_df, val_df, test_df = temporal_split(
                df=df,
                date_column='date',
                train_end='2021-06-30',
                val_end='2022-06-30',
                target_column=None
            )


class TestIntegration:
    """Integration tests with real-world scenarios"""

    def test_full_ml_pipeline(self, sample_injury_data):
        """Test complete ML pipeline with temporal splits"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = get_temporal_classification_splits(
            df=sample_injury_data,
            date_column='Date of Injury',
            target_column='injury_label',
            train_end='2022-06-30',
            val_end='2023-06-30'
        )

        # Drop non-numeric columns for simple model
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train_num = X_train[numeric_cols]
        X_val_num = X_val[numeric_cols]
        X_test_num = X_test[numeric_cols]

        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train_num, y_train)

        # Validate
        val_probs = model.predict_proba(X_val_num)[:, 1]
        val_auc = roc_auc_score(y_val, val_probs)

        # Test
        test_probs = model.predict_proba(X_test_num)[:, 1]
        test_auc = roc_auc_score(y_test, test_probs)

        # Basic sanity checks
        assert 0 <= val_auc <= 1, "AUC should be between 0 and 1"
        assert 0 <= test_auc <= 1, "AUC should be between 0 and 1"

    def test_comparison_function(self, sample_injury_data):
        """Test random vs temporal comparison function"""
        comparison = compare_random_vs_temporal_split(
            df=sample_injury_data,
            date_column='Date of Injury',
            target_column='injury_label',
            train_end='2022-06-30',
            val_end='2023-06-30'
        )

        # Check output structure
        assert isinstance(comparison, pd.DataFrame)
        assert 'Split Method' in comparison.columns
        assert 'Train Size' in comparison.columns
        assert len(comparison) == 2  # Random and Temporal


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '--tb=short'])
