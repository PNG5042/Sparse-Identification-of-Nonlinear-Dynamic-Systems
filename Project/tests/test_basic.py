"""Basic tests to verify environment setup"""
import numpy as np
import pandas as pd
import pytest


def test_numpy_available():
    """Test that numpy is installed and working"""
    arr = np.array([1, 2, 3, 4, 5])
    assert arr.mean() == 3.0
    assert arr.sum() == 15


def test_pandas_available():
    """Test that pandas is installed and working"""
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    assert len(df) == 3
    assert list(df.columns) == ['a', 'b']


def test_scipy_available():
    """Test that scipy is available"""
    from scipy import optimize
    assert optimize is not None


def test_matplotlib_available():
    """Test that matplotlib is available"""
    import matplotlib.pyplot as plt
    assert plt is not None


class TestNumpyOperations:
    """Test numpy mathematical operations"""
    
    def test_array_creation(self):
        arr = np.linspace(0, 1, 11)
        assert len(arr) == 11
        assert arr[0] == 0.0
        assert arr[-1] == 1.0
    
    def test_array_math(self):
        arr = np.array([1, 2, 3, 4, 5])
        assert np.sum(arr) == 15
        assert np.mean(arr) == 3.0
        assert np.std(arr) > 0


class TestPandasOperations:
    """Test pandas data operations"""
    
    def test_dataframe_creation(self):
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [4, 5, 6]
        })
        assert len(df) == 3
        assert 'x' in df.columns
        assert 'y' in df.columns
    
    def test_dataframe_operations(self):
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        assert df['a'].mean() == 3.0
        assert df['a'].sum() == 15
