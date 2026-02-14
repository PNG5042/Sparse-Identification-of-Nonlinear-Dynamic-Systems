import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# will work on it more when I get back home, also need to go through old Tensile Test to see what I can reuse