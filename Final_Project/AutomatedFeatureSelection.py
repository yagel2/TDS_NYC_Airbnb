import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def remove_low_correlated_features(X, y, corr_threshold=0.05):
  """
  Removes features that are not collerated with the target feature.
  """

  correlations = X.corrwith(y)
  important_corr_features = correlations[abs(correlations) > corr_threshold].index.tolist()
  new_X = X[important_corr_features]

  return new_X

def remove_highly_correlated_features(X, y, corr_threshold=0.85):
    """
    Removes one of each pair of features with high correlation.
    Keeps the feature that is more correlated with the target.
    """
    # Absolute correlation matrix
    corr_matrix = X.corr().abs()

    # Correlation of each feature with target
    target_corr = X.corrwith(y).abs()

    # Set of columns to remove
    to_remove = set()

    # Iterate through correlation matrix
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            feature_1 = corr_matrix.columns[i]
            feature_2 = corr_matrix.columns[j]

            # Check if the pair is highly correlated
            if corr_matrix.iloc[i, j] > corr_threshold:
                # Compare correlation with target
                if target_corr[feature_1] < target_corr[feature_2]:
                    to_remove.add(feature_1)
                else:
                    to_remove.add(feature_2)

    # Drop less relevant correlated features
    X_filtered = X.drop(columns=list(to_remove))

    return X_filtered

def calculate_feature_importance(X, y):
  """
  Calculates for each feature its importance in the XGBRegressor.
  """

  # train-test split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Initialize and train XGBRegressor
  xgb_model = XGBRegressor()
  xgb_model.fit(X_train, y_train)

  # Predictions (optional, if you need them)
  y_pred_test = xgb_model.predict(X_test)
  y_pred_train = xgb_model.predict(X_train)

  # SHAP analysis using the trained regressor
  explainer = shap.TreeExplainer(xgb_model)
  shap_values = explainer.shap_values(X_train)

  # Aggregate SHAP values
  shap_importance = np.abs(shap_values).mean(axis=0)

  # Rank features by SHAP importance
  shap_importance_df = pd.DataFrame({
      'feature': X_train.columns,
      'importance': shap_importance
  }).sort_values(by='importance', ascending=False)

  return shap_importance_df

def calculate_best_features(X, y, shap_importance_df):
  """
  Calculates the features to select.
  """

  # Store the best r^2 we got in this variable(init to -1 as we got no r^2 yet)
  best_r2 = -1

  for top_k in range(1, len(shap_importance_df) + 1):
    selected_features = shap_importance_df['feature'].iloc[:top_k].tolist()

    # Final dataset
    X_final = X[selected_features]

    # train-test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

    # Retrain final model on training set
    xgb_model = XGBRegressor()
    xgb_model.fit(X_train, y_train)

    # Predictions
    y_pred = xgb_model.predict(X_test)

    # Evaluate r^2 metric
    r2 = r2_score(y_test, y_pred)

    # If model did better, update he value of the best r^2 we got.
    if r2 > best_r2:
      best_r2 = r2

    # If model did worse, return the features without the last one.
    else:
      return selected_features[:-1]
  return selected_features

def select_features(X, y):
  """
  Returns the features to select.
  """
  X = remove_low_correlated_features(X, y, corr_threshold=0.05)
  X = remove_highly_correlated_features(X, y, corr_threshold=0.85)
  shap_importance_df = calculate_feature_importance(X, y)
  selected_features = calculate_best_features(X, y, shap_importance_df)
  return selected_features
