import pandas as pd
import numpy as np
import logging
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import xgboost as xgb
from preprocessing import preprocess

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_model(train_path=None, df=None, model_dir='/opt/ml/model'):

    # Create output directory
    os.makedirs(model_dir, exist_ok=True)

    default_path = os.path.join("/opt/ml/input/data/train", "train_data.csv")
    if (train_path is None or not os.path.exists(train_path)) and os.path.exists(default_path):
        train_path = default_path

    # Load training data
    if df is None and train_path is None:
        raise ValueError("Either 'train_path' or 'df' must be provided.")

    if df is None:
        logger.info(f"Loading training data from: {train_path}")
        df = pd.read_csv(train_path)

    logger.info(f"Training on data shape: {df.shape}")

    # Preprocess data and get back fitted imputer/scaler
    df, log, imputer, scaler = preprocess(
        df, is_training=True, scale=True, return_diff=True)
    logger.info(f"Data shape after preprocessing: {df.shape}")

    # Define targets and features
    regression_target = 'NewPayUp'
    classification_target = 'NewPayUp_missing'
    exclude_cols = [regression_target, classification_target]
    numeric_cols = [col for col in df.select_dtypes(
        include=['float64', 'int64']).columns if col not in exclude_cols]
    X = df[numeric_cols]
    y = df[classification_target]
    y_reg = df[regression_target]

    # Save feature columns used during training
    feature_path = os.path.join(model_dir, 'feature_columns.joblib')
    joblib.dump(numeric_cols, feature_path)
    logger.info(f"Saved feature columns to: {feature_path}")

    # Split for classification
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(
        f"Split classification data: train={len(X_train)}, val={len(X_val)}")

    # Train classifier
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    clf = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=5,
        scale_pos_weight=pos_weight
    )
    clf.fit(X_train, y_train)
    logger.info("Trained XGBoost classifier")

    # Evaluate classifier
    val_accuracy = clf.score(X_val, y_val)
    logger.info(f"Classifier validation accuracy: {val_accuracy:.4f}")
    logger.info("Classification report:\n" +
                classification_report(y_val, clf.predict(X_val)))

    cv_scores = cross_val_score(
        clf, X_train, y_train, cv=5, scoring='accuracy')
    logger.info(
        f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': clf.feature_importances_
    }).sort_values(by='importance', ascending=False)
    logger.info("Top 5 features:\n" +
                feature_importance.head(5).to_string(index=False))

    # Prepare data for regression
    # selecting only the rows where we do have a value for 'NewPayUp'
    mask = ~df[regression_target].isna()
    # Gets input features only for valid regression rows
    X_reg = df[mask][numeric_cols]
    # target vector
    y_reg = df[mask][regression_target]
    # Splits into train/val sets for supervised regression training
    X_reg_train, X_reg_val, y_reg_train, y_reg_val = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    logger.info(
        f"Regression data split: train={len(X_reg_train)}, val={len(X_reg_val)}")

    # Train regressor
    reg = xgb.XGBRegressor(objective='reg:squarederror',
                           n_estimators=100, max_depth=5)
    reg.fit(X_reg_train, y_reg_train)
    logger.info("Trained XGBoost regressor")

    # Evaluate regressor
    y_reg_pred = reg.predict(X_reg_val)
    mae = mean_absolute_error(y_reg_val, y_reg_pred)
    r2 = r2_score(y_reg_val, y_reg_pred)
    logger.info(f"Regression MAE: {mae:.4f}, R2 Score: {r2:.4f}")

    # Save models and preprocessing objects
    joblib.dump(clf, os.path.join(model_dir, 'classifier_model.joblib'))
    joblib.dump(reg, os.path.join(model_dir, 'regressor_model.joblib'))
    joblib.dump(imputer, os.path.join(model_dir, 'imputer.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    logger.info(f"Saved all model artifacts to: {model_dir}")


if __name__ == "__main__":
    local_data_path = 'data/train_data.csv'
    if os.path.exists(local_data_path):
        train_model(train_path=local_data_path)
    else:
        train_model()
