import pandas as pd
import numpy as np
import logging
import os
import joblib
import boto3
from io import StringIO
import json
import time
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, balanced_accuracy_score,
                             r2_score, mean_absolute_error, mean_squared_error,
                             mean_absolute_percentage_error)
from preprocessing import preprocess


# CONFIGURE  LOGGING
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def predict(input_path=None, df=None, model_dir="/opt/ml/model", output_path=None):
    t_start = time.time()
    logger.info(" Starting prediction pipeline")
    os.makedirs(model_dir, exist_ok=True)

    # LOAD  OR  RECEIVE  DATA
    if df is None and input_path is None:
        raise ValueError("Either 'input_path' or 'df' must be provided.")

    if df is None:
        logger.info(f"Loading input data from: {input_path}")
        df = pd.read_csv(input_path)
    else:
        logger.info(f"Using provided DataFrame with shape: {df.shape}")
    logger.info(f"Initial data shape: {df.shape}")

    newpayup_missing_mask = df["NewPayUp"].isnull().astype(int)
    df_raw_input = df.copy()

    # LOAD  MODEL  ARTIFACTS
    clf = joblib.load(os.path.join(model_dir, "classifier_model.joblib"))
    reg = joblib.load(os.path.join(model_dir, "regressor_model.joblib"))
    imputer = joblib.load(os.path.join(model_dir, "imputer.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    logger.info("Loaded model artifacts")

    feature_path = os.path.join(model_dir, 'feature_columns.joblib')
    trained_feature_cols = joblib.load(feature_path)
    logger.info("Loaded feature columns used at training time")

    # PRE-PROCESSING
    df, _log, _imp, _scl = preprocess(df, is_training=False, scale=True,
                                      return_diff=True, imputer=imputer, scaler=scaler)
    logger.info(f"Data shape after preprocessing: {df.shape}")

    regression_target = "NewPayUp"
    classification_target = "NewPayUp_missing"
    exclude_cols = [regression_target, classification_target]
    # Align input features with those used during training
    numeric_cols = trained_feature_cols
    missing_feats = [c for c in numeric_cols if c not in df.columns]
    if missing_feats:
        logger.warning(
            f"Input data missing expected feature columns: {missing_feats}. "
            "Filling with zeros.")
        for c in missing_feats:
            df[c] = 0.0

    extra_feats = [c for c in df.select_dtypes(include=["float64", "int64"]).columns
                   if c not in numeric_cols + exclude_cols]
    if extra_feats:
        logger.info(f"Ignoring extra feature columns not seen in training: {extra_feats}")

    X = df[numeric_cols]
    logger.info(f"Numeric columns used in X: {numeric_cols}")

    # CLASSIFY WHETHER PayUp IS MISSING
    missing_pred = clf.predict(X)
    df["is_missing_predicted"] = missing_pred
    logger.info(
        f"Missingness predictions – missing: {missing_pred.sum()} "
        f"({missing_pred.mean()*100:.2f}%), present: {(missing_pred == 0).sum()} "
        f"({(missing_pred == 0).mean()*100:.2f}%)")

    # REGRESS PayUp ONLY WHERE IT’S PRESENT
    df["predicted_NewPayUp"] = np.nan
    mask = (missing_pred == 0) & (newpayup_missing_mask == 0)
    if mask.any():
        df.loc[mask, "predicted_NewPayUp"] = reg.predict(X[mask])
        logger.info(f"Predicted regression values for {mask.sum()} rows")

    # Residuals
    df["NewPayUp"] = df["NewPayUp"].round(4)
    df["predicted_NewPayUp"] = df["predicted_NewPayUp"].round(4)
    df["residual"] = df[regression_target] - df["predicted_NewPayUp"]
    df.loc[df["is_missing_predicted"] == 1, "residual"] = 0.0
    df["residual"] = df["residual"].round(5)
    logger.info("Calculated residuals")

    # STEP 3 -- ANOMALY  DETECTION
    excluded_from_anomaly = ["UPB"]
    X_anomaly = X.drop(columns=excluded_from_anomaly, errors="ignore")
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    iso_forest.fit(X_anomaly)
    anomaly_scores = -iso_forest.score_samples(X_anomaly)

    threshold_99 = np.percentile(anomaly_scores, 99)
    df["is_anomaly"] = (anomaly_scores > threshold_99).astype(int)
    df["anomaly_residuals"] = anomaly_scores.round(4)
    df["anomaly_threshold"] = round(threshold_99, 4)
    logger.info(
        f"Anomaly detection completed with threshold: {threshold_99:.4f}")

    # RESTORE  HUMAN-FRIENDLY  /  LEGACY COLS
    cols_to_restore = [
        "ObservationDate", "Agency", "Maturity", "Coupon", "OrigYear", "ActualSpread",
        "Dealer1SourceSpread", "Dealer2AnalyticsSpread", "EpsilonBenchSpread",
        "AllDealerAverage", "AllDealerAverageDispersion", "Dealer1Dealer2Average",
        "Dealer1Dealer2AverageDispersion", "AllDealerAverageThresh", "Dealer1Dealer2Thresh",
        "UPB", "UPBThreshold", "TicksThreshold", "OrigYearThreshold",
        "UseAllDealerAverage", "UseDealer1Dealer2Average", "UseDealer2AnalyticsSpread",
        "UseDealer1SourceSpread"
    ]
    for col in cols_to_restore:
        if col in df.columns and col in df_raw_input.columns:
            df[col] = df_raw_input[col]

    # Target-first ordering for readability
    desired_order = [
        "ObservationDate", "Agency", "Maturity", "Coupon", "OrigYear", "ActualSpread",
        "NewPayUp", "predicted_NewPayUp", "residual",
        "NewPayUp_missing", "is_missing_predicted", "is_anomaly", "anomaly_residuals",
        "anomaly_threshold", "Dealer1SourceSpread", "Dealer2AnalyticsSpread",
        "EpsilonBenchSpread", "AllDealerAverage", "AllDealerAverageDispersion",
        "Dealer1Dealer2Average", "Dealer1Dealer2AverageDispersion",
        "AllDealerAverageThresh", "Dealer1Dealer2Thresh", "UPB",
        "UPBThreshold", "TicksThreshold", "OrigYearThreshold",
        "UseAllDealerAverage", "UseDealer1Dealer2Average",
        "UseDealer2AnalyticsSpread", "UseDealer1SourceSpread",
        # engineered flags follow …
    ]
    missing_cols = [c for c in desired_order if c not in df.columns]
    if missing_cols:
        logger.warning(f"Missing expected columns in output: {missing_cols}")

    ordered_cols = [c for c in desired_order if c in df.columns] + \
                   [c for c in df.columns if c not in desired_order]
    df = df[ordered_cols]
    logger.info("Reordered columns for final output")

    # METRICS  BLOCK
    # >>> Ground-truth & predictions for classification
    y_true = newpayup_missing_mask.values
    y_pred = missing_pred

    # >>> Ground-truth & predictions for regression (only valid rows)
    y_reg_true = df.loc[mask,
                        "NewPayUp"].values if mask.any() else np.array([])
    y_reg_pred = df.loc[mask, "predicted_NewPayUp"].values if mask.any(
    ) else np.array([])

    # Compute confusion matrix for richer storytelling
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Mean-squared-error (for RMSE)
    mse_val = mean_squared_error(
        y_reg_true, y_reg_pred) if mask.any() else np.nan

    stats = {
        # DATA  SNAPSHOTS
        "Input Data Dimensions": df_raw_input.shape,
        "Post-Processed Data Dimensions": df.shape,
        "Final Output Feature Count": len(df.columns),
        "Final Output Columns": list(df.columns),

        #  SOURCES
        "Data Source": input_path if input_path else "in-memory DataFrame",
        "Classifier Model Artifact": "classifier_model.joblib",
        "Regressor Model Artifact": "regressor_model.joblib",
        "Scaler / Imputer Artifacts": ["scaler.joblib", "imputer.joblib"],

        #  CLASSIFICATION
        "Missing Target Values (NewPayUp)": int(newpayup_missing_mask.sum()),
        "Rows Predicted Missing": int(tp + fp),
        "Rows Predicted Present": int(tn + fn),
        "Classifier Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Balanced Accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "Precision (Missing class)": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall (Missing class)": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1-Score (Missing class)": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "True Positives": int(tp),
        "False Positives": int(fp),
        "True Negatives": int(tn),
        "False Negatives": int(fn),

        # REGRESSION
        "Rows Regressed (NewPayUp present)": int(mask.sum()),
        "PayUp Prediction R²": round(r2_score(y_reg_true, y_reg_pred), 4) if mask.any() else None,
        "Mean Absolute Error (MAE)": round(mean_absolute_error(y_reg_true, y_reg_pred), 4) if mask.any() else None,
        "Root Mean Squared Error (RMSE)": round(np.sqrt(mse_val), 4) if mask.any() else None,
        "Mean Absolute % Error (MAPE)": round(mean_absolute_percentage_error(y_reg_true, y_reg_pred), 4) if mask.any() else None,
        "Residual Mean": round(df.loc[mask, "residual"].mean(), 5) if mask.any() else None,
        "Residual Std Dev": round(df.loc[mask, "residual"].std(), 5) if mask.any() else None,

        # UNSUPERVISED  OUTLIERS
        "Anomaly Detection Threshold (99th %)": round(threshold_99, 4),
        "Total Anomalies Detected": int(df["is_anomaly"].sum()),
        "Anomaly Detection Rate": round(df["is_anomaly"].mean(), 4),
        "Average Anomaly Score": round(np.mean(anomaly_scores), 4),
        "Anomaly Score Std Dev": round(np.std(anomaly_scores), 4),
        "Maximum Anomaly Score": round(np.max(anomaly_scores), 4),

        #  PERFORMANCE
        "Total Runtime (seconds)": round(time.time() - t_start, 2),
        "Output Path": output_path if output_path else "local_default"
    }

    # SAVE  OUTPUT
    try:
        if output_path and output_path.startswith("s3://"):
            s3 = boto3.client("s3")
            bucket, key = output_path[5:].split("/", 1)
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
            logger.info(f"Saved prediction output to S3: {output_path}")
        else:
            # Local fallback
            local_dir = output_path if output_path else os.path.abspath(
                os.path.join(os.getcwd(), ".", "output"))
            os.makedirs(local_dir, exist_ok=True)
            local_file = os.path.join(local_dir, "predicted_data_output.csv")
            df.to_csv(local_file, index=False)
            logger.info(f"Saved prediction output locally to: {local_file}")

            # Save statistics
            metrics_file = os.path.join(
                local_dir, "predicted_data_output_metrics.csv")
            pd.DataFrame([stats]).to_csv(metrics_file, index=False)
            logger.info(f"Saved prediction metrics locally to: {metrics_file}")

        logger.info(" Prediction pipeline completed")

        return {
            "status": "success",
            "message": "Prediction completed successfully",
            "records": len(df),
            "columns": list(df.columns),
            "statistics": stats
        }

    except Exception as e:
        logger.error(f"Error during output saving: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "statistics": stats
        }


if __name__ == "__main__":
    # Example local run
    predict(input_path="data/test_data.csv")
