import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def preprocess(df, is_training=True, scale=False, return_diff=False, imputer=None, scaler=None):
    # Log the start of preprocessing
    logger.info("Starting preprocessing")

    # Preserve original dataframe for comparison
    original_df = df.copy()
    df = df.copy()

    # Initialize log for tracking transformations
    log = {
        "dropped_rows": [],
        "rows_with_missing_before_impute": [],
        "imputed": {},
        "converted": [],
        "encoded": [],
        "scaled": [],
        "dropped_columns": []
    }

    # Drop derived columns if they exist
    derived_cols = ['PayupDifference', 'Impact', 'AnamolyFlag', 'SourceFile']
    for col in derived_cols:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
            logger.info(f"Dropped derived column: {col}")

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    logger.info("Replaced inf/-inf values with NaN")

    # Create missing flag for target if applicable
    if 'NewPayUp' in df.columns:
        df["NewPayUp_missing"] = df["NewPayUp"].isnull().astype(int)
        logger.info("Created 'NewPayUp_missing' flag")

    # Define columns where negative values are valid and should not have flags
    valid_negative_cols = {
        "ActualSpread",
        "Dealer1SourceSpread",
        "Dealer2AnalyticsSpread",
        "EpsilonBenchSpread",
        "PayupDifference",
        "Impact"
    }

    # Add binary flags for negative values in numeric columns (excluding valid-negative ones)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ["NewPayUp", "NewPayUp_missing"] and col not in valid_negative_cols:
            df[f"{col}_is_negative"] = (df[col] < 0).astype(int)
            log["converted"].append(f"{col}_is_negative")
    logger.info(
        "Added negative value flags for numeric columns (excluding valid-negative columns)")

    # Preserve original data types
    original_dtypes = original_df.dtypes

    # Attempt to convert object columns to numeric where possible
    for col in df.select_dtypes(include=['object']).columns:
        try:
            converted = pd.to_numeric(df[col])
            if not converted.equals(df[col]):
                log["converted"].append(col)
            df[col] = converted
            logger.info(f"Converted object column to numeric: {col}")
        except:
            continue

    # Encode remaining object columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        log["encoded"].append(col)
        logger.info(f"Encoded categorical column: {col}")

    # Record rows with missing values before imputation
    log["rows_with_missing_before_impute"] = df[df.isnull().any(axis=1)
                                                ].index.tolist()
    logger.info(
        f"Rows with missing values before imputation: {len(log['rows_with_missing_before_impute'])}")

    # Re-evaluate numeric columns after conversions
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in [
        "NewPayUp", "NewPayUp_missing"]]

    # ONLY drop all-NaN columns if training
    if is_training:
        dropped_nan_cols = [
            col for col in numeric_cols if df[col].isna().all()]
        if dropped_nan_cols:
            df.drop(columns=dropped_nan_cols, inplace=True)
            log["dropped_columns"].extend(dropped_nan_cols)
            logger.info(
                f"Dropped columns with all NaN values: {dropped_nan_cols}")

    # Final numeric column set before imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in [
        "NewPayUp", "NewPayUp_missing"]]

    # Impute missing values
    if imputer is None:
        imputer = SimpleImputer(strategy="median")
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        logger.info("Fitted and applied median imputer")
    else:
        df[numeric_cols] = imputer.transform(df[numeric_cols])
        logger.info("Applied existing median imputer")

    # Log columns that were imputed
    for col in numeric_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            log["imputed"][col] = int(df[col].isnull().sum())
    logger.info(f"Columns imputed: {list(log['imputed'].keys())}")

    # Scale features if requested
    if scale:
        scale_cols = [col for col in df.columns if col not in [
            "NewPayUp", "NewPayUp_missing"]]
        if scaler is None:
            scaler = StandardScaler()
            df[scale_cols] = scaler.fit_transform(df[scale_cols])
            logger.info("Fitted and applied standard scaler")
        else:
            df[scale_cols] = scaler.transform(df[scale_cols])
            logger.info("Applied existing standard scaler")
        log["scaled"] = scale_cols

    # Return results
    if return_diff:
        logger.info("Returning dataframe, log, imputer, and scaler")
        return df, log, imputer, scaler

    logger.info("Returning preprocessed dataframe")
    return df
