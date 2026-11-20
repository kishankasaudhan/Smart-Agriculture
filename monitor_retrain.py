"""Model-quality monitoring and retraining pipeline for Crop Yield Prediction.

Supports all Assessment Tasks A through E:
Task A: Prepare new data (validate columns, missing, duplicates, size, distribution).
Task B: Evaluate existing champion model (MAE, RMSE, R2, save report).
Task C: Retrain candidate on combined data, save candidate model artifact, track in MLflow if available.
Task D: Implement quality gate (RMSE comparison, R2 guardrail, decision logging).
Task E: Demonstration, execution evidence, reports and artifact generation.
"""
from pathlib import Path
import argparse
import json
import os
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Try importing MLflow if available in the environment
try:
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

DATA_PATH = Path(os.getenv("DATA_PATH", "data/crop_yield.csv"))
MODEL_DIR = Path("models")
CHAMPION_MODEL_PATH = MODEL_DIR / "crop_yield_model.joblib"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:6001")
EXPERIMENT_NAME = "smart-agriculture-yield"
REGISTERED_MODEL = "crop-yield-predictor"

NUMERIC_FEATURES = [
    "nitrogen", "phosphorus", "potassium", "temperature",
    "humidity", "ph", "rainfall"
]
CATEGORICAL_FEATURES = ["crop_type", "soil_type"]
TARGET = "yield_tons_per_hectare"

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED = FEATURES + [TARGET]


def load_historical_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"Historical dataset missing columns: {missing}")
    df = df[REQUIRED].copy()
    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[TARGET])
    return df


def build_pipeline(model):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES)
    ])
    return Pipeline([("preprocess", preprocess), ("model", model)])


# ==========================================
# Task A: Prepare new data
# ==========================================
def load_and_validate_new_data(path: str) -> pd.DataFrame:
    print("=" * 60)
    print("Task A: Preparing & Validating New Incoming Data")
    print("=" * 60)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Incoming batch file not found at: {path}")

    df = pd.read_csv(path)
    print(f"1. Raw incoming batch loaded: {len(df)} rows, {len(df.columns)} columns.")

    # 1. Validate required columns
    missing_cols = sorted(set(REQUIRED) - set(df.columns))
    if missing_cols:
        raise ValueError(f"Validation Error: Incoming dataset missing required columns: {missing_cols}")
    print("   [PASSED] Required columns present:", REQUIRED)

    df = df[REQUIRED].copy()

    # 2. Convert and validate missing / invalid numeric fields
    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    missing_counts = df.isna().sum()
    if missing_counts.any():
        bad_cols = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(f"Validation Error: Missing or non-numeric values detected: {bad_cols}")
    print("   [PASSED] Zero missing or invalid numeric values.")

    # Check non-empty categorical features
    if df[CATEGORICAL_FEATURES].astype(str).apply(lambda s: s.str.strip().eq("")).any().any():
        raise ValueError("Validation Error: Incoming batch contains blank categorical entries.")
    print("   [PASSED] Categorical values validated.")

    # 3. Check duplicate records
    duplicate_count = int(df.duplicated().sum())
    print(f"   [PASSED] Duplicate records detected: {duplicate_count}")
    if duplicate_count > 0:
        print(f"   Removing {duplicate_count} duplicate rows...")
        df = df.drop_duplicates()

    # Check minimum batch size and valid bounds
    if len(df) < 10:
        raise ValueError(f"Incoming data has only {len(df)} records. Minimum 10 required.")
    if (df[TARGET] < 0).any():
        raise ValueError("Validation Error: Negative crop yield values detected in target column.")

    # 4. Display dataset size and target distribution
    print(f"2. Validated dataset size: {len(df)} records.")
    print("3. Target distribution summary (yield_tons_per_hectare):")
    stats = df[TARGET].describe().round(3)
    for k, v in stats.items():
        print(f"      {k:7s}: {v}")
    print()
    return df


def calculate_metrics(y_true, predictions):
    return {
        "mae": float(round(mean_absolute_error(y_true, predictions), 4)),
        "rmse": float(round(np.sqrt(mean_squared_error(y_true, predictions)), 4)),
        "r2": float(round(r2_score(y_true, predictions), 4)),
    }


# ==========================================
# Task B: Evaluate the existing model
# ==========================================
def load_champion_model(tracking_uri: str):
    # Attempt MLflow loading first if MLflow is available and server reachable
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient(tracking_uri=tracking_uri)
            version_info = client.get_model_version_by_alias(REGISTERED_MODEL, "champion")
            model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL}@champion")
            print(f"Loaded champion from MLflow Registry ({REGISTERED_MODEL}@champion, version {version_info.version})")
            return model, f"v{version_info.version}"
        except Exception as e:
            print(f"MLflow registry load failed ({e}), falling back to local champion artifact...")

    # Fallback to local saved champion model
    if not CHAMPION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Champion model not found at {CHAMPION_MODEL_PATH}. Run training first."
        )
    model = joblib.load(CHAMPION_MODEL_PATH)
    print(f"Loaded existing champion model from: {CHAMPION_MODEL_PATH}")
    return model, "v1_local"


def run_pipeline(
    new_data_path="data/new_crop_yield.csv",
    output_dir="reports",
    candidate_dir="models/candidate",
    tracking_uri=TRACKING_URI,
    min_rmse_improvement=0.0
):
    print("\n" + "=" * 60)
    print("MLOps Model-Quality Monitoring & Retraining Pipeline")
    print("Theme: Smart Agriculture Crop Yield Prediction")
    print("=" * 60 + "\n")

    # Task A: Prepare new data
    new_df = load_and_validate_new_data(new_data_path)
    old_df = load_historical_data(DATA_PATH)

    # Task B: Evaluate existing model
    print("=" * 60)
    print("Task B: Evaluating Existing Champion Model on New Batch")
    print("=" * 60)
    champion, champion_version = load_champion_model(tracking_uri)

    # Split incoming batch: 30% held out as untouched evaluation test set, 70% for retraining
    update_rows, eval_rows = train_test_split(new_df, test_size=0.30, random_state=42)
    X_eval, y_eval = eval_rows[FEATURES], eval_rows[TARGET]

    champion_preds = champion.predict(X_eval)
    champion_scores = calculate_metrics(y_eval, champion_preds)
    print(f"Champion ({champion_version}) Evaluation on New Unseen Batch ({len(eval_rows)} samples):")
    print(f"   MAE  : {champion_scores['mae']} tonnes/hectare")
    print(f"   RMSE : {champion_scores['rmse']} tonnes/hectare")
    print(f"   R²   : {champion_scores['r2']}")
    print()

    # Task C: Retrain the model
    print("=" * 60)
    print("Task C: Retraining Candidate Model on Combined Data")
    print("=" * 60)
    combined_df = pd.concat([old_df, update_rows], ignore_index=True).drop_duplicates()
    print(f"Historical training rows: {len(old_df)}")
    print(f"New incoming batch rows : {len(new_df)} (Retrain portion: {len(update_rows)}, Eval holdout: {len(eval_rows)})")
    print(f"Combined training set   : {len(combined_df)} records after deduplication.")

    X_train, y_train = combined_df[FEATURES], combined_df[TARGET]
    candidate = build_pipeline(RandomForestRegressor(
        n_estimators=250, max_depth=12, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    ))

    mlflow_run_id = None
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name="monitor-retrain-candidate") as active_run:
                candidate.fit(X_train, y_train)
                candidate_preds = candidate.predict(X_eval)
                candidate_scores = calculate_metrics(y_eval, candidate_preds)
                mlflow.log_params({
                    "workflow": "monitor_retrain",
                    "candidate_model": "RandomForestRegressor",
                    "historical_rows": len(old_df),
                    "incoming_rows": len(new_df),
                    "combined_rows": len(combined_df),
                    "champion_version": champion_version
                })
                mlflow.log_metrics({f"champion_{k}": v for k, v in champion_scores.items()})
                mlflow.log_metrics({f"candidate_{k}": v for k, v in candidate_scores.items()})
                mlflow.sklearn.log_model(candidate, name="candidate_model")
                mlflow_run_id = active_run.info.run_id
                print(f"[MLflow] Logged run {mlflow_run_id} to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            print(f"[MLflow Notice] MLflow tracking skipped or unavailable ({e}); proceeding with local training.")
            candidate.fit(X_train, y_train)
            candidate_preds = candidate.predict(X_eval)
            candidate_scores = calculate_metrics(y_eval, candidate_preds)
    else:
        candidate.fit(X_train, y_train)
        candidate_preds = candidate.predict(X_eval)
        candidate_scores = calculate_metrics(y_eval, candidate_preds)

    # Save Candidate Model Separately
    candidate_path = Path(candidate_dir)
    candidate_path.mkdir(parents=True, exist_ok=True)
    candidate_artifact_file = candidate_path / "crop_yield_candidate.joblib"
    joblib.dump(candidate, candidate_artifact_file)
    print(f"Candidate model artifact saved separately to: {candidate_artifact_file}")
    print()

    # Task D: Implement a Quality Gate
    print("=" * 60)
    print("Task D: Quality Gate Evaluation & Decision")
    print("=" * 60)
    rmse_improvement = champion_scores["rmse"] - candidate_scores["rmse"]
    mae_improvement = champion_scores["mae"] - candidate_scores["mae"]
    r2_improvement = candidate_scores["r2"] - champion_scores["r2"]

    print(f"Comparison Summary on Unseen Batch:")
    print(f"   Metric       | Champion ({champion_version}) | Candidate | Delta (Improvement)")
    print(f"   -------------+------------------------+-----------+--------------------")
    print(f"   RMSE (lower) | {champion_scores['rmse']:<22} | {candidate_scores['rmse']:<9} | {rmse_improvement:+.4f}")
    print(f"   MAE  (lower) | {champion_scores['mae']:<22} | {candidate_scores['mae']:<9} | {mae_improvement:+.4f}")
    print(f"   R²   (higher)| {champion_scores['r2']:<22} | {candidate_scores['r2']:<9} | {r2_improvement:+.4f}")

    # Decision rule:
    # 1. Primary: Candidate RMSE must be lower than Champion RMSE (rmse_improvement > min_rmse_improvement).
    # 2. Guardrail: Candidate R² must not degrade (candidate R² >= champion R²).
    is_eligible = (rmse_improvement > min_rmse_improvement) and (candidate_scores["r2"] >= champion_scores["r2"])
    decision = "ELIGIBLE_FOR_PROMOTION" if is_eligible else "RETAIN_EXISTING_CHAMPION"

    metric_justification = (
        "Selected Primary Metric: Root Mean Squared Error (RMSE). "
        "Justification: In agricultural crop yield forecasting, yield predictions guide mission-critical supply-chain, "
        "storage, and market pricing commitments. Large prediction errors impose disproportionately severe financial penalties "
        "on farming cooperatives. RMSE heavily penalizes large outlier errors via quadratic squaring, making it the most reliable "
        "measure of model stability. R² is enforced as a secondary guardrail to guarantee that overall explained variance does not regress."
    )

    print(f"\nQuality Gate Decision: {decision}")
    print(f"Decision Rule: Candidate RMSE must beat Champion by > {min_rmse_improvement} and R² >= Champion R²")
    print(f"Safety Guarantee: Champion model alias was NOT overwritten automatically.")
    print()

    # Save Reports
    report_path = Path(output_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    report_data = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_metadata": {
            "incoming_data_file": str(new_data_path),
            "historical_rows": len(old_df),
            "incoming_batch_rows": len(new_df),
            "combined_training_rows": len(combined_df),
            "evaluation_holdout_rows": len(eval_rows)
        },
        "champion_model": {
            "version": champion_version,
            "metrics": champion_scores
        },
        "candidate_model": {
            "artifact_path": str(candidate_artifact_file),
            "metrics": candidate_scores,
            "mlflow_run_id": mlflow_run_id
        },
        "quality_gate": {
            "primary_metric": "RMSE",
            "metric_justification": metric_justification,
            "threshold_min_rmse_improvement": min_rmse_improvement,
            "observed_rmse_improvement": round(rmse_improvement, 4),
            "observed_r2_improvement": round(r2_improvement, 4),
            "decision": decision,
            "champion_overwritten": False
        }
    }

    report_json_file = report_path / "monitoring_report.json"
    with open(report_json_file, "w") as f:
        json.dump(report_data, f, indent=2)

    pd.DataFrame([
        {"model": f"champion_{champion_version}", **champion_scores},
        {"model": "candidate", **candidate_scores}
    ]).to_csv(report_path / "evaluation_metrics.csv", index=False)

    with open(report_path / "quality_gate_decision.json", "w") as f:
        json.dump(report_data["quality_gate"], f, indent=2)

    print("=" * 60)
    print("Task E: Artifacts and Summary Report Generated")
    print("=" * 60)
    print(f"1. Monitoring Report JSON   : {report_json_file}")
    print(f"2. Evaluation Metrics CSV   : {report_path / 'evaluation_metrics.csv'}")
    print(f"3. Quality Gate Decision    : {report_path / 'quality_gate_decision.json'}")
    print(f"4. Candidate Model Artifact : {candidate_artifact_file}")
    print()
    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Monitoring & Retraining Pipeline")
    parser.add_argument("--new-data", default="data/new_crop_yield.csv", help="Path to new incoming batch CSV")
    parser.add_argument("--min-rmse-improvement", type=float, default=0.0, help="Minimum RMSE improvement required")
    args = parser.parse_args()
    run_pipeline(new_data_path=args.new_data, min_rmse_improvement=args.min_rmse_improvement)
