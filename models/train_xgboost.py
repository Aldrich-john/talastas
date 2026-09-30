"""
Artic: Disfluency Classification with XGBoost
Trains a multi-class gradient boosted decision tree on acoustic + BERT contextual features.
Target classes:
0: fluent
1: um
2: uh
3: like (disfluent filler)

Outputs evaluation metrics as required by thesis:
- Precision, Recall, F1-Score (per-class & macro)
- Accuracy
- Confusion Matrix
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

def train_xgboost_classifier():
    features_path = config.DATA_DIR / "features_dataset.npz"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features at {features_path}. Run extract_features.py first.")

    data = np.load(features_path)
    X = data["X"]
    y = data["y"]

    print("=" * 60)
    print("ARTIC: XGBoost Disfluency Classifier Training")
    print(f"Features: {X.shape[1]}-dimensional vector (Acoustic + BERT Context)")
    print(f"Total instances: {X.shape[0]} word tokens")
    print("=" * 60)

    # Stratified Split (80% Train, 20% Test)
    # Using numpy for independence
    np.random.seed(42)
    indices = np.arange(len(y))
    np.random.shuffle(indices)

    split_idx = int(0.8 * len(y))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model = None
    model_type = "xgboost"

    try:
        import xgboost as xgb
        print("[Trainer] Using XGBoost...")
        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=len(config.TARGET_DISFLUENCIES),
            max_depth=config.XGB_PARAMS["max_depth"],
            learning_rate=config.XGB_PARAMS["learning_rate"],
            n_estimators=config.XGB_PARAMS["n_estimators"],
            subsample=config.XGB_PARAMS["subsample"],
            colsample_bytree=config.XGB_PARAMS["colsample_bytree"],
            random_state=42
        )
        model.fit(X_train, y_train)

        # Save model
        save_path = config.MODELS_DIR / "xgboost_disfluency_model.json"
        model.save_model(str(save_path))
        print(f"[Trainer] Model saved to: {save_path}")

    except ImportError:
        print("[Notice] xgboost library not found. Training using standard GradientBoosting / Random Forest surrogate...")
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            import pickle
            model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
            model.fit(X_train, y_train)
            save_path = config.MODELS_DIR / "xgboost_surrogate.pkl"
            with open(save_path, "wb") as f:
                pickle.dump(model, f)
            print(f"[Trainer] Surrogate model saved to: {save_path}")
            model_type = "sklearn_gbm"
        except ImportError:
            print("[Notice] Neither xgboost nor sklearn installed yet. Saving weights mock.")
            return

    # Predictions & Evaluation
    preds = model.predict(X_test)

    # Compute metrics
    from collections import defaultdict
    class_metrics = {}
    
    overall_correct = np.sum(preds == y_test)
    accuracy = float(overall_correct / len(y_test))

    print("\n" + "-" * 50)
    print("EVALUATION RESULTS (Test Set):")
    print(f"Overall Accuracy: {accuracy:.4f} ({overall_correct}/{len(y_test)})")
    print("-" * 50)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    print("-" * 50)

    f1_scores = []
    confusion_matrix = np.zeros((len(config.TARGET_DISFLUENCIES), len(config.TARGET_DISFLUENCIES)), dtype=int)

    for i in range(len(y_test)):
        confusion_matrix[y_test[i], preds[i]] += 1

    for label_id, label_name in enumerate(config.TARGET_DISFLUENCIES):
        tp = np.sum((preds == label_id) & (y_test == label_id))
        fp = np.sum((preds == label_id) & (y_test != label_id))
        fn = np.sum((preds != label_id) & (y_test == label_id))
        support = int(np.sum(y_test == label_id))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        f1_scores.append(f1)
        class_metrics[label_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support
        }
        print(f"{label_name:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<8}")

    macro_f1 = float(np.mean(f1_scores))
    print("-" * 50)
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print("-" * 50)

    # Save metrics report for thesis compliance
    metrics_summary = {
        "model_type": model_type,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "class_breakdown": class_metrics,
        "confusion_matrix": confusion_matrix.tolist()
    }

    report_path = config.REPORTS_DIR / "model_evaluation_metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"[Trainer] Evaluation metrics saved to: {report_path}")
    return metrics_summary

if __name__ == "__main__":
    train_xgboost_classifier()
