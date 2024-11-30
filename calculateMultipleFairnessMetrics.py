import numpy as np
from sklearn.metrics import confusion_matrix



# Function to calculate multiple fairness metrics
def calculate_multiple_fairness_metrics(model, X_test, y_test, sensitive_feature):
    # Predictions before mitigation
    preds_before = model.predict(X_test)
    
    # Confusion matrix for FP and FN rates
    cm = confusion_matrix(y_test, preds_before)
    tn, fp, fn, tp = cm.ravel()

    # Calculate Demographic Parity (DP)
    dp_before = abs(np.mean(preds_before[X_test[sensitive_feature] == 1]) - np.mean(preds_before[X_test[sensitive_feature] == 0]))

    # Calculate Equalized Odds (EO)
    fpr_before = fp / (fp + tn)  # False Positive Rate
    fnr_before = fn / (fn + tp)  # False Negative Rate
    eo_fpr_before = abs(fpr_before - fpr_before)  # EO based on FPR
    eo_fnr_before = abs(fnr_before - fnr_before)  # EO based on FNR

    # Calculate Disparate Impact (DI)
    di_before = np.mean(preds_before[X_test[sensitive_feature] == 1]) / np.mean(preds_before[X_test[sensitive_feature] == 0])

    # False Positive Rate Difference (FPRD)
    fprd_before = abs(fpr_before - fpr_before)

    # False Negative Rate Difference (FNRD)
    fnrd_before = abs(fnr_before - fnr_before)

    # Return metrics
    metrics_before = {
        "Demographic Parity": dp_before,
        "Equalized Odds FPR": eo_fpr_before,
        "Equalized Odds FNR": eo_fnr_before,
        "Disparate Impact": di_before,
        "FPRD": fprd_before,
        "FNRD": fnrd_before
    }
    
    return metrics_before