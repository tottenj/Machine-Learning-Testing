#Produces Fairness Across Mitigation Strategies

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import EqualizedOdds, GridSearch
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]
data = pd.read_csv(url, names=column_names, sep=',\s+', engine='python')

# Preprocessing: Convert categorical columns to numerical using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Split data into features (X) and target (y)
X = data.drop('income_>50K', axis=1)  # Assuming 'income_>50K' is our target
y = data['income_>50K']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Base model (Logistic Regression) - fit without the sensitive feature
base_model = LogisticRegression(max_iter=1000)
base_model.fit(X_train, y_train)  # Fit without sensitive feature in model

# Mitigation strategies to evaluate
mitigation_strategies = {
    'Threshold Optimizer (Equalized Odds)': ThresholdOptimizer(
        estimator=base_model,
        constraints='equalized_odds'  # Apply Equalized Odds post-processing
    ),
    'GridSearch (Equalized Odds)': GridSearch(
        estimator=base_model,
        constraints=EqualizedOdds(),
        grid_size=20
    ),
}

# Function to calculate fairness metrics
def calculate_fairness_metrics(model, X_test, y_test, sensitive_feature):
    # Get predictions from the model
    preds = model.predict(X_test) if not isinstance(model, ThresholdOptimizer) else model.predict(X_test, sensitive_features=X_test[sensitive_feature])
    
    # Initialize a dictionary to store the metrics
    metrics = {}
    
    # Split data into groups based on the sensitive feature
    group_0 = X_test[sensitive_feature] == 0
    group_1 = X_test[sensitive_feature] == 1
    
    # Calculate Demographic Parity (DP)
    rate_group_0 = np.mean(preds[group_0])
    rate_group_1 = np.mean(preds[group_1])
    dp = abs(rate_group_0 - rate_group_1)
    metrics['Demographic Parity'] = dp

    # Calculate confusion matrix components for each group
    cm_group_0 = confusion_matrix(y_test[group_0], preds[group_0], labels=[0, 1])
    cm_group_1 = confusion_matrix(y_test[group_1], preds[group_1], labels=[0, 1])

    # Extract TN, FP, FN, TP for each group
    tn_0, fp_0, fn_0, tp_0 = cm_group_0.ravel()
    tn_1, fp_1, fn_1, tp_1 = cm_group_1.ravel()

    # Calculate False Positive Rates (FPR) and False Negative Rates (FNR) for each group
    fpr_0 = fp_0 / (fp_0 + tn_0) if (fp_0 + tn_0) > 0 else 0
    fpr_1 = fp_1 / (fp_1 + tn_1) if (fp_1 + tn_1) > 0 else 0
    fnr_0 = fn_0 / (fn_0 + tp_0) if (fn_0 + tp_0) > 0 else 0
    fnr_1 = fn_1 / (fn_1 + tp_1) if (fn_1 + tp_1) > 0 else 0

    # Equalized Odds (EO) differences
    eo_fpr = abs(fpr_0 - fpr_1)
    eo_fnr = abs(fnr_0 - fnr_1)
    metrics['Equalized Odds FPR'] = eo_fpr
    metrics['Equalized Odds FNR'] = eo_fnr

    # Disparate Impact (DI)
    di = (rate_group_1 / rate_group_0) if rate_group_0 > 0 else 0
    metrics['Disparate Impact'] = di

    return metrics

# Store fairness metrics for each strategy
fairness_metrics = []

# Evaluate each strategy for fairness
for strategy_name, strategy in mitigation_strategies.items():
    if strategy is None:  # No mitigation, use base model
        preds = base_model.predict(X_test)  # Use base model
        fairness_metrics_before = calculate_fairness_metrics(base_model, X_test, y_test, sensitive_feature='sex_Male')
        for metric in fairness_metrics_before:
            fairness_metrics.append({
                'Strategy': strategy_name,
                'Metric': metric,
                'Fairness Value': fairness_metrics_before[metric],
                'Mitigation': 'Before Mitigation'
            })
    else:
        if strategy_name == 'Threshold Optimizer (Equalized Odds)':
            strategy.fit(X_train, y_train, sensitive_features=X_train['sex_Male'])
            preds = strategy.predict(X_test, sensitive_features=X_test['sex_Male'])
        else:  # For GridSearch or other mitigators
            strategy.fit(X_train, y_train, sensitive_features=X_train['sex_Male'])
            preds = strategy.predict(X_test)
        
        fairness_metrics_after = calculate_fairness_metrics(strategy, X_test, y_test, sensitive_feature='sex_Male')
        for metric in fairness_metrics_after:
            fairness_metrics.append({
                'Strategy': strategy_name,
                'Metric': metric,
                'Fairness Value': fairness_metrics_after[metric],
                'Mitigation': 'After Mitigation'
            })

# Convert fairness metrics to DataFrame for easy comparison
fairness_df = pd.DataFrame(fairness_metrics)

# Plot fairness metrics comparison
plt.figure(figsize=(14, 7))
sns.barplot(x='Fairness Value', y='Strategy', hue='Mitigation', data=fairness_df, ci=None)
plt.title('Fairness Metrics Comparison Across Mitigation Strategies')
plt.ylabel('Mitigation Strategy')
plt.xlabel('Fairness Metric Value')
plt.tight_layout()
plt.show()
