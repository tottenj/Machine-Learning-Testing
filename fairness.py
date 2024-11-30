import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from fairlearn.metrics import MetricFrame
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from evaluateFairness import evaluate_fairness
from fairlearn.reductions import EqualizedOdds, GridSearch
from fairlearn.reductions import DemographicParity
#from plotMultiple import plot_multiple_fairness_metrics_comparison
from scipy import stats


def calculate_multiple_fairness_metrics(model, X_test, y_test, sensitive_feature):
    # Get predictions from the model
    preds = model.predict(X_test)
    
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

    # False Positive Rate Difference (FPRD)
    metrics['FPRD'] = eo_fpr

    # False Negative Rate Difference (FNRD)
    metrics['FNRD'] = eo_fnr

    return metrics


# Function to calculate fairness metrics
def evaluate_fairness(model, X_test, y_test, sensitive_feature):
    # Predictions
    preds = model.predict(X_test)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    
    # MetricFrame for fairness evaluation
    metric_frame = MetricFrame(
        metrics={'accuracy': accuracy_score},
        y_true=y_test,
        y_pred=preds,
        sensitive_features=X_test[sensitive_feature]
    )
    
    print(f"Fairness Metrics for Sensitive Feature: {sensitive_feature}")
    print(metric_frame.by_group)
    
    return metric_frame


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


# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_preds)

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)

# Neural Network
nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
nn.fit(X_train, y_train)
nn_preds = nn.predict(X_test)
nn_accuracy = accuracy_score(y_test, nn_preds)

print(f"Logistic Regression Accuracy: {lr_accuracy}")
print(f"Random Forest Accuracy: {rf_accuracy}")
print(f"Neural Network Accuracy: {nn_accuracy}")



# For simplicity, let’s use 'sex' as the sensitive attribute (binary gender: male, female)
evaluate_fairness(lr, X_test, y_test, sensitive_feature='sex_Male')
evaluate_fairness(rf, X_test, y_test, sensitive_feature='sex_Male')
evaluate_fairness(nn, X_test, y_test, sensitive_feature='sex_Male')


# Initialize the fairness mitigation algorithm
mitigator = GridSearch(
    estimator=LogisticRegression(max_iter=1000),
    constraints=EqualizedOdds(),
    grid_size=20
)

# Fit the model with fairness constraints
mitigator.fit(X_train, y_train, sensitive_features=X_train['sex_Male'])

# Re-evaluate fairness after mitigation
mitigator_predictions = mitigator.predict(X_test)
mitigate_metric_frame = MetricFrame(
    metrics={'accuracy': accuracy_score},
    y_true=y_test,
    y_pred=mitigator_predictions,
    sensitive_features=X_test['sex_Male']
)

print("Fairness after mitigation:")
print(mitigate_metric_frame.by_group)





# Collect accuracy results before and after mitigation for each model
accuracy_results = {
    'Model': ['Logistic Regression', 'Random Forest', 'Neural Network'],
    'Before Mitigation': [lr_accuracy, rf_accuracy, nn_accuracy],
    'After Mitigation': [accuracy_score(y_test, mitigator.predict(X_test))] * 3  # Assuming the same mitigator
}

accuracy_df = pd.DataFrame(accuracy_results)

# Plot Accuracy comparison
plt.figure(figsize=(8, 6))
sns.barplot(x='Model', y='value', hue='Mitigation', data=accuracy_df.melt(id_vars=['Model'], value_vars=['Before Mitigation', 'After Mitigation'], var_name='Mitigation', value_name='value'))
plt.title('Accuracy Comparison Before and After Fairness Mitigation')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.show()









# Define the metrics you want to track for fairness (before and after mitigation)
metrics = ['Demographic Parity', 'Equalized Odds FPR', 'Equalized Odds FNR', 'Disparate Impact', 'FPRD', 'FNRD']

# Initialize data storage
fairness_data = []

# Loop through each model
for model, name in zip([lr, rf, nn], ['Logistic Regression', 'Random Forest', 'Neural Network']):
    
    # Fairness metrics before mitigation
    metrics_before = calculate_multiple_fairness_metrics(model, X_test, y_test, sensitive_feature='sex_Male')
    
    # Fairness metrics after mitigation
    metrics_after = calculate_multiple_fairness_metrics(mitigator, X_test, y_test, sensitive_feature='sex_Male')
    
    for metric in metrics:
        fairness_data.append({
            'Model': name,
            'Metric': metric,
            'Before Mitigation': metrics_before.get(metric, 0),
            'After Mitigation': metrics_after.get(metric, 0)
        })

# Create a DataFrame for plotting
fairness_df = pd.DataFrame(fairness_data)

# Melt the DataFrame for easier plotting
fairness_df_melted = pd.melt(fairness_df, id_vars=['Model', 'Metric'], value_vars=['Before Mitigation', 'After Mitigation'], 
                             var_name='Mitigation', value_name='Metric Value')

# Plot fairness metrics comparison
plt.figure(figsize=(14, 7))
sns.barplot(x='Metric', y='Metric Value', hue='Mitigation', data=fairness_df_melted, ci=None)
plt.title('Fairness Metrics Comparison Before and After Mitigation')
plt.ylabel('Fairness Metric Value')
plt.xlabel('Fairness Metric')
plt.xticks(rotation=45)
plt.legend(title='Mitigation', loc='upper left')
plt.show()









# List of fairness metrics
metrics = ['Demographic Parity', 'Equalized Odds FPR', 'Equalized Odds FNR', 'Disparate Impact', 'FPRD', 'FNRD']

# Initialize a list to store accuracy values for each metric before and after mitigation
accuracy_comparison_data = []

# Loop through each fairness metric and calculate accuracy before and after mitigation
for metric in metrics:
    # Get accuracy before mitigation for each model
    accuracy_before = [
        accuracy_score(y_test, model.predict(X_test)) for model in [lr, rf, nn]
    ]
    
    # Get accuracy after mitigation for each model
    accuracy_after = [
        accuracy_score(y_test, mitigator.predict(X_test)) for model in [lr, rf, nn]
    ]
    
    # Append data for each metric to the list
    accuracy_comparison_data.append({
        'Metric': metric,
        'Accuracy Before Mitigation': np.mean(accuracy_before),  # Average of all models
        'Accuracy After Mitigation': np.mean(accuracy_after)   # Average of all models
    })

# Create a DataFrame for plotting
accuracy_df = pd.DataFrame(accuracy_comparison_data)

# Melt the DataFrame for easier plotting
accuracy_df_melted = pd.melt(accuracy_df, id_vars=['Metric'], value_vars=['Accuracy Before Mitigation', 'Accuracy After Mitigation'],
                             var_name='Mitigation', value_name='Accuracy')

# Plot accuracy comparison for each fairness metric
plt.figure(figsize=(14, 7))
sns.barplot(x='Metric', y='Accuracy', hue='Mitigation', data=accuracy_df_melted, ci=None)
plt.title('Accuracy Comparison for Each Fairness Metric (Before and After Mitigation)')
plt.ylabel('Accuracy')
plt.xlabel('Fairness Metric')
plt.xticks(rotation=45)
plt.legend(title='Mitigation', loc='upper left')
plt.show()

