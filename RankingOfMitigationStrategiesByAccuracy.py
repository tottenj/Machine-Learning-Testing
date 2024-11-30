import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import DemographicParity, EqualizedOdds, GridSearch

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

# Base model (Logistic Regression)
base_model = LogisticRegression(max_iter=1000)
base_model.fit(X_train.drop(columns=['sex_Male']), y_train)  # Fit without sensitive feature in model

# Predict using base model
base_preds = base_model.predict(X_test.drop(columns=['sex_Male']))

# Mitigation strategies to evaluate
mitigation_strategies = {
    'No Mitigation (Base Model)': None,
    'Threshold Optimizer (Equalized Odds)': ThresholdOptimizer(
        estimator=base_model,
        constraints='equalized_odds'  # Apply Equalized Odds post-processing
    ),
    'GridSearch (Equalized Odds)': GridSearch(
        estimator=base_model,
        constraints=EqualizedOdds(),
        grid_size=20
    ),
    # Add other mitigation strategies if needed
}

# Store the results
results = []

# Evaluate each strategy
for strategy_name, strategy in mitigation_strategies.items():
    if strategy is None:  # No mitigation, use base model
        preds = base_preds
        accuracy = accuracy_score(y_test, preds)
    else:
        if strategy_name == 'Threshold Optimizer (Equalized Odds)':
            strategy.fit(X_train.drop(columns=['sex_Male']), y_train, sensitive_features=X_train['sex_Male'])
            preds = strategy.predict(X_test.drop(columns=['sex_Male']), sensitive_features=X_test['sex_Male'])
        else:  # For GridSearch or other mitigators
            strategy.fit(X_train.drop(columns=['sex_Male']), y_train, sensitive_features=X_train['sex_Male'])
            preds = strategy.predict(X_test.drop(columns=['sex_Male']))
        
        accuracy = accuracy_score(y_test, preds)

    # Store accuracy for each strategy
    results.append({
        'Strategy': strategy_name,
        'Accuracy': accuracy
    })

# Create a DataFrame for results
results_df = pd.DataFrame(results)

# Sort strategies by accuracy
results_df = results_df.sort_values(by='Accuracy', ascending=False)

# Display the rankings
print(results_df)

# Plot rankings
plt.figure(figsize=(10, 6))
sns.barplot(x='Accuracy', y='Strategy', data=results_df)
plt.title('Ranking of Mitigation Strategies by Accuracy')
plt.xlabel('Accuracy')
plt.ylabel('Mitigation Strategy')
plt.tight_layout()
plt.show()




def calculate_fairness_metrics(model, X_test, y_test, sensitive_feature):
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

    return metrics



def compare_fairness_across_strategies(models, mitigators, X_test, y_test, sensitive_feature):
    # Store fairness metrics for each strategy
    fairness_metrics = []
    model_names = [model.__class__.__name__ for model in models]
    
    for i, model in enumerate(models):
        metrics_before = calculate_fairness_metrics(model, X_test, y_test, sensitive_feature)
        metrics_after = calculate_fairness_metrics(mitigators[i], X_test, y_test, sensitive_feature)
        
        for metric in metrics_before:
            fairness_metrics.append({
                'Model': model_names[i],
                'Metric': metric,
                'Fairness Value': metrics_before[metric],
                'Mitigation': 'Before Mitigation'
            })
        
        for metric in metrics_after:
            fairness_metrics.append({
                'Model': model_names[i],
                'Metric': metric,
                'Fairness Value': metrics_after[metric],
                'Mitigation': 'After Mitigation'
            })
    
    # Convert to DataFrame for plotting
    fairness_df = pd.DataFrame(fairness_metrics)
    
    # Plotting fairness metrics comparison
    plt.figure(figsize=(14, 7))
    sns.barplot(x='Metric', y='Fairness Value', hue='Mitigation', data=fairness_df, ci=None)
    plt.title('Fairness Metrics Before and After Mitigation')
    plt.ylabel('Fairness Metric Value')
    plt.xlabel('Fairness Metric')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



