import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'final_dataset_attrition.csv'
df = pd.read_csv(file_path)

# Drop columns that have no observed values
df = df.drop(columns=['Date_of_termination', 'Unnamed: 32'], errors='ignore')

# Handle missing values by imputing with the mean for numerical columns and the most frequent value for categorical columns
imputer_num = SimpleImputer(strategy='mean')
imputer_cat = SimpleImputer(strategy='most_frequent')

# Separate numerical and categorical columns
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Impute missing values
df[num_cols] = imputer_num.fit_transform(df[num_cols])
df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

# Encode categorical variables
label_encoders = {}
for column in cat_cols:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Prepare the dataset for logistic regression
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

# Predict the test set results
y_pred_logreg = logreg.predict(X_test_scaled)

# Evaluate the logistic regression model
logreg_accuracy = accuracy_score(y_test, y_pred_logreg)
logreg_classification_report = classification_report(y_test, y_pred_logreg)
logreg_confusion_matrix = confusion_matrix(y_test, y_pred_logreg)

# Logistic regression coefficients and odds ratios
logit_model = sm.Logit(y_train, sm.add_constant(X_train_scaled))
result = logit_model.fit()
odds_ratios = pd.DataFrame({
    'Feature': X.columns,
    'Odds Ratio': result.params[1:].apply(lambda x: round(np.exp(x), 2)),
    'P-value': result.pvalues[1:],
    '95% CI Lower': result.conf_int()[0][1:].apply(lambda x: round(np.exp(x), 2)),
    '95% CI Upper': result.conf_int()[1][1:].apply(lambda x: round(np.exp(x), 2))
})

# K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaler.fit_transform(X))

# Analyze cluster characteristics
cluster_analysis = df.groupby('Cluster').mean()

# Display results
print("Logistic Regression Accuracy: ", logreg_accuracy)
print("Logistic Regression Classification Report:\n", logreg_classification_report)
print("Logistic Regression Confusion Matrix:\n", logreg_confusion_matrix)
print("\nLogistic Regression Coefficients and Odds Ratios:\n", odds_ratios)
print("\nCluster Analysis:\n", cluster_analysis)


# create html tables for the results
odds_ratios.to_html('odds_ratios.html')
cluster_analysis.to_html('cluster_analysis.html')

# generate a summary report
with open('summary_report.txt', 'w') as f:
    f.write("Logistic Regression Accuracy: {}\n".format(logreg_accuracy))
    f.write("Logistic Regression Classification Report:\n{}\n".format(logreg_classification_report))
    f.write("Logistic Regression Confusion Matrix:\n{}\n".format(logreg_confusion_matrix))
    f.write("\nLogistic Regression Coefficients and Odds Ratios:\n{}\n".format(odds_ratios))
    f.write("\nCluster Analysis:\n{}\n".format(cluster_analysis))
    f.write("\nOdds Ratios HTML Table: odds_ratios.html\n")
    f.write("Cluster Analysis HTML Table: cluster_analysis.html\n")

# Plot the logistic regression coefficients
plt.figure(figsize=(12, 8))
plt.errorbar(odds_ratios['Feature'], odds_ratios['Odds Ratio'], yerr=[odds_ratios['Odds Ratio'] - odds_ratios['95% CI Lower'], odds_ratios['95% CI Upper'] - odds_ratios['Odds Ratio']], fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
plt.title('Logistic Regression Coefficients and Odds Ratios')
plt.xlabel('Feature')
plt.ylabel('Odds Ratio')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('logistic_regression_odds_ratios.png')

# Plot the cluster analysis
plt.figure(figsize=(12, 8))
cluster_analysis.T.plot(kind='bar', figsize=(12, 8))
plt.title('Cluster Analysis')
plt.xlabel('Feature')
plt.ylabel('Mean Value')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('cluster_analysis.png')