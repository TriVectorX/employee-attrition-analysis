import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'final_dataset_attrition.csv'
data = pd.read_csv(file_path)

# Dropping irrelevant columns
data = data.drop(columns=['Unnamed: 32', 'Date_of_Hire', 'Date_of_termination'])

# Converting categorical columns to numerical using one-hot encoding
categorical_columns = data.select_dtypes(include=['object']).columns
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Define the features and target variable
X = data_encoded.drop(columns=['Attrition_Yes'])
y = data_encoded['Attrition_Yes']

# Function to evaluate KNN with different k values, sampling ratios, and distance metrics
def evaluate_knn(X, y, k_values, sample_ratios, metrics):
    results = []
    
    for sample_ratio in sample_ratios:
        # Handle class imbalance using SMOTE with specific sampling ratios
        smote = SMOTE(sampling_strategy=sample_ratio, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
        
        # Standardizing the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        for k in k_values:
            for metric in metrics:
                # Creating and training the KNN model
                knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
                knn.fit(X_train, y_train)
                
                # Making predictions
                y_pred = knn.predict(X_test)
                
                # Evaluating the model
                accuracy = accuracy_score(y_test, y_pred)
                results.append((k, sample_ratio, metric, accuracy))
                print(f"k: {k}, sample_ratio: {sample_ratio}, metric: {metric}, accuracy: {accuracy}")
    
    return results

# Define the range of k values, sampling ratios, and distance metrics to test
k_values = range(3, 12, 1)  # Example: 3, 5, 7, 9, 11
sample_ratios = [0.38, 0.6, 1.0]  # Valid oversampling ratios
metrics = ['euclidean', 'manhattan', 'minkowski']  # Example: different distance metrics

# Evaluate the KNN model with different k values, sampling ratios, and distance metrics
results = evaluate_knn(X, y, k_values, sample_ratios, metrics)

# Convert results to DataFrame for easier plotting
results_df = pd.DataFrame(results, columns=['k', 'sample_ratio', 'metric', 'accuracy'])

# Plot the results
plt.figure(figsize=(12, 8))
for metric in metrics:
    for sample_ratio in sample_ratios:
        subset = results_df[(results_df['sample_ratio'] == sample_ratio) & (results_df['metric'] == metric)]
        plt.plot(subset['k'], subset['accuracy'], marker='o', label=f'Metric: {metric}, Sample Ratio: {sample_ratio}')
plt.title('KNN Accuracy for Different k Values, Sampling Ratios, and Distance Metrics')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
# plt.show()
# save plot to file
plt.savefig('knn_accuracy_plot.png')