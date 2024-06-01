## IBM Employee Attrition Analysis

This repository contains the Python source code for a logistic regression model aimed at predicting employee attrition. The analysis utilizes various machine learning techniques to preprocess the data, build predictive models, and evaluate their performance. The code includes the use of logistic regression to understand the impact of various factors on employee attrition and the application of K-means clustering for further insights.

### Structure of the Code

1. **Import Libraries**
    - Essential libraries such as Pandas, Scikit-learn, Statsmodels, and Matplotlib are imported for data manipulation, model building, and visualization.

2. **Load the Dataset**
    - The dataset is loaded from a CSV file named `final_dataset_attrition.csv`.

3. **Data Cleaning**
    - Unnecessary columns are removed, and missing values are handled to prepare the data for analysis. Numerical columns are imputed with their mean values, and categorical columns are imputed with their most frequent values.

4. **Encoding Categorical Variables**
    - Categorical variables, such as job roles and departments, are encoded into numerical format using LabelEncoder.

5. **Prepare Data for Logistic Regression**
    - The dataset is split into features (X) and the target variable (y), followed by a train-test split and feature scaling to standardize the features.

6. **Train Logistic Regression Model**
    - A logistic regression model is trained on the scaled training data, and predictions are made on the test data. The model's performance is evaluated using metrics such as accuracy, classification report, and confusion matrix.

7. **Logistic Regression Coefficients and Odds Ratios**
    - The coefficients and odds ratios of the logistic regression model are calculated to understand the impact of each feature on employee attrition. The model also provides confidence intervals for these estimates.

8. **K-Means Clustering**
    - K-means clustering is applied to the dataset to identify clusters of employees with similar characteristics. The characteristics of each cluster are analyzed to gain further insights into employee attrition patterns.

9. **Display and Save Results**
    - The results, including logistic regression metrics and cluster analysis, are printed, saved as HTML tables, and visualized through plots. This includes generating a summary report, plotting logistic regression coefficients and odds ratios, and visualizing cluster analysis.

### Outcomes
- **Logistic Regression Model**: Achieved high accuracy in predicting employee attrition, with detailed insights into significant predictors through odds ratios and confidence intervals.
- **K-Means Clustering**: Provided further insights by categorizing employees into clusters based on similar characteristics, aiding in understanding different attrition patterns.
- **Visualizations and Reports**: Generated comprehensive visualizations and reports to facilitate the interpretation of results and support decision-making.

This code structure enables a thorough analysis of employee attrition, leveraging both logistic regression and clustering techniques to provide actionable insights.
