import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('preprocessed_data.csv')

correlation_matrix = data.corr()

attrition_correlation = correlation_matrix['Attrition_Yes'].sort_values(ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x=attrition_correlation.values, y=attrition_correlation.index)
plt.title('Correlation of Features with Attrition')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')
plt.show()

attrition_correlation