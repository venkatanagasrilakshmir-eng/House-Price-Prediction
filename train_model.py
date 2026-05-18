import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
import seaborn as sns

# Load dataset
data = pd.read_csv('data/house_data.csv')

# Exploratory Data Analysis and Image Generation
corr = data.corr()
os.makedirs('images', exist_ok=True)
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='Blues')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('images/correlation_heatmap.png')
plt.close()

# Splitting data into features and target
X = data.drop(['Price'], axis=1)
y = data['Price']

# Split train/test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/random_forest_model.pkl')

# Feature importance plot
importances = model.feature_importances_
feat_names = X.columns
plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=feat_names)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('images/feature_importance.png')
plt.close()

# Prediction graph
y_pred = model.predict(X_test)
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.tight_layout()
plt.savefig('images/prediction_graph.png')
plt.close()

# Print RMSE
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
