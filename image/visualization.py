# images/visualization.py

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("../data/house_data.csv")

# -----------------------------
# Correlation Heatmap
# -----------------------------
plt.figure(figsize=(8,6))

sns.heatmap(
    data.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig("correlation_heatmap.png")

plt.close()

# -----------------------------
# Price Distribution
# -----------------------------
plt.figure(figsize=(8,5))

sns.histplot(
    data["Price"],
    kde=True
)

plt.title("House Price Distribution")
plt.xlabel("Price")

plt.savefig("price_distribution.png")

plt.close()

# -----------------------------
# Area vs Price
# -----------------------------
plt.figure(figsize=(8,5))

sns.scatterplot(
    x=data["Area"],
    y=data["Price"]
)

plt.title("Area vs Price")
plt.xlabel("Area")
plt.ylabel("Price")

plt.savefig("area_vs_price.png")

plt.close()

# -----------------------------
# Train Model
# -----------------------------
X = data.drop("Price", axis=1)
y = data["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Feature Importance
# -----------------------------
importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))

sns.barplot(
    x=importance,
    y=features
)

plt.title("Feature Importance")

plt.savefig("feature_importance.png")

plt.close()

# -----------------------------
# Actual vs Predicted Prices
# -----------------------------
predictions = model.predict(X_test)

plt.figure(figsize=(8,5))

plt.plot(
    y_test.values,
    label="Actual Prices"
)

plt.plot(
    predictions,
    label="Predicted Prices"
)

plt.legend()

plt.title("Actual vs Predicted Prices")

plt.savefig("model_prediction.png")

plt.close()

print("All visualization images saved successfully!")
