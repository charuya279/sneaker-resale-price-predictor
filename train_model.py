import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

# Load dataset
df = pd.read_csv("sneaker_resale_dataset.csv")

X = df[
    ["retail_price", "production_qty", "release_year",
     "brand_popularity", "condition"]
]
y = df["resale_price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))

# Save model
joblib.dump(model, "model.pkl")