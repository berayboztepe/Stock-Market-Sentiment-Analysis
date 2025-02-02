import os
import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

file_path = "data/final_stock_sentiment.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset file `{file_path}` not found!")

df = pd.read_csv(file_path)

if "Sentiment_Score" not in df.columns or "Stock Movement" not in df.columns:
    raise ValueError("Missing required columns: 'Sentiment_Score' or 'Stock Movement'.")

df["Stock Movement"] = df["Stock Movement"].astype(int)

X = df[["Sentiment_Score"]]
y = df["Stock Movement"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if y.value_counts().min() == 1:
    print("Only one sample in minority class! Upsampling minority class to balance dataset.")
    
    df_minority = df[df["Stock Movement"] == 0]
    df_minority_upsampled = resample(df_minority, 
                                     replace=True, 
                                     n_samples=5,
                                     random_state=42)

    df = pd.concat([df, df_minority_upsampled])
    df = df.sample(frac=1).reset_index(drop=True)

    X = df[["Sentiment_Score"]]
    y = df["Stock Movement"]
    X_scaled = scaler.fit_transform(X)

print("Applying SMOTE to balance classes.")
smote = SMOTE(random_state=42, k_neighbors=1)  
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

rf_model = RandomForestClassifier(random_state=42)
params = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}

cv_splits = min(2, len(y_train))
grid_search = GridSearchCV(rf_model, params, cv=cv_splits, scoring="accuracy")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_stock_sentiment_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

y_pred = best_model.predict(X_test)
print("Model Training Complete!")
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
