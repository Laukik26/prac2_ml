import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def train_model():
    # Load processed data
    df = pd.read_csv("data/processed/diabetes_processed.csv")
    
    # Separate features and target variable
    X = df.drop(columns=["target"], axis=1)
    y = df["target"]
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Save the trained model
    joblib.dump(model, "models/linear_regression_model.pkl")
    print("Model Training Complete")

if __name__ == "__main__":
    train_model()