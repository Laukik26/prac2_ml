import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

def evaluate_model():
    # Load processed data
    df = pd.read_csv("data/processed/diabetes_processed.csv")
    
    # Separate features and target variable
    X = df.drop(columns=["target"], axis=1)
    y = df["target"]
    
    # Load the trained model
    model = joblib.load("models/linear_regression_model.pkl")
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate Mean Squared Error
    mse = mean_squared_error(y, y_pred)
    
    print(f"Mean Squared Error: {mse}")

if __name__ == "__main__":
    evaluate_model()    
    print("Model Evaluation Complete")