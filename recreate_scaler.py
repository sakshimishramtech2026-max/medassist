import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

BASE_DIR = os.path.dirname(__file__)
csv_path = os.path.join(BASE_DIR, "parkinsons data.csv")
output_path = os.path.join(BASE_DIR, "scaler.pkl")

def recreate_scaler():
    print(f"Reading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Preprocessing as per training notebook:
    # 1. Drop the 'name' column
    # 2. Drop the 'status' column (target)
    # The remaining columns should be our features in their original CSV order
    
    if 'name' in df.columns:
        df = df.drop('name', axis=1)
    
    # Features (X)
    X = df.drop('status', axis=1).values
    
    print(f"Features shape: {X.shape}")
    print(f"Feature order: {list(df.drop('status', axis=1).columns)}")
    
    # Initialize and fit the scaler
    scaler = MinMaxScaler()
    scaler.fit(X)
    
    # Save the scaler
    with open(output_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"✅ Scaler recreated and saved to {output_path}")

if __name__ == "__main__":
    recreate_scaler()
