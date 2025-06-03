# scripts/train_and_upload.py
import boto3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import tarfile
import os
import tempfile

def create_sample_data():
    """Create sample classification data"""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    return X, y

def train_model():
    """Train a simple RandomForest model"""
    print("Creating sample data...")
    X, y = create_sample_data()
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    return model

def create_inference_script():
    """Create inference script for SageMaker"""
    inference_code = '''
import joblib
import numpy as np
import json
import os

def model_fn(model_dir):
    """Load model from the model directory"""
    try:
        model = joblib.load(os.path.join(model_dir, "model.pkl"))
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def input_fn(request_body, content_type):
    """Parse input data for inference"""
    if content_type == "application/json":
        try:
            input_data = json.loads(request_body)
            return np.array(input_data["instances"])
        except Exception as e:
            print(f"Error parsing input: {e}")
            raise ValueError(f"Error parsing input: {e}")
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """Make predictions"""
    try:
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
        
        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist()
        }
    except Exception as e:
        print(f"Error making predictions: {e}")
        raise

def output_fn(prediction, content_type):
    """Format prediction output"""
    if content_type == "application/json":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
'''
    return inference_code

def upload_model_to_s3():
    """Upload trained model to S3"""
    # Get bucket name from Terraform output
    s3_client = boto3.client('s3')
    
    # List buckets to find our ML bucket
    buckets = s3_client.list_buckets()
    ml_bucket = None
    for bucket in buckets['Buckets']:
        if 'ml-bucket' in bucket['Name']:
            ml_bucket = bucket['Name']
            break
    
    if not ml_bucket:
        print("No ML bucket found. Make sure Terraform has been applied.")
        return
    
    print(f"Using bucket: {ml_bucket}")
    
    # Train model
    model = train_model()
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = os.path.join(temp_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, "model.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")
        
        # Save inference script
        inference_path = os.path.join(model_dir, "inference.py")
        with open(inference_path, 'w') as f:
            f.write(create_inference_script())
        print(f"Inference script saved to: {inference_path}")
        
        # Create tar.gz file
        tar_path = os.path.join(temp_dir, "model.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(model_dir, arcname=".")
        
        print(f"Model archive created: {tar_path}")
        
        # Upload to S3
        s3_key = "model/model.tar.gz"
        print(f"Uploading to s3://{ml_bucket}/{s3_key}")
        
        s3_client.upload_file(tar_path, ml_bucket, s3_key)
        print("Model uploaded successfully!")

if __name__ == "__main__":
    upload_model_to_s3()