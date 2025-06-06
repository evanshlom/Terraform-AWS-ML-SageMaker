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
import time

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
import os
import json
import joblib
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """Load model from the model directory"""
    try:
        logger.info(f"Loading model from {model_dir}")
        model_path = os.path.join(model_dir, "model.pkl")
        logger.info(f"Model path: {model_path}")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def input_fn(request_body, content_type="application/json"):
    """Parse input data for inference"""
    try:
        logger.info(f"Parsing input with content_type: {content_type}")
        
        if content_type == "application/json":
            input_data = json.loads(request_body)
            logger.info(f"Input data: {input_data}")
            
            if "instances" in input_data:
                return np.array(input_data["instances"])
            else:
                return np.array(input_data)
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
    except Exception as e:
        logger.error(f"Error parsing input: {e}")
        raise

def predict_fn(input_data, model):
    """Make predictions"""
    try:
        logger.info(f"Making prediction on data shape: {input_data.shape}")
        
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
        
        result = {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist()
        }
        
        logger.info("Prediction successful")
        return result
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise

def output_fn(prediction, content_type="application/json"):
    """Format prediction output"""
    try:
        if content_type == "application/json":
            return json.dumps(prediction)
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
    except Exception as e:
        logger.error(f"Error formatting output: {e}")
        raise

# Health check endpoint
def ping():
    """Health check function"""
    return {"status": "healthy"}
'''
    return inference_code

def upload_model_to_s3():
    """Upload trained model to S3"""
    import os
    
    # Get bucket name from environment variable or find it
    bucket_name = os.environ.get('BUCKET_NAME')
    s3_client = boto3.client('s3')
    
    if not bucket_name:
        # Fallback: find ML bucket
        buckets = s3_client.list_buckets()
        for bucket in buckets['Buckets']:
            if 'ml-demo-2025-ml-bucket' in bucket['Name']:
                bucket_name = bucket['Name']
                break
    
    if not bucket_name:
        raise Exception("No ML bucket found")
    
    print(f"Using bucket: {bucket_name}")
    
    # Check if model already exists
    try:
        s3_client.head_object(Bucket=bucket_name, Key="model/model.tar.gz")
        print("Model already exists, skipping upload")
        return
    except:
        pass  # Model doesn't exist, continue with upload
    
    # Train model
    model = train_model()
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = os.path.join(temp_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, "model.pkl")
        joblib.dump(model, model_path)
        
        # Save inference script
        inference_path = os.path.join(model_dir, "inference.py")
        with open(inference_path, 'w') as f:
            f.write(create_inference_script())
        
        # Create tar.gz file
        tar_path = os.path.join(temp_dir, "model.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(model_dir, arcname=".")
        
        # Upload to S3 with retry
        s3_key = "model/model.tar.gz"
        for attempt in range(3):
            try:
                print(f"Upload attempt {attempt + 1}/3 to s3://{bucket_name}/{s3_key}")
                s3_client.upload_file(
                    tar_path, 
                    bucket_name, 
                    s3_key,
                    ExtraArgs={'ServerSideEncryption': 'AES256'}
                )
                
                # Verify upload
                response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
                print(f"Upload successful! Size: {response['ContentLength']} bytes")
                return
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise
                time.sleep(5)

if __name__ == "__main__":
    upload_model_to_s3()