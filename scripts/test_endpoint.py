# scripts/test_endpoint.py
import boto3
import json
import numpy as np

def test_sagemaker_endpoint():
    """Test the deployed SageMaker endpoint"""
    
    # Initialize SageMaker runtime client
    runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')
    
    # Find the endpoint name (assumes it contains 'simple-ml-demo')
    sagemaker_client = boto3.client('sagemaker', region_name='us-east-1')
    
    endpoints = sagemaker_client.list_endpoints()
    endpoint_name = None
    
    for endpoint in endpoints['Endpoints']:
        if 'simple-ml-demo' in endpoint['EndpointName']:
            endpoint_name = endpoint['EndpointName']
            break
    
    if not endpoint_name:
        print("No endpoint found. Make sure the deployment is complete.")
        return
    
    print(f"Testing endpoint: {endpoint_name}")
    
    # Create sample data for prediction (10 features as expected by the model)
    test_data = np.random.randn(5, 10).tolist()  # 5 samples, 10 features each
    
    payload = {
        "instances": test_data
    }
    
    try:
        # Make prediction
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        print("Prediction successful!")
        print(f"Predictions: {result['predictions']}")
        print(f"Probabilities: {result['probabilities']}")
        
    except Exception as e:
        print(f"Error making prediction: {e}")

if __name__ == "__main__":
    test_sagemaker_endpoint()