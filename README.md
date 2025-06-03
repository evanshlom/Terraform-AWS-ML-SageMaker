### Simple ML Service with Terraform and AWS SageMaker

### A minimal demonstration project that deploys a machine learning service using Terraform, AWS SageMaker, and GitHub Actions.

*** Remember: To avoid accidental expensive AWS costs, run `terraform destroy` after deploying the ML model to AWS (this is just a demo)

## Project Structure

```
├── .github/workflows/deploy.yml  # GitHub Actions pipeline
├── terraform/
│   ├── main.tf                   # Main Terraform configuration
│   ├── variables.tf              # Terraform variables
│   └── outputs.tf                # Terraform outputs
├── scripts/
│   ├── train_and_upload.py       # Train ML model and upload to S3
│   └── test_endpoint.py          # Test the deployed endpoint
└── README.md                     # This file
```

## What This Project Does

1. **Infrastructure**: Creates AWS S3 bucket and SageMaker resources using Terraform
2. **ML Model**: Trains a simple RandomForest classifier on synthetic data
3. **Deployment**: Deploys the model to a SageMaker endpoint
4. **CI/CD**: Automates everything through GitHub Actions

## Prerequisites

### Local Development
1. **Install Terraform** (Windows):
   ```powershell
   choco install terraform
   ```

2. **Install AWS CLI**:
   ```powershell
   choco install awscli
   ```

3. **Install Python dependencies**:
   ```bash
   pip install boto3 scikit-learn joblib pandas numpy
   ```

### AWS Setup
1. Create an IAM user with programmatic access
2. Attach the following policies:
   - `AmazonS3FullAccess`
   - `AmazonSageMakerFullAccess` 
   - `IAMFullAccess` (for creating roles)

### GitHub Setup
1. Fork this repository
2. Add these secrets to your GitHub repository:
   - `AWS_ACCESS_KEY_ID`: Your AWS access key
   - `AWS_SECRET_ACCESS_KEY`: Your AWS secret key

## Local Testing

1. **Configure AWS credentials**:
   ```bash
   aws configure
   ```

2. **Initialize and apply Terraform**:
   ```bash
   cd terraform
   terraform init
   terraform plan
   terraform apply
   ```

3. **Train and upload the model**:
   ```bash
   python scripts/train_and_upload.py
   ```

4. **Test the endpoint** (wait 5-10 minutes after model upload):
   ```bash
   python scripts/test_endpoint.py
   ```

## GitHub Actions Deployment

1. Push to the `main` branch
2. GitHub Actions will:
   - Run Terraform to create infrastructure
   - Train and upload the ML model
   - Deploy to SageMaker endpoint

## The ML Model

- **Type**: Random Forest Classifier
- **Data**: Synthetic classification dataset (1000 samples, 10 features)
- **Purpose**: Binary classification demo
- **Accuracy**: ~85-95% on test data

## API Usage

Once deployed, you can make predictions by sending POST requests to the SageMaker endpoint:

```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime')
payload = {
    "instances": [[1.2, -0.5, 0.8, ...]]  # 10 features
}

response = runtime.invoke_endpoint(
    EndpointName='simple-ml-demo-endpoint',
    ContentType='application/json',
    Body=json.dumps(payload)
)
```

## Cost Considerations

- **SageMaker Endpoint**: ~$35-50/month (ml.t2.medium instance)
- **S3 Storage**: <$1/month for model files
- **Total**: ~$40-55/month when running

**Important**: Remember to destroy resources when not needed:
```bash
terraform destroy
```

## Troubleshooting

1. **Endpoint creation fails**: Check IAM permissions and S3 bucket access
2. **Model upload fails**: Ensure Terraform has created the S3 bucket first
3. **Predictions fail**: Wait 5-10 minutes after model upload for endpoint to be ready
4. **GitHub Actions fails**: Check AWS credentials in repository secrets

## Customization

- Modify `train_and_upload.py` to use your own dataset
- Change model type in the training script
- Adjust instance types in `terraform/main.tf` for different performance/cost
- Add more features like model versioning, monitoring, etc.

## Clean Up

To avoid ongoing charges:
```bash
cd terraform
terraform destroy
```