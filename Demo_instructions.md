# AWS Setup Instructions

Follow these steps to configure AWS and GitHub for automated ML deployment.

## Prerequisites

- AWS Account with administrative access
- GitHub account
- This repository forked to your GitHub account

## Step 1: Create AWS IAM User

### 1.1 Create User
1. Go to AWS Console → IAM → Users → **Create User**
2. **User name**: `terraform-aws-ml-sagemaker-user`
3. **Access type**: ✅ Programmatic access
4. Click **Next**

### 1.2 Attach Policies
Attach these **3 policies** to your user:
- `AmazonS3FullAccess`
- `AmazonSageMakerFullAccess`
- `IAMFullAccess`

### 1.3 Save Your Keys
After user creation, AWS displays:
- **AWS_ACCESS_KEY_ID** (starts with `AKIA...`)
- **AWS_SECRET_ACCESS_KEY** (long random string)

⚠️ **IMPORTANT**: Copy these immediately - AWS only shows the secret key once!

## Step 2: Configure GitHub Secrets

### 2.1 Fork This Repository
1. Fork this repository to your GitHub account
2. Go to your forked repo

### 2.2 Add AWS Secrets
1. Navigate to: **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Add these **2 secrets**:

   **Secret 1:**
   - Name: `AWS_ACCESS_KEY_ID`
   - Value: `[your AWS access key from Step 1.3]`

   **Secret 2:**
   - Name: `AWS_SECRET_ACCESS_KEY`
   - Value: `[your AWS secret key from Step 1.3]`

## Step 3: Deploy Your ML Service

### 3.1 Trigger Deployment
Push any change to your main branch:
```bash
git add .
git commit -m "Deploy ML service"
git push origin main
```

### 3.2 Monitor Deployment
1. Go to your repo's **Actions** tab
2. Watch the `Deploy ML Service` workflow run
3. ✅ Green checkmark = successful deployment (takes ~8-12 minutes)

## Step 4: Verify Deployment

Once the GitHub Actions workflow completes successfully:

```bash
# Test your deployed ML endpoint
python scripts/test_endpoint.py
```

## What Gets Created

Your `terraform-aws-ml-sagemaker-user` will automatically deploy:
- **S3 Bucket**: Stores your ML model
- **SageMaker Model**: RandomForest classifier
- **SageMaker Endpoint**: REST API for predictions
- **IAM Roles**: Proper security permissions

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Actions workflow fails | Check AWS secrets are correctly named and valid |
| Terraform errors | Verify IAM user has all 3 required policies |
| Endpoint not found | Wait 5-10 minutes after deployment completes |
| Permission denied | Ensure IAMFullAccess policy is attached |

## Cost Estimation

- **SageMaker Endpoint**: ~$35-50/month (ml.t2.medium)
- **S3 Storage**: <$1/month
- **Total**: ~$40-55/month when running

## Clean Up

To avoid ongoing charges:
```bash
cd terraform
terraform destroy
```

---

**Ready to impress?** Just fork, add secrets, and push! Your automated ML pipeline will be live in minutes. 🚀