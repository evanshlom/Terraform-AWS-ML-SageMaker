# terraform/main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# S3 bucket for ML models and data
resource "aws_s3_bucket" "ml_bucket" {
  bucket = "${var.project_name}-ml-bucket-${random_string.bucket_suffix.result}"
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

resource "aws_s3_bucket_versioning" "ml_bucket_versioning" {
  bucket = aws_s3_bucket.ml_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "ml_bucket_encryption" {
  bucket = aws_s3_bucket.ml_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# IAM role for SageMaker
resource "aws_iam_role" "sagemaker_role" {
  name = "${var.project_name}-sagemaker-role-${random_string.bucket_suffix.result}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_execution_policy" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy" "s3_access_policy" {
  name = "${var.project_name}-s3-access-${random_string.bucket_suffix.result}"
  role = aws_iam_role.sagemaker_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.ml_bucket.arn,
          "${aws_s3_bucket.ml_bucket.arn}/*"
        ]
      }
    ]
  })
}

# SageMaker Model
resource "aws_sagemaker_model" "ml_model" {
  name               = "${var.project_name}-model"
  execution_role_arn = aws_iam_role.sagemaker_role.arn

  primary_container {
    image = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
    model_data_url = "s3://${aws_s3_bucket.ml_bucket.bucket}/model/model.tar.gz"
    environment = {
      SAGEMAKER_PROGRAM = "inference.py"
    }
  }

  depends_on = [aws_s3_bucket.ml_bucket]
}

# SageMaker Endpoint Configuration
resource "aws_sagemaker_endpoint_configuration" "ml_endpoint_config" {
  name = "${var.project_name}-endpoint-config"

  production_variants {
    variant_name           = "primary"
    model_name            = aws_sagemaker_model.ml_model.name
    initial_instance_count = 1
    instance_type         = "ml.t2.medium"
  }
}

# SageMaker Endpoint
resource "aws_sagemaker_endpoint" "ml_endpoint" {
  name                 = "${var.project_name}-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.ml_endpoint_config.name
}