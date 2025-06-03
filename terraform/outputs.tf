# terraform/outputs.tf
output "s3_bucket_name" {
  description = "Name of the S3 bucket"
  value       = aws_s3_bucket.ml_bucket.bucket
}

output "sagemaker_endpoint_name" {
  description = "Name of the SageMaker endpoint"
  value       = aws_sagemaker_endpoint.ml_endpoint.name
}

output "sagemaker_endpoint_url" {
  description = "URL of the SageMaker endpoint"
  value       = "https://runtime.sagemaker.${var.aws_region}.amazonaws.com/endpoints/${aws_sagemaker_endpoint.ml_endpoint.name}/invocations"
}