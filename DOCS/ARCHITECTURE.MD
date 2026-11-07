# ScamShield

## MVP Architecture

**Server:** Serverless Architecture

### AWS Services

| AWS Service | Purpose in ScamShield | Cost |
|---|---|---|
| **API Gateway** | Receives requests from third-party applications and sends them to Lambda. | Pay per API request. The HTTP API option is generally cheaper than REST API. Eligible free-tier usage or credits may apply. |
| **AWS Lambda** | Runs the backend logic, coordinates the detection models, and returns a risk assessment. | Pay for requests and execution time. Low usage may fit within eligible free-tier allowances. |
| **IAM** | Controls permissions so services and applications can access only what they need. | No additional charge for standard IAM use. |
| **CloudWatch** | Monitors errors, performance, and logs to help us identify issues. | Basic monitoring may fit within free allowances; additional logs, metrics, dashboards, and alarms can incur charges. |
| **DynamoDB** | Stores and retrieves phone-number reputation records if ScamShield maintains its own reputation database. | Pay for reads, writes, and storage. On-demand billing avoids paying for a database instance while idle. |

**Cost note:** These services do not have one fixed total price. Actual charges depend on request volume, model execution time, storage, and logging. We will monitor usage and set budget alerts to help control costs.

## Post-Production Architecture

These services will be added only when the project needs them.

| AWS Service | When and Why We Will Use It | Cost |
|---|---|---|
| **Amazon S3 + CloudFront** | S3 will store model files, datasets, and other files when needed. CloudFront will be considered alongside S3 when we introduce a frontend or need faster content delivery. | S3 charges for storage, requests, and applicable data transfer. CloudFront charges mainly for data transfer and requests; free allowances may apply. |
| **Amazon SageMaker AI** | We will consider SageMaker after selecting and testing the exact machine-learning models, to determine whether they need managed model hosting or inference. | Pay for the compute instances, storage, and other SageMaker features used. Costs depend on the model and deployment setup. |
| **Amazon SQS** | We will introduce SQS when we have background tasks that should run separately from the immediate API response, such as processing reports or batch jobs. | Pay per request. The AWS SQS Free Tier includes up to one million requests per month. |

## Final Decision

Start with API Gateway, Lambda, IAM, DynamoDB and CloudWatch.

The remaining services will be introduced when their specific needs arise, helping us avoid unnecessary costs during the initial development stage.



### NB: THIS FILE IS PRONE TO CHANGE AND REGULAR UPDATE WILL BE MADE

**Last Update: 07th - November, 2026**