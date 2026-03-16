import boto3
import json

client = boto3.client("bedrock-runtime", region_name="us-east-1")

body = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"text": "Explain REST APIs in one sentence."}
            ]
        }
    ]
}

response = client.invoke_model(
    modelId="global.amazon.nova-2-lite-v1:0",
    contentType="application/json",
    accept="application/json",
    body=json.dumps(body)
)

result = json.loads(response["body"].read())

print(result)