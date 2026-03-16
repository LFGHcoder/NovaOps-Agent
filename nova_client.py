import boto3
import json

# Create Bedrock runtime client
client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"
)

# Use the system inference profile you showed earlier
MODEL_ID = "global.amazon.nova-2-lite-v1:0"


def call_nova(prompt: str) -> str:
    """
    Sends a prompt to Amazon Nova 2 Lite via Bedrock
    and returns the generated text response.
    """

    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": prompt}
                ]
            }
        ]
    }

    response = client.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )

    # Parse Bedrock response
    result = json.loads(response["body"].read())

    try:
        return result["output"]["message"]["content"][0]["text"]
    except Exception:
        # Fallback safe return
        return json.dumps(result)