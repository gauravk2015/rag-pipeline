"""
Below script can be executed to query AWS Bedrock managed knowledge base (KB). 
It uses the Bedrock Agentic Retrieve API to retrieve relevant information from the knowledge base and generate a response.
Install boto3 library if not already installed using the command: pip install boto3
"""
import boto3

# Set your AWS region and knowledge base ID
REGION = ""
KNOWLEDGE_BASE_ID = ""

client = boto3.client(
    "bedrock-agent-runtime",
    region_name=REGION,
)

def ask_question(question: str) -> None:
    response = client.agentic_retrieve_stream(
        messages=[
            {
                "role": "user",
                "content": {"text": question},
            }
        ],
        retrievers=[
            {
                "configuration": {
                    "knowledgeBase": {
                        "knowledgeBaseId": KNOWLEDGE_BASE_ID
                    }
                }
            }
        ],
        agenticRetrieveConfiguration={
            "foundationModelType": "MANAGED",
            "maxAgentIteration": 5,
        },
        generateResponse=True,
    )

    for event in response["stream"]:
        if "responseEvent" in event:
            print(event["responseEvent"].get("text", ""), end="", flush=True)

        elif "result" in event:
            for item in event["result"].get("results", []):
                source = item.get("location", {})
                content = item.get("content", {}).get("text", "")
                print(f"\n\nSource: {source}")
                print(f"Retrieved text: {content[:300]}...")

    print("\n")

while True:
    question = input("\nAsk an HR policy question (or 'exit'): ").strip()

    if question.lower() == "exit":
        break

    if question:
        try:
            ask_question(question)
        except Exception as exc:
            print(f"\nError: {exc}")
