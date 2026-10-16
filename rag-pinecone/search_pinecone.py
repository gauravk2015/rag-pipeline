"""
Search Pinecone records and use the LLM to generate a response.
Pinecone internally generates the query embedding using its integrated embedding model.
Therefore, we pass the query as text rather than generating and sending the embedding vector ourselves.
Oct 2026
"""

import os

from dotenv import load_dotenv
from pinecone import Pinecone
from anthropic import Anthropic

load_dotenv()

# Validate configuration
required = [
    "PINECONE_API_KEY",
    "PINECONE_INDEX_HOST",
    "PINECONE_NAMESPACE",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
]

missing = [key for key in required if not os.getenv(key)]
if missing:
    raise ValueError(f"Missing .env settings: {', '.join(missing)}")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(host=os.getenv("PINECONE_INDEX_HOST"))
namespace = os.getenv("PINECONE_NAMESPACE")

# Initialize Claude
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com",
)
model = os.getenv("ANTHROPIC_MODEL")


def search_products(user_query: str) -> str:
    # 1. Search Pinecone; integrated embedding handles the query embedding
    results = index.search(
        namespace=namespace,
        top_k=3,
        inputs={"text": user_query},
    )

    # 2. Extract matching records using the documented response attributes
    hits = results.result.hits

    if not hits:
        return "No matching products were found."

    context = []

    for hit in hits:
        fields = hit.fields or {}

        context.append({
            "id": hit.id,
            "score": hit.score,
            "text": fields.get("text", ""),
            "brand": fields.get("brand", ""),
            "category": fields.get("category", ""),
            "subcategory": fields.get("subcategory", ""),
            "price": fields.get("price"),
            "currency": fields.get("currency", ""),
        })

    # 3. Ask Claude to answer using the retrieved records
    response = client.messages.create(
        model=model,
        max_tokens=400,
        system=(
            "You are a product assistant. Answer using only the "
            "retrieved product records. The products are retrieved using vector db and may contain results not valid for user query. "
            "Hence, do not invent product details or prices. " 
            "Strictly parse the results or product records which are valid for user query and do not talk about any other products which are mentioned in the results "  
            "Only if the records do not match the question then "
            "state that the available data is insufficient. Dont say that some data is missing for some fields. "
            "Keep a friendly tone and always ask that do you want to equire any other product? "
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question: {user_query}\n\n"
                    f"Retrieved product records:\n{context}"
                ),
            }
        ],
    )

    return "\n".join(
        block.text
        for block in response.content
        if block.type == "text"
    )


if __name__ == "__main__":
    while True:
        query = input("\nEnter product query (or 'exit'): ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        try:
            print("\nClaude:", search_products(query))
        except Exception as exc:
            print(f"\nRequest failed: {exc}")