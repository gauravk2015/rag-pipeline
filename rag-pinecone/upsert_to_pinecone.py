"""
Upsert records to Pinecone.
Creating chunks, embedding etc. is managed by Pinecone.
Using integrated embedding model- llama-text-embed-v2, which can be selected while creating index.
Oct 2026
"""

import os

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
index_host = os.getenv("PINECONE_INDEX_HOST")

# Get the namespace to save records
# Embedding will be created by Pinecone itself, hence no need to send the embedding
# Chunk, dimension are all defined in Pinecone console for the namespace

namespace = os.getenv("PINECONE_NAMESPACE")

if not api_key or not index_host or not namespace:
    raise ValueError(
        "Missing PINECONE_API_KEY or PINECONE_INDEX_HOST or PINECODE_NAMESPACE in .env"
    )

# Connect to Pinecone
pc = Pinecone(api_key=api_key)
index = pc.Index(host=index_host)

"""
Two sample product records.
text is the field used by Pinecone to create embedding values
and also used in metadata to return original text value while searching.
""" 
products = [
    {
        "_id": "B01N6SJ4NU",
        "text": (
            "Philips Lumea Prestige IPL Hair Removal Device "
            "with 4 Attachments. Brand: Philips. "
            "Category: Personal Care Appliances. "
            "Subcategory: Hair Removal. Price: 482.18 EUR."
        ),
        "brand": "Philips",
        "category": "Personal Care Appliances",
        "subcategory": "Hair Removal",
        "asin": "B01N6SJ4NU",
        "ean": "8710100000000",
        "price": 482.18,
        "currency": "EUR",
    },
    {
        "_id": "sample-product-002",
        "text": (
            "Samsung Galaxy S24 Smartphone. "
            "Brand: Samsung. "
            "Category: Electronics. "
            "Subcategory: Smartphones."
        ),
        "brand": "Samsung",
        "category": "Electronics",
        "subcategory": "Smartphones",
    },
]

# Upsert records using the integrated embedding model
response = index.upsert_records(
    namespace = namespace,
    records = products,
)

print("Upsert request submitted.")
print(response)
print(f"Products submitted: {len(products)}")

"""
Sample data saved in pinecone as:
values is the embedding generated from text field
{
  "id": "sample-product-002",
  "values": [
    -0.0363769531,
    -0.0294036865,
 .............
  ],
  "metadata": {
    "brand": "Samsung",
    "category": "Electronics",
    "subcategory": "Smartphones",
    "text": "Samsung Galaxy S24 Smartphone. Brand: Samsung. Category: Electronics. Subcategory: Smartphones."
  }
}
"""