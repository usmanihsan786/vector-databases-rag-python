import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv


# deploymentmodel ="text-embedding-3-small"

# openai= OpenAI(
#     api_key=os.environ["Azure_Open_AI_API_KEY"],
#     base_url=os.environ["AZURE_OPENAI_ENDPOINT"]
# )
# steps
# 1. creaet a persistent database
client = chromadb.PersistentClient(
    path="chroma_db"
)
# Create collection
client.delete_collection("course-documents")
collection = client.get_or_create_collection(
    name="course-documents"
)

# 3. insert documnet in collection 
collection.add(
    ids=["doc1","doc2","doc3"],
    documents=[
        "Python tutorial",
        "Docker guide",
        "Vector database basic"
    ],
    metadatas=[
        {"category":"programming"},{"category":"Devops"},{"category":"AI"} 
    ]
)


collection.update(
    ids=["doc1"],
    documents=[
        "Python is widely used for AI and machine learning",
    ]
)
collection.update(
    ids=["doc1"],
    metadatas=[
        {"category":"AI","level":"begineer"},
    ]
)

# 4. run a similarity search
results = collection.query(
    query_texts=[
        "how do vectordatabase works?"
        ],
        # filtering by metadata 
        where={"category":"AI"},
        n_results=2
)

print(results["documents"])
print(results.keys())