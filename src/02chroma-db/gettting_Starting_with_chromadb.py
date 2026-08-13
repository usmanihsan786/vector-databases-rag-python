import chromadb

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_or_create_collection(
    name="course-documents"
)

collection.add(
    ids=["doc1"],
    documents=["Python is Popular for AI"],
    metadatas=[
        {
            "category": "python",
            "chapter": "1",
            "author": "sudip"
        }
    ]
)

print("Document added successfully!")