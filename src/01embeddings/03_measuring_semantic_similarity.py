import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import ollama

load_dotenv()



deployementName = "text-embedding-3-small"
# deployementName = "qwen3-embedding:4b"
text1 = "how can I learn python? "

results = []

# response1 = ollama.embed(
#     model=deployementName,
#     input=text1
# )

client = OpenAI(
    api_key=os.environ["Azure_Open_AI_API_KEY"],
    base_url=os.environ["AZURE_OPENAI_ENDPOINT"]
)
# query_embedding = response1.embeddings[0]
response1 =  client.embeddings.create(
    model= deployementName,
    input=text1
)

query_embedding =  response1.data[0].embedding

def cosine_similarity(vec1, vec2):
    arr1= np.array(vec1)
    arr2 = np.array(vec2)

    return np.dot(arr1,arr2) / (
    np.linalg.norm(arr1)*  np.linalg.norm(arr2) )

documents = [

    "Python programming tutorial",
    "Learn Java in 30 Days",
    "Healthy cooking recipies",
    "Begineer python course",
    "Elephants live in forests",
    "The weather is sunny today",
    "A car has four wheels"

]


for doc in documents:
    response2 = client.embeddings.create(
    model=deployementName,
    input=doc
    )
    doc_embedding = response2.data[0].embedding
    score = cosine_similarity(query_embedding,doc_embedding)

    results.append((doc,score))

results.sort(
        key= lambda item: item[1] , 
        reverse= True   
    )

for doc,score in results:
    print(f"{score:.3f} {doc}")



          
# similarity = cosine_similarity(embedding1,embedding2)

# print(len(embedding1))
# print(embedding1[:10])
# print("--------------------------------------------")
# print(len(embedding2))
# print(embedding2[:10])
# print("---------------------------------------------")
# print(similarity)




