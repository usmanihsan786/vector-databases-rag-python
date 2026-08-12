import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import ollama


load_dotenv()
# client = OpenAI(

#     api_key=os.environ["Azure_Open_AI_API_KEY"],
#     base_url=os.environ["AZURE_OPENAI_ENDPOINT"]
# )

deployementName = "qwen3-embedding:4b"
text1 = "I love programming "
text2 = "elephent is giant animal"


response1 = ollama.embed(
    model=deployementName,
    input=text1
)
embedding1 = response1.embeddings[0]

response2 = ollama.embed(
    model= deployementName,
    input=text2
)

embedding2 = response2.embeddings[0]


def cosine_similarity(vec1, vec2):
    arr1= np.array(vec1)
    arr2 = np.array(vec2)

    return np.dot(arr1,arr2) / (
        np.linalg.norm(arr1)*  np.linalg.norm(arr2)
        )


similarity = cosine_similarity(embedding1,embedding2)

print(len(embedding1))
print(embedding1[:10])
print("--------------------------------------------")
print(len(embedding2))
print(embedding2[:10])
print("---------------------------------------------")
print(similarity)


