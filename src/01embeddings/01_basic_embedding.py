import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


client = OpenAI(
        api_key=os.environ["Azure_Open_AI_API_KEY"],
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"]
    )
deployment_name = "text-embedding-3-small"
text = "Artificial Intelligence is transforming"

response = client.embeddings.create(
        model=deployment_name,
        input=text
    )

embedding = response.data[0].embedding

print(len(embedding))
print(response.usage)
print(response.model)
print("First 10 values:", [f"{x:.3f}" for x in embedding[:20]])





