from langchain_huggingface import HuggingFaceEmbeddings , HuggingFacePipeline

import os
os.environ['HF_HOME'] = "D:/jina_embedding_cache"

embedding = HuggingFaceEmbeddings(model_name = 'jinaai/jina-embeddings-v3')

text  = "Delhi is the capital of India"
vector = embedding.embed_query(text=text)

print(str(vector))