from langchain_huggingface import HuggingFaceEmbeddings , HuggingFacePipeline

import os
os.environ['HF_HOME'] = "D:/sentence_tansformer_cache"

embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

text  = "Delhi is the capital of India"
vector = embedding.embed_query(text=text)

print(str(vector))