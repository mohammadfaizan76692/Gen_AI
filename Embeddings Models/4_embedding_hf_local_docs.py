from langchain_huggingface import HuggingFaceEmbeddings , HuggingFacePipeline

# import os
# os.environ['HF_HOME'] = "D:/sentence_tansformer_cache"

# embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
embedding = HuggingFaceEmbeddings(
    model_name='D:\sentence_tansformer_cache\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf'  # full path to the downloaded model
)

text  = "Delhi is the capital of India"
documents =[
    "Delhi is the capital of India",
    # "Kolkata is the capital of West Bengal",
    # "Paris is the capital of France"
]
vector = embedding.embed_documents( documents)
print(type(vector[0]))
print("length of Embeddings: ", len(vector[0]))


print(str(vector))