from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np




embedding_model = HuggingFaceEmbeddings(model_name='D:\sentence_tansformer_cache\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf')


documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "which indian captain know for his calm demeanor"

## doc embeddings
doc_emb = embedding_model.embed_documents(documents)

query_emb = embedding_model.embed_query(query)

similarity_scores = cosine_similarity([query_emb], doc_emb) ### have put 2D lists
print(f"similarity_scores {similarity_scores}")

index = np.argmax(np.array(similarity_scores[0])) ## getting index with highest similarity

print(documents[index])
