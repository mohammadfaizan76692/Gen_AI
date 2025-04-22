from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
## changing cache directory very very important
os.environ['HF_HOME'] = 'D:/huggingface_cache_locally'
llm = HuggingFacePipeline.from_model_id(
    
    model_id  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = 'text-generation',                                   
    pipeline_kwargs= dict(
        temperature  = 0.2,
        max_new_tokens = 200)

)

model= ChatHuggingFace(llm = llm)

reponse = model.invoke("What is the Capital of India")
print(reponse.content)