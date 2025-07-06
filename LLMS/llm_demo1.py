from langchain.llms import OpenAI
from dotenv import load_dotenv # used for getting environment varibale
load_dotenv()

## laoding openAI llm
llm = OpenAI(model = "gpt-3.5-turbo-instruct")

## just for hitting invoke taking pron 
response = llm.invoke("What is the capital of Spain")

print(response) 