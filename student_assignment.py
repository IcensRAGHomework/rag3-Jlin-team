import datetime
import chromadb
import traceback
from chromadb.utils import embedding_functions
from model_configurations import get_model_configuration
import pandas as pd
from datetime import datetime
from chromadb.config import Settings

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"

def generate_hw01():
    settings = Settings(persist_directory="chroma.sqlite3")
    client = chromadb.Client(settings)
    collection = client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space":"cosine"}
    )
    df=pd.read_csv("COA_OpenData.csv")
    
    documents=[]
    metadatas=[]
    ids=[]
    for idx, row in df.iterrows():
        doc_text=row["HostWords"]
        documents.append(doc_text)
    
        try:
            dt=pd.to_datetime(row["CreateDate"])
            timestamp=int(dt.timestamp())
        except Exception:
            timestamp=None
            
        metadata={
            "file_name":"COA_OpenData.csv",
            "name":row["Name"],
            "type":row["Type"],
            "address":row["Address"],
            "tel":row["Tel"],
            "city":row["City"],
            "town":row["Town"],
            "date":timestamp
        }
        metadatas.append(metadata)
        ids.append(str(idx))
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    return collection
        
    
def generate_hw02(question, city, store_type, start_date, end_date):
    pass
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    pass
    
def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    return collection
