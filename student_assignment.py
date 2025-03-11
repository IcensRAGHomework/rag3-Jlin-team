import datetime
import chromadb
import traceback
from chromadb.utils import embedding_functions
from model_configurations import get_model_configuration
import pandas as pd
import datetime
from chromadb.config import Settings

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"
def generate_hw01():
    client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space":"cosine"},
        embedding_function=openai_ef
    )
    
    if collection.count() == 0:
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
    else:
        print("Collection contains data, no need generate.")
    
    return collection

        
    
def generate_hw02(question, city, store_type, start_date, end_date):
    collection=generate_hw01()
    start_ts=int(start_date.timestamp())
    end_ts=int(end_date.timestamp())
    where_filter={
        "$and":[
        {"city":{"$in":city}},
        {"type":{"$in":store_type}},
        {"date":{"$gte":start_ts}},
        {"date":{"$lte":end_ts}}
        ]
    }
    
    query_results = collection.query(
        query_texts=[question],
        n_results=10,
        where=where_filter,
        include=["metadatas", "distances"]
    )
    
    filtered=[]
    for i, distance in enumerate(query_results["distances"][0]):
        similarity = 1 - distance
        if similarity >= 0.80:
            filtered.append((query_results["metadatas"][0][i]["name"], similarity))
            
    filtered.sort(key=lambda x: x[1], reverse=True)
    store_names = [name for name, _ in filtered]
    return store_names
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    collection = generate_hw01()
    
    query_results = collection.query(
        query_texts = [question],
        n_results = 10,
        where={"name": {"$eq": store_name}},
        include=["metadatas"]
    )
    
    if query_results["ids"]:
        store_id=query_results["ids"][0][0]
        metadata = query_results["metadatas"][0][0]
        metadata["new_store_name"] = new_store_name
        collection.update(ids=[store_id], metadatas=[metadata])
        print(f"Update {store_name} with new_store_name: {new_store_name}")
    else:
        print(f"Store {store_name} not found.")
    
    where_filer = {
        "$and":[
            {"city":{"$in":city}},
            {"type":{"$in":store_type}}
        ]
    }
    
    results = collection.query(
        query_texts = [question],
        n_results = 10,
        where =where_filer,
        include=["metadatas","distances"]
    )
    
    if results.get("metadatas") and results.get("distances"):
        filter_stores = []
        for i, distance in enumerate(results["distances"][0]):
            similarity = 1 - distance
            if similarity >= 0.80:
                metadata = results["metadatas"][0][i]
                store_name = metadata.get("new_store_name", metadata["name"])
                filter_stores.append((store_name, similarity))
            
        # sort by similarity to desc order
        filter_stores.sort(key=lambda x:x[1], reverse=True)
        store_names = [name for name,_ in filter_stores]
        return store_names
    
    print("No store found.")
    return []
    
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




print(generate_hw03(
    question="我想要找南投縣的田媽媽餐廳，招牌是蕎麥麵",
    store_name="耄饕客棧",
    new_store_name="田媽媽（耄饕客棧）",
    city=["南投縣"],
    store_type=["美食"]
))