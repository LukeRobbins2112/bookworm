import os
import openai
import chromadb
from chromadb.utils import embedding_functions
import pysqlite3
import sys
sys.modules['sqlite3'] = pysqlite3
import json
from dotenv import load_dotenv
load_dotenv()

collections = {}
openai_ef = None
key_list = ["meta", "characters", "plot", "timeline", "setting", "chapters"]

def initialize_db():
    # Disable telemetry to avoid the PostHog error in Python 3.8
    chromadb_settings = chromadb.config.Settings(
        anonymized_telemetry=False
    )
    
    # Initialize the OpenAI embedding function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name="text-embedding-3-small"
    )
    
    # Create a persistent client with telemetry disabled
    client = chromadb.EphemeralClient()
    
    for key in key_list:
        collections[key] = client.get_or_create_collection(name=key, embedding_function=openai_ef)


# helper for add_document() -- allows single or multiple string/dict/...
def _to_list(x):
    if isinstance(x, list):
        return x
    return [x]

# helper for add_document() -- generates IDs for each document entry
def _generate_id(prefix="coll"):
    # Initialize the counter if it doesn't exist
    if not hasattr(_generate_id, "counter"):
        _generate_id.counter = 0
    
    id_val = f"{prefix}_{_generate_id.counter}"
    _generate_id.counter += 1
    return id_val

def add_db_document(key, docs, metadatas_in=None):
    # ensure documents is in list format
    doc_list = _to_list(docs)

    # use provided metadatas, or fallback to minimial
    if metadatas_in is None:
        metadatas_in = [{"type": key} for _ in doc_list]
    else:
        mdata_list = _to_list(metadatas_in)
        
    if len(doc_list) != len(mdata_list):
        print("Error adding document: metadata list size != document list size")
        return
    
    id_list = [_generate_id(key) for _ in doc_list]
    
    collections[key].add(
        documents=doc_list,
        ids=id_list,
        metadatas=mdata_list,
    )

def load_db_documents(key):
    collection_path = os.path.join(os.getcwd(), key)
    for filename in os.listdir(collection_path):
        with open(os.path.join(collection_path, filename), 'r') as f:
            text = f.read()
            metadatas = [{
                "type": key,
                "name" : filename
            }]
            add_db_document(key, text, metadatas)
            print(f"Added document to db: {filename}")

def load_all_db_documents():
    for key in key_list:
        load_db_documents(key)


def lookup_entry(target, content):
    entry_result = "Not found"
    
    if target in collections:
        query_results = collections[target].query(
            query_texts=[content],
            n_results=1
        )
        entry_result = query_results["documents"][0]
        
    # print(f"Lookup for {target}: {entry_result}")
    return entry_result

def list_entries(target):
    if target not in collections:
        print(f"{target} not in collections")
        return

    collection = collections[target]
    
    # Get entries with pagination
    page_size = 100
    offset = 0
    
    while True:
        page = collection.get(limit=page_size, offset=offset)
        
        # If no more results, break the loop
        if len(page['ids']) == 0:
            break
        
        # Process this page of results
        print(f"Page {offset // page_size + 1}:")
        for i, doc_id in enumerate(page['ids']):
            print(f"ID: {doc_id}, Document: {page['documents'][i]}")
            
            # Move to the next page
            offset += page_size
