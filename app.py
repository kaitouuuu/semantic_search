import chromadb
import json
import time
from concurrent.futures import ProcessPoolExecutor
from transformers import AutoTokenizer, AutoModel
import torch
from fastapi import FastAPI
from pydantic import BaseModel
import signal
import sys
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from multiprocessing import Pool, freeze_support
import os
from contextlib import asynccontextmanager
import logging
import psutil

# Replace the ProcessPoolExecutor with a Pool
# process_num = os.cpu_count()
process_num = 5
process_pool = None

# Define the local repository path
local_repo_path = 'local_data'  # This should match the mount point for the Docker volume

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
model = AutoModel.from_pretrained("BAAI/bge-m3")

# Ensure the model is in evaluation mode
model.eval()

# Instantiate ChromaDB instance. Data is stored on disk (a folder named 'local_data' will be used).
chroma_client = chromadb.PersistentClient(path=local_repo_path)

# Create the collection vector
chroma_client.list_collections()
collection_list = chroma_client.list_collections()
collection_filename = []
collection = []
for c in collection_list:
    collection_filename.append(c.name)
    collection_tmp = chroma_client.get_collection(name=c.name)
    collection.append(collection_tmp)

# Set up logging
log_folder = 'log'
os.makedirs(log_folder, exist_ok=True)
log_file = os.path.join(log_folder, 'process.log')

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def embed_text(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

    # Run the text through the model to get embeddings
    with torch.no_grad():
        model_output = model(**inputs)

    # Extract the embeddings from the [CLS] token (first token in the sequence)
    cls_embedding = model_output.last_hidden_state[:, 0, :]  # Get the [CLS] token representation

    return cls_embedding.numpy().flatten().tolist()  # Convert to a list after flattening

def compare_strings(s, t):
    s_words = s.split()
    t_words = t.split()

    len_s = len(s_words)
    len_t = len(t_words)

    i, j = 0, 0
    while i < len_s and j < len_t:
        if s_words[i] == t_words[j]:
            j += 1
        i += 1

    return j == len_t

def process_description(args):
    start_time = time.time()

    input_description, collection_name, input_constraint, input_price1, input_price2, input_id, input_is_combo, input_number_items, num_descriptions, num_items = args
    
    # Recreate the ChromaDB client and get the collection
    chroma_client = chromadb.PersistentClient(path=local_repo_path)
    target_collection = chroma_client.get_collection(name=collection_name)
    
    matched_items = []
    # First check if input_description matches the 'name' field in lowercase
    for idx, metadata in enumerate(target_collection.get()['metadatas']):
        if compare_strings(metadata['name'].lower(), input_description.lower()) and (input_price1 <= metadata['price'] <= input_price2):
            matched_items.append({
                "product_id": metadata['item_id'],
                "vendor": metadata['vendor'],
                "product_title": metadata['name'],
                "product_price_sold": metadata['price'],
                "product_price_original": metadata['price_0'],
                "discount": metadata['discount'],
                "product_description": metadata['full_description'],
                "product_combo": metadata['combo'],
                "product_link": metadata['link_product'],
                "product_image": metadata['link_image'],
                "product_type": metadata['type']
            })

    # If matches are found, skip embedding search and return matched_items
    if matched_items:
        return matched_items, []

    # Embed the input description
    embed = embed_text(input_description)

    if input_id == "":
        if input_constraint == "higher":
            query_condition = {"$and": [{"is_combo": {"$eq": input_is_combo}}, {"price": {"$gte": input_price1}}]}
        elif input_constraint == "lower":
            query_condition = {"$and": [{"is_combo": {"$eq": input_is_combo}}, {"price": {"$lte": input_price2}}]}
        elif input_constraint == "between":
            query_condition = {"$and": [{"is_combo": {"$eq": input_is_combo}}, {"$and": [{"price": {"$gte": min(input_price1, input_price2)}}, {"price": {"$lte": max(input_price1, input_price2)}}]}]}
        elif input_constraint == "equal":
            query_condition = {"$and": [{"is_combo": {"$eq": input_is_combo}}, {"price": {"$eq": input_price1}}]}
        else:
            query_condition = {"is_combo": {"$eq": input_is_combo}}

        # Use query embeddings instead of text
        query_results = target_collection.query(
            query_embeddings=[embed],  # Here, we use the precomputed embeddings
            n_results=min(max(5, input_number_items) * num_descriptions, num_items),
            include=['documents', 'distances', 'metadatas'],
            where=query_condition
        )
    else:
        query_results = target_collection.query(
            query_embeddings=[embed],  # Here, we use the precomputed embeddings
            n_results=min(max(5, input_number_items) * num_descriptions, num_items),
            include=['documents', 'distances', 'metadatas'],
            where={"item_id": {"$eq": input_id}}
        )

    # Store the top results for each description
    description_results = []
    for idx, metadata in enumerate(query_results['metadatas'][0]):
        product_info = {
            "product_id": metadata['item_id'],
            "vendor": metadata['vendor'],
            "product_title": metadata['name'],
            "product_price_sold": metadata['price'],
            "product_price_original": metadata['price_0'],
            "discount": metadata['discount'],
            "product_description": metadata['full_description'],
            "product_combo": metadata['combo'],
            "product_link": metadata['link_product'],
            "product_image": metadata['link_image'],
            "product_type": metadata['type']
        }
        description_results.append(product_info)

    end_time = time.time()
    logger.info(f'Total time taken for processing description {input_description}: {end_time - start_time} seconds')
    logger.info(f'Memory usage: {psutil.virtual_memory().percent}%')
    logger.info(f'CPU usage: {psutil.cpu_percent()}%')
    return matched_items, description_results

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global process_pool
    process_pool = Pool(processes=process_num, maxtasksperchild=None)
    yield
    # Shutdown
    if process_pool:
        process_pool.close()
        process_pool.join()

app = FastAPI(lifespan=lifespan)

def result_for_semantic_search(Filename, Description, Input_constraint, Input_price1, Input_price2, Input_id, Input_is_combo, Input_number_items):
    start = time.time()

    # Split the description string into an array of individual item descriptions
    descriptions = [desc.strip() for desc in Description.split(';')]
    num_descriptions = len(descriptions)

    # Input the price and its constraint for keyword search
    input_constraint = Input_constraint
    try:
        input_price1 = float(Input_price1)
    except (ValueError, TypeError):
        input_price1 = 0.0

    try:
        input_price2 = float(Input_price2)
    except (ValueError, TypeError):
        input_price2 = 0.0

    # Input the id for keyword search
    input_id = Input_id
    input_is_combo = Input_is_combo
    try:
        input_number_items = int(Input_number_items)
    except (ValueError, TypeError):
        input_number_items = 3

    all_results = []

    target_collection = None

    # Find the collection to search within
    for coll in collection:
        if coll.name == Filename:
            target_collection = coll
            break

    if target_collection is None:
        return json.dumps({"success": False, "error": f"No collection found with the name {Filename}"}, ensure_ascii=False)

    num_items = target_collection.count()

    # Prepare the arguments for each process
    process_args = [(desc, Filename, input_constraint, input_price1, input_price2, input_id, input_is_combo, input_number_items, num_descriptions, num_items) for desc in descriptions]

    global process_pool
    if process_pool is None:
        # If the pool doesn't exist, create it (this shouldn't happen normally)
        process_pool = Pool(processes=process_num, maxtasksperchild=None)
    
    results = process_pool.map(process_description, process_args)

    # Collect the results
    for matched_items, description_results in results:
        all_results.append(matched_items)
        all_results.append(description_results)

    # Interleave the results and apply deduplication
    final_results = []
    seen_product_ids = set()

    for i in range(input_number_items):
        for description_results in all_results:
            if len(description_results) > i:
                product_info = description_results[i]

                # Ensure distinct products
                if product_info['product_id'] not in seen_product_ids:
                    final_results.append(product_info)
                    seen_product_ids.add(product_info['product_id'])

    # If the results are not enough, ensure we still have n * num_descriptions
    if len(final_results) < input_number_items * num_descriptions:
        for description_results in all_results:
            for product_info in description_results:
                if product_info['product_id'] not in seen_product_ids:
                    final_results.append(product_info)
                    seen_product_ids.add(product_info['product_id'])
                if len(final_results) == input_number_items * num_descriptions:
                    break
            if len(final_results) == input_number_items * num_descriptions:
                break

    output = json.dumps({"success": True, "data": final_results[:input_number_items * num_descriptions]}, ensure_ascii=False)

    end = time.time()
    logger.info(f'Total time taken: {end - start} seconds')
    return output

# Define Pydantic model for search request
class SearchRequest(BaseModel):
    filename: str
    description: str
    input_constraint: str
    input_price1: float
    input_price2: float
    input_id: str
    input_is_combo: str
    input_number_items: int

# FastAPI endpoint for search
@app.post("/run/click")
async def search(request: SearchRequest):
    logger.info(f"Received search request: {request.json()}")
    
    result = result_for_semantic_search(
        request.filename,
        request.description,
        request.input_constraint,
        request.input_price1,
        request.input_price2,
        request.input_id,
        request.input_is_combo,
        request.input_number_items
    )
    return json.loads(result)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def signal_handler(sig, frame):
    logger.info('Shutting down gracefully...')
    global process_pool
    if process_pool:
        process_pool.close()
        process_pool.join()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def reload_collections():
    global chroma_client, collection, collection_filename

    # Re-instantiate ChromaDB client
    chroma_client = chromadb.PersistentClient(path=local_repo_path)

    # Clear the collections
    collection_filename.clear()
    collection.clear()

    # Reload the collections
    collection_list = chroma_client.list_collections()
    for c in collection_list:
        collection_filename.append(c.name)
        collection_tmp = chroma_client.get_collection(name=c.name)
        collection.append(collection_tmp)

# Reload collections on startup
reload_collections()

# At the end of your script
if __name__ == "__main__":
    freeze_support()  # This is necessary for Windows
    uvicorn.run(app, host="0.0.0.0", port=7860)
