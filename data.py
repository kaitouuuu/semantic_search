import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import csv
import json
import numpy as np
import gradio as gr
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer, AutoModel
import torch

# Define the local repository path
local_repo_path = 'local_data'  # This should match the mount point for the Docker volume

# Select the embedding model to use
# sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-m3")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
model = AutoModel.from_pretrained("BAAI/bge-m3")

# Ensure the model is in evaluation mode
model.eval()

# Instantiate ChromaDB instance. Data is stored on disk (a folder named 'local_data' will be used).
chroma_client = chromadb.PersistentClient(path=local_repo_path)
# chroma_client = chromadb.HttpClient(host='apps.etc.run', port=8000)

# Create the collection vector
chroma_client.list_collections()
collection_list = chroma_client.list_collections()
collection_filename = []
collection = []
for c in collection_list:
    collection_filename.append(c.name)
    collection_tmp = chroma_client.get_collection(name=c.name)
    collection.append(collection_tmp)

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

    return gr.Dropdown(choices=collection_filename, interactive=True), gr.Dropdown(choices=collection_filename, interactive=True)

def push_data(CSV_File):
    # Create the collection, aka vector database. Or, if database already exists, then use it. Specify the model that we want to use to do the embedding.
    collection_name = os.path.splitext(os.path.basename(CSV_File.name))[0]

    if collection_name in collection_filename:
        chroma_client.delete_collection(name=collection_name)
        # Update the collections list to remove the deleted collection
        collection[:] = [coll for coll in collection if coll.name != collection_name]
        # Update the collection filenames list to remove the deleted collection's name
        collection_filename[:] = [name for name in collection_filename if name != collection_name]

    collection_tmp = chroma_client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    with open(CSV_File.name) as file:
        lines = csv.reader(file)

        documents = []
        metadatas = []
        ids = []
        embeddings = []  # New array to store the embeddings

        # Loop through each line and populate the 3 arrays.
        for i, line in enumerate(lines):
            if i == 0:
                continue

            document = line[1].lower() + ", " + line[2] + ", " + line[4]
            documents.append(document)
            metadatas.append({"item_id": line[0], "vendor": line[1], "name": line[2], "price": float(line[3]), "full_description": line[5], "combo": line[6], "link_product": line[7], "link_image": line[8], "type": line[9], "price_0": line[10], "discount": line[11], "is_combo": line[12]})
            ids.append(line[0])

            # Embed each document and store the result
            embeddings.append(embed_text(document))

    # Add all the data to the vector database. ChromaDB automatically converts and stores the text as vector embeddings. This may take a few minutes.
    collection_tmp.add(embeddings=embeddings, metadatas=metadatas, ids=ids)

    # Add to collection
    collection.append(collection_tmp)
    collection_filename.append(collection_name)

    # Update the dropdown lists with the new collection name
    df = pd.read_csv(CSV_File.name, encoding='utf-8')

    return df.to_csv(index=False), gr.Dropdown(choices=collection_filename, interactive=True), gr.Dropdown(choices=collection_filename, interactive=True)

def delete_data(filename):
    is_delete = "Cannot find " + filename
    if filename in collection_filename:
        chroma_client.delete_collection(name=filename)
        # Update the collections list to remove the deleted collection
        collection[:] = [coll for coll in collection if coll.name != filename]
        # Update the collection filenames list to remove the deleted collection's name
        collection_filename[:] = [name for name in collection_filename if name != filename]
        is_delete = "Deleted " + filename

    return is_delete, gr.Dropdown(choices=collection_filename, interactive=True), gr.Dropdown(choices=collection_filename, interactive=True)

def process_description(input_description, target_collection, input_constraint, input_price1, input_price2, input_id, input_is_combo, input_number_items, num_descriptions, num_items):
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

    return matched_items, description_results

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
        input_price1 = 0.0  # or any default value you prefer

    try:
        input_price2 = float(Input_price2)
    except (ValueError, TypeError):
        input_price2 = 0.0  # or any default value you prefer

    # Input the id for keyword search
    input_id = Input_id
    input_is_combo = Input_is_combo
    try:
        input_number_items = int(Input_number_items)
    except (ValueError, TypeError):
        input_number_items = 3  # or any default value you prefer

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

    # # Use multithreading to process descriptions concurrently
    # with ThreadPoolExecutor(max_workers=2) as executor:
    #     # Map futures to their descriptions
    #     future_to_description = {executor.submit(process_description, desc, target_collection, input_constraint, input_price1, input_price2, input_id, input_is_combo, input_number_items, num_descriptions, num_items): desc for desc in descriptions}

    #     for future in as_completed(future_to_description):
    #         input_description = future_to_description[future]
    #         try:
    #             matched_items, description_results = future.result()
    #             all_results.append(matched_items)
    #             all_results.append(description_results)
    #         except Exception as exc:
    #             print(f'{input_description} generated an exception: {exc}')

    # Use singlethreading to process descriptions
    for input_description in descriptions:
        matched_items, description_results = process_description(input_description, target_collection, input_constraint, input_price1, input_price2, input_id, input_is_combo, input_number_items, num_descriptions, num_items)
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
    print(f'Total time taken: {end - start} seconds')
    return output

demo = gr.Blocks()

with demo:
    gr.Markdown("### Search engine.")

    with gr.Tabs():
        with gr.TabItem("Push data"):
            gr.Markdown("""
               ### Instructions for Push Data
                - Use this tab to upload your CSV file.
                - Once the file is uploaded, you can move to the 'Search' tab to perform searches.
            """)
            with gr.Row():
                file_input = gr.File()
            push_data_button = gr.Button("Upload")
            output_csv = gr.Textbox(label="Result")
        with gr.TabItem("Delete"):
            gr.Markdown("""
                ### Instructions for Delete
                - Use this tab to delete your CSV file.
            """)
            with gr.Row():
                file_delete = gr.Dropdown(label="File to delete", choices=collection_filename)
            delete_data_button = gr.Button("Delete")
            output_delete = gr.Textbox(label="Result")
        with gr.TabItem("Search"):
            gr.Markdown("""
                ### Instructions for Search
                - **Keyword:** Provide text input for semantic search. If empty, it will be treated as "".
                - **Constraint:** Can be one of the following:
                    - "Lower" for values less than the given price.
                    - "Higher" for values greater than the given price.
                    - "Equal" for values equal to the given price.
                    - "Between" for values between two given prices.
                    - "None" if no constraint is needed.
                - **From price:** Input a float value as the first price.
                - **To price:** Input a float value as the second price (only used if constraint is "Between").
                - **Product ID:** Input the unique ID to search for an exact match. If empty, it will be treated as "".
            """)
            with gr.Row():
                file_search = gr.Dropdown(label="File to search", choices=collection_filename)
                description_input = gr.Textbox(label="Keyword")
                constraint_input = gr.Dropdown(label="Price filter type", choices=["between", "lower", "higher", "equal", "none"], value="none")
                price1 = gr.Number(label="From price", value="0")
                price2 = gr.Number(label="To price", value="0")
                id_input = gr.Textbox(label="Product ID", type="text", value="")
                is_combo_input = gr.Dropdown(label="Is Combo?", choices=["True", "False"], value="False", interactive=True)
                number_items_input = gr.Textbox(label="Number of items", value="3")
            search_button = gr.Button("Search")
            output = gr.Textbox(label="Result")

    push_data_button.click(push_data, inputs=file_input, outputs=[output_csv, file_delete, file_search])
    delete_data_button.click(delete_data, inputs=file_delete, outputs=[output_delete, file_delete, file_search])
    search_button.click(api_name="click", fn=result_for_semantic_search, inputs=[file_search, description_input, constraint_input, price1, price2, id_input, is_combo_input, number_items_input], outputs=output)
    demo.load(reload_collections, inputs=None, outputs=[file_delete, file_search])

demo.queue(api_open=True)
demo.launch(
    debug=True,
    show_error=True,
    share=True,
    show_api=True,
    server_name="0.0.0.0",
    server_port=7861
)