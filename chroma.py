import os
import chromadb
from chromadb.utils import embedding_functions
import chromadb.config
import pysqlite3
import sys
sys.modules['sqlite3'] = pysqlite3
import anthropic
import openai
import json
import client_setup
from dotenv import load_dotenv
load_dotenv()

# LOOKUP_FLAG options: NEVER, ALWAYS, ON_DEMAND
LOOKUP_FLAG = "ON_DEMAND" 

API_KEY_RAG = os.getenv("OPENAI_API_KEY_RAG_LOOKUP")
API_KEY_CHAT = os.getenv("OPENAI_API_KEY_WRITING_ASSISTANT")

claude_client = anthropic.Anthropic()
gpt_db_client = openai.OpenAI(api_key=API_KEY_RAG)
gpt_writer_client = openai.OpenAI(api_key=API_KEY_CHAT)

claude_conversation = []
gpt_db_conversation = []
gpt_writer_conversation = []

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


collections = {}
key_list = ["meta", "characters", "plot", "timeline", "setting", "chapters"]
for key in key_list:
    collections[key] = client.get_or_create_collection(name=key, embedding_function=openai_ef)

# Set up OpenAI LLMs with their system prompt and seed conversation    
client_setup.initialize_db_llm(gpt_db_conversation, key_list)
client_setup.initialize_writer_llm(gpt_writer_conversation)


def generate_id(prefix="coll"):
    # Initialize the counter if it doesn't exist
    if not hasattr(generate_id, "counter"):
        generate_id.counter = 0
    
    id_val = f"{prefix}_{generate_id.counter}"
    generate_id.counter += 1
    return id_val

# Create a collection with the OpenAI embedding function
meta_collection_name = "meta_documents"
character_collection_name = "character_documents"


# Add some documents to the collection
meta_documents = [
    "Book Title: 'Earth is Fucked (and So Am I)'",
    "Plot Elevator Pitch: An astronaut scientist is on a mission to a foreign planet to do research on the planet and its primitive alien inhabitants. When it comes time to leave, he accidentally gets left behind, with no help coming. To get home, he's going to have to do so with the help of the unwitting civilization of the planet. The planet? Earth. The primitive civilization? Humans, 2035AD.",
]

character_documents = [
    "The main character's name is Twain, he is the protagonist and first person narrator. He s cynical and sarcastic, with wit, but not mean spirited.",
    "The supporting character is Gulla, Twain's research partner and the only member of his species on Earth with him. Gulla is a bit absent-minded, though brilliant in technical areas.",
]

# Add documents with unique IDs
collections["meta"].add(
    documents=meta_documents,
    ids=["meta1", "meta2"],
    metadatas=[
        {"type": "overview"},
        {"type": "overview"},
    ]
)

collections["characters"].add(
    documents=character_documents,
    ids=["meta3", "meta4"],
    metadatas=[
        {"type": "character"},
        {"type": "character"},
    ]
)

chapters_path = os.getcwd() + "/chapters"
for filename in os.listdir(chapters_path):
    with open(os.path.join(chapters_path, filename), 'r') as f:
        text = f.read()
        chap_idx = filename.removeprefix("chapter")
        collections["chapters"].add(
            documents = [text],
            ids = [generate_id("chapter")],
            metadatas = [{
                "type": "chapter",
                "chapter_index": chap_idx
            }]
        )
        sample = text[0:50]
        # print(f"chapter {chap_idx}: {sample}")


def add_entry(target, content):
    if target in collections:
        collections[target].add(
            documents = [content],
            ids = [generate_id(target)],
            metadatas=[{"type": target}]
        )
    print(f"Added document to db: {target}")


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

def analyze(user_question, context=None, backend="openai"):
    result = "no result"

    should_lookup = LOOKUP_FLAG
    
    if context is not None:
        llm_query += f"No lookup needed, context is {context}"
        should_lookup = "NEVER"

    llm_query =f"Lookup Flag: {should_lookup}.\n Here is my question: {user_question}"
        
    if backend == "openai":
        # print("using openai writing assistant")
        gpt_writer_conversation.append({"role": "user", "content": llm_query})
        completion_analysis = gpt_writer_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=gpt_writer_conversation,
            max_completion_tokens=200,
            n=1,
        )
        
        # Get response and print helpful part
        result = completion_analysis.choices[0].message.content
        gpt_writer_conversation.append({"role": "assistant", "content": result})
    else:
        # use Claude
       #  print("using claude writing assistant")
        message = claude_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=200,
            temperature=1,
            system="You are my creative writing partner helping me digest plot, characters, etc. I'll ask you questions, prefixed with context looked up from vector db. Respond with helpful comments or reminders about things I may have forgotten",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": llm_query
                        }
                    ]
                }
            ],
        )
        result = message.content[0].text

    print(f"\n\nANALYSIS: {result}")
    return result

def query_db(lookup_target):
    lookup_gen = f"Lookup: {lookup_target}"
    gpt_db_conversation.append({"role": "user", "content": lookup_gen})
    completion_lookup = gpt_db_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=gpt_db_conversation,
        max_completion_tokens=200,
        n=1,
    )
    lookup_string = completion_lookup.choices[0].message.content 
    # print(lookup_string)
    lookups = json.loads(lookup_string)
    
    context = ""
    
    # Process the lookups
    for lookup in lookups:
        collection = lookup["collection"]
        query = lookup["query"]
        print(f"Querying {collection}: {query}\n")
        
        coll_context = lookup_entry(collection, query)
        context += f"{collection}: {coll_context}\n"
        
    print(f"\nCONTEXT: {context}\n")
    
    # append assistant response to conversation to grow context
    gpt_db_conversation.append({"role": "assistant", "content": lookup_string})

def discuss():
    print("What do you want to discuss?\n")
    
    while True:
        question = input("User> ").lower()
        if question.lower() == "quit":
            break

        # Do initial submission -- if no additional context needed, just provide response. Otherwise, do a lookup and resubmit.
        analysis_result = analyze(question)
        initial_json = json.loads(analysis_result)

        # 
        if initial_json["action"].lower() == "answer":
            print("ANSWER: ")
            print(initial_json["content"])
        elif initial_json["action"].lower() == "lookup":
            print("LOOKING UP CONTEXT...")
            db_context = query_db(question)  
            analysis_result = analyze(question, db_context)
        else:
            print("Error: Invalid action type")

        
        

# def query_db(target, content):
#     question = input("What do you want to know? ").lower()
    
#     context = lookup_entry(target, content)
#     llm_query = f"Use the following context: {context} for my question: {question}"

#     gpt_conversation.append({"role": "user", "content": llm_query})
#     completion = gpt_client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=gpt_conversation,
#         max_completion_tokens=200,
#         n=1,
#     )
#     print(completion.choices[0].message.content)

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

while True:
    user_input = input("Add, Lookup, Query, List, Discuss? ").lower()
    if user_input.lower() == "q":
        print("Exiting")
        break

    if user_input == "discuss":
        discuss()
        continue

    target = input("Which collection? ('Show' to show options) ").lower()
    if target.lower() == "show":
        for key in collections.keys():
            print(f"{key}\n")
        target = input("Which collection? ").lower()
                  
    content = input("Entry details: ")

    if user_input == "add":
        add_entry(target, content)
    elif user_input == "lookup":
        lookup_entry(target, content)
    elif user_input == "query":
        query_db(content)
    elif user_input == "list":
        list_entries(target)
    elif user_input == "discuss":
        discuss()
    else:
        print("Invalid option")
        continue

