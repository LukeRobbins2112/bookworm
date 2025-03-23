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
import client_setup # custom helper
import chroma #custom helper
import formatter #custom helper
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

chroma.initialize_db()

# Set up OpenAI LLMs with their system prompt and seed conversation    
client_setup.initialize_db_llm(gpt_db_conversation, chroma.key_list)
client_setup.initialize_writer_llm(gpt_writer_conversation)


# Add all existing documents to respective collections
chroma.load_all_db_documents()


def analyze(user_question, context=None, backend="openai"):
    result = "no result"

    should_lookup = LOOKUP_FLAG
    
    if context is not None:
        llm_query += f"No lookup needed, context is {context}"
        should_lookup = "NEVER"

    llm_query = f"Lookup Flag: {should_lookup}.\n Here is my question: {user_question}"
        
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
        
        coll_context = chroma.lookup_entry(collection, query)
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
        for key in chroma.key_list:
            print(f"{key}\n")
        target = input("Which collection? ").lower()
                  
    content = input("Entry details: ")

    if user_input == "add":
        chroma.add_db_document(target, content)
    elif user_input == "lookup":
        chroma.lookup_entry(target, content)
    elif user_input == "query":
        query_db(content)
    elif user_input == "list":
        chroma.list_entries(target)
    else:
        print("Invalid option")
        continue

