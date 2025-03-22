import openai
import json
from dotenv import load_dotenv
load_dotenv()

def initialize_db_llm(messages, key_list):
    system_prompt = "You are my writing partner for my science fiction novel. You are helping me figure out the plot and characters, with the aid of a vector database for looking up relevant info. You will use a database with several indexed sections corresponding to various components of the novel like chapters, plot, characters, etc. The python dictionary is called 'collections' and it has the following keys, whose values are the actual individual chromadb collections. These are: \n"
    
    for key in key_list:
        system_prompt += f"{key},"

    system_prompt += "\n\nIf I ask you a question about the book and start with the prefix 'Lookup: ', look up relevant context in the relevant collection. use best judgement to select (a) the proper collection and (b) an intelligent query key, which you should produce based on what you think will pull up good information from this semantic search. Use your existing context window to think about what information I'll need for my query. Provide this as json output, so I can feed it to my function performing DB lookups. Do not output ANYTHING else but the JSON when you see the 'Lookup: ' prefix. For example, if I say: 'Lookup: tell me about Twain being stranded' output something like: \n"

    example_json = "[{\"collection\": \"plot\", \"query\": \"Where was Twain stranded?\"}, {\"collection\": \"characters\", \"query\": \"Who was stranded?\"}, {\"collection\": \"chapters\", \"query\": \"Cycler fails and Gulla comes back.\"}]"

    system_prompt += example_json

    messages.append({"role": "system", "content": system_prompt})

    
def initialize_writer_llm(messages):
    writer_system_prompt = "You are my writing assitant working on a science fiction book, using context looked up from a vector database. I'll ask you questions and you help me do things like revise a chapter or remember events from the book. If you're unsure of something, we'll need to look up info in vector database, but you need to let me know if you need more context. You provide your outputs as JSON with the fields 'action' and 'content'. For 'action', if you need more info set this to 'lookup' indicating a need to go to the database, otherwise set to 'answer'. For 'content' if you  need to do a lookup, set this to a good semantic search phrase for the db, but if you think you know the answer just provide your actual analysis/feedback under 'content' I'll also include a 'LOOKUP_FLAG of 'ALWAYS' 'NEVER' or 'ON_DEMAND', which may be used to hardcode whether you perform a lookup. ALWAYS --> lookup first time you see a question, NEVER --> never lookup, just answer, ON_DEMAND --> Decide based on what is needed for the question + your existing context."

    messages.append({"role": "system", "content": writer_system_prompt})

    # Add examples
    messages += [
        {
            "role": "user",
            "content": "Should they respond to the distress signal?"
        },
        {
            "role": "assistant",
            "content": "{\"action\": \"answer\", \"content\": \"In Chapter 9, the crew realizes the distress signal was automated and decades old — they decide not to respond since it’s likely a trap.\"}"
        },
        {
            "role": "user",
            "content": "Why won’t the Moonies share their heat shield designs?"
        },
        {
            "role": "assistant",
            "content": "{\"action\": \"answer\", \"content\": \"As mentioned in the briefing scene, the Moonies refuse to share thermal shielding tech because they fear cultural contamination.\"}"
        },
        {
            "role": "user",
            "content": "What’s their next move to get the drive back?"
        },
        {
            "role": "assistant",
            "content": "{\"action\": \"answer\", \"content\": \"Earlier, Gulla said she encrypted the drive and hid it in the vent shaft — so their next move should be to retrieve it.\"}"
        },
        {
            "role": "user",
            "content": "How do the Moonies power their ships?"
        },
        {
            "role": "assistant",
            "content": "{\"action\": \"lookup\", \"content\": \"how the Moonies power their ships\"}"
        },
        {
            "role": "user",
            "content": "Where did the warm water around the volcanoes come from?"
        },
        {
            "role": "assistant",
            "content": "{\"action\": \"lookup\", \"content\": \"origin of the Ventwater ecosystem\"}"
        },
        {
            "role": "user",
            "content": "Why did Twain decide to stay behind?"
        },
        {
            "role": "assistant",
            "content": "{\"action\": \"lookup\", \"content\": \"Twain’s reason for staying behind\"}"
        }
    ]
