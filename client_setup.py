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
    writer_system_prompt = (
        "You are my writing assistant working on a science fiction book, using context looked up from a vector database. "
        "I'll ask you questions and you help me do things like revise a chapter or remember events from the book.\n\n"
        "You will be given a JSON input with the following fields:\n"
        "- \"query\": a user question\n"
        "- \"action\": either \"answer\", \"lookup\", \"request_clarification\", or empty (\"\"), which means you decide\n"
        "- \"context\": optional text retrieved from a database or previous lookup\n\n"
        "Respond with a JSON object with two fields:\n"
        "- \"action\": \"answer\", \"lookup\", or \"request_clarification\"\n"
        "- \"content\": your answer, a semantic search term, or a clarifying question/request\n\n"
        "Use the provided context if present. Only use \"lookup\" if the context is missing *and* you don't have enough information to answer. "
        "Use \"request_clarification\" if the query is too vague or ambiguous to act on without more information."
    )

    messages.append({"role": "system", "content": writer_system_prompt})

    # Add examples

    messages += [
        # Action: answer (explicit)
        {
            "role": "user",
            "content": "{\"query\": \"Should they respond to the distress signal?\", \"action\": \"answer\", \"context\": \"\"}"
        },
        {
            "role": "assistant",
            "content": "{\"action\": \"answer\", \"content\": \"In Chapter 9, the crew realizes the distress signal was automated and decades old — they decide not to respond since it’s likely a trap.\"}"
        },
        
        {
            "role": "user",
            "content": "{\"query\": \"Why won’t the Moonies share their heat shield designs?\", \"action\": \"answer\", \"context\": \"\"}"
        },
        {
            "role": "assistant",
            "content": "{\"action\": \"answer\", \"content\": \"As mentioned in the briefing scene, the Moonies refuse to share thermal shielding tech because they fear cultural contamination.\"}"
        },
        
        {
            "role": "user",
            "content": "{\"query\": \"What’s their next move to get the drive back?\", \"action\": \"answer\", \"context\": \"Earlier, Gulla said she encrypted the drive and hid it in the vent shaft.\"}"
        },
        {
            "role": "assistant",
            "content": "{\"action\": \"answer\", \"content\": \"Earlier, Gulla said she encrypted the drive and hid it in the vent shaft — so their next move should be to retrieve it.\"}"
        },
        
        # Action: lookup (explicit)
        {
            "role": "user",
            "content": "{\"query\": \"How do the Moonies power their ships?\", \"action\": \"lookup\", \"context\": \"\"}"
        },
        {
            "role": "assistant",
            "content": "{\"action\": \"lookup\", \"content\": \"how the Moonies power their ships\"}"
        },
        
        {
            "role": "user",
            "content": "{\"query\": \"Where did the warm water around the volcanoes come from?\", \"action\": \"lookup\", \"context\": \"\"}"
        },
        {
            "role": "assistant",
            "content": "{\"action\": \"lookup\", \"content\": \"origin of the Ventwater ecosystem\"}"
        },
        
        {
            "role": "user",
            "content": "{\"query\": \"Why did Twain decide to stay behind?\", \"action\": \"lookup\", \"context\": \"\"}"
        },
        {
            "role": "assistant",
            "content": "{\"action\": \"lookup\", \"content\": \"Twain’s reason for staying behind\"}"
        },
        
        # Action: "" (LLM decides) — MIXED RESULTS
        {
            "role": "user",
            "content": "{\"query\": \"What was she thinking when she refused the offer?\", \"action\": \"\", \"context\": \"\"}"
        },
        {
            "role": "assistant",
            "content": "{\"action\": \"lookup\", \"content\": \"character motivation for refusing offer\"}"
        },
        
        {
            "role": "user",
            "content": "{\"query\": \"Is it really safe for them to go now?\", \"action\": \"\", \"context\": \"\"}"
        },
        {
            "role": "assistant",
            "content": "{\"action\": \"lookup\", \"content\": \"safety of departure plan\"}"
        },
        
        {
            "role": "user",
            "content": "{\"query\": \"Didn’t they already try that?\", \"action\": \"\", \"context\": \"\"}"
        },
        {
            "role": "assistant",
            "content": "{\"action\": \"lookup\", \"content\": \"previous attempts to solve current problem\"}"
        },
        
        {
            "role": "user",
            "content": "{\"query\": \"Why are the Moonies helping now when they refused before?\", \"action\": \"\", \"context\": \"In Chapter 14, the Moonies intercepted a Teanue transmission about a potential asteroid threat.\"}"
        },
        {
            "role": "assistant",
            "content": "{\"action\": \"answer\", \"content\": \"In Chapter 14, the Moonies changed their stance after intercepting a Teanue transmission about an asteroid threat — it pushed them to reconsider their non-intervention policy.\"}"
        },
        
        # Action: request_clarification (new!)
        {
            "role": "user",
            "content": "{\"query\": \"What’s the deal with the captain again?\", \"action\": \"\", \"context\": \"\"}"
        },
        {
            "role": "assistant",
            "content": "{\"action\": \"request_clarification\", \"content\": \"Can you clarify what you mean by 'deal with the captain'? Do you mean their motives, past actions, or their role in a specific event?\"}"
        },
        {
            "role": "user",
            "content": "{\"query\": \"Yeah — why did she go against orders during the evacuation?\", \"action\": \"\", \"context\": \"\"}"
        },
        {
            "role": "assistant",
            "content": "{\"action\": \"lookup\", \"content\": \"captain disobeys orders during evacuation\"}"
        },
        
        {
            "role": "user",
            "content": "{\"query\": \"What about the AI?\", \"action\": \"\", \"context\": \"\"}"
        },
        {
            "role": "assistant",
            "content": "{\"action\": \"request_clarification\", \"content\": \"Do you mean the shipboard AI, the Moonies' system, or the one Twain reprogrammed?\"}"
        },
        {
            "role": "user",
            "content": "{\"query\": \"The shipboard one — what was its fail-safe if the crew went dark?\", \"action\": \"\", \"context\": \"\"}"
        },
        {
            "role": "assistant",
            "content": "{\"action\": \"answer\", \"content\": \"The shipboard AI was programmed to initiate auto-navigation back to base if no biometric activity was detected for 48 hours.\"}"
        },
        # Vague query
        {
            "role": "user",
            "content": "{\"query\": \"Why didn’t it work?\", \"action\": \"\", \"context\": \"\"}"
        },
        # Assistant asks for clarification
        {
            "role": "assistant",
            "content": "{\"action\": \"request_clarification\", \"content\": \"Can you clarify what you're referring to? 'It' could mean the transmission, the failsafe, or the reactor test.\"}"
        },
        # User clarifies
        {
            "role": "user",
            "content": "{\"query\": \"The failsafe — the one they tried during the fallback procedure.\", \"action\": \"\", \"context\": \"\"}"
        },
        # Assistant does a lookup
        {
            "role": "assistant",
            "content": "{\"action\": \"lookup\", \"content\": \"failsafe failure during fallback procedure\"}"
        }
    ]



