import json

def create_user_query(action="", context=""):
    question = input("User> ").lower()
    if question.lower() == "quit":
        return None

    action = action.lower()
    if action not in {"lookup", "answer", ""}:
        print(f"Invalid input action {action}, aborting query creation")
        return None

    question_dict = {
        "query" : question,
        "action" : action,
        "context" : context
    }

    question_string = json.dumps(question_dict)
    print(question_string)
    return question_string
    
