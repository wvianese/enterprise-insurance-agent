import json
from dotenv import load_dotenv
from openai import OpenAI

from tools.claims_api import get_claims
from tools.customer_api import get_customer
from tools.policy_search import search_policy_documents

load_dotenv()

class Agent:
    def __init__(self):
        self.client = OpenAI()
        self.tools = [
            {
                "type": "function",
                "name": "get_claims",
                "description": "Retrieve the claims history for a customer using their customer ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_id": {
                            "type": "string",
                            "description": "The customer's ID, for example '104'"
                        }
                    },
                    "required": ["customer_id"],
                    "additionalProperties": False
                },
                "strict": True
            },
            {
                "type": "function",
                "name": "get_customer",
                "description": "Retrieve the customer information using their customer ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_id": {
                            "type": "string",
                            "description": "The customer's ID, for example '104'"
                        }
                    },
                    "required": ["customer_id"],
                    "additionalProperties": False
                },
                "strict": True
            },
            {
                "type": "function",
                "name": "search_policy_documents",
                "description": "Search insurance policy documents for information relevant to the user's question",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The policy information to search for"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "The maximum number of relevant policy chunks to return"
                        }
                    },
                    "required": ["query", "top_k"],
                    "additionalProperties": False
                },
                "strict": True
            }
        ]
    def run(self, user_input):

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an insurance claims assistant. Use the available tools whenever "
                    "they are needed to answer the user's question."
                ),
            },
            {
                "role": "user",
                "content": user_input
            }
        ]

        while True:

            response = self.client.responses.create(
                model="gpt-5.4-mini",
                input=messages,
                tools=self.tools
            )

            messages += response.output

            function_calls = [item for item in response.output if item.type == "function_call"]

            if not function_calls:
                return response.output_text #Loops until all function calls are resolved and returns the final output text

            for item in function_calls:

                arguments = json.loads(item.arguments)

                if item.name == "get_claims":
                    result = get_claims(**arguments)

                elif item.name == "get_customer":
                    result = get_customer(**arguments)

                elif item.name == "search_policy_documents":
                    result = search_policy_documents(**arguments)

                messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps(result)
                    }
                )