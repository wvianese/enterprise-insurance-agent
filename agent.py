import json
from dotenv import load_dotenv
from openai import OpenAI

from tools.claims_api import get_claims

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

        response = self.client.responses.create(
            model="gpt-5.4-mini",
            input=messages,
            tools=self.tools
        )

        messages += response.output

        tool_requested = False

        for item in response.output:
            if item.type != "function_call":
                continue

            tool_requested = True

            arguments = json.loads(item.arguments)

            if item.name == "get_claims":
                result = get_claims(**arguments)

            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(result)
                }
            )

        if not tool_requested:
            return response.output_text
        
        response = self.client.responses.create(
            model="gpt-5.4-mini",
            input=messages,
            tools=self.tools

        )

        return response.output_text