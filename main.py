import os
from dotenv import load_dotenv
from openai import OpenAI

# Load variables from .env file
load_dotenv()

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.getenv("OPENAI_API_KEY"),
)

response = client.responses.create(
    model="gpt-5-mini",
    instructions="You are a coding assistant that talks like a pirate.",
    input="How do I check if a Python object is an instance of a class?",
)

print(response.output_text)
