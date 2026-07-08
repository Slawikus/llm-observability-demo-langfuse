import os
from dotenv import load_dotenv
from openai import OpenAI

# Load variables from .env file
load_dotenv()

# Create OpenAI client instance
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Talk to LLM
response = client.responses.create(
    model="gpt-4o-mini",
    instructions="You are a coding assistant that talks like a pirate.",
    input="How do I check if a Python object is an instance of a class?",
)

print(response.output_text)
