import os
from dotenv import load_dotenv
from langfuse.openai import OpenAI
from output import nicer_print

# Load variables from .env file
load_dotenv()

# Create OpenAI client instance
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# nicer_print(user_input, available_dishes, response.output_text)
