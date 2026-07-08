import os
import random
from dotenv import load_dotenv
from langfuse import get_client
from langfuse.openai import OpenAI
from output import nicer_print

# Load variables from .env file
load_dotenv()

# Create OpenAI client instance
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Generic catalogue of dishes the kitchen can cook
ALL_DISHES = [
    "Bolognese",
    "Shawarma",
    "Doner Kebab",
    "Sushi",
    "Margherita Pizza",
    "Pad Thai",
    "Chicken Curry",
    "Caesar Salad",
    "Beef Tacos",
    "Ramen",
]

# The trick: only expose 5 random dishes to the LLM on each call,
# so the available "menu" changes from run to run.
available_dishes = random.sample(ALL_DISHES, 5)


def _generate_user_input() -> str:
    input_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    response = input_client.responses.create(
        model="gpt-4o-mini",
        instructions=(
            "You generate a single, realistic meal request as if written by "
            "a hungry person deciding what to eat. Vary the mood, cravings, "
            "and phrasing each time. "
            "Return only the request, with no extra commentary."
        ),
        input="Generate one random meal request.",
    )

    return response.output_text.strip()


langfuse = get_client()

# Group both OpenAI calls under a single parent trace so the whole
# request cycle is observable as one unit in Langfuse.
with langfuse.start_as_current_observation(name="meal-planning"):
    user_input = _generate_user_input()
    # user_input = "I want a big bowl of creamy mac and cheese with a crispy topping"

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=(
            "You are a helpful meal planning assistant. "
            "Create a meal plan of exactly 2 dishes chosen ONLY from the "
            "following available dishes:\n"
            f"{', '.join(available_dishes)}\n"
            "Tailor the choices to the user's request. "
            "Do not invent dishes that are not in the list. "
            "For each dish, add a short one-line reason why it fits."
        ),
        input=user_input,
    )

nicer_print(user_input, available_dishes, response.output_text)
