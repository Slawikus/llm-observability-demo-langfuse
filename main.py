import os
import uuid
from dotenv import load_dotenv
from langfuse.openai import OpenAI
from pydantic import BaseModel
from langfuse import get_client, observe, propagate_attributes

# Load variables from .env file
load_dotenv()

DEPARTMENTS = {
    "rental-contracts":
        "Rental Contracts — lease questions, renewals, amendments",
    "terminations-moveout":
        "Terminations & Move-out — cancellations, move-out dates, deposit returns",
    "tenant-complaints":
        "Tenant Complaints — noise, neighbour disputes, general grievances",
    "energy-heating":
        "Energy & Heating — heating failures, hot water, utility billing",
    "repairs-maintenance":
        "Repairs & Maintenance — broken fixtures, structural damage, general repairs",
}

class AuthResult(BaseModel):
    verified: bool
    customer_name: str
    reason: str

class RoutingDecision(BaseModel):
    department: str        # one of the five keys above
    routing_reason: str    # why this department — key for later analysis
    issue_summary: str
    confidence: str        # "low" / "medium" / "high"

# Create OpenAI client instance
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = os.getenv("LLM_MODEL", "gpt-4o-mini")

langfuse = get_client()

@observe(name="step-greeting")
def greet_and_collect_name() -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system", 
                "content": (
                    "You are a friendly receptionist at Stadtquartier Wohnservice, "
                    "a residential property management company. Greet the tenant warmly. "
                    "If they haven't given their name, ask for it."
                )
            },
        ]
    )
    return response.choices[0].message.content


@observe(name="step-auth")
def verify_tenant(name: str, address: str) -> AuthResult:
    # Manual sub-span inside the @observe function
    with langfuse.start_as_current_observation(name="address-format-check") as span:
        is_plausible = len(address.split()) >= 2
        span.update(metadata={"raw_address": address, "passed_format": is_plausible})

    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You simulate a tenant verification system. "
                    "If the address sounds plausible (street name + house number + city), "
                    "mark as verified. "
                    "If it is clearly fictional or incomplete, mark as not verified."
                )
            },
            {"role": "user", "content": f"Name: {name}\nAddress: {address}"}
        ],
        response_format=AuthResult
    )
    return response.choices[0].message.parsed


@observe(name="step-routing")
def route_to_department(transcript: str) -> RoutingDecision:
    dept_list = "\n".join(f"- {k}: {v}" for k, v in DEPARTMENTS.items())

    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a routing agent at Stadtquartier Wohnservice.\n"
                    f"Based on the conversation, route the tenant to the correct department.\n\n"
                    f"Available departments:\n{dept_list}\n\n"
                    f"Choose exactly one department key. Explain your reasoning in routing_reason."
                )
            },
            {"role": "user", "content": f"Conversation:\n{transcript}"}
        ],
        response_format=RoutingDecision
    )
    return response.choices[0].message.parsed


@observe(name="tenant-routing-call")
def handle_call():
    session_id = uuid.uuid4().hex[:8]
    with propagate_attributes(session_id=session_id):
        print("\n🏠 Stadtquartier Wohnservice — Tenant Support")
        print(f"Session: {session_id}")
        print("-" * 45)

        greeting = greet_and_collect_name()
        print(f"Agent: {greeting}")

        name = input("\nTenant (name): ").strip()
        address = input("Tenant (address): ").strip()

        auth = verify_tenant(name, address)
        if not auth.verified:
            print(f"Agent: I'm sorry, I couldn't verify your details. Please call us directly.")
            with propagate_attributes(tags=["auth-failed"]):
                return

        print(f"Agent: Thank you {auth.customer_name}. How can I help you today?")
        issue = input("Issue: ").strip()

        transcript = (
            f"Tenant name: {name}\n"
            f"Address: {address}\n"
            f"Issue: {issue}\n"
        )

        print(transcript)

        routing = route_to_department(transcript)

        # Attach routing result to the parent trace for dashboard filtering
        propagate_attributes(
            tags=[routing.department, f"confidence-{routing.confidence}"],
            metadata={
                "routing_department": routing.department,
                "routing_reason": routing.routing_reason,
                "confidence": routing.confidence,
                "customer_name": auth.customer_name,
            }
        )

        print(f"\n✅ Routing to: {DEPARTMENTS[routing.department]}")
        print(f"   Reason: {routing.routing_reason}")
        print(f"   Confidence: {routing.confidence}")
        print(f"   Issue summary: {routing.issue_summary}")

if __name__ == "__main__":
    handle_call()
    langfuse.flush()

# nicer_print(user_input, available_dishes, response.output_text)
