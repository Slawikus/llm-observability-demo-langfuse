import os
from dotenv import load_dotenv
from langfuse.openai import OpenAI
from pydantic import BaseModel

# Load variables from .env file
load_dotenv()

# Create OpenAI client instance
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
model = os.getenv("LLM_MODEL", "gpt-4o-mini")

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


TEST_SCENARIOS = [
    {
        "id": "TC-001",
        "description": "Clear heating issue, cooperative tenant",
        "persona": (
            "You are Lisa Müller, a tenant at Berliner Str. 12, Berlin. "
            "Your heating stopped working 2 days ago. You are polite but worried because it's cold. "
            "Give your name and address when asked. Keep answers short and factual."
        ),
        "expected_department": "energy-heating",
        "max_turns": 5,
        "min_friendliness": 3,
    },
    {
        "id": "TC-002",
        "description": "Lease termination, calm and direct",
        "persona": (
            "You are Thomas Bauer, a tenant at Hauptstr. 7, Hamburg. "
            "You want to end your lease. You are calm and businesslike. "
            "Give your name and address when asked."
        ),
        "expected_department": "terminations-moveout",
        "max_turns": 5,
        "min_friendliness": 2,
    },
    {
        "id": "TC-003",
        "description": "Noise complaint, frustrated tone",
        "persona": (
            "You are Emre Yilmaz, a tenant at Gartenweg 3, Munich. "
            "Your upstairs neighbour makes noise every night and you haven't slept in a week. "
            "You are frustrated and slightly impatient. Give your details when asked."
        ),
        "expected_department": "tenant-complaints",
        "max_turns": 6,
        "min_friendliness": 3,
    },
    {
        "id": "TC-004",
        "description": "Auth failure — wrong address",
        "persona": (
            "You are Sandra Koch. You want to report a broken window. "
            "When asked for your address, say 'Musterstr. 99, Berlin' — this is not a real tenant address. "
            "Do not correct it even if the agent says it can't be verified."
        ),
        "expected_department": "auth-failed",  # special value: we expect the call to end here
        "max_turns": 3,
        "min_friendliness": 3,
    },
    {
        "id": "TC-005",
        "description": "Ambiguous issue — agent must ask, not guess",
        "persona": (
            "You are Klaus Werner, a tenant at Friedrichstr. 45, Berlin. "
            "Your opening message is just: 'I have a problem with my apartment.' "
            "Only give more details if the agent asks a direct question. Be cooperative."
        ),
        "expected_department": None,  # any correct department is acceptable — we test that the agent asks first
        "max_turns": 6,
        "min_friendliness": 3,
    },
    {
        "id": "TC-006",
        "description": "Tricky phrasing — sounds like terminations but is energy",
        "persona": (
            "You are Yuki Tanaka, a tenant at Rosenweg 8, Frankfurt. "
            "You want to cancel your energy supply contract, not your apartment lease. "
            "Say: 'I want to cancel my energy contract.' Give your details when asked."
        ),
        "expected_department": "energy-heating",
        "max_turns": 5,
        "min_friendliness": 2,
    },
]


ROUTING_SYSTEM_PROMPT = (
    "You are a routing agent at Stadtquartier Wohnservice, "
    "a residential property management company.\n"
    "Based on the conversation transcript, "
    "decide which department should handle this tenant's issue.\n\n"
    "Available departments:\n"
    + "\n".join(f"- {k}: {v}" for k, v in DEPARTMENTS.items())
    + "\n\nChoose exactly one department key from the list above. "
    "In routing_reason, explain specifically why this department is the right fit."
)


class RoutingDecision(BaseModel):
    department: str
    routing_reason: str
    issue_summary: str
    confidence: str  # "low" / "medium" / "high"


def route_conversation(transcript: list[dict]) -> RoutingDecision | None:
    """Takes a list of {"role": "tenant"/"agent", "text": "..."} dicts,
    returns routing decision."""

    transcript_text = "\n".join(
        f"{t['role'].upper()}: {t['text']}" for t in transcript
    )

    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": ROUTING_SYSTEM_PROMPT},
                {"role": "user", "content": f"Conversation:\n{transcript_text}"}
            ],
            response_format=RoutingDecision
        )
        return response.choices[0].message.parsed
    except Exception:
        return None

class AuthResult(BaseModel):
    verified: bool
    customer_name: str
    reason: str


def check_auth(name: str, address: str) -> AuthResult:
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You simulate a tenant database check. "
                    "If the address has a recognisable street name and number, "
                    "mark verified. "
                    "Clearly fictional or incomplete addresses "
                    "(like 'Musterstr. 99') are not verified."
                )
            },
            {"role": "user", "content": f"Name: {name}\nAddress: {address}"}
        ],
        response_format=AuthResult
    )
    return response.choices[0].message.parsed


def test_agent_message(persona: str, conversation_history: list[dict]) -> str:
    """Given a persona and the conversation so far, generate the next customer message."""

    messages = [
        {
            "role": "system",
            "content": (
                f"{persona}\n\n"
                "You are participating in a customer support call. "
                "Respond naturally as this person would. "
                "Keep each response short — 1–3 sentences maximum. "
                "Do not break character."
            )
        }
    ]

    # Add conversation history:
    # agent messages become 'user' from the test agent's perspective
    for turn in conversation_history:
        role = "assistant" if turn["role"] == "tenant" else "user"
        messages.append({"role": role, "content": turn["text"]})

    messages.append({"role": "user", "content": "What do you say next?"})

    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content.strip()
