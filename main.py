import os
from dotenv import load_dotenv
from langfuse.openai import OpenAI

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
