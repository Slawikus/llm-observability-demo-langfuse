import os
import json
import re
from typing import Literal
from dotenv import load_dotenv
from langfuse.openai import OpenAI
from pydantic import BaseModel
from rich import print

# Load variables from .env file
load_dotenv()

# Create OpenAI client instance
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
model = os.getenv("LLM_MODEL", "gpt-4o-mini")
# The LLM-as-Judge model is configured separately: you generally want to hold
# the judge fixed while swapping the model under test.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", model)

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
    confidence: Literal["low", "medium", "high"]


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

# give routing a shot
routing_decision = route_conversation(
    transcript=[
        {"role": "tenant", "text": "I want to terminate my lease."}
    ]
)
print(routing_decision)


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


AGENT_GREETING = (
    "Thank you for calling Stadtquartier Wohnservice. "
    "My name is Alex, how can I help you today?"
)

# Placeholder for the real step-auth check: has the tenant given us something
# address-shaped (German street suffix followed by a house number)? Routing is
# gated on this, so the agent can't hand a tenant off before collecting an
# address the auth step can actually reject.
ADDRESS_PATTERN = re.compile(
    r"(?:str\.|(?:straße|strasse|weg|platz|allee|gasse|ring)\b)\s*\d+",
    re.IGNORECASE,
)


def run_test_conversation(scenario: dict) -> dict:
    """Runs one full test conversation. Returns transcript + routing result.

    Every turn carries a "scripted" flag: True for hardcoded agent lines
    (the greeting, the auth rejection), False for anything a model wrote.
    The friendliness judge only scores unscripted turns — otherwise it is
    grading string literals rather than agent behaviour.
    """
    conversation = []
    routing_result = None
    auth_failed = False
    clarifying_turns = 0

    # Agent opens
    conversation.append(
        {"role": "agent", "text": AGENT_GREETING, "scripted": True}
    )

    # One turn of headroom past max_turns, so blowing the budget is
    # observable in the results instead of being clipped by the loop bound.
    for turn_num in range(scenario["max_turns"] + 2):
        # Test agent responds as customer
        customer_msg = test_agent_message(scenario["persona"], conversation)
        conversation.append(
            {"role": "tenant", "text": customer_msg, "scripted": False}
        )

        # Simple auth extraction (look for name + address pattern)
        # In a real system this would be the step-auth span from session 2
        tenant_messages = " ".join(
            t["text"] for t in conversation if t["role"] == "tenant"
        )
        if "Musterstr" in tenant_messages:
            auth_failed = True
            conversation.append({
                "role": "agent",
                "text": (
                    "I'm sorry — I couldn't match that address to a tenant record, "
                    "so I'm not able to go further on this call. If you check the "
                    "address on your lease and call us back, we'll pick it up right away."
                ),
                "scripted": True,
            })
            break

        # Auth comes before routing: don't hand the tenant off until they've
        # actually given an address, so the auth check above gets a chance to
        # reject it. After that, confidence decides when to route rather than a
        # fixed conversation length — which is what makes max_turns a real budget.
        if ADDRESS_PATTERN.search(tenant_messages):
            attempt = route_conversation(conversation)

            if attempt and attempt.confidence in ("medium", "high"):
                routing_result = attempt
                agent_reply = (
                    f"Thank you. I'm routing you to our "
                    f"{DEPARTMENTS.get(routing_result.department, routing_result.department)} "
                    f"team. "
                    f"{routing_result.routing_reason}"
                )
                conversation.append(
                    {"role": "agent", "text": agent_reply, "scripted": False}
                )
                break

        # Router isn't confident enough yet: the agent asks a clarifying
        # question and we go round again.
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a warm, empathetic receptionist. "
                        "Briefly acknowledge what the tenant has told you. "
                        "Before anything else you must collect their full name AND "
                        "their street address, including street name and house "
                        "number — ask for whichever is still missing, and do not "
                        "accept an apartment number in place of a street address. "
                        "Only once you have both may you ask about the issue itself. "
                        "Two sentences maximum."
                    )
                },
                {
                    "role": "user",
                    "content": "\n".join(
                        f"{t['role']}: {t['text']}" for t in conversation
                    )
                }
            ]
        )
        agent_reply = response.choices[0].message.content.strip()
        conversation.append(
            {"role": "agent", "text": agent_reply, "scripted": False}
        )
        clarifying_turns += 1

    return {
        "scenario_id": scenario["id"],
        "transcript": conversation,
        "routing": routing_result,
        "auth_failed": auth_failed,
        "clarifying_turns": clarifying_turns,
    }

# give test conversation a shot:
test_conversation_result = run_test_conversation(TEST_SCENARIOS[0])
print(test_conversation_result)


FRIENDLINESS_RUBRIC = """
Score the AGENT's tone on a scale of 1 to 5:
  1 = cold or dismissive
  3 = professional but flat
  5 = warm and empathetic
Return JSON: {"score": <1-5>, "reason": "<one sentence>"}
"""


def evaluate_result(scenario: dict, result: dict) -> dict:
    transcript = result["transcript"]
    routing = result["routing"]
    auth_failed = result["auth_failed"]
    clarifying_turns = result["clarifying_turns"]

    # An auth failure is not a routing decision, so it doesn't belong in the
    # routing-accuracy denominator.
    is_routing_test = scenario["expected_department"] != "auth-failed"

    # Routing match
    if scenario["expected_department"] == "auth-failed":
        routing_correct = auth_failed
    elif scenario["expected_department"] is None:
        # TC-005: the department is genuinely open, but the agent must not guess
        # from an opening as vague as "I have a problem with my apartment" —
        # it has to ask at least one clarifying question and still land a
        # confident routing decision.
        routing_correct = routing is not None and clarifying_turns >= 1
    else:
        routing_correct = (
                routing is not None and routing.department == scenario["expected_department"]
        )

    # Turn count
    turn_count = len([t for t in transcript if t["role"] == "tenant"])
    within_budget = turn_count <= scenario["max_turns"]

    # Friendliness (LLM-as-Judge). Scripted agent lines are dropped: judging
    # them scores a string literal, which no change to the agent can improve.
    convo_text = "\n".join(
        f"{t['role'].upper()}: {t['text']}"
        for t in transcript
        if not (t["role"] == "agent" and t["scripted"])
    )
    judge_response = client.chat.completions.create(
        model=JUDGE_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": FRIENDLINESS_RUBRIC},
            {"role": "user", "content": convo_text}
        ]
    )
    friendliness_result = json.loads(judge_response.choices[0].message.content)
    friendliness_score = friendliness_result["score"]

    passed = (routing_correct
              and within_budget
              and friendliness_score >= scenario["min_friendliness"])

    return {
        "id": scenario["id"],
        "description": scenario["description"],
        "expected": scenario["expected_department"] or "any",
        "actual": routing.department if routing else ("auth-failed" if auth_failed else "no-routing"),
        "routing_correct": routing_correct,
        "is_routing_test": is_routing_test,
        "turns": turn_count,
        "clarifying_turns": clarifying_turns,
        "within_budget": within_budget,
        "friendliness": friendliness_score,
        "routing_reason": routing.routing_reason if routing else "—",
        "passed": passed,
    }


def run_suite(scenarios: list) -> list:
    results = []

    for scenario in scenarios:
        print(f"Running {scenario['id']}: {scenario['description']}...")
        conversation_result = run_test_conversation(scenario)
        eval_result = evaluate_result(scenario, conversation_result)
        results.append(eval_result)
        status = "✅ PASS" if eval_result["passed"] else "❌ FAIL"
        print(f"  {status} — routing: {eval_result['actual']} | turns: {eval_result['turns']} | friendliness: {eval_result['friendliness']}/5")

    return results


def print_report(results: list):
    print("\n" + "=" * 82)
    print(
        f"{'ID':<8} {'Expected':<22} {'Actual':<22} "
        f"{'Turns':>5} {'Asked':>6} {'Friend':>7} {'Result':>6}"
    )
    print("-" * 82)

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"{r['id']:<8} {r['expected']:<22} {r['actual']:<22} "
            f"{r['turns']:>5} {r['clarifying_turns']:>6} "
            f"{r['friendliness']:>7} {status:>6}"
        )

    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    # Routing accuracy counts only scenarios that actually exercise a routing
    # decision — auth-failure scenarios end the call before routing happens.
    routing_tests = [r for r in results if r["is_routing_test"]]
    routing_hits = sum(1 for r in routing_tests if r["routing_correct"])
    routing_acc = routing_hits / len(routing_tests) if routing_tests else 0.0

    avg_turns = sum(r["turns"] for r in results) / total
    avg_friendliness = sum(r["friendliness"] for r in results) / total

    print("=" * 82)
    print(
        f"SUMMARY: {passed}/{total} passed | "
        f"routing accuracy: {routing_acc:.0%} ({routing_hits}/{len(routing_tests)}) | "
        f"avg turns: {avg_turns:.1f} | avg friendliness: {avg_friendliness:.1f}/5"
    )

if __name__ == "__main__":
    results = run_suite(TEST_SCENARIOS)
    print_report(results)
