import os
import json
from openai import OpenAI
from langfuse import get_client
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
langfuse = get_client()
model = os.getenv("LLM_MODEL", "gpt-4o-mini")

VALID_DEPARTMENTS = {
    "rental-contracts",
    "terminations-moveout",
    "tenant-complaints",
    "energy-heating",
    "repairs-maintenance"
}

# Four conversations — two good, two with problems
TRANSCRIPTS = [
    {
        "id": "trace-001",
        "expected_department": "energy-heating",
        "actual_department": "energy-heating",
        "description": "Clear heating issue, warm agent",
        "turns": [
            {"role": "tenant", "text": "Hi, my heating has been broken for two days, it's freezing in here."},
            {"role": "agent",  "text": "Oh no, I'm so sorry to hear that — two days without heat is really uncomfortable. Can I take your name and address so I can pull up your file?"},
            {"role": "tenant", "text": "Lisa Müller, Berliner Str. 12, Berlin."},
            {"role": "agent",  "text": "Thank you Lisa, I've verified your details. I'm routing you to our Energy & Heating team right now — they handle exactly this and will prioritise your case. Is there anything else before I connect you?"},
        ]
    },
    {
        "id": "trace-002",
        "expected_department": "terminations-moveout",
        "actual_department": "terminations-moveout",
        "description": "Lease termination, direct and professional",
        "turns": [
            {"role": "tenant", "text": "I want to terminate my lease."},
            {"role": "agent",  "text": "Understood. Could I get your name and address to verify your tenancy?"},
            {"role": "tenant", "text": "Thomas Bauer, Hauptstr. 7, Hamburg."},
            {"role": "agent",  "text": "Thank you Mr Bauer. I'm connecting you to our Terminations & Move-out department. They'll walk you through the notice period and next steps."},
        ]
    },
    {
        "id": "trace-003",
        "expected_department": "tenant-complaints",
        "actual_department": "repairs-maintenance",  # WRONG routing
        "description": "Noise complaint, routed incorrectly",
        "turns": [
            {"role": "tenant", "text": "My upstairs neighbour is making noise constantly. I can't sleep."},
            {"role": "agent",  "text": "I understand. What's your name and address?"},
            {"role": "tenant", "text": "Emre Yilmaz, Gartenweg 3, Munich."},
            {"role": "agent",  "text": "Thank you. I'm sending you to Repairs & Maintenance — they handle apartment issues."},
        ]
    },
    {
        "id": "trace-004",
        "expected_department": "energy-heating",
        "actual_department": "energy-heating",
        "description": "Agent loops on the same question, then invents a department",
        "turns": [
            {"role": "tenant", "text": "My hot water isn't working."},
            {"role": "agent",  "text": "Can you tell me more about the issue?"},
            {"role": "tenant", "text": "The hot water stopped yesterday."},
            {"role": "agent",  "text": "Can you tell me more about the issue?"},  # loop
            {"role": "tenant", "text": "I already told you. Hot water. Not working."},
            {"role": "agent",  "text": "I see. I'll route you to our Plumbing & Utilities Emergency Response Centre."},  # hallucinated department
        ]
    },
]

def rule_based_eval(transcript: dict) -> dict:
    turns = transcript["turns"]
    turn_count = len(turns)

    # Routing match
    routing_correct = transcript["actual_department"] == transcript["expected_department"]

    # Loop detection: did the agent say the same thing twice?
    agent_messages = [t["text"] for t in turns if t["role"] == "agent"]
    has_loop = len(agent_messages) != len(set(agent_messages))

    return {
        "turn_count": turn_count,
        "routing_correct": routing_correct,
        "has_loop": has_loop,
    }

# Test it
for t in TRANSCRIPTS:
    result = rule_based_eval(t)
    print(f"{t['id']} | routing_correct={result['routing_correct']} | turns={result['turn_count']} | loop={result['has_loop']}")


FRIENDLINESS_RUBRIC = """
You are evaluating the friendliness and empathy of an AI customer support agent.

Score the AGENT's side of the conversation on a scale of 1 to 5:
  1 = Cold, dismissive, or impatient — makes the tenant feel like a burden
  2 = Curt — answers the question but no warmth
  3 = Professional but flat — polite, neutral, adequate
  4 = Warm — acknowledges the tenant's situation, sounds like a real person
  5 = Excellent — empathetic, makes the tenant feel heard and cared for

Respond in JSON: {"score": <1-5>, "reason": "<one sentence>"}
Only evaluate the agent, not the tenant.
"""

def judge_friendliness(transcript: dict) -> dict:
    conversation_text = "\n".join(
        f"{t['role'].upper()}: {t['text']}" for t in transcript["turns"]
    )

    print(conversation_text)
    response = client.chat.completions.create(
        model="gpt-5-mini",          # different model from the agent (gpt-4o-mini)
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": FRIENDLINESS_RUBRIC},
            {"role": "user", "content": f"Conversation:\n{conversation_text}"}
        ]
    )

    return json.loads(response.choices[0].message.content)

result = judge_friendliness(TRANSCRIPTS[0])
print(f"Friendliness: {result['score']}/5 — {result['reason']}")


HALLUCINATION_PROMPT = f"""
You are checking whether a customer support agent invented information that doesn't exist.

The ONLY valid department names are:
- rental-contracts
- terminations-moveout
- tenant-complaints
- energy-heating
- repairs-maintenance

Check if the agent mentioned any department name, policy, or procedure that is NOT on this list.

Respond in JSON: {{"hallucination_detected": true/false, "reason": "<one sentence>"}}
"""

def judge_hallucination(transcript: dict) -> dict:
    agent_text = " ".join(t["text"] for t in transcript["turns"] if t["role"] == "agent")

    response = client.chat.completions.create(
        model="gpt-5-mini",          # different model from the agent (gpt-4o-mini)
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": HALLUCINATION_PROMPT},
            {"role": "user", "content": f"Agent messages:\n{agent_text}"}
        ]
    )

    return json.loads(response.choices[0].message.content)

result = judge_hallucination(TRANSCRIPTS[3])  # the bad one
print(f"Hallucination: {result['hallucination_detected']} — {result['reason']}")


def evaluate_conversation(transcript: dict) -> dict:
    rules = rule_based_eval(transcript)
    friendliness = judge_friendliness(transcript)
    hallucination = judge_hallucination(transcript)

    return {
        "id": transcript["id"],
        "description": transcript["description"],
        "routing_correct": rules["routing_correct"],
        "turn_count": rules["turn_count"],
        "has_loop": rules["has_loop"],
        "friendliness": friendliness["score"],
        "friendliness_reason": friendliness["reason"],
        "hallucination": hallucination["hallucination_detected"],
        "hallucination_reason": hallucination["reason"],
    }


evaluation_results = []
for t in TRANSCRIPTS:
    evaluation_results.append(
        evaluate_conversation(t)
    )


print(f"\n{'ID':<12} {'Routing':>8} {'Turns':>6} {'Loop':>6} {'Friend':>7} {'Halluc':>7}")
print("-" * 55)

for r in evaluation_results:
    print(
        f"{r['id']:<12} "
        f"{'✅' if r['routing_correct'] else '❌':>7} "
        f"{r['turn_count']:>6} "
        f"{'❌' if r['has_loop'] else '✅':>5} "
        f"{r['friendliness']:>7} "
        f"{'❌' if r['hallucination'] else '✅':>6}"
    )
