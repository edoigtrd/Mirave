from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

state = {
    "message": "I was charged twice and need the duplicate refunded today.",
    "account_tier": "business",
}

# Pointed at the local mock server (inference/server.py) instead of the real
# api.typesafe.ai — same MIRAVE_API_KEY the server was started with.
with TypeSafeClient(base_url="http://127.0.0.1:8000", api_key="dev-key") as client:
    response = client.system_one(
        state=state,
        questions={
            "intent": Choice(
                instructions="What is the customer's main request?",
                criteria={
                    "refund": "The customer wants money returned.",
                    "technical_help": "The customer needs a bug or integration fixed.",
                    "other": "None of the options clearly fits.",
                },
            ),
            "is_urgent": Noul(
                instructions="Does `message` explicitly communicate time pressure?"
            ),
            "frustration": Score(
                instructions="How frustrated does the customer appear?",
                criteria=["Calm and neutral", "Concerned but civil", "Very angry"],
            ),
        },
    )

print("intent:", response.choices["intent"].choice, response.choices["intent"].confidence)
print("is_urgent:", response.nouls["is_urgent"].noul)
print("frustration:", response.scores["frustration"].score, response.scores["frustration"].confidence)