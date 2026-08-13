import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Sample context: Key provisions of India's Digital Personal Data Protection (DPDP) Act 2023
DPDP_ACT_DOC = """
Digital Personal Data Protection (DPDP) Act 2023 Key Provisions:
1. Data Fiduciary Obligations: A Data Fiduciary must implement reasonable security safeguards to prevent personal data breaches and notify the Data Protection Board and affected Data Principals in the event of a breach.
2. Rights of Data Principal: Data Principals (individuals) have the right to access information about their processed data, seek correction or erasure, and nominate another individual to exercise their rights in case of death or incapacity.
3. Penalties for Breach: Section 33 mandates penalties up to ₹250 Crore for failure to implement reasonable security safeguards to prevent data breaches, and up to ₹200 Crore for failure to notify the Board and affected individuals.
4. Processing Children's Data: Processing personal data of a child requires verifiable consent of a parent/lawful guardian. Data Fiduciaries are strictly prohibited from behavioral tracking or targeted advertising directed at children.
"""

def query_with_manual_context(question: str, context: str | None = None) -> str:
    """
    Builds a prompt injecting manual context (if provided) and enforces grounding rules.
    """
    system_prompt = (
        "You are an expert Legal AI Assistant specializing in Indian Data Privacy law.\n"
        "STRICT GROUNDING RULE: Answer the user's question ONLY using the provided context.\n"
        "If the answer cannot be directly derived from the context, state: 'I cannot answer this based on the provided document.'"
    )

    if context:
        user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
    else:
        user_prompt = f"QUESTION:\n{question}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.0, # Greedy zero-entropy sampling for strict grounding
        max_tokens=250
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    question = "What is the maximum financial penalty under Section 33 of the DPDP Act for failing to prevent a personal data breach?"

    print("==================================================")
    print("CASE 1: Question WITH Relevant DPDP Act Context")
    print("==================================================")
    answer_1 = query_with_manual_context(question, context=DPDP_ACT_DOC)
    print(f"Question: {question}")
    print(f"Answer:   {answer_1}\n")

    print("==================================================")
    print("CASE 2: Question WITHOUT Context (Model relying on general training memory)")
    print("==================================================")
    answer_2 = query_with_manual_context(question, context=None)
    print(f"Question: {question}")
    print(f"Answer:   {answer_2}\n")

    print("==================================================")
    print("CASE 3: Question WITH Irrelevant Context")
    print("==================================================")
    irrelevant_context = "The Information Technology (IT) Act of 2000 primarily addresses cybercrime and electronic commerce in India."
    answer_3 = query_with_manual_context(question, context=irrelevant_context)
    print(f"Question: {question}")
    print(f"Answer:   {answer_3}\n")

