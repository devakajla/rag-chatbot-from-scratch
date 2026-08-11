import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def run_chat_session():
    """
    Demonstrates how a backend application maintains conversation history
    for a stateless LLM API.
    """
    # History array lives in our backend memory (stateful wrapper around stateless API)
    conversation_history = [
        {"role": "system", "content": "You are a helpful Python mentor."}
    ]

    print("--- INTERACTIVE GROQ CHATBOT (Type 'exit' or 'quit' to stop) ---")
    print("Notice how we append every user prompt and assistant response to conversation_history.\n")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Ending chat session.")
            break

        # 1. Append user message to memory
        conversation_history.append({"role": "user", "content": user_input})

        # 2. Send ENTIRE history array to the stateless API
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=conversation_history,
            temperature=0.7,
            max_tokens=300
        )

        assistant_reply = response.choices[0].message.content
        print(f"\nAssistant: {assistant_reply}\n")

        # 3. Append assistant response to memory so the next turn remembers it
        conversation_history.append({"role": "assistant", "content": assistant_reply})

        # 4. Print token consumption for transparency
        usage = response.usage
        print(f"[Tokens used in this request turn: Prompt={usage.prompt_tokens}, Completion={usage.completion_tokens}, Total={usage.total_tokens}]")
        print("-" * 60)

if __name__ == "__main__":
    run_chat_session()
