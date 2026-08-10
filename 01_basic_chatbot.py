import os
from dotenv import load_dotenv
from groq import Groq

# 1. Load environment variables from .env file
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key :
    raise ValueError(
        "Please set a valid GROQ_API_KEY in your .env file."
    )

# 2. Initialize the Groq client
# Under the hood, this sets up an HTTP client pointing to Groq's API base URL.
client = Groq(api_key=api_key)

# 3. Define our prompt and message payload
# Messages array represents the full conversation turn history sent to the API.
messages = [
    {
        "role": "system",
        "content": "You are a concise AI Backend Engineering mentor."
    },
    {
        "role": "user",
        "content": "Explain what a stateless API means in 2 bullet points."
    }
]

print("--- SENDING REQUEST TO GROQ API ---")
print(f"Model Target: llama-3.3-70b-versatile")
print(f"Messages Payload:\n{messages}\n")

# 4. Make the synchronous API request
# Groq exposes an OpenAI-compatible Chat Completions API schema.
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
    temperature=0.0,      # Controls randomness (0.0 = deterministic, 1.0 = creative)
    max_tokens=350        # Upper bound on completion tokens generated
)

# 5. Extract and inspect the response payload
assistant_message = response.choices[0].message.content
usage = response.usage

print("--- GROQ RESPONSE RECEIVED ---")
print(f"Assistant Answer:\n{assistant_message}\n")

print("--- METADATA & TOKEN USAGE ---")
print(f"Prompt Tokens (Input):     {usage.prompt_tokens}")
print(f"Completion Tokens (Output): {usage.completion_tokens}")
print(f"Total Tokens:               {usage.total_tokens}")
print(f"Finish Reason:             {response.choices[0].finish_reason}")
