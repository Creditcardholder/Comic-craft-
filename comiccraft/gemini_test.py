import os
from google import genai

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("API key missing")
    exit()

client = genai.Client(api_key=api_key)

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input="Give me one short idea for a comic story."
)

print(interaction.output_text)
