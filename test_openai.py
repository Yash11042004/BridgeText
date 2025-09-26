# test_openai_chat.py
import os
from openai import OpenAI   # uses SDK style you used earlier
from dotenv import load_dotenv

load_dotenv() 
openai_api_key = os.getenv("OPENAI_API_KEY")
assert openai_api_key, "OPENAI_API_KEY is not set in the environment"

client = OpenAI(api_key=openai_api_key)

try:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",   # choose a model you have access to; gpt-4o-mini is an example
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'hello world' in one sentence."}
        ],
        max_tokens=64,
        temperature=0.2
    )
    # print raw for debugging
    print("RAW RESPONSE:", resp)
    # safe print of the assistant text
    print("\nASSISTANT:", resp.choices[0].message.content)
except Exception as e:
    print("ERROR:", type(e).__name__, e)
