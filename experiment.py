import os
from dotenv import load_dotenv
load_dotenv()

print("TRACING:", os.getenv("LANGCHAIN_TRACING_V2"))
print("API KEY:", os.getenv("LANGSMITH_API_KEY", "NOT FOUND")[:20])
print("PROJECT:", os.getenv("LANGSMITH_PROJECT"))