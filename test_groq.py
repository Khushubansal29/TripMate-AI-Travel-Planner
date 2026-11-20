
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERROR: Groq API key not found!")
else:
    try:
        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0
        )

        response = llm.invoke(
            "Say hello to TripMate in one sentence."
        )

        print("\nSUCCESS! GROQ CONNECTION WORKING")
        print(response.content)

    except Exception as e:
        print("\nCONNECTION FAILED:")
        print(e)
