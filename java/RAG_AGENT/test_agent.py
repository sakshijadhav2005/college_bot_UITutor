import sys
import io
sys.path.append('.')

# Set up encoding to prevent Windows console encoding crashes
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from My_RAG.agent import run_study_agent

def main():
    print("==================================================")
    print("🤖 TESTING LANGCHAIN GEMINI STUDY TUTOR AGENT")
    print("==================================================")
    
    # Test query 1 (Conversational start)
    query_1 = "Hello! Who are you and what can you help me with?"
    print(f"\nUser: {query_1}")
    print("Assistant: ", end="", flush=True)
    try:
        response_1 = run_study_agent(query_1, thread_id="test-thread-cli")
        print(response_1)
    except Exception as e:
        print(f"\n❌ Error running agent: {e}")
        print("Please check that GOOGLE_API_KEY is correctly set in your .env file.")
        return

    # Test query 2 (Exam Quick Action test)
    query_2 = "Can you search the document and generate 3 quick MCQs for me?"
    print(f"\nUser: {query_2}")
    print("Assistant: ", end="", flush=True)
    try:
        response_2 = run_study_agent(query_2, thread_id="test-thread-cli")
        print(response_2)
    except Exception as e:
        print(f"\n❌ Error running agent: {e}")
        
    print("\n==================================================")
    print("🎉 AGENT CLI TEST COMPLETED!")
    print("==================================================")

if __name__ == "__main__":
    main()
