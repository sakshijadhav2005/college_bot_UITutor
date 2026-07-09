import os
from typing import List, Dict, Any
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
try:
    from My_RAG.config import PINECONE_INDEX_NAME
    from My_RAG.embeddings import EmbeddingManager
except ImportError:
    from config import PINECONE_INDEX_NAME
    from embeddings import EmbeddingManager
from langchain_pinecone import PineconeVectorStore


# Initialize checkpointer for agent memory
checkpointer = InMemorySaver()

@tool
def search_documents(query: str) -> str:
    """Search the uploaded document vector store to retrieve relevant contexts for the query.
    Use this tool whenever the user asks questions, requests a quiz, MCQs, or key points about the document.
    """
    try:
        # Load embedding model
        emb_manager = EmbeddingManager()
        embeddings = emb_manager.get_embeddings()
        
        # Connect to Pinecone
        vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
        
        # Perform similarity search
        results = vectorstore.similarity_search(query, k=5)
        
        if not results:
            return "No matching content found in the documents."
            
        # Format results
        formatted_results = []
        for doc in results:
            source = doc.metadata.get("source_file", "Unknown Source")
            formatted_results.append(f"[Source: {source}]:\n{doc.page_content}")
            
        return "\n\n---\n\n".join(formatted_results)
    except Exception as e:
        return f"Error retrieving document content: {str(e)}"

# System prompt for our study coach agent
SYSTEM_PROMPT = """You are a brilliant, supportive, and highly structured AI study tutor and exam coach specifically designed for college students. 
Your goal is to help college students study, digest, and prepare for their college exams using ONLY the content of the uploaded documents.

## Strict Grounding Rules:
1. ALWAYS use the `search_documents` tool to query the Pinecone vector database for document content before answering ANY question, generating quizzes, MCQs, or key points.
2. You must ground all explanations, definitions, and questions strictly in the retrieved document context. 
3. If the user asks a question that cannot be answered or verified by the document content retrieved from Pinecone, you must state: "I could not find the answer to this in the uploaded document. Here is what I do know from the document: [provide any related info]..." 
4. Do NOT make up external facts or use external training knowledge to answer academic questions unless it directly supports the concepts retrieved from the document.
5. CRITICAL DIRECTIVE: If the user requests a quiz, MCQs, or exam key points/summaries, you MUST search the document immediately using `search_documents` (with query terms like 'key concepts', 'main topics', or key subject words). Do not ask the user for permission or topic choice first. Retrieve data first, then generate the quiz or questions from the retrieved context.

## College Student Pedagogy:
- Use clear, academic, yet encouraging language suitable for undergraduate and graduate-level courses.
- Highlight key definitions, formulas, methodologies, and theories.
- Guide students step-by-step through complex concepts.

## Formatting Guidelines
1. Always respond in a clear, well-structured, and standard format.
2. Use markdown headings, bullet points, and bold text to make your response easy to read.
3. For quizzes, MCQs, and cheat sheets, follow these structures:
   - **Quiz:** Provide clear fill-in-the-blank or short-answer questions. Hide or separate the answers (e.g., in a spoiler or at the bottom) so the student can test themselves.
   - **MCQs:** Create high-quality multiple-choice questions with 4 options (A, B, C, D) and specify the correct answer at the very end.
   - **Key Points:** Provide concise, bulleted explanations of crucial definitions, equations, theories, or concepts.
"""


def get_study_agent():
    """Initialize and return the study agent using init_chat_model and google-genai."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key or google_api_key == "your_gemini_api_key_here":
        raise ValueError("GOOGLE_API_KEY is not set. Please set it in your .env file.")
        
    model = init_chat_model(
        "gemini-2.5-flash",
        model_provider="google-genai",
        temperature=0.5,
    )
    
    agent = create_agent(
        model=model,
        tools=[search_documents],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    
    return agent

def run_study_agent(prompt: str, thread_id: str) -> str:
    """Run the study agent with a given prompt and thread_id for conversation history."""
    agent = get_study_agent()
    
    # Run the agent
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    
    # Extract response content
    messages = result.get("messages", [])
    if messages:
        last_message = messages[-1]
        
        # Get content
        content = ""
        if hasattr(last_message, "content"):
            content = last_message.content
        elif isinstance(last_message, dict) and "content" in last_message:
            content = last_message["content"]
            
        # Parse content if it is a list of dictionary blocks
        if isinstance(content, list):
            text_blocks = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_blocks.append(block.get("text", ""))
                    elif "text" in block:
                        text_blocks.append(block["text"])
                elif isinstance(block, str):
                    text_blocks.append(block)
            return "\n".join(text_blocks)
        elif isinstance(content, str):
            return content
            
    return "No response could be generated."

