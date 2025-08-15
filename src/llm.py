"""LLM module for generating final answers based on retrieved documents"""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")


def final_answer(retrieved_docs, user_query):
    """Generate a final answer using the retrieved documents and user query"""
    llm_model = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.3
    )

    prompt = f"""
  You are a highly reliable, context‑driven assistant. Your goal is to answer user questions **only** using the information in the retrieved context.  

  1. ROLE  
    • Use the provided CONTEXT to craft your answer.  
    • Never introduce external information, assumptions, or hallucinations.  

  2. PROCESS  
    • **Step 1:** Check if the CONTEXT contains explicit information answering the question.  
    • **Step 2:** If it does, extract the relevant passages, and present them.  
    • **Step 3:** If the CONTEXT is incomplete or ambiguous, ask a clarifying question or say:  
      “I’m sorry, but the context doesn’t give enough information to answer that. Could you clarify or provide more details?”  

  3. STYLE & FORMAT  
    • Use a one‑line heading summarizing your answer.  
    • Use bullet points (one per line) for each distinct fact or step.  
    • Keep each bullet concise (1–2 sentences).  

  4. FORMAT
    If context is sufficient, provide a clear, structured, or conversational answer, optionally referencing document titles or sections if helpful. Use bullets, short paragraphs, or quotes as needed — whichever feels most appropriate.

    If context is lacking, politely explain that there’s not enough information to answer confidently. Encourage clarification or suggest possible directions.

    Avoid overly formal tone unless required. Keep it natural, helpful, and direct.

  ---  
  **CONTEXT:**  
  {retrieved_docs}

  **USER QUESTION:**  
  {user_query}

  """

    llm_response = llm_model.invoke(prompt)
    return llm_response.content
