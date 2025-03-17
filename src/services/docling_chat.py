import openai
import os

# Load API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Check if API key is available and print a message if not
if not openai.api_key:
    print("Warning: OPENAI_API_KEY environment variable not found. Chat functionality may not work.")

def chat_with_document(message, history, document_text_state):
    history = history or []
    history.append({"role": "user", "content": message})

    context = f"Document: {document_text_state}\n\nUser: {message}"

    # Add error handling for API calls
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[{"role": "system", "content": context}] + history
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"Error: Could not generate response. Please check your OpenAI API key. Details: {str(e)}"
        print(f"OpenAI API error: {str(e)}")
    
    history.append({"role": "assistant", "content": reply})
    return history, history
