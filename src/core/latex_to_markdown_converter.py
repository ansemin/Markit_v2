import os
import logging
from typing import Optional
from google import genai

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Load API key from environment variable
api_key = os.getenv("GOOGLE_API_KEY")

# Check if API key is available
if not api_key:
    logger.warning("GOOGLE_API_KEY environment variable not found. LaTeX to Markdown conversion may not work.")

def convert_latex_to_markdown(latex_content: str) -> Optional[str]:
    """
    Convert LaTeX content to Markdown using Gemini API.
    
    Args:
        latex_content: The LaTeX content to convert
        
    Returns:
        Converted markdown content or None if conversion fails
    """
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable not set")
        return None
        
    try:
        # Create a client
        client = genai.Client(api_key=api_key)
        
        # Set up the prompt
        prompt = """
        Convert this LaTeX content to clean, well-formatted Markdown.
        Preserve all tables, lists, and formatting.
        For tables, use standard Markdown table syntax.
        For mathematical expressions, use $ for inline and $$ for display math.
        Keep the structure and hierarchy of the content. Return only the markdown content, no other text.
        """
        
        # Generate the response
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                prompt,
                latex_content
            ],
            config={
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
        )
        
        # Extract the markdown text from the response
        markdown_text = response.text
        
        logger.info("Successfully converted LaTeX to Markdown")
        return markdown_text
        
    except Exception as e:
        logger.error(f"Error converting LaTeX to Markdown: {str(e)}")
        return None 