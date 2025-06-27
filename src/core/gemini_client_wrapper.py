"""
Gemini client wrapper that mimics OpenAI client interface for MarkItDown compatibility.
This allows us to use Gemini Flash 2.5 for image processing in MarkItDown.
"""

import logging
import base64
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from google import genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

from src.core.config import config
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class GeminiChatCompletions:
    """Chat completions interface that mimics OpenAI's chat.completions API."""
    
    def __init__(self, client):
        self.client = client
    
    def create(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> 'GeminiResponse':
        """Create a chat completion that mimics OpenAI's API."""
        if not messages:
            raise ValueError("Messages cannot be empty")
        
        # Extract the user message (MarkItDown sends a single user message with text + image)
        user_message = None
        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg
                break
        
        if not user_message:
            raise ValueError("No user message found")
        
        content = user_message.get("content", [])
        if not isinstance(content, list):
            content = [{"type": "text", "text": str(content)}]
        
        # Extract text prompt and image
        text_prompt = ""
        image_data = None
        
        for item in content:
            if item.get("type") == "text":
                text_prompt = item.get("text", "")
            elif item.get("type") == "image_url":
                image_url = item.get("image_url", {}).get("url", "")
                if image_url.startswith("data:image/"):
                    # Extract base64 data from data URI
                    try:
                        header, data = image_url.split(",", 1)
                        image_data = base64.b64decode(data)
                    except Exception as e:
                        logger.error(f"Failed to decode image data: {e}")
                        raise ValueError("Invalid image data URI")
        
        if not text_prompt:
            text_prompt = "Describe this image in detail."
        
        if not image_data:
            raise ValueError("No image data found in request")
        
        try:
            # Use Gemini to process the image
            response = self.client.models.generate_content(
                model=config.model.gemini_model,
                contents=[
                    {
                        "parts": [
                            {"text": text_prompt},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",  # Assume JPEG for now
                                    "data": base64.b64encode(image_data).decode()
                                }
                            }
                        ]
                    }
                ],
                config={
                    "temperature": config.model.temperature,
                    "max_output_tokens": 1024,  # Reasonable limit for image descriptions
                }
            )
            
            # Extract text from Gemini response
            response_text = ""
            if hasattr(response, "text") and response.text:
                response_text = response.text
            elif hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and candidate.content:
                    if hasattr(candidate.content, "parts") and candidate.content.parts:
                        response_text = candidate.content.parts[0].text
            
            if not response_text:
                logger.warning("Empty response from Gemini, using fallback")
                response_text = "Image processing completed but no description generated."
            
            return GeminiResponse(response_text)
            
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            # Return a fallback response to avoid breaking MarkItDown
            return GeminiResponse(f"Image description unavailable due to processing error: {str(e)}")


class GeminiChoice:
    """Mimics OpenAI's Choice object."""
    
    def __init__(self, content: str):
        self.message = GeminiMessage(content)


class GeminiMessage:
    """Mimics OpenAI's Message object."""
    
    def __init__(self, content: str):
        self.content = content


class GeminiResponse:
    """Mimics OpenAI's ChatCompletion response."""
    
    def __init__(self, content: str):
        self.choices = [GeminiChoice(content)]


class GeminiClientWrapper:
    """
    Gemini client wrapper that mimics OpenAI client interface for MarkItDown.
    
    This allows MarkItDown to use Gemini for image processing while thinking
    it's using an OpenAI client.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        if not HAS_GEMINI:
            raise ImportError("google-genai package is required for Gemini support")
        
        api_key = api_key or config.api.google_api_key
        if not api_key:
            raise ValueError("Google API key is required for Gemini client")
        
        self.client = genai.Client(api_key=api_key)
        self.chat = GeminiChatCompletions(self.client)
        
        logger.info("Gemini client wrapper initialized for MarkItDown compatibility")
    
    @property
    def completions(self):
        """Alias for chat to match some OpenAI client patterns."""
        return self.chat


def create_gemini_client_for_markitdown() -> Optional[GeminiClientWrapper]:
    """
    Create a Gemini client wrapper for use with MarkItDown.
    
    Returns:
        GeminiClientWrapper if Gemini is available and configured, None otherwise.
    """
    if not HAS_GEMINI:
        logger.warning("Gemini not available for MarkItDown image processing")
        return None
    
    if not config.api.google_api_key:
        logger.warning("No Google API key found for MarkItDown image processing")
        return None
    
    try:
        return GeminiClientWrapper()
    except Exception as e:
        logger.error(f"Failed to create Gemini client for MarkItDown: {e}")
        return None


# For testing purposes
if __name__ == "__main__":
    # Test the wrapper
    try:
        client = create_gemini_client_for_markitdown()
        if client:
            print("✅ Gemini client wrapper created successfully")
            print("✅ Ready for MarkItDown integration")
        else:
            print("❌ Failed to create Gemini client wrapper")
    except Exception as e:
        print(f"❌ Error: {e}")