from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import os
import base64
import tempfile
import json
from PIL import Image
import io

from src.parsers.parser_interface import DocumentParser
from src.parsers.parser_registry import ParserRegistry

# Import the Mistral AI client
try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False

# Load API key from environment variable
api_key = os.getenv("MISTRAL_API_KEY")

# Check if API key is available and print a message if not
if not api_key:
    print("Warning: MISTRAL_API_KEY environment variable not found. Mistral OCR parser may not work.")

class MistralOcrParser(DocumentParser):
    """Parser that uses Mistral OCR to convert documents to markdown."""

    @classmethod
    def get_name(cls) -> str:
        return "Mistral OCR (pdf, jpg, png)"

    @classmethod
    def get_supported_ocr_methods(cls) -> List[Dict[str, Any]]:
        return [
            {
                "id": "ocr",
                "name": "OCR Only",
                "default_params": {}
            },
            {
                "id": "understand",
                "name": "Document Understanding",
                "default_params": {}
            }
        ]
    
    @classmethod
    def get_description(cls) -> str:
        return "Mistral OCR parser for extracting text from documents and images with optional document understanding"
    
    def encode_image(self, image_path):
        """Encode the image to base64."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            print(f"Error: The file {image_path} was not found.")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def parse(self, file_path: Union[str, Path], ocr_method: Optional[str] = None, **kwargs) -> str:
        """Parse a document using Mistral OCR."""
        if not MISTRAL_AVAILABLE:
            raise ImportError(
                "The Mistral AI client is not installed. "
                "Please install it with 'pip install mistralai'."
            )
        
        # Use the globally loaded API key
        if not api_key:
            raise ValueError(
                "MISTRAL_API_KEY environment variable is not set. "
                "Please set it to your Mistral API key."
            )
        
        # Check the OCR method
        use_document_understanding = ocr_method == "understand"
        
        try:
            # Initialize the Mistral client
            client = Mistral(api_key=api_key)
            
            # Determine file type based on extension
            file_path = Path(file_path)
            file_extension = file_path.suffix.lower()
            
            # Process the document with OCR
            if use_document_understanding:
                # Use document understanding via chat API for enhanced extraction
                return self._extract_with_document_understanding(client, file_path, file_extension)
            else:
                # Use regular OCR for basic text extraction
                return self._extract_with_ocr(client, file_path, file_extension)
            
        except Exception as e:
            error_message = f"Error parsing document with Mistral OCR: {str(e)}"
            print(error_message)
            return f"# Error\n\n{error_message}\n\nPlease check your API key and try again."
    
    def _extract_with_ocr(self, client, file_path, file_extension):
        """Extract document content using basic OCR."""
        try:
            # Process according to file type
            if file_extension in ['.pdf']:
                # For PDFs, we need to upload the file to the Mistral API first
                try:
                    # Upload the file to Mistral API
                    uploaded_pdf = client.files.upload(
                        file={
                            "file_name": file_path.name,
                            "content": open(file_path, "rb"),
                        },
                        purpose="ocr"
                    )
                    
                    # Get signed URL for the file
                    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
                    
                    # Use the signed URL for OCR processing
                    ocr_response = client.ocr.process(
                        model="mistral-ocr-latest",
                        document={
                            "type": "document_url",
                            "document_url": signed_url.url
                        }
                    )
                except Exception as e:
                    # If file upload fails, try to use a direct URL method with base64
                    print(f"Failed to upload PDF, trying alternate method: {str(e)}")
                    base64_pdf = self.encode_image(file_path)
                    
                    if base64_pdf:
                        ocr_response = client.ocr.process(
                            model="mistral-ocr-latest",
                            document={
                                "type": "image_url",
                                "image_url": f"data:application/pdf;base64,{base64_pdf}"
                            }
                        )
                    else:
                        return "# Error\n\nFailed to process PDF document."
            else:
                # For images (jpg, png, etc.), use image_url with base64
                base64_image = self.encode_image(file_path)
                if not base64_image:
                    return "# Error\n\nFailed to encode the image."
                
                mime_type = self._get_mime_type(file_extension)
                
                ocr_response = client.ocr.process(
                    model="mistral-ocr-latest",
                    document={
                        "type": "image_url",
                        "image_url": f"data:{mime_type};base64,{base64_image}"
                    }
                )
            
            # Process the OCR response
            # The Mistral OCR response is structured with pages that contain text content
            markdown_text = ""
            
            # Check if the response contains pages
            if hasattr(ocr_response, 'pages') and ocr_response.pages:
                for page in ocr_response.pages:
                    # Add page number as heading
                    page_num = page.index if hasattr(page, 'index') else "Unknown"
                    markdown_text += f"## Page {page_num}\n\n"
                    
                    # Add text content if available
                    if hasattr(page, 'text'):
                        markdown_text += page.text + "\n\n"
                    
                    # Or markdown content if that's how it's structured
                    elif hasattr(page, 'markdown'):
                        markdown_text += page.markdown + "\n\n"
                    
                    # Add any extracted tables with markdown formatting
                    if hasattr(page, 'tables') and page.tables:
                        for i, table in enumerate(page.tables):
                            markdown_text += f"### Table {i+1}\n\n"
                            if hasattr(table, 'markdown'):
                                markdown_text += table.markdown + "\n\n"
                            elif hasattr(table, 'data'):
                                # Convert table data to markdown format
                                markdown_text += self._convert_table_to_markdown(table.data) + "\n\n"
            
            # If no markdown was generated, check for raw content
            if not markdown_text and hasattr(ocr_response, 'content'):
                markdown_text = ocr_response.content
            
            # If still no content, try to access any available data
            if not markdown_text:
                # Try to get a JSON representation to extract data
                try:
                    response_dict = ocr_response.to_dict() if hasattr(ocr_response, 'to_dict') else ocr_response.__dict__
                    markdown_text = "# Extracted Content\n\n"
                    
                    # Look for content or text in the response dictionary
                    if 'content' in response_dict:
                        markdown_text += response_dict['content']
                    elif 'text' in response_dict:
                        markdown_text += response_dict['text']
                    elif 'pages' in response_dict:
                        for page in response_dict['pages']:
                            if 'text' in page:
                                markdown_text += page['text'] + "\n\n"
                    else:
                        # Just dump what we got as JSON
                        markdown_text += f"```json\n{json.dumps(response_dict, indent=2)}\n```"
                except Exception as e:
                    markdown_text = f"# Error Processing Response\n\nCould not process the OCR response: {str(e)}"
            
            # If we still have no content, return an error
            if not markdown_text:
                return "# Error\n\nNo text was extracted from the document."
            
            return f"# Document Content\n\n{markdown_text}"
        
        except Exception as e:
            return f"# OCR Extraction Error\n\n{str(e)}"
    
    def _extract_with_document_understanding(self, client, file_path, file_extension):
        """Extract and understand document content using chat completion."""
        try:
            # For PDFs and images, we'll use Mistral's document understanding capability
            if file_extension in ['.pdf']:
                # Upload PDF first
                try:
                    # Upload the file
                    uploaded_pdf = client.files.upload(
                        file={
                            "file_name": file_path.name,
                            "content": open(file_path, "rb"),
                        },
                        purpose="ocr"
                    )
                    
                    # Get the signed URL
                    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
                    
                    # Send to chat completion API with document understanding prompt
                    chat_response = client.chat.complete(
                        model="mistral-large-latest",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Convert this document to well-formatted markdown. Preserve all important content, structure, headings, lists, and tables. Include brief descriptions of any images."
                                    },
                                    {
                                        "type": "document_url",
                                        "document_url": signed_url.url
                                    }
                                ]
                            }
                        ]
                    )
                    
                    # Get the markdown result
                    return chat_response.choices[0].message.content
                
                except Exception as e:
                    # Fall back to OCR if document understanding fails
                    print(f"Document understanding failed, falling back to OCR: {str(e)}")
                    return self._extract_with_ocr(client, file_path, file_extension)
            
            else:
                # For images, encode to base64
                base64_image = self.encode_image(file_path)
                if not base64_image:
                    return "# Error\n\nFailed to encode the image."
                
                mime_type = self._get_mime_type(file_extension)
                
                # Use the chat API with the image for document understanding
                chat_response = client.chat.complete(
                    model="mistral-large-latest",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Extract all text from this image and convert it to well-formatted markdown. Preserve the structure and layout as much as possible."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ]
                )
                
                # Get the markdown result
                return chat_response.choices[0].message.content
        
        except Exception as e:
            return f"# Document Understanding Error\n\n{str(e)}\n\nFalling back to OCR method."
    
    def _get_mime_type(self, file_extension: str) -> str:
        """Get the MIME type for a file extension."""
        mime_types = {
            ".pdf": "application/pdf",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
        }
        
        return mime_types.get(file_extension, "application/octet-stream")
    
    def _convert_table_to_markdown(self, table_data) -> str:
        """Convert a table data structure to markdown format."""
        if not table_data or not isinstance(table_data, list):
            return ""
        
        # Create markdown table
        markdown = ""
        
        # Add header row
        if table_data and isinstance(table_data[0], list):
            header = table_data[0]
            markdown += "| " + " | ".join(str(cell) for cell in header) + " |\n"
            
            # Add separator row
            markdown += "| " + " | ".join(["---"] * len(header)) + " |\n"
            
            # Add data rows
            for row in table_data[1:]:
                markdown += "| " + " | ".join(str(cell) for cell in row) + " |\n"
        
        return markdown


# Register the parser with the registry
if MISTRAL_AVAILABLE:
    ParserRegistry.register(MistralOcrParser)
else:
    print("Mistral OCR parser not registered: mistralai package not installed") 