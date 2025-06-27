"""Content formatting and rendering utilities for the Markit application."""

import markdown
import json
import base64
import html
import logging

from src.core.logging_config import get_logger

logger = get_logger(__name__)


def format_markdown_content(content):
    """Convert markdown content to HTML."""
    if not content:
        return content
    
    # Convert the content to HTML using markdown library
    html_content = markdown.markdown(str(content), extensions=['tables'])
    return html_content


def render_latex_to_html(latex_content):
    """Convert LaTeX content to HTML using Mathpix Markdown like GOT-OCR demo."""
    # Clean up the content similar to GOT-OCR demo
    content = latex_content.strip()
    if content.endswith("<|im_end|>"):
        content = content[:-len("<|im_end|>")]
    
    # Fix unbalanced delimiters exactly like GOT-OCR demo
    right_num = content.count("\\right")
    left_num = content.count("\\left")
    
    if right_num != left_num:
        content = (
            content.replace("\\left(", "(")
            .replace("\\right)", ")")
            .replace("\\left[", "[")
            .replace("\\right]", "]")
            .replace("\\left{", "{")
            .replace("\\right}", "}")
            .replace("\\left|", "|")
            .replace("\\right|", "|")
            .replace("\\left.", ".")
            .replace("\\right.", ".")
        )
    
    # Process content like GOT-OCR demo: remove $ signs and replace quotes
    content = content.replace('"', "``").replace("$", "")
    
    # Split into lines and create JavaScript string like GOT-OCR demo
    outputs_list = content.split("\n")
    js_text_parts = []
    for line in outputs_list:
        # Escape backslashes and add line break
        escaped_line = line.replace("\\", "\\\\")
        js_text_parts.append(f'"{escaped_line}\\n"')
    
    # Join with + like in GOT-OCR demo
    js_text = " + ".join(js_text_parts)
    
    # Create HTML using Mathpix Markdown like GOT-OCR demo
    html_content = f"""<!DOCTYPE html>
<html lang="en" data-lt-installed="true">
<head>
    <meta charset="UTF-8">
    <title>LaTeX Content</title>
    <script>
        const text = {js_text};
    </script>
    <style>
        #content {{
            max-width: 800px;
            margin: auto;
            padding: 20px;
        }}
        body {{
            font-family: 'Times New Roman', serif;
            line-height: 1.6;
            background-color: #ffffff;
            color: #333;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        td, th {{
            border: 1px solid #333;
            padding: 8px 12px;
            text-align: center;
            vertical-align: middle;
        }}
    </style>
    <script>
        let script = document.createElement('script');
        script.src = "https://cdn.jsdelivr.net/npm/mathpix-markdown-it@1.3.6/es5/bundle.js";
        document.head.append(script);
        script.onload = function() {{
            const isLoaded = window.loadMathJax();
            if (isLoaded) {{
                console.log('Styles loaded!')
            }}
            const el = window.document.getElementById('content-text');
            if (el) {{
                const options = {{
                    htmlTags: true
                }};
                const html = window.render(text, options);
                el.outerHTML = html;
            }}
        }};
    </script>
</head>
<body>
    <div id="content">
        <div id="content-text"></div>
    </div>
</body>
</html>"""
    
    return html_content


def format_latex_content(content):
    """Format LaTeX content for display in UI using MathJax rendering like GOT-OCR demo."""
    if not content:
        return content
    
    try:
        # Generate rendered HTML
        rendered_html = render_latex_to_html(content)
        
        # Encode for iframe display (similar to GOT-OCR demo)
        encoded_html = base64.b64encode(rendered_html.encode("utf-8")).decode("utf-8")
        iframe_src = f"data:text/html;base64,{encoded_html}"
        
        # Create the display with both rendered and raw views
        formatted_content = f"""
        <div style="background-color: #f8f9fa; border-radius: 8px; border: 1px solid #e9ecef; margin: 10px 0;">
            <div style="background-color: #e9ecef; padding: 10px; border-radius: 8px 8px 0 0; font-weight: bold; color: #495057;">
                📄 LaTeX Content (Rendered with MathJax)
            </div>
            <div style="padding: 0;">
                <iframe src="{iframe_src}" width="100%" height="500px" style="border: none; border-radius: 0 0 8px 8px;"></iframe>
            </div>
            <div style="background-color: #e9ecef; padding: 8px 15px; border-radius: 0; font-size: 12px; color: #6c757d; border-top: 1px solid #dee2e6;">
                💡 LaTeX content rendered with MathJax. Tables and formulas are displayed as they would appear in a LaTeX document.
            </div>
            <details style="margin: 0; border-top: 1px solid #dee2e6;">
                <summary style="padding: 8px 15px; background-color: #e9ecef; cursor: pointer; font-size: 12px; color: #6c757d;">
                    📝 View Raw LaTeX Source
                </summary>
                <div style="padding: 15px; background-color: #f8f9fa;">
                    <pre style="background-color: transparent; margin: 0; padding: 0;
                                font-family: 'Courier New', monospace; font-size: 12px; line-height: 1.4; 
                                white-space: pre-wrap; word-wrap: break-word; color: #2c3e50; max-height: 200px; overflow-y: auto;">
{content}
                    </pre>
                </div>
            </details>
        </div>
        """
        
    except Exception as e:
        # Fallback to simple formatting if rendering fails
        logger.error(f"Error rendering LaTeX content: {e}")
        escaped_content = html.escape(str(content))
        formatted_content = f"""
        <div style="background-color: #f8f9fa; border-radius: 8px; border: 1px solid #e9ecef; margin: 10px 0;">
            <div style="background-color: #e9ecef; padding: 10px; border-radius: 8px 8px 0 0; font-weight: bold; color: #495057;">
                📄 LaTeX Content (Fallback View)
            </div>
            <div style="padding: 15px;">
                <pre style="background-color: transparent; margin: 0; padding: 0;
                            font-family: 'Courier New', monospace; font-size: 14px; line-height: 1.4; 
                            white-space: pre-wrap; word-wrap: break-word; color: #2c3e50;">
{escaped_content}
                </pre>
            </div>
            <div style="background-color: #e9ecef; padding: 8px 15px; border-radius: 0 0 8px 8px; font-size: 12px; color: #6c757d;">
                ⚠️ Rendering failed, showing raw LaTeX. Error: {str(e)}
            </div>
        </div>
        """
    
    return formatted_content