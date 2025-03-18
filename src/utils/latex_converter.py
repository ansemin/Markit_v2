import re
import logging

# Configure logging
logger = logging.getLogger(__name__)

def latex_to_markdown(latex_text):
    """
    Convert LaTeX formatted text from GOT-OCR to Markdown.
    
    Args:
        latex_text (str): LaTeX formatted text
        
    Returns:
        str: Markdown formatted text
    """
    if not latex_text:
        return ""
    
    logger.info("Converting LaTeX to Markdown")
    
    # Make a copy of the input text
    md_text = latex_text
    
    # Handle LaTeX tables
    md_text = convert_latex_tables(md_text)
    
    # Handle LaTeX math environments
    md_text = convert_math_environments(md_text)
    
    # Handle LaTeX formatting commands
    md_text = convert_formatting_commands(md_text)
    
    # Handle LaTeX lists
    md_text = convert_latex_lists(md_text)
    
    # Clean up any remaining LaTeX-specific syntax
    md_text = cleanup_latex(md_text)
    
    logger.info("LaTeX to Markdown conversion completed")
    return md_text

def convert_latex_tables(latex_text):
    """Convert LaTeX tables to Markdown tables."""
    result = latex_text
    
    # Detect and convert tabular environments
    tabular_pattern = r'\\begin\{(tabular|table)\}(.*?)\\end\{(tabular|table)\}'
    
    def replace_table(match):
        table_content = match.group(2)
        
        # Extract rows
        rows = re.split(r'\\\\', table_content)
        md_rows = []
        
        # Create header separator after first row
        if rows:
            first_row = rows[0]
            # Count columns based on & separators
            col_count = first_row.count('&') + 1
            
            # Process rows
            for i, row in enumerate(rows):
                # Skip empty rows
                if not row.strip():
                    continue
                    
                # Split by & to get cells
                cells = row.split('&')
                # Clean cell content
                cells = [cell.strip().replace('\\hline', '') for cell in cells]
                
                # Join cells with | for Markdown table format
                md_row = '| ' + ' | '.join(cells) + ' |'
                md_rows.append(md_row)
                
                # Add header separator after first row
                if i == 0:
                    md_rows.append('| ' + ' | '.join(['---'] * col_count) + ' |')
        
        return '\n'.join(md_rows)
    
    # Replace all tabular environments
    result = re.sub(tabular_pattern, replace_table, result, flags=re.DOTALL)
    return result

def convert_math_environments(latex_text):
    """Convert LaTeX math environments to Markdown math syntax."""
    result = latex_text
    
    # Convert equation environments to $$ ... $$ format
    result = re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}', r'$$\1$$', result, flags=re.DOTALL)
    result = re.sub(r'\\begin\{align\}(.*?)\\end\{align\}', r'$$\1$$', result, flags=re.DOTALL)
    result = re.sub(r'\\begin\{eqnarray\}(.*?)\\end\{eqnarray\}', r'$$\1$$', result, flags=re.DOTALL)
    
    # Convert inline math $ ... $ (if not already in right format)
    result = re.sub(r'\\(\(|\))', '$', result)
    
    # Handle standalone math expressions
    result = re.sub(r'\\begin\{math\}(.*?)\\end\{math\}', r'$\1$', result, flags=re.DOTALL)
    
    return result

def convert_formatting_commands(latex_text):
    """Convert LaTeX formatting commands to Markdown syntax."""
    result = latex_text
    
    # Bold: \textbf{text} -> **text**
    result = re.sub(r'\\textbf\{([^}]*)\}', r'**\1**', result)
    result = re.sub(r'\\bf\{([^}]*)\}', r'**\1**', result)
    
    # Italic: \textit{text} -> *text*
    result = re.sub(r'\\textit\{([^}]*)\}', r'*\1*', result)
    result = re.sub(r'\\it\{([^}]*)\}', r'*\1*', result)
    result = re.sub(r'\\emph\{([^}]*)\}', r'*\1*', result)
    
    # Underline: don't have direct equivalent in MD, use emphasis
    result = re.sub(r'\\underline\{([^}]*)\}', r'_\1_', result)
    
    # Section headings
    result = re.sub(r'\\section\{([^}]*)\}', r'## \1', result)
    result = re.sub(r'\\subsection\{([^}]*)\}', r'### \1', result)
    result = re.sub(r'\\subsubsection\{([^}]*)\}', r'#### \1', result)
    
    # Remove \title command
    result = re.sub(r'\\title\{([^}]*)\}', r'# \1', result)
    
    return result

def convert_latex_lists(latex_text):
    """Convert LaTeX lists to Markdown lists."""
    result = latex_text
    
    # Handle itemize (unordered lists)
    itemize_pattern = r'\\begin\{itemize\}(.*?)\\end\{itemize\}'
    
    def replace_itemize(match):
        list_content = match.group(1)
        items = re.findall(r'\\item\s+(.*?)(?=\\item|$)', list_content, re.DOTALL)
        return '\n' + '\n'.join([f'- {item.strip()}' for item in items]) + '\n'
    
    result = re.sub(itemize_pattern, replace_itemize, result, flags=re.DOTALL)
    
    # Handle enumerate (ordered lists)
    enumerate_pattern = r'\\begin\{enumerate\}(.*?)\\end\{enumerate\}'
    
    def replace_enumerate(match):
        list_content = match.group(1)
        items = re.findall(r'\\item\s+(.*?)(?=\\item|$)', list_content, re.DOTALL)
        return '\n' + '\n'.join([f'{i+1}. {item.strip()}' for i, item in enumerate(items)]) + '\n'
    
    result = re.sub(enumerate_pattern, replace_enumerate, result, flags=re.DOTALL)
    
    return result

def cleanup_latex(latex_text):
    """Clean up any remaining LaTeX-specific syntax."""
    result = latex_text
    
    # Remove LaTeX document structure commands
    result = re.sub(r'\\begin\{document\}|\\end\{document\}', '', result)
    result = re.sub(r'\\maketitle', '', result)
    result = re.sub(r'\\documentclass\{[^}]*\}', '', result)
    result = re.sub(r'\\usepackage\{[^}]*\}', '', result)
    
    # Convert special characters
    latex_special_chars = {
        r'\&': '&',
        r'\%': '%',
        r'\$': '$',
        r'\#': '#',
        r'\_': '_',
        r'\{': '{',
        r'\}': '}',
        r'~': ' ',
        r'\ldots': '...'
    }
    
    for latex_char, md_char in latex_special_chars.items():
        result = result.replace(latex_char, md_char)
    
    # Fix extra whitespace
    result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)
    
    return result 