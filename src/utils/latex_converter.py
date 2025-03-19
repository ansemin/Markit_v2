import re
import logging
from typing import Dict, List, Tuple, Optional
import latex2markdown

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LatexConverter:
    """Enhanced LaTeX to Markdown converter that handles complex LaTeX structures."""
    
    @staticmethod
    def convert(latex_text: str) -> str:
        """
        Convert LaTeX text to Markdown, with special handling for tables and other structures.
        
        Args:
            latex_text: Raw LaTeX text from the GOT-OCR model
            
        Returns:
            str: Converted Markdown text
        """
        if not latex_text or not isinstance(latex_text, str):
            return ""
            
        # Process the text in stages
        processed_text = latex_text
        
        # Stage 1: Pre-process tables before standard conversion
        processed_text = LatexConverter._preprocess_tables(processed_text)
        
        # Stage 2: Convert using latex2markdown library
        try:
            # Use the standard latex2markdown library as a base - FOLLOWING OFFICIAL DOCUMENTATION
            l2m = latex2markdown.LaTeX2Markdown(processed_text)
            processed_text = l2m.to_markdown()
        except Exception as e:
            logger.error(f"Error in standard latex2markdown conversion: {str(e)}")
            # Continue with our custom processing even if the standard library fails
        
        # Stage 3: Post-process to fix any remaining issues
        processed_text = LatexConverter._postprocess_markdown(processed_text)
        
        return processed_text
    
    @staticmethod
    def _preprocess_tables(latex_text: str) -> str:
        """
        Pre-process LaTeX tables to ensure they convert correctly.
        
        Args:
            latex_text: Raw LaTeX text
            
        Returns:
            str: Pre-processed LaTeX text with table modifications
        """
        processed_text = latex_text
        
        # Find all tabular environments
        table_pattern = r'\\begin{tabular}(.*?)\\end{tabular}'
        tables = re.findall(table_pattern, processed_text, re.DOTALL)
        
        for i, table_content in enumerate(tables):
            # Extract the column specification
            col_spec_match = re.search(r'{([^}]*)}', table_content)
            if not col_spec_match:
                continue
                
            # Process the table content
            rows_text = re.sub(r'{[^}]*}', '', table_content, count=1)  # Remove the column spec
            
            # Split into rows by \\ or \hline
            rows = re.split(r'\\\\|\\hline', rows_text)
            rows = [row.strip() for row in rows if row.strip()]
            
            # Calculate number of columns based on the number of & in the first non-empty row plus 1
            for row in rows:
                if '&' in row:
                    num_cols = row.count('&') + 1
                    break
            else:
                num_cols = 1  # Default if no & found
            
            # Create a clean tabular environment that's easier to parse
            clean_table = f"\\begin{{tabular}}{{{'|'.join(['c'] * num_cols)}}}\n"
            
            for row in rows:
                if row.strip():
                    clean_row = ' & '.join([cell.strip() for cell in row.split('&')])
                    clean_table += clean_row + " \\\\\n"
            
            clean_table += "\\end{tabular}"
            
            # Replace the original table with the clean one
            processed_text = processed_text.replace(
                f"\\begin{tabular}{table_content}\\end{tabular}",
                clean_table
            )
        
        return processed_text
    
    @staticmethod
    def _postprocess_markdown(markdown_text: str) -> str:
        """
        Post-process the converted Markdown to fix any remaining issues.
        
        Args:
            markdown_text: Converted Markdown text
            
        Returns:
            str: Post-processed Markdown text
        """
        processed_text = markdown_text
        
        # Fix common issues with tables
        # 1. Fix pipe tables that may be malformed
        table_lines = []
        in_table = False
        
        for line in processed_text.split('\n'):
            if '|' in line and not line.strip().startswith('|') and not in_table:
                # This might be the start of a table, add the missing pipe
                line = '| ' + line
                in_table = True
                
            if in_table:
                if '|' in line:
                    # Ensure line ends with pipe
                    if not line.strip().endswith('|'):
                        line = line + ' |'
                    table_lines.append(line)
                else:
                    # End of table
                    in_table = False
                    
                    # If this is a table, add a header separator row after the first row
                    if len(table_lines) > 0:
                        col_count = table_lines[0].count('|') - 1
                        separator = '| ' + ' | '.join(['---'] * col_count) + ' |'
                        table_lines.insert(1, separator)
                    
                    # Add the current line and the processed table
                    for table_line in table_lines:
                        processed_text = processed_text.replace(table_line, table_line)
                    table_lines = []
            
        # Fix math blocks
        processed_text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', processed_text, flags=re.DOTALL)
        processed_text = re.sub(r'\\\((.*?)\\\)', r'$\1$', processed_text, flags=re.DOTALL)
        
        # Fix formatting issues
        processed_text = processed_text.replace('\\textbf{', '**')
        processed_text = processed_text.replace('\\textit{', '*')
        processed_text = processed_text.replace('}', '')  # Remove closing braces
        
        # Fix escape sequences
        processed_text = processed_text.replace('\\%', '%')
        processed_text = processed_text.replace('\\$', '$')
        processed_text = processed_text.replace('\\&', '&')
        
        return processed_text
        
def convert_latex_to_markdown(latex_text: str) -> str:
    """
    Convenience function to convert LaTeX to Markdown.
    
    Args:
        latex_text: Raw LaTeX text from the GOT-OCR model
        
    Returns:
        str: Converted Markdown text
    """
    return LatexConverter.convert(latex_text) 