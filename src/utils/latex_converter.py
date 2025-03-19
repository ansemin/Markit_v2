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
        processed_text, tables_dict = LatexConverter._extract_tables(processed_text)
        
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
        
        # Stage 4: Reinsert tables as markdown tables
        processed_text = LatexConverter._reinsert_tables(processed_text, tables_dict)
        
        return processed_text
    
    @staticmethod
    def _extract_tables(latex_text: str) -> tuple:
        """
        Extract tables from LaTeX and replace with placeholders.
        
        Args:
            latex_text: Raw LaTeX text
            
        Returns:
            tuple: (processed text with placeholders, dict of tables)
        """
        processed_text = latex_text
        tables_dict = {}
        
        # Find all tabular environments
        table_pattern = r'\\begin{tabular}(.*?)\\end{tabular}'
        tables = re.findall(table_pattern, processed_text, re.DOTALL)
        
        for i, table_content in enumerate(tables):
            placeholder = f"TABLE_PLACEHOLDER_{i}"
            tables_dict[placeholder] = table_content
            
            # Replace the table with a placeholder
            processed_text = processed_text.replace(
                f"\\begin{{tabular}}{table_content}\\end{{tabular}}",
                placeholder
            )
        
        return processed_text, tables_dict
    
    @staticmethod
    def _reinsert_tables(markdown_text: str, tables_dict: dict) -> str:
        """
        Convert LaTeX tables to Markdown tables and reinsert them.
        
        Args:
            markdown_text: Processed markdown text with placeholders
            tables_dict: Dictionary of tables extracted from LaTeX
            
        Returns:
            str: Markdown text with tables converted and reinserted
        """
        processed_text = markdown_text
        
        for placeholder, table_content in tables_dict.items():
            # Convert LaTeX table to Markdown table
            markdown_table = LatexConverter._convert_table_to_markdown(table_content)
            
            # Replace the placeholder with the Markdown table
            processed_text = processed_text.replace(placeholder, markdown_table)
        
        return processed_text
    
    @staticmethod
    def _convert_table_to_markdown(table_content: str) -> str:
        """
        Convert a LaTeX table to Markdown format.
        
        Args:
            table_content: LaTeX table content
            
        Returns:
            str: Markdown table
        """
        # Extract the column specification
        col_spec_match = re.search(r'{([^}]*)}', table_content)
        if not col_spec_match:
            return f"[Table conversion failed]"
            
        # Process the table content
        rows_text = re.sub(r'{[^}]*}', '', table_content, count=1)  # Remove the column spec
        
        # Split into rows by \\ or \hline
        rows = re.split(r'\\\\|\\hline', rows_text)
        rows = [row.strip() for row in rows if row.strip()]
        
        if not rows:
            return "[Empty table]"
        
        # Calculate number of columns based on the number of & in the first non-empty row plus 1
        num_cols = 1  # Default
        for row in rows:
            if '&' in row:
                num_cols = row.count('&') + 1
                break
        
        # Build markdown table
        markdown_table = []
        
        # Add header row
        if rows:
            first_row = rows[0]
            cells = [cell.strip() for cell in first_row.split('&')]
            markdown_table.append("| " + " | ".join(cells + [""] * (num_cols - len(cells))) + " |")
            
            # Add separator row
            markdown_table.append("| " + " | ".join(["---"] * num_cols) + " |")
            
            # Add data rows
            for row in rows[1:]:
                cells = [cell.strip() for cell in row.split('&')]
                markdown_table.append("| " + " | ".join(cells + [""] * (num_cols - len(cells))) + " |")
        
        return "\n".join(markdown_table)
    
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