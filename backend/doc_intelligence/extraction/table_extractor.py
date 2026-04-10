import camelot
import pandas as pd
from ..config import logger

class TableExtractor:
    def __init__(self, file_path):
        self.file_path = file_path

    def extract_tables(self):
        """Extracts tables from the PDF using Camelot's Lattice and Stream methods."""
        all_tables_data = []
        
        try:
            logger.info("Extracting tables using Camelot (Lattice method)...")
            tables = camelot.read_pdf(self.file_path, pages='all', flavor='lattice')
            
            # If lattice finds nothing, try stream
            if len(tables) == 0:
                logger.info("No lattice tables found. Trying Stream method...")
                tables = camelot.read_pdf(self.file_path, pages='all', flavor='stream')
            
            for table in tables:
                page_number = table.page
                # Convert table to markdown-like text
                df = table.df
                table_text = df.to_csv(index=False, sep="|")
                
                all_tables_data.append({
                    "page": page_number,
                    "table_text": table_text,
                    "accuracy": table.accuracy,
                    "whitespace": table.whitespace
                })
                
            logger.info(f"Extracted {len(all_tables_data)} tables total.")
            
        except Exception as e:
            logger.error(f"Error extracting tables with Camelot: {e}")
            logger.warning("Ensure Ghostscript is installed and in your PATH.")
            
        return all_tables_data

    def merge_tables_into_pages(self, pages_data, tables_data):
        """Merges extracted tables into the corresponding pages' refined text."""
        for page in pages_data:
            page_num = page["page"]
            # Find all tables for this page
            page_tables = [t["table_text"] for t in tables_data if t["page"] == page_num]
            
            if page_tables:
                # Append tables to the end of the page text
                tables_combined = "\n\n[EXTRACTED TABLES]\n" + "\n\n".join(page_tables)
                page["text"] += tables_combined
                
        return pages_data

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        extractor = TableExtractor(sys.argv[1])
        tables = extractor.extract_tables()
        for t in tables:
            print(f"--- Page {t['page']} (Accuracy: {t['accuracy']}) ---")
            print(t['table_text'][:500])
