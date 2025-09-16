"""
EASY CONFIGURATION - Chapter Detection System
===========================================

🔧 CHANGE YOUR DEFAULT TEXTBOOK HERE:
Just modify the PDF_PATH below to point to your textbook.

Example paths:
- "../data_files/my_textbook.pdf"
- "C:/Users/wa/Documents/medical_book.pdf"
- "/path/to/your/textbook.pdf"
"""

# 📚 DEFAULT TEXTBOOK PATH (EDIT THIS!)
# PDF_PATH = "../data_files/Robert Hockberger,Ron M. Walls,Susan Wilcox, - ROSEN's Emergency medicine Concepts and clinical practice both volumes TENTH edition (2023, Elsevier) - libgen.li.pdf"
PDF_PATH = "../data_files/Nursing-Skills-2e-1720739371._print.pdf"

# 📁 OUTPUT SETTINGS
OUTPUT_DIR = "output"
CREATE_JSON = True
CREATE_CSV = True
SHOW_ALL_CHAPTERS = False  # Set to True to see full chapter list

# 🎯 DETECTION SETTINGS
VERBOSE = True  # Show detailed progress
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to accept chapters
