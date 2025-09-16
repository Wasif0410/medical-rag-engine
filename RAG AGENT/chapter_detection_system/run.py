#!/usr/bin/env python3
"""
🚀 SUPER FAST RUNNER - Chapter Detection System
==============================================

SIMPLEST WAY TO RUN:
    python run.py

CHANGE TEXTBOOK:
    1. Edit settings.py (change PDF_PATH)
    2. python run.py

CUSTOM PDF:
    python run.py "path/to/your/book.pdf"
"""

import sys
from pathlib import Path
from chapter_detector import detect_chapters, save_chapters

# Import settings
try:
    from settings import PDF_PATH, OUTPUT_DIR, CREATE_JSON, CREATE_CSV, SHOW_ALL_CHAPTERS, VERBOSE
except ImportError:
    # Fallback defaults
    PDF_PATH = "../data_files/Robert Hockberger,Ron M. Walls,Susan Wilcox, - ROSEN's Emergency medicine Concepts and clinical practice both volumes TENTH edition (2023, Elsevier) - libgen.li.pdf"
    OUTPUT_DIR = "output"
    CREATE_JSON = True
    CREATE_CSV = True
    SHOW_ALL_CHAPTERS = False
    VERBOSE = True

def run_detection(pdf_path: str = None):
    """Run chapter detection - FAST AND SIMPLE"""
    
    # Use settings default or command line argument
    if pdf_path is None:
        pdf_path = PDF_PATH
    
    pdf_file = Path(pdf_path)
    
    # Quick file check
    if not pdf_file.exists():
        print(f"❌ File not found: {pdf_path}")
        print(f"💡 Edit settings.py to change default textbook")
        return False
    
    # Create output folder
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    if VERBOSE:
        print(f"🔍 Processing: {pdf_file.name}")
        print("-" * 50)
    
    # MAIN DETECTION
    chapters = detect_chapters(str(pdf_file))
    
    if not chapters:
        print("❌ No chapters found!")
        return False
    
    # Results
    if VERBOSE:
        print(f"✅ Found {len(chapters)} chapters")
        print(f"📄 {chapters[-1].end_page + 1} total pages")
        print(f"🎯 Method: {chapters[0].method} (confidence: {chapters[0].confidence:.1f})")
    
    # Show chapters (limited or all)
    if SHOW_ALL_CHAPTERS:
        for i, ch in enumerate(chapters, 1):
            pages = ch.end_page - ch.start_page + 1
            print(f"{i:3d}. {ch.title} (pages {ch.start_page+1}-{ch.end_page+1})")
    else:
        # Show first 3 and last 3
        show_chapters = chapters[:3] + chapters[-3:] if len(chapters) > 6 else chapters
        for i, ch in enumerate(show_chapters):
            if i == 3 and len(chapters) > 6:
                print("    ... (edit settings.py: SHOW_ALL_CHAPTERS = True for full list)")
            chapter_num = chapters.index(ch) + 1
            pages = ch.end_page - ch.start_page + 1
            print(f"{chapter_num:3d}. {ch.title} (pages {ch.start_page+1}-{ch.end_page+1})")
    
    # Save files
    base_name = pdf_file.stem.replace(" ", "_").replace(",", "")[:25]
    
    saved_files = []
    if CREATE_JSON:
        json_file = output_dir / f"{base_name}_chapters.json"
        save_chapters(chapters, json_file, 'json')
        saved_files.append(str(json_file))
    
    if CREATE_CSV:
        csv_file = output_dir / f"{base_name}_chapters.csv"
        save_chapters(chapters, csv_file, 'csv')
        saved_files.append(str(csv_file))
    
    if VERBOSE:
        print(f"\n💾 Saved to: {', '.join(saved_files)}")
        print("🚀 Ready for RAG!")
    
    return True

def main():
    """Main runner"""
    
    # Handle command line
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    if pdf_path and pdf_path in ["--help", "-h", "help"]:
        print(__doc__)
        return
    
    # RUN IT!
    success = run_detection(pdf_path)
    
    if not success:
        print("\n💡 TIPS:")
        print("   - Check file path in settings.py")
        print("   - Make sure PDF file exists")
        print("   - Try: python run.py --help")

if __name__ == "__main__":
    main()
