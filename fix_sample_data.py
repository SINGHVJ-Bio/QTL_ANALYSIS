#!/usr/bin/env python3
"""
Quick script to fix the annotations file header
"""

def fix_annotations_file():
    """Fix the annotations.bed file header"""
    
    with open("data/annotations.bed", "r") as f:
        content = f.read()
    
    # Remove # from the header line
    content = content.replace("#chr", "chr")
    
    with open("data/annotations.bed", "w") as f:
        f.write(content)
    
    print("✅ Fixed annotations.bed file header")

if __name__ == "__main__":
    fix_annotations_file()
    