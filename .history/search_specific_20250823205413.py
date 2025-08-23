#!/usr/bin/env python3
"""
Script to search for the exact problematic content
"""

def search_notebook():
    # Read the notebook line by line
    with open('CVD_ASIF_Enhanced.ipynb', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Notebook has {len(lines)} lines")
    
    # Search for lines containing specific patterns
    for i, line in enumerate(lines):
        if '⟪' in line:
            print(f"Line {i+1}: {line.strip()}")
        if 'SyntaxError' in line:
            print(f"Line {i+1}: {line.strip()}")
        if 'invalid syntax' in line:
            print(f"Line {i+1}: {line.strip()}")
        if 'from sklearn.model_selection import GridSearchCV' in line:
            print(f"Line {i+1}: {line.strip()}")
        if 'Bayesian search with cross-validation' in line:
            print(f"Line {i+1}: {line.strip()}")

if __name__ == "__main__":
    search_notebook()
