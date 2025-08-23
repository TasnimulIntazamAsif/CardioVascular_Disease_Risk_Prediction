#!/usr/bin/env python3
"""
Debug script to understand the notebook structure and fix the syntax error
"""

import json
import re

def debug_notebook():
    # Read the notebook
    with open('CVD_ASIF_Enhanced.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    print(f"Notebook has {len(notebook['cells'])} cells")
    
    # Search for cells with execution_count: 173
    for i, cell in enumerate(notebook['cells']):
        if cell.get('execution_count') == 173:
            print(f"\n=== Cell {i} with execution_count: 173 ===")
            print(f"Cell type: {cell.get('cell_type')}")
            print(f"Source length: {len(cell['source'])}")
            print("Source content:")
            for j, line in enumerate(cell['source']):
                print(f"  {j}: {repr(line)}")
            break
    else:
        print("No cell found with execution_count: 173")
    
    # Search for any cells containing the error message
    print("\n=== Searching for cells with error messages ===")
    for i, cell in enumerate(notebook['cells']):
        for j, line in enumerate(cell['source']):
            if 'SyntaxError' in line or 'invalid syntax' in line:
                print(f"Cell {i}, line {j}: {repr(line)}")
                print(f"  Cell execution_count: {cell.get('execution_count')}")
                print(f"  Cell type: {cell.get('cell_type')}")
    
    # Search for cells containing the import statement
    print("\n=== Searching for cells with GridSearchCV import ===")
    for i, cell in enumerate(notebook['cells']):
        for j, line in enumerate(cell['source']):
            if 'GridSearchCV' in line:
                print(f"Cell {i}, line {j}: {repr(line)}")
                print(f"  Cell execution_count: {cell.get('execution_count')}")
                print(f"  Cell type: {cell.get('cell_type')}")
    
    # Search for cells containing the comment
    print("\n=== Searching for cells with 'Bayesian search' comment ===")
    for i, cell in enumerate(notebook['cells']):
        for j, line in enumerate(cell['source']):
            if 'Bayesian search with cross-validation' in line:
                print(f"Cell {i}, line {j}: {repr(line)}")
                print(f"  Cell execution_count: {cell.get('execution_count')}")
                print(f"  Cell type: {cell.get('cell_type')}")

if __name__ == "__main__":
    debug_notebook()
