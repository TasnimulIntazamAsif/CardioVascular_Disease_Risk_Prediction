#!/usr/bin/env python3
"""
Script to fix the syntax error in CVD_ASIF_Enhanced.ipynb
The issue is that an error message got embedded in the notebook source around line 3795
"""

import json
import re

def fix_notebook():
    # Read the notebook
    with open('CVD_ASIF_Enhanced.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    print(f"Notebook has {len(notebook['cells'])} cells")
    
    # Find the problematic cell (execution_count: 173)
    problematic_cell = None
    for i, cell in enumerate(notebook['cells']):
        if cell.get('execution_count') == 173:
            problematic_cell = i
            print(f"Found problematic cell at index {i} with execution_count: 173")
            print(f"Cell type: {cell.get('cell_type')}")
            print(f"Source length: {len(cell['source'])}")
            print(f"Original source: {cell['source']}")
            break
    
    if problematic_cell is None:
        print("Could not find cell with execution_count: 173")
        return
    
    # Fix the source by removing any error messages and ensuring proper imports
    cell = notebook['cells'][problematic_cell]
    fixed_source = []
    
    for line in cell['source']:
        # Remove any lines that contain error messages or ANSI codes
        if not re.search(r'⟪.*⟫|\\u001b\[|SyntaxError|invalid syntax', line):
            # Clean up the import statement if it has extra spaces
            if 'from sklearn.model_selection import GridSearchCV' in line:
                line = '    from sklearn.model_selection import GridSearchCV\n'
            fixed_source.append(line)
        else:
            print(f"Removing problematic line: {line}")
    
    # Ensure the cell has the proper structure
    if not any('from sklearn.model_selection import GridSearchCV' in line for line in fixed_source):
        # Add the import if it's missing
        print("Adding missing import statement")
        fixed_source.insert(0, '    from sklearn.model_selection import GridSearchCV\n')
    
    cell['source'] = fixed_source
    print(f"Fixed source: {cell['source']}")
    
    # Write the fixed notebook
    with open('CVD_ASIF_Enhanced_fixed.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print("Notebook fixed and saved as CVD_ASIF_Enhanced_fixed.ipynb")

if __name__ == "__main__":
    fix_notebook()
