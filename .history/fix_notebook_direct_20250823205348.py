#!/usr/bin/env python3
"""
Direct fix script to remove the error message from the notebook
"""

def fix_notebook():
    # Read the notebook line by line
    with open('CVD_ASIF_Enhanced.ipynb', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Original file has {len(lines)} lines")
    
    # Find and fix the problematic line
    fixed_lines = []
    changes_made = 0
    
    for i, line in enumerate(lines):
        # Check if this line contains the error message
        if '⟪' in line and 'SyntaxError' in line and 'invalid syntax' in line:
            print(f"Found problematic line {i+1}: {line.strip()}")
            # Replace with the correct import statement
            fixed_lines.append('    "from sklearn.model_selection import GridSearchCV\\n",\n')
            changes_made += 1
        else:
            fixed_lines.append(line)
    
    print(f"Made {changes_made} changes")
    
    # Write the fixed notebook
    with open('CVD_ASIF_Enhanced_fixed_v3.ipynb', 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print("Notebook fixed and saved as CVD_ASIF_Enhanced_fixed_v3.ipynb")

if __name__ == "__main__":
    fix_notebook()
