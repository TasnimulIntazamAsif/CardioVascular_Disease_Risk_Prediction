#!/usr/bin/env python3
"""
Targeted script to fix the syntax error in CVD_ASIF_Enhanced.ipynb
"""

import json
import re

def fix_notebook():
    # Read the notebook
    with open('CVD_ASIF_Enhanced.ipynb', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Original file size:", len(content))
    
    # Remove the problematic error message that got embedded
    # This pattern matches the error output that got mixed into the source
    pattern = r'⟪.*⟫\[39m\n\\u001b\[31m\s*\\u001b\[39m\\u001b\[31mfrom sklearn\.model_selection import GridSearchCV\\u001b\[39m\n\s*\^\n\\u001b\[31mSyntaxError\\u001b\[31m:\s*\\u001b\[31m invalid syntax\n'
    
    # Replace with the correct import statement
    replacement = '    from sklearn.model_selection import GridSearchCV\n'
    
    # Perform the replacement
    new_content = re.sub(pattern, replacement, content)
    
    print("New file size:", len(new_content))
    print("Changes made:", content.count(pattern))
    
    # Write the fixed notebook
    with open('CVD_ASIF_Enhanced_fixed_v2.ipynb', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("Notebook fixed and saved as CVD_ASIF_Enhanced_fixed_v2.ipynb")

if __name__ == "__main__":
    fix_notebook()
