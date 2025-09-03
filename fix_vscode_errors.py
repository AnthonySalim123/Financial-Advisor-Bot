# Create fix_vscode_errors.py
"""Fix VS Code import errors"""

# Fix 1: Add missing imports to analysis.py
print("Fixing analysis.py...")
with open('pages/analysis.py', 'r') as f:
    content = f.read()

if 'from typing import Dict' not in content:
    # Add after the other imports
    content = content.replace(
        'from typing import Dict, Tuple, Optional, List',
        'from typing import Dict, Tuple, Optional, List'
    )
    if 'from typing import' not in content:
        content = content.replace(
            'import sys',
            'from typing import Dict, List, Optional\nimport sys'
        )
    with open('pages/analysis.py', 'w') as f:
        f.write(content)
    print("✅ Fixed typing imports in analysis.py")

# Fix 2: Add missing imports to backtesting.py
print("Fixing backtesting.py...")
with open('pages/backtesting.py', 'r') as f:
    content = f.read()

if 'import plotly.graph_objects as go' not in content:
    # Add after the plotly.express import if it exists
    content = content.replace(
        'import sys',
        'import plotly.graph_objects as go\nimport sys'
    )
    with open('pages/backtesting.py', 'w') as f:
        f.write(content)
    print("✅ Fixed plotly imports in backtesting.py")

print("\nAll fixes applied! Reload VS Code window.")