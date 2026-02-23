"""Extract code cells from LSTM_TMS.ipynb to a single Python file."""
import json

with open(r'c:\project_WWTP\python\notebook\DL\LSTM_TMS.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print(f'Total code cells: {len(code_cells)}')
for i, c in enumerate(code_cells):
    src_lines = c['source']
    first_line = src_lines[0].strip() if src_lines else '(empty)'
    print(f'Cell {i}: {len(src_lines)} lines | {first_line[:100]}')

# Write all code cells to a single .py file
with open(r'c:\project_WWTP\python\notebook\DL\_lstm_tms_extracted.py', 'w', encoding='utf-8') as out:
    for i, c in enumerate(code_cells):
        out.write(f'\n# {"="*70}\n')
        out.write(f'# Cell {i}\n')
        out.write(f'# {"="*70}\n')
        out.write(''.join(c['source']))
        out.write('\n')
