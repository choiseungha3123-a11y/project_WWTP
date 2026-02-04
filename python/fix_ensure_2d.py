import json

# 노트북 로드
with open('notebook/DL/LSTM.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ensure_2d_y 함수는 그대로 두고, main에서 잘못 사용된 부분만 수정
# 이미 main 함수에서 수정했으므로 확인만 함

found = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell['source'])
        
        if 'def main():' in source_text:
            # X_va_s를 ensure_2d_y로 처리하는 버그가 있는지 확인
            if 'X_va_s = ensure_2d_y(X_va_s)' in source_text:
                print(f"Found bug in cell {i}: X_va_s should not use ensure_2d_y")
                # 이미 수정된 코드에서는 이 버그가 없어야 함
            else:
                print(f"Cell {i} (main function) looks correct - only y variables use ensure_2d_y")
                found = True

if not found:
    print("Warning: Could not verify main function")
else:
    print("\n✓ ensure_2d_y usage is correct!")
