import os

def check_file_encoding(filepath):
    try:
        with open(filepath, 'rb') as file:
            raw = file.read()
            if raw.startswith(b'\xef\xbb\xbf'):
                print(f"File {filepath} has UTF-8 BOM")
            elif b'\x00' in raw:
                print(f"File {filepath} contains null bytes at positions: {[i for i, b in enumerate(raw) if b == 0]}")
            else:
                print(f"File {filepath} looks OK")
    except Exception as e:
        print(f"Error checking {filepath}: {e}")

# 只检查项目文件，不检查 venv
for root, dirs, files in os.walk('.'):
    if 'venv' in root:
        continue
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            check_file_encoding(filepath)