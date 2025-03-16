def fix_file_encoding(filepath):
    try:
        # 尝试以二进制模式读取文件
        with open(filepath, 'rb') as f:
            content = f.read()
        
        # 尝试检测和删除任何 BOM 标记
        if content.startswith(b'\xef\xbb\xbf'):  # UTF-8 BOM
            content = content[3:]
        elif content.startswith(b'\xff\xfe') or content.startswith(b'\xfe\xff'):  # UTF-16 BOM
            content = content[2:]
            
        # 删除所有无效字符
        content = content.decode('utf-8', errors='ignore').encode('utf-8')
        
        # 写回文件
        with open(filepath, 'wb') as f:
            f.write(content)
            
        print(f"已修复文件编码: {filepath}")
    except Exception as e:
        print(f"修复文件编码时出错 {filepath}: {e}")

# 修复 game_env.py
fix_file_encoding('src/environment/game_env.py')