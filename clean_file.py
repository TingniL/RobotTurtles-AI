def clean_file(filepath):
    try:
        # 读取文件内容
        with open(filepath, 'rb') as f:
            content = f.read()
        
        # 移除所有的空字节
        cleaned_content = content.replace(b'\x00', b'')
        
        # 写回文件
        with open(filepath, 'wb') as f:
            f.write(cleaned_content)
            
        print(f"已清理文件: {filepath}")
    except Exception as e:
        print(f"清理文件时出错 {filepath}: {e}")

# 清理 game_env.py
clean_file('src/environment/game_env.py')