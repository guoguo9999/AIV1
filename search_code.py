import os

def search_code_in_files(directory, code_snippet):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if code_snippet in content:
                        print(f"找到代码片段在文件: {file_path}")
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")

# 指定要搜索的目录
search_directory = '.'  # 当前目录
# 要搜索的代码片段
code_to_search = """"""

search_code_in_files(search_directory, code_to_search)