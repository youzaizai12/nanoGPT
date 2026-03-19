import os
import tiktoken
import numpy as np

# __file__ 表示当前脚本（prepare.py）的路径，dirname取其所在文件夹
script_dir = os.path.dirname(__file__)
input_file_path = os.path.join(script_dir, 'tianlong.txt')

# 指定UTF-8编码读取文件（解决Windows默认GBK解码失败）
try:
    # 优先用UTF-8编码读取（中文文本最常用）
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
except UnicodeDecodeError:
    # 备用：如果文件是GBK编码，自动切换
    with open(input_file_path, 'r', encoding='gbk') as f:
        data = f.read()

# 原有核心逻辑
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# 用gpt2编码文本为tokens
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)

# 输出tokens数量，验证读取和编码成功
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# 将tokens保存为二进制文件（保存到脚本所在目录）
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(script_dir, 'train.bin'))
val_ids.tofile(os.path.join(script_dir, 'val.bin'))

# 额外提示：确认文件保存成功
print(f"文件已保存到：{script_dir}")
print("生成的文件：train.bin、val.bin")