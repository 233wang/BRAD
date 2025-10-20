# src/cleaned_dataset.py
# 清洗 dataset/original.code 文件中每行的 <BEG> 和 <END> 标记

from pathlib import Path

# 构造跨目录的路径
input_file = fr'./../dataset/JCSD/valid/similar1.code'
output_file = fr'./../dataset/JCSD/valid/similar2.code'

# 读取和处理文件
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        stripped = line.strip()
        if stripped.startswith('<BEG>'):
            stripped = stripped[len('<BEG>'):]
        if stripped.endswith('<END>'):
            stripped = stripped[:-len('<END>')]
        outfile.write(stripped.strip() + '\n')