#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_code_train_to_keywords.py

针对预分词+带 <BEG>/<END> 标记的 code_train.txt 数据集，
批量生成与原始项目格式完全一致的 source.keywords 文件，
每行由空格分隔的“子标识符”列表，无任何 JSON、无占位。
"""

import os
import re
import argparse
from tqdm import tqdm

# 拆分 CamelCase 与 snake_case 的正则：在小写-大写边界或下划线处拆分
_SPLIT_TOKEN = re.compile(r'(?<=[a-z])(?=[A-Z])|_+')

# 要过滤掉的 Java 保留字 & 常见类型名（全部小写比较）
JAVA_KEYWORDS = {
    'abstract','assert','boolean','break','byte','case','catch','char','class','const','continue',
    'default','do','double','else','enum','extends','final','finally','float','for','goto','if',
    'implements','import','instanceof','int','interface','long','native','new','package','private',
    'protected','public','return','short','static','strictfp','super','switch','synchronized','this',
    'throw','throws','transient','try','void','volatile','while',
    # 常见类型名称
    'string','object','list','map','set','array','integer','boolean'
}

# 允许的标识符模式：字母/下划线开头，后面字母数字下划线
_VALID_IDENT = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

def split_subtokens(tok: str) -> list:
    """
    将一个原始标识符按 CamelCase 或 snake_case 拆分成更小的子 token 并小写。
    例如： "getUser_Name" -> ["get","user","name"]
    """
    parts = _SPLIT_TOKEN.split(tok)
    return [p.lower() for p in parts if p and not p.isdigit()]

def extract_keywords_from_tokens(tokens: list) -> list:
    """
    从一行 tokens（已经 split()）中，过滤出合规的标识符，
    再拆分子 token、去重、保持原序，返回列表。
    """
    seen = set()
    keywords = []
    for tok in tokens:
        # 1) 基本合法性：符合标识符模式
        if not _VALID_IDENT.match(tok):
            continue
        # 2) 过滤保留字 / 类型名
        low = tok.lower()
        if low in JAVA_KEYWORDS:
            continue
        # 3) 拆分子 token
        for sub in split_subtokens(tok):
            if sub not in seen:
                seen.add(sub)
                keywords.append(sub)
    return keywords

def process_file(in_path: str, out_path: str):
    """
    逐行读取输入文件 in_path（带 <BEG> 及 <END> 标记），
    提取 keywords，写到 out_path，每行空格分隔。
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # 统计行数，方便 tqdm
    total = sum(1 for _ in open(in_path, 'r', encoding='utf-8'))
    with open(in_path, 'r', encoding='utf-8') as fin, \
         open(out_path, 'w', encoding='utf-8') as fout:
        for line_no, line in enumerate(tqdm(fin, total=total, desc="转换 keywords"), 1):
            raw = line.strip()
            if not raw:
                fout.write("\n")
                continue
            # 去掉 <BEG> 和 <END>
            raw = raw.replace("<BEG>", "").replace("<END>", "").strip()
            # 分词（code_train.txt 已经是空格分词）
            tokens = raw.split()
            # 提取、拆分、去重
            kws = extract_keywords_from_tokens(tokens)
            # 写出：与原项目 source.keywords 格式完全一致
            fout.write(" ".join(kws) + "\n")
    print(f"✅ 完成：{in_path} → {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="将 code_train.txt 批量转换为与原项目格式一致的 source.keywords"
    )
    p.add_argument("-i","--input",  required=True,
                   help="输入 code_train.txt 文件路径")
    p.add_argument("-o","--output", required=True,
                   help="输出 source.keywords 文件路径")
    args = p.parse_args()
    process_file(args.input, args.output)
