#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bm25_retrieval_dir.py

数据目录结构：
  data_dir/
    train/
      bytecode.txt
      comment.txt
    valid/
      bytecode.txt
      comment.txt
    test/
      bytecode.txt
      comment.txt

输出：
  train/train.similar.bytecode
  train/train.similar.comment
  valid/valid.similar.bytecode
  valid/valid.similar.comment
  test/test.similar.bytecode
  test/test.similar.comment
"""

import os
import argparse
from rank_bm25 import BM25Okapi, BM25L  # BM25 索引构建库 :contentReference[oaicite:0]{index=0}
from tqdm import tqdm              # 进度条展示库 :contentReference[oaicite:1]{index=1}

DATA_DIR = fr'./../dataset/JCSD/'

def load_lines(path):
    """按行读取文本（去除末尾换行符）"""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]  # open() 阅读文件 :contentReference[oaicite:2]{index=2}

def write_lines(path, lines):
    """按行写入文本（自动创建父目录并添加换行）"""
    os.makedirs(os.path.dirname(path), exist_ok=True)  # 递归创建目录 :contentReference[oaicite:3]{index=3}
    with open(path, 'w', encoding='utf-8') as f:
        for l in lines:
            f.write(l + '\n')  # 写入文件并换行 :contentReference[oaicite:4]{index=4}

def build_bm25_index(train_bytecodes):
    """对训练集字节码分词后构建 BM25 索引"""
    tokenized = [bc.split() for bc in train_bytecodes]        # split() 分词 :contentReference[oaicite:5]{index=5}
    return BM25Okapi(tokenized), tokenized                                # 初始化 BM25 :contentReference[oaicite:6]{index=6}

def retrieve_for_split(bm25, train_bc, train_cm, query_bc, train_sc, split_name):
    """对单个 split 的字节码列表做检索，返回相似 bytecode & comment 列表"""
    sim_bc, sim_cm, sim_sc = [], [], []
    for i, bc in enumerate(tqdm(query_bc, desc=f"检索 {split_name}", ncols=80)):  # tqdm 用于进度条 :contentReference[oaicite:7]{index=7}
        tokens = bc.split()
        top_n = bm25.get_top_n(tokens, train_bc, n=5)
        # 选择第一个不是自身的
        for candidate in top_n:
            if candidate != bc:
                idx = train_bc.index(candidate)
                sim_bc.append(train_bc[idx])
                sim_cm.append(train_cm[idx])
                sim_sc.append(train_sc[idx])
                break
        else:
            # 极端情况：全都相等（极少发生），保留自身
            sim_bc.append(bc)
            sim_cm.append(train_cm[i])
            sim_sc.append(train_sc[i])
    return sim_bc, sim_cm, sim_sc

def main():
    # data_dir = args.data_dir.rstrip('/')  # 格式化目录路径 :contentReference[oaicite:9]{index=9}

    # 1. 加载 train/valid/test 的 bytecode/comment
    train_bc = load_lines(os.path.join(DATA_DIR, 'train', 'cfg_train.txt'))  # os.path.join 拼接路径 :contentReference[oaicite:10]{index=10}
    train_cm = load_lines(os.path.join(DATA_DIR, 'train', 'source.comment'))
    # valid_bc = load_lines(os.path.join(DATA_DIR, 'valid', 'cfg_valid.txt'))
    # valid_cm = load_lines(os.path.join(DATA_DIR, 'valid', 'source.comment'))
    test_bc  = load_lines(os.path.join(DATA_DIR, 'test',  'cfg_test.txt'))
    test_cm  = load_lines(os.path.join(DATA_DIR, 'test',  'source.comment'))

    train_sc = load_lines(os.path.join(DATA_DIR, 'train', 'source.code'))
    test_sc  = load_lines(os.path.join(DATA_DIR, 'test',  'source.code'))
    
    # 2. 构建 BM25 索引（仅用 train）
    bm25, train_tokenized  = build_bm25_index(train_bc)
#     bm25, train_tokenized  = build_bm25_index(train_sc)

    # 3. 分别对三个 split 进行检索
#     train_sim_bc, train_sim_cm = retrieve_for_split(bm25, train_bc, train_cm, train_bc, 'train')
    # valid_sim_bc, valid_sim_cm = retrieve_for_split(bm25, train_bc, train_cm, valid_bc, 'valid')
    test_sim_bc,  test_sim_cm, test_sourcecode  = retrieve_for_split(bm25, train_bc, train_cm, test_bc, train_sc,  'test')

#     test_sim_sc,  test_sim_cm  = retrieve_for_split(bm25, train_sc, train_cm, test_sc,  'test')

    # 4. 写回各自目录
#     write_lines(os.path.join(DATA_DIR, 'train', 'similar.bytecode'), train_sim_bc)
#     write_lines(os.path.join(DATA_DIR, 'train', 'similar.comment' ), train_sim_cm)
    # write_lines(os.path.join(DATA_DIR, 'valid', 'similar.bytecode'), valid_sim_bc)
    # write_lines(os.path.join(DATA_DIR, 'valid', 'similar.comment' ), valid_sim_cm)
    write_lines(os.path.join(DATA_DIR, 'test',  'similar.bytecode' ), test_sim_bc)
    write_lines(os.path.join(DATA_DIR, 'test',  'similar.comment'  ), test_sim_cm)
    write_lines(os.path.join(DATA_DIR, 'test',  'bytesimilar.code'  ), test_sourcecode)

#     write_lines(os.path.join(DATA_DIR, 'test',  'codesimilar.code' ), test_sim_sc)
#     write_lines(os.path.join(DATA_DIR, 'test',  'codesimilar.comment' ), test_sim_cm)
    
    print("检索并写回完成。")  # argparse 用于脚本参数 :contentReference[oaicite:11]{index=11}

if __name__ == "__main__":
    main()
