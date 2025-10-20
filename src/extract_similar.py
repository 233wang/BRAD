#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 code_train.txt / tgt_train.txt 中，基于 TF-IDF + 最近邻检索，
为每条样本找到最相似（但非自身）的源码/注释对，生成 similar.code 和 similar.comment
输出格式与原始项目一致：每行一个 '<BEG> … <END>'。
"""
import os
import sys
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def load_lines(path):
    print(f"🔍 正在读取文件: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f]
    print(f"✅ 读取完成，共 {len(lines)} 行")
    return lines

def wrap_be(l):
    return f"<BEG> {l} <END>"

def main(code_in, comment_in, out_sim_code, out_sim_comment, k=2):
    # 1. 读入
    codes = load_lines(code_in)
    comments = load_lines(comment_in)
    assert len(codes) == len(comments), "❌ 错误：源码和注释行数需一致"
    n_samples = len(codes)
    print(f"📦 样本总数: {n_samples}")

    # 2. TF-IDF 建立向量空间
    print("🛠️  构建 TF-IDF 向量... 请稍候")
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=False)
    X = tfidf.fit_transform(codes)
    print(f"✅ TF-IDF 构建完成，向量形状: {X.shape}")
    print(f"词表大小: {len(tfidf.vocabulary_)}")

    # 3. 最近邻检索（排除自身，取第 2 最近邻）
    print("🔎 开始最近邻检索...")
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X, return_distance=True)
    # 打印前 5 个样本的近邻信息
    print("前 5 个样本的近邻 (排除自身后第1近邻):")
    for i in range(min(5, n_samples)):
        # 第一近邻是自身，取第二
        sim = indices[i][1] if indices[i][0] == i else indices[i][0]
        print(f"  样本 {i} -> 近邻 {sim}，距离 {distances[i][1]:.4f}")

    # 4. 输出 similar.code / similar.comment
    print(f"✍️ 写出文件: {out_sim_code} 和 {out_sim_comment}")
    with open(out_sim_code, 'w', encoding='utf-8') as f_code, \
         open(out_sim_comment, 'w', encoding='utf-8') as f_com:
        for i, nbr_idxs in enumerate(tqdm(indices, desc="写出相似对", ncols=80)):
            sim_idx = nbr_idxs[1] if nbr_idxs[0] == i else nbr_idxs[0]
            f_code.write(wrap_be(codes[sim_idx]) + "\n")
            f_com.write(wrap_be(comments[sim_idx]) + "\n")
            if (i + 1) % 10000 == 0:
                print(f"  已处理 {i+1}/{n_samples} 条")

    print("🎉 完成 similar.code 和 similar.comment 的生成！")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="基于 TF-IDF + 近邻检索构建 similar.code/similar.comment")
    p.add_argument("-i_code",   required=True, help="输入：code_train.txt")
    p.add_argument("-i_com",    required=True, help="输入：tgt_train.txt")
    p.add_argument("-o_simc",   required=True, help="输出：similar.code")
    p.add_argument("-o_simcm",  required=True, help="输出：similar.comment")
    p.add_argument("-k",        type=int, default=2, help="近邻数，默认2（排除自身后取第1近邻）")
    args = p.parse_args()

    main(args.i_code, args.i_com, args.o_simc, args.o_simcm, k=args.k)
