#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from tqdm import tqdm

_SPLIT_TOKEN = re.compile(r'(?<=[a-z])(?=[A-Z])|_+')

JAVA_KEYWORDS = {
    'abstract','assert','boolean','break','byte','case','catch','char','class','const','continue',
    'default','do','double','else','enum','extends','final','finally','float','for','goto','if',
    'implements','import','instanceof','int','interface','long','native','new','package','private',
    'protected','public','return','short','static','strictfp','super','switch','synchronized','this',
    'throw','throws','transient','try','void','volatile','while',
    'string','object','list','map','set','array','integer','boolean'
}

_VALID_IDENT = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

def split_subtokens(tok: str) -> list:
    parts = _SPLIT_TOKEN.split(tok)
    return [p.lower() for p in parts if p and not p.isdigit()]

def extract_keywords_from_tokens(tokens: list) -> list:
    seen = set()
    keywords = []
    for tok in tokens:
        if not _VALID_IDENT.match(tok):
            continue
        low = tok.lower()
        if low in JAVA_KEYWORDS:
            continue
        for sub in split_subtokens(tok):
            if sub not in seen:
                seen.add(sub)
                keywords.append(sub)
    return keywords

def process_file(in_path: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    total = sum(1 for _ in open(in_path, 'r', encoding='utf-8'))
    with open(in_path, 'r', encoding='utf-8') as fin, \
         open(out_path, 'w', encoding='utf-8') as fout:
        for line_no, line in enumerate(tqdm(fin, total=total, desc="trans keywords"), 1):
            raw = line.strip()
            if not raw:
                fout.write("\n")
                continue
            raw = raw.replace("<BEG>", "").replace("<END>", "").strip()
            tokens = raw.split()
            kws = extract_keywords_from_tokens(tokens)
            fout.write(" ".join(kws) + "\n")
    print(f"✅ finish：{in_path} → {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="trans code_train.txt to source.keywords"
    )
    p.add_argument("-i","--input",  required=True,
                   help="input code_train.txt dir")
    p.add_argument("-o","--output", required=True,
                   help="output source.keywords dir")
    args = p.parse_args()
    process_file(args.input, args.output)
