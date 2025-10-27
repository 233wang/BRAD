#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
from rank_bm25 import BM25Okapi, BM25L  
from tqdm import tqdm              

DATA_DIR = fr'./../dataset/JCSD/'

def load_lines(path):
    
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]  

def write_lines(path, lines):
    
    os.makedirs(os.path.dirname(path), exist_ok=True)  
    with open(path, 'w', encoding='utf-8') as f:
        for l in lines:
            f.write(l + '\n')  

def build_bm25_index(train_bytecodes):
    
    tokenized = [bc.split() for bc in train_bytecodes]        
    return BM25Okapi(tokenized), tokenized                                

def retrieve_for_split(bm25, train_bc, train_cm, query_bc, train_sc, split_name):
    
    sim_bc, sim_cm, sim_sc = [], [], []
    for i, bc in enumerate(tqdm(query_bc, desc=f"retrieval {split_name}", ncols=80)): 
        tokens = bc.split()
        top_n = bm25.get_top_n(tokens, train_bc, n=5)
        
        for candidate in top_n:
            if candidate != bc:
                idx = train_bc.index(candidate)
                sim_bc.append(train_bc[idx])
                sim_cm.append(train_cm[idx])
                sim_sc.append(train_sc[idx])
                break
        else:
            
            sim_bc.append(bc)
            sim_cm.append(train_cm[i])
            sim_sc.append(train_sc[i])
    return sim_bc, sim_cm, sim_sc

def main():
    train_bc = load_lines(os.path.join(DATA_DIR, 'train', 'cfg_train.txt'))  
    train_cm = load_lines(os.path.join(DATA_DIR, 'train', 'source.comment'))
    # valid_bc = load_lines(os.path.join(DATA_DIR, 'valid', 'cfg_valid.txt'))
    # valid_cm = load_lines(os.path.join(DATA_DIR, 'valid', 'source.comment'))
    test_bc  = load_lines(os.path.join(DATA_DIR, 'test',  'cfg_test.txt'))
    test_cm  = load_lines(os.path.join(DATA_DIR, 'test',  'source.comment'))
    train_sc = load_lines(os.path.join(DATA_DIR, 'train', 'source.code'))
    test_sc  = load_lines(os.path.join(DATA_DIR, 'test',  'source.code'))
    
    bm25, train_tokenized  = build_bm25_index(train_bc)
#     bm25, train_tokenized  = build_bm25_index(train_sc)


#     train_sim_bc, train_sim_cm = retrieve_for_split(bm25, train_bc, train_cm, train_bc, 'train')
    # valid_sim_bc, valid_sim_cm = retrieve_for_split(bm25, train_bc, train_cm, valid_bc, 'valid')
    test_sim_bc,  test_sim_cm, test_sourcecode  = retrieve_for_split(bm25, train_bc, train_cm, test_bc, train_sc,  'test')
#     test_sim_sc,  test_sim_cm  = retrieve_for_split(bm25, train_sc, train_cm, test_sc,  'test')


#     write_lines(os.path.join(DATA_DIR, 'train', 'similar.bytecode'), train_sim_bc)
#     write_lines(os.path.join(DATA_DIR, 'train', 'similar.comment' ), train_sim_cm)
    # write_lines(os.path.join(DATA_DIR, 'valid', 'similar.bytecode'), valid_sim_bc)
    # write_lines(os.path.join(DATA_DIR, 'valid', 'similar.comment' ), valid_sim_cm)
    write_lines(os.path.join(DATA_DIR, 'test',  'similar.bytecode' ), test_sim_bc)
    write_lines(os.path.join(DATA_DIR, 'test',  'similar.comment'  ), test_sim_cm)
    write_lines(os.path.join(DATA_DIR, 'test',  'bytesimilar.code'  ), test_sourcecode)
#     write_lines(os.path.join(DATA_DIR, 'test',  'codesimilar.code' ), test_sim_sc)
#     write_lines(os.path.join(DATA_DIR, 'test',  'codesimilar.comment' ), test_sim_cm)
    print("Finish")

if __name__ == "__main__":
    main()
