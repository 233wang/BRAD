import json
from collections import Counter

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle


class c2cDataset(Dataset):
    def __init__(self, code_word2id, comment_word2id, dataset, max_code_num, max_comment_len, max_keywords_len, file, use_cfg=True):
        self.ids = []
        self.source_code = []
        self.comment = []
        self.template = []
        self.keywords = []
        self.max_code_num = max_code_num
        self.max_comment_len = max_comment_len
        self.max_keywords_len = max_keywords_len
        self.code_word2id = code_word2id
        self.comment_word2id = comment_word2id
        self.file = file

        #wj:增加读取CFG的逻辑
        self.use_cfg = use_cfg
        if self.use_cfg:
            # 读取CFG文件
            cfg_path = f'./../dataset/{dataset}/{file}/source.cfg'
            with open(cfg_path, 'r', encoding='utf-8') as f:
                self.cfg_lines = f.readlines()
            self.cfg_adj = []  # 邻接矩阵列表
            self.cfg_nodes = []  # 每个节点的 token ID 列表
            self.cfg_len = []  # 每个样本的节点数量
        else:
            self.cfg_lines = None


        with open(fr'./../dataset/{dataset}/{file}/source.code', 'r', encoding="ISO-8859-1") as f:
            source_code_lines = f.readlines()
        with open(fr'./../dataset/{dataset}/{file}/source.comment', 'r', encoding="ISO-8859-1") as f:
            comment_lines = f.readlines()
        with open(fr'./../dataset/{dataset}/{file}/similar.comment', 'r', encoding="ISO-8859-1") as f:
            template_lines = f.readlines()
        with open(fr'./../dataset/{dataset}/{file}/source.keywords', 'r', encoding="ISO-8859-1") as f:
            identifier_lines = f.readlines()


        count_id = 0
        if self.use_cfg:
            iterator = zip(source_code_lines, comment_lines, template_lines, identifier_lines, self.cfg_lines)
        else:
            iterator = zip(source_code_lines, comment_lines, template_lines, identifier_lines)
        for idx, items in enumerate(tqdm(iterator)):
            if self.use_cfg:
                code_line, comment_line, template_line, identifier_line, cfg_line = items
            else:
                code_line, comment_line, template_line, identifier_line = items
#         for code_line, comment_line, template_line, identifier_line in tqdm(
#                 zip(source_code_lines, comment_lines, template_lines, identifier_lines)):
            count_id += 1
            self.ids.append(count_id)

            code_token_list = code_line.strip().split(' ')
            source_code_list = [code_word2id[token] if token in code_word2id else code_word2id['<UNK>']
                                for token in code_token_list[:self.max_code_num]]
            self.source_code.append(source_code_list)

            if file != 'test':
                comment_token_list = comment_line.strip().split(' ')
                comment_list = [comment_word2id[token] if token in comment_word2id else comment_word2id['<UNK>']
                                for token in comment_token_list[:self.max_comment_len]] + [comment_word2id['<EOS>']]
                self.comment.append(comment_list)
            else:
                comment_token_list = comment_line.strip().split(' ')
                self.comment.append(comment_token_list)

            template_token_list = template_line.strip().split(' ')
            template_list = [comment_word2id[token] if token in comment_word2id else comment_word2id['<UNK>']
                             for token in template_token_list[:self.max_comment_len]]
            self.template.append(template_list)

            identifier_token_list = identifier_line.strip().split(' ')
            identifier_list = [comment_word2id[token] for token in identifier_token_list
                               if token in comment_word2id]
            self.keywords.append(identifier_list[:self.max_keywords_len])

            # 【新增】处理 CFG
            if self.use_cfg:
                # 1) 解析 JSON
                cfg_obj = json.loads(cfg_line)
                adj_list = cfg_obj['adj']  # 邻接表：list of list
                nodes_list = cfg_obj['nodes']  # 节点 token 序列 list
                # 2) 构建邻接矩阵
                N = len(adj_list)
                mat = [[0] * N for _ in range(N)]
                for i, neigh in enumerate(adj_list):
                    for j in neigh:
                        if 0 <= j < N:
                            mat[i][j] = 1
                self.cfg_adj.append(mat)
                # 3) 将每个节点的 token 列表转为 ID，并收集
                node_ids_batch = []
                for token_seq in nodes_list:
                    # 如果 cfg_line 中 nodes 是字符串，用空格分词
                    tokens = token_seq.strip().split()
                    ids = [code_word2id.get(tok, code_word2id['<UNK>']) for tok in tokens]
                    # ids = [
                    #     code_word2id.get(tok, code_word2id['<UNK>'])
                    #     for tok in token_seq
                    # ]
                    node_ids_batch.append(ids)
                self.cfg_nodes.append(node_ids_batch)
                # 4) 记录节点数
                self.cfg_len.append(N)

    def __getitem__(self, index):
        # return self.source_code[index], \
        #        self.comment[index], \
        #        self.template[index], \
        #        self.keywords[index], \
        #        len(self.source_code[index]), \
        #        len(self.comment[index]), \
        #        len(self.template[index]), \
        #        len(self.keywords[index]), \
        #        self.ids[index]

        #wj: 根据use_cfg决定返回内容
        if not self.use_cfg:
            return self.source_code[index], self.comment[index], self.template[index], self.keywords[index], \
                   len(self.source_code[index]), len(self.comment[index]), len(self.template[index]), len(self.keywords[index]), \
                   self.ids[index]
        else:
            return self.source_code[index], self.comment[index], self.template[index], self.keywords[index], \
                   self.cfg_adj[index], self.cfg_nodes[index], \
                   len(self.source_code[index]), len(self.comment[index]), len(self.template[index]), len(self.keywords[index]), \
                   self.cfg_len[index], self.ids[index]

    def __len__(self):
        return len(self.ids)

    def collate_fn(self, data):

        # dat = pd.DataFrame(data)
        # return_list = []
        # for i in dat:
        #     if i < 4:
        #         if i == 1 and self.file == 'test':
        #             return_list.append(dat[i].tolist())
        #         else:
        #             return_list.append(pad_sequence([torch.tensor(x, dtype=torch.int64) for x in dat[i].tolist()], True))
        #     elif i < 8:
        #         if i == 5 and self.file == 'test':
        #             return_list.append(dat[i].tolist())
        #         else:
        #             return_list.append(torch.tensor(dat[i].tolist()))
        #     else:
        #         return_list.append(dat[i].tolist())
        # return return_list
        # 将data列表转换为DataFrame以逐列处理
        dat = pd.DataFrame(data)
        batch_data = []
        if not self.use_cfg:
            # 原始处理：0-3序列数据padding，4-7长度转tensor，8为id列表
            for i in dat:
                if i < 4:
                    # 序列数据（代码/注释/模板/关键词）
                    if i == 1 and self.file == 'test':
                        batch_data.append(dat[i].tolist())  # 测试集comment保留原文本
                    else:
                        batch_data.append(pad_sequence([torch.tensor(x, dtype=torch.int64)
                                                        for x in dat[i]], batch_first=True))
                elif i < 8:
                    if i == 5 and self.file == 'test':
                        batch_data.append(dat[i].tolist())  # 测试集comment长度用列表
                    else:
                        batch_data.append(torch.tensor(dat[i].tolist(), dtype=torch.int64))
                else:
                    batch_data.append(dat[i].tolist())
        else:
            # 扩展处理包含CFG数据
            # 序列数据索引0-3同上，索引4邻接矩阵，5节点列表，6-9长度，10节点数，11 ids
            # 先处理0-3列（代码/注释/模板/关键词）与未使用CFG时相同
            for i in range(0, 4):
                if i == 1 and self.file == 'test':
                    batch_data.append(dat[i].tolist())
                else:
                    batch_data.append(pad_sequence([torch.tensor(x, dtype=torch.int64)
                                                    for x in dat[i]], batch_first=True))
            # 处理第4列CFG邻接矩阵：转换为tensor并pad到统一大小
            adj_matrices = dat[4].tolist()  # 每项是list of list
            max_nodes = max(len(mat) for mat in adj_matrices)
            # 构造 (batch, max_nodes, max_nodes) 全零tensor
            adj_tensor = torch.zeros((len(adj_matrices), max_nodes, max_nodes), dtype=torch.int64)
            for i, mat in enumerate(adj_matrices):
                N_i = len(mat)
                for r in range(N_i):
                    adj_tensor[i, r, :N_i] = torch.tensor(mat[r], dtype=torch.int64)
            batch_data.append(adj_tensor)
            # 处理第5列CFG节点token列表：pad每个节点的token序列，并pad节点数量
            node_lists = dat[5].tolist()  # 每项是list of node_id_list
            max_nodes = max(len(node_list) for node_list in node_lists)
            max_node_token_len = 0
            for node_list in node_lists:
                for ids in node_list:
                    max_node_token_len = max(max_node_token_len, len(ids))
            # 构造 (batch, max_nodes, max_node_token_len) 的PAD张量
            node_tensor = torch.full((len(node_lists), max_nodes, max_node_token_len),
                                     fill_value=self.code_word2id['<PAD>'], dtype=torch.int64)
            for i, node_list in enumerate(node_lists):
                for j, ids in enumerate(node_list):
                    node_tensor[i, j, :len(ids)] = torch.tensor(ids, dtype=torch.int64)
            batch_data.append(node_tensor)
            # 处理长度相关列：6-9为code/comment/template/keywords长度，10为节点数
            for i in range(6, 11):
                if i == 7 and self.file == 'test':  # 测试集的comment长度
                    batch_data.append(dat[i].tolist())
                else:
                    batch_data.append(torch.tensor(dat[i].tolist(), dtype=torch.int64))
            # 处理ID列
            batch_data.append(dat[11].tolist())
        return batch_data


# class c2cEvalDataset(Dataset):
#     def __init__(self, file, code_word2id, comment_word2id, dataset='rencos_java',
#                  max_code_num=100, max_comment_len=50, max_keywords_len=30):
#         self.ids = []
#         self.source_code = []
#         self.comment = []
#         self.template = []
#         self.keywords = []
#         self.max_code_num = max_code_num
#         self.max_comment_len = max_comment_len
#         self.max_keywords_len = max_keywords_len
#         self.code_word2id = code_word2id
#         self.comment_word2id = comment_word2id
#
#         with open(fr'./dataset/{dataset}/{file}/source.code_original', 'r') as f:
#             source_code_lines = f.readlines()
#         with open(fr'./dataset/{dataset}/{file}/source.comment', 'r') as f:
#             comment_lines = f.readlines()
#         with open(fr'./dataset/{dataset}/{file}/similar.comment', 'r') as f:
#             template_lines = f.readlines()
#         with open(fr'./dataset/{dataset}/{file}/source.identifier', 'r') as f:
#             identifier_lines = f.readlines()
#
#         count_id = 0
#         for code_line, comment_line, template_line, identifier_line in tqdm(
#                 zip(source_code_lines, comment_lines, template_lines, identifier_lines)):
#             count_id += 1
#             self.ids.append(count_id)
#
#             code_token_list = code_line.strip().split(' ')
#             source_code_list = [code_word2id[token] if token in code_word2id else code_word2id['<UNK>']
#                                 for token in code_token_list[:self.max_code_num]]
#             self.source_code.append(source_code_list)
#             # self.source_code.append(source_code_list)
#
#             comment_token_list = comment_line.strip().split(' ')
#             self.comment.append(comment_token_list)
#
#             template_token_list = template_line.strip().split(' ')
#             template_list = [comment_word2id[token] if token in comment_word2id else comment_word2id['<UNK>']
#                              for token in template_token_list[:self.max_comment_len]]
#             self.template.append(template_list)
#             # self.template.append([common_word2id['<PAD>']])
#
#             identifier_token_list = identifier_line.strip().split(' ')
#             identifier_list = [comment_word2id[token] for token in identifier_token_list
#                                if token in comment_word2id]
#             if len(identifier_list) == 0:
#                 print(count_id, identifier_line)
#             self.keywords.append(identifier_list[:self.max_keywords_len])
#
#     def __getitem__(self, index):
#         return self.source_code[index], \
#                self.template[index], \
#                self.keywords[index], \
#                len(self.source_code[index]), \
#                len(self.template[index]), \
#                len(self.keywords[index]), \
#                self.comment[index], \
#                len(self.comment[index]), \
#                self.ids[index]
#
#     def __len__(self):
#         return len(self.ids)
#
#     # dataloader自定义padding操作
#     def collate_fn(self, data):
#         dat = pd.DataFrame(data)
#         return_list = []
#         for i in dat:
#             if i < 3:
#                 return_list.append(pad_sequence([torch.tensor(x) for x in dat[i].tolist()], True))
#             elif i < 6:
#                 return_list.append(torch.tensor(dat[i].tolist()))
#             else:
#                 return_list.append(dat[i].tolist())
#         return return_list
