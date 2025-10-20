import random
import time
import json
import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from dataloader import c2cDataset
from DECOM_model import DECOM
from utils import MaskedSoftmaxCELoss, DAMSMLoss, eval_bleu_rouge_meteor
import os

seed = 12345


def seed_everything(seed=seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def get_loaders(code_word2id, comment_word2id, dataset, max_code_len, max_comment_len, max_keywords_len,
                batch_size=32, num_workers=0, pin_memory=False, use_cfg=False):
    train_set = c2cDataset(code_word2id, comment_word2id, dataset, max_code_len, max_comment_len, max_keywords_len,
                           'train', use_cfg)
    test_set = c2cDataset(code_word2id, comment_word2id, dataset, max_code_len, max_comment_len, max_keywords_len,
                          'test', use_cfg)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=train_set.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             collate_fn=test_set.collate_fn,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, test_loader


def train_FineTune(model, seq2seq_loss, evaluator_loss, dataloader, bos_token, optimizer, epoch, cuda, max_iter_num, use_cfg):
    """
    对模型进行第二阶段微调：在第1阶段训练基础上，
    微调整个模型参数，支持CFG模块和多轮审议。
    返回每轮的平均 loss。
    """
    # losses_list 用于存储每一轮 seq2seq loss 以及 evaluator loss
    losses_list = [[] for _ in range(max_iter_num + 1)]
    model.train()

    seed_everything(seed + epoch)
    for data in tqdm(dataloader, desc=f"Epoch {epoch+1} FineTune"):
        # source_code, comment, template, keywords, source_code_len, comment_len, template_len, keywords_len = \
        #     [d.cuda() for d in data[:8]] if cuda else data[:8]
        # ===== 解包 batch，包括可选的 CFG =====
        if use_cfg:
            data = [d.to('cuda') if isinstance(d, torch.Tensor) else d for d in data]
            (source_code, comment, template, keywords, cfg_adj, cfg_nodes,
             source_code_len, comment_len, template_len, keywords_len, cfg_len,
             ids) = data
        else:
            (source_code, comment, template, keywords,
             source_code_len, comment_len,
             template_len, keywords_len, _ids) = (
                d.cuda() for d in data[:8]
            ) if cuda else data[:8]
            cfg_adj = cfg_nodes = cfg_len = None
        # code_id = data[-1]
        '''wj:新修改代码-start'''
        # ===== 准备 Decoder 的 teacher forcing 输入 =====
        B = comment.size(0)
        bos = torch.full((B, 1), bos_token,
                         dtype=torch.long,
                         device=comment.device)
        # 输入 = <BOS> + 真值评论的前 L-1 个 token
        comment_input = torch.cat([bos, comment[:, :-1]], dim=1)
        comment_input_len = comment_len - 1

        # ===== 前向计算 =====
        optimizer.zero_grad()
        if use_cfg:
            memories, anchor, positive, negative = model(
                source_code, comment_input,
                template, keywords,
                source_code_len, comment_input_len,
                template_len, keywords_len,
                cfg_adj, cfg_nodes, cfg_len
            )
        else:
            memories, anchor, positive, negative = model(
                source_code, comment_input,
                template, keywords,
                source_code_len, comment_input_len,
                template_len, keywords_len
            )

        # ===== 计算损失 =====
        total_loss = None
        # seq2seq 多轮 loss
        for i in range(max_iter_num):
            loss_i = seq2seq_loss(memories[i], comment, comment_len)
            losses_list[i].append(loss_i.item())
            total_loss = loss_i if total_loss is None else total_loss + loss_i
        # evaluator loss 加权
        loss_e = evaluator_loss(anchor, positive, negative) * 0.1
        losses_list[-1].append(loss_e.item())
        total_loss = total_loss + loss_e

        # ===== 反向传播 + 优化 =====
        total_loss.backward()
        optimizer.step()

    # ===== 统计平均 loss =====
    avg_loss = [round(np.mean(lst), 4) if lst else 0.0
                for lst in losses_list]
    '''wj:新修改代码-end'''
    # 原始代码
    #     bos = torch.tensor([bos_token] * comment.size(0), device=comment.device).reshape(-1, 1)
    #     comment_input = torch.cat([bos, comment[:, :-1]], 1)
    #     comment_input_len = torch.add(comment_len, -1)
    #
    #     optimizer.zero_grad()
    #     memory, anchor, positive, negative = model(source_code, comment_input, template, keywords,
    #                                                source_code_len, comment_input_len, template_len, keywords_len)
    #
    #     loss = None
    #     for iter_idx in range(max_iter_num):
    #         loss_idx = seq2seq_loss(memory[iter_idx], comment, comment_len)
    #         losses_list[iter_idx].append(loss_idx.item())
    #         if loss is None:
    #             loss = loss_idx
    #         else:
    #             loss += loss_idx
    #
    #     loss_e = evaluator_loss(anchor, positive, negative) * 0.1
    #     losses_list[-1].append(loss_e.item())
    #
    #     loss += loss_e
    #     # accumulate the grad
    #     loss.backward()
    #     # optimizer the parameters
    #     optimizer.step()
    #
    # avg_loss = [round(np.sum(losses) / len(losses), 4) for losses in losses_list]

    return avg_loss


def evaluate_model(model, dataloader, bos_token, commont_id2word, cuda, max_iter_num, use_cfg):
    losses, comment_reference, ids = [], [], [] 
    comment_prediction = {i: [] for i in range(max_iter_num + 1)}
    model.eval()

    seed_everything(seed)
    with torch.no_grad():
        for data in tqdm(dataloader, desc=f"evaluating"):
            if use_cfg:
                data = [d.to('cuda') if isinstance(d, torch.Tensor) else d for d in data]
                (source_code, comment, template, keywords, cfg_adj, cfg_nodes,
                 source_code_len, comment_len, template_len, keywords_len, cfg_len, code_id) = data
                # code_id = data[-1]
            else:
                # 无CFG情况下与原流程相同
                (source_code, comment, template, keywords,
                 source_code_len, comment_len, template_len, keywords_len) = [d.cuda() for d in
                                                                              data[:8]] if cuda else data[:8]
                cfg_adj = cfg_nodes = cfg_len = None
                code_id = data[-1]

            bos = torch.tensor([bos_token] * len(comment), device=template.device).reshape(-1, 1)
            if use_cfg:
                memories = model(
                    source_code, bos,
                    template, keywords,
                    source_code_len, comment_len,
                    template_len, keywords_len,
                    cfg_adj, cfg_nodes, cfg_len)
            else:
                memories = model(
                    source_code, bos,
                    template, keywords,
                    source_code_len, comment_len,
                    template_len, keywords_len)

            for i in range(len(comment)):
                ref = comment[i]
                comment_reference.append([ref])

                for j, comment_pred in enumerate(memories):
                    if isinstance(comment_pred[i], (int, torch.Tensor)) and not isinstance(comment_pred[i],(list, tuple)):
                        pre = [commont_id2word[int(comment_pred[i])]]
                    else:
                        pre = [commont_id2word[id] for id in comment_pred[i]]
                    # pre = [common_id2word[id] for id in comment_pred[i]]
                    comment_prediction[j].append(pre)

            ids += code_id

    for ii, comment_pred in enumerate(comment_prediction.values()):
        assert len(ids) == len(comment_pred) == len(comment_reference)
        bleu, rouge, meteor, _, _ = eval_bleu_rouge_meteor(ids, comment_pred, comment_reference)
        print(bleu, rouge, meteor)

    return bleu, rouge, meteor, comment_prediction


class Config(object):
    def __init__(self, dataset_config):
        dataset, max_code_len, max_comment_len, max_keywords_len = dataset_config.values()
        self.cuda = True
        self.dataset = dataset
        with open(fr'./../dataset/{dataset}/code.word2id', 'rb') as f:
            code_word2id = pickle.load(f)
        with open(fr'./../dataset/{dataset}/code.id2word', 'rb') as f:
            code_id2word = pickle.load(f)
        with open(fr'./../dataset/{dataset}/comment.word2id', 'rb') as f:
            comment_word2id = pickle.load(f)
        with open(fr'./../dataset/{dataset}/comment.id2word', 'rb') as f:
            comment_id2word = pickle.load(f)
        self.code_word2id = code_word2id
        self.code_id2word = code_id2word
        self.comment_word2id = comment_word2id
        self.comment_id2word = comment_id2word
        self.bos_token = self.comment_word2id['<BOS>']
        self.eos_token = self.comment_word2id['<EOS>']

        self.d_model = 512
        self.d_ff = 2048
        self.head_num = 8
        self.encoder_layer_num = 4
        self.decoder_layer_num = 6
        self.max_code_len = max_code_len
        self.max_comment_len = max_comment_len
        self.max_keywords_len = max_keywords_len
        self.code_vocab_size = len(code_word2id)
        self.comment_vocab_size = len(comment_word2id)
        self.beam_width = 5
        self.lr = 1e-4
        self.fineTune_lr = 1e-5
        self.batch_size = 32
        self.max_iter_num = 3
        self.dropout = 0.2
        self.epochs = 100
        self.clipping_distance = 16
        self.use_cfg = True
        self.fusion_mode = 'concat'


if __name__ == '__main__':
    print("开始第二阶段训练")
    jcsd_config = {'name': 'JCSD', 'max_code_len': 300, 'max_comment_len': 50, 'max_keywords_len': 30}
    pcsd_config = {'name': 'PCSD', 'max_code_len': 100, 'max_comment_len': 50, 'max_keywords_len': 30}
    config = Config(jcsd_config)
    cuda = torch.cuda.is_available() and config.cuda
    if cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    seed_everything(seed)

    model = DECOM(config.d_model, config.d_ff, config.head_num, config.encoder_layer_num,
                  config.decoder_layer_num, config.code_vocab_size, config.comment_vocab_size,
                  config.bos_token, config.eos_token, config.max_comment_len,
                  config.clipping_distance, config.max_iter_num, config.dropout, beam_width=4,
                  use_cfg=config.use_cfg, fusion_mode=config.fusion_mode)

    print("load the best model parameters!")
    model.load_state_dict(torch.load(f"./../saved_model/{config.dataset}/first_step_params.pkl"))
    if cuda:
        model.cuda()

    seq2seq_loss = MaskedSoftmaxCELoss()
    evaluator_loss = DAMSMLoss()

    optimizer = optim.Adam(model.parameters(), lr=config.fineTune_lr)

    print(get_parameter_number(model))
    train_loader, test_loader = get_loaders(config.code_word2id, config.comment_word2id, config.dataset,
                                            config.max_code_len, config.max_comment_len, config.max_keywords_len,
                                            config.batch_size, num_workers=0, pin_memory=False, use_cfg=config.use_cfg)

    last_improve = 0
    best_valid_bleu = 0
    best_test_bleu = 0
    for e in range(config.epochs):
        start_time = time.time()

        train_loss = train_FineTune(model, seq2seq_loss, evaluator_loss, train_loader, config.bos_token,
                                        optimizer, e, cuda, config.max_iter_num, config.use_cfg)
        print('epoch:{},train_loss:{},time:{}sec'.format(e + 1, train_loss, round(time.time() - start_time, 2)))

        if (e + 1) % 5 == 0 or e >= 0:
            # validation
            valid_bleu, valid_rouge, valid_meteor, valid_prediction = \
                evaluate_model(model, test_loader, config.bos_token, config.comment_id2word, cuda, config.max_iter_num,
                               config.use_cfg)

            print('epoch:{},valid_bleu:{},valid_rouge:{},valid_meteor:{},time:{}sec'.
                  format(e + 1, valid_bleu, valid_rouge, valid_meteor, round(time.time() - start_time, 2)))

            if valid_bleu > best_valid_bleu:
                best_valid_bleu = valid_bleu
                last_improve = e
                # save the best model parameters
                torch.save(model.state_dict(), f"./../saved_model/{config.dataset}/second_step_params.pkl")
                # output the prediction of comments for test set
                for ii, comment_pred in enumerate(valid_prediction.values()):
                    with open(f'./../results/{config.dataset}/second_step_result.{ii}', 'w') as w:
                        for comment_list in comment_pred:
                            comment = ' '.join(comment_list)
                            w.write(comment + '\n')

            if e - last_improve >= 20:
                print("No optimization for 20 epochs, auto-stopping and save model parameters")
                break

    print("finish!!!")
    print("best_valid_bleu:", best_valid_bleu)
    print("best_test_bleu:", best_test_bleu)
