import math

import torch
from torch import nn
import torch.nn.functional as F
import operator
from queue import PriorityQueue
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


def transpose_qkv(X, num_heads):
    """Transposition for parallel computation of multiple attention heads.

    Defined in :numref:`sec_multihead-attention`"""
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3)

    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`.

    Defined in :numref:`sec_multihead-attention`"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis.

    Defined in :numref:`sec_attention-scoring-functions`"""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
#     if valid_lens is None:
#         return nn.functional.softmax(X, dim=-1)
#     else:
#         shape = X.shape
#         if valid_lens.dim() == 1:
#             valid_lens = torch.repeat_interleave(valid_lens, shape[1])
#         else:
#             valid_lens = valid_lens.reshape(-1)
#         # On the last axis, replace masked elements with a very large negative
#         # value, whose exponentiation outputs 0
#         X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
#         return nn.functional.softmax(X.reshape(shape), dim=-1)

    """在最后一维做 softmax，同时屏蔽掉超出 valid_lens 的位置。"""
    # X: （batch_heads, Q, K）
    if valid_lens is None:
        return torch.softmax(X, dim=-1)
    shape = X.shape  # (batch_heads, Q, K)
    # valid_lens 可以是 1D: (batch_heads,) 或 2D: (batch_heads, Q)
    if valid_lens.dim() == 1:
        # (batch_heads,) -> (batch_heads*Q,)
        valid_lens = valid_lens.repeat_interleave(shape[1])
    else:
        # (batch_heads, Q) -> (batch_heads*Q,)
        valid_lens = valid_lens.reshape(-1)
    # 均一化到 2D
    X_flat = X.reshape(-1, shape[-1])  # (batch_heads*Q, K)
    valid_lens = valid_lens.to(X_flat.device)  # 防止 device 不一致
    # 屏蔽
    X_masked = sequence_mask(X_flat, valid_lens, value=-1e6)
    return torch.softmax(X_masked.reshape(shape), dim=-1)


def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.

    Defined in :numref:`sec_seq2seq_decoder`"""
#     maxlen = X.size(1)
#     mask = torch.arange((maxlen), dtype=torch.float32,
#                         device=X.device)[None, :] < valid_len[:, None]
#     X[~mask] = value
#     return X
    """
    对 X 进行序列掩码，将超过 valid_len 的部分替换为 value。
    X: shape = [B, T]
    valid_len: shape = [B]，表示每一行的有效长度
    """
    maxlen = X.size(1)  # 第二维度作为掩码上限
    mask = torch.arange(maxlen, device=X.device)[None, :].expand(X.size(0), maxlen)
    valid_len = valid_len[:, None].expand_as(mask)
    X[mask >= valid_len] = value
    return X


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFFN, self).__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.W_2(self.dropout(F.relu(self.W_1(x))))


class AddNorm(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        return self.norm(x + self.dropout(y))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, d_model))
        x = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / \
            torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
        self.P[:, :, 0::2] = torch.sin(x)
        self.P[:, :, 1::2] = torch.cos(x)

    def forward(self, x):
        x = x + self.P[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=100):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x, pos):
        # x -> batch * seq * dim
        # pos -> batch * seq
        x = x + self.pos_embedding(pos)
        return self.dropout(x)


class KeywordsEncoding(nn.Module):
    def __init__(self, d_model, dropout, keywords_type=6):
        super(KeywordsEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.type_embedding = nn.Embedding(keywords_type, d_model)

    def forward(self, x, keywords_type):
        x = x + self.type_embedding(keywords_type)
        return self.dropout(x)


class MultiHeadAttentionWithRPR(nn.Module):
    """Multi-head attention.

    Defined in :numref:`sec_multihead-attention`"""

    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, clipping_distance,
                 dropout, bias=False, **kwargs):
        super(MultiHeadAttentionWithRPR, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention_rpr = DotProductAttentionWithRPR(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.relative_pos_v = nn.Embedding(2 * clipping_distance + 1, num_hiddens // num_heads)
        self.relative_pos_k = nn.Embedding(2 * clipping_distance + 1, num_hiddens // num_heads)
        self.clipping_distance = clipping_distance

    def forward(self, queries, keys, values, valid_lens):
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        # Shape of `valid_lens`:
        # (`batch_size`,) or (`batch_size`, no. of queries)
        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        # relative position matrix
        range_queries = torch.arange(queries.size(1), device=queries.device)
        range_keys = torch.arange(keys.size(1), device=keys.device)
        distance_mat = range_keys[None, :] - range_queries[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.clipping_distance, self.clipping_distance) + \
                               self.clipping_distance
        # pos_k, pos_v -> seq_q * seq_k * dim
        pos_k = self.relative_pos_k(distance_mat_clipped)
        pos_v = self.relative_pos_v(distance_mat_clipped)

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for
            # `num_heads` times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries,
        # `num_hiddens` / `num_heads`)
        output = self.attention_rpr(queries, keys, values, pos_k, pos_v, valid_lens)

        # Shape of `output_concat`:
        # (`batch_size`, no. of queries, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class DotProductAttentionWithRPR(nn.Module):
    """Scaled dot product attention.

    Defined in :numref:`subsec_additive-attention`"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttentionWithRPR, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def forward(self, queries, keys, values, pos_k, pos_v, valid_lens=None):
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = torch.bmm(queries, keys.transpose(1, 2))
        scores_pos = torch.bmm(queries.transpose(0, 1), pos_k.transpose(1, 2)).transpose(0, 1)
        scores = (scores + scores_pos) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        output = torch.bmm(self.dropout(self.attention_weights), values)
        output_pos = torch.bmm(self.dropout(self.attention_weights.transpose(0, 1)), pos_v).transpose(0, 1)
        return output + output_pos


class MultiHeadAttention(nn.Module):
    """Multi-head attention.

    Defined in :numref:`sec_multihead-attention`"""

    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        # Shape of `valid_lens`:
        # (`batch_size`,) or (`batch_size`, no. of queries)
        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for
            # `num_heads` times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries,
        # `num_hiddens` / `num_heads`)
        output = self.attention(queries, keys, values, valid_lens)

        # Shape of `output_concat`:
        # (`batch_size`, no. of queries, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class DotProductAttention(nn.Module):
    """Scaled dot product attention.

    Defined in :numref:`subsec_additive-attention`"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, head_num, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, d_model, d_model, d_model, head_num, dropout)
        self.feedForward = PositionWiseFFN(d_model, d_ff, dropout)
        self.add_norm1 = AddNorm(d_model)
        self.add_norm2 = AddNorm(d_model)

    def forward(self, x, valid_lens):
        y = self.self_attention(x, x, x, valid_lens)
        y = self.add_norm1(x, y)
        z = self.feedForward(y)
        z = self.add_norm2(y, z)
        return z


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, head_num, N=6, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        # self.embedding = nn.Embedding(vocab_size, d_model)
        # self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([EncoderBlock(d_model, d_ff, head_num, dropout) for _ in range(N)])

    def forward(self, x, valid_lens):
        # encoder编码时,输入x是CNN输出的embedding
        # x = self.pos_encoding(self.embedding(x) * math.sqrt(self.d_model))
        for layer in self.layers:
            x = layer(x, valid_lens)

        return x


class EncoderBlockWithRPR(nn.Module):
    def __init__(self, d_model, d_ff, head_num, clipping_distance, dropout=0.1):
        super(EncoderBlockWithRPR, self).__init__()
        self.self_attention = MultiHeadAttentionWithRPR(d_model, d_model, d_model, d_model, head_num,
                                                        clipping_distance, dropout)
        self.feedForward = PositionWiseFFN(d_model, d_ff, dropout)
        self.add_norm1 = AddNorm(d_model)
        self.add_norm2 = AddNorm(d_model)

    def forward(self, x, valid_lens):
        y = self.self_attention(x, x, x, valid_lens)
        y = self.add_norm1(x, y)
        z = self.feedForward(y)
        z = self.add_norm2(y, z)
        return z


class EncoderWithRPR(nn.Module):
    def __init__(self, d_model, d_ff, head_num, clipping_distance, N=6, dropout=0.1):
        super(EncoderWithRPR, self).__init__()
        self.d_model = d_model
        # self.embedding = nn.Embedding(vocab_size, d_model)
        # self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList(
            [EncoderBlockWithRPR(d_model, d_ff, head_num, clipping_distance, dropout) for _ in range(N)])

    def forward(self, x, valid_lens):
        # encoder编码时,输入x是CNN输出的embedding
        # x = self.pos_encoding(self.embedding(x) * math.sqrt(self.d_model))
        for layer in self.layers:
            x = layer(x, valid_lens)

        return x

# -----------------------
# 修改: DecoderBlockWithKeywords 增加 CFG 融合门控
# -----------------------
class DecoderBlockWithKeywords(nn.Module):
    def __init__(self, i, d_model, d_ff, head_num, dropout=0.1, use_cfg=False, fusion_mode = 'concat'):
        super(DecoderBlockWithKeywords, self).__init__()
        self.i = i
        self.use_cfg = use_cfg
        self.fusion_mode = fusion_mode
        self.masked_self_attention = MultiHeadAttention(d_model, d_model, d_model, d_model, head_num, dropout)
        self.cross_attention_code = MultiHeadAttention(d_model, d_model, d_model, d_model, head_num, dropout)
        self.cross_attention_template = MultiHeadAttention(d_model, d_model, d_model, d_model, head_num, dropout)
        self.cross_attention_keywords = MultiHeadAttention(d_model, d_model, d_model, d_model, head_num, dropout)
        # self.gate = nn.Linear(d_model + d_model, 1)
        if self.use_cfg and self.fusion_mode == 'gate':
            self.cross_attention_cfg = MultiHeadAttention(d_model, d_model, d_model, d_model, head_num, dropout)
            self.gate_cg = nn.Linear(d_model * 2, 1)
            self.gate_ck = nn.Linear(d_model * 2, 1)
        else:
            self.gate = nn.Linear(d_model * 2, 1)
        self.feedForward = PositionWiseFFN(d_model, d_ff, dropout)
        self.add_norm1 = AddNorm(d_model)
        self.add_norm2 = AddNorm(d_model)
        self.add_norm3 = AddNorm(d_model)
        self.add_norm4 = AddNorm(d_model)

    def forward(self, x, state):
        source_code_enc, source_code_len = state[0], state[1]
        template_enc, template_len = state[2], state[3]
        keywords_enc, keywords_len = state[4], state[5]

        idx_offset = 6
        if self.use_cfg and self.fusion_mode == 'gate':
            cfg_enc, cfg_len = state[6], state[7]
            idx_offset = 8
        memories = state[idx_offset]
        prev = memories[self.i]
        if prev is None:
            key_values = x
        else:
#             key_values = torch.cat((state[6][self.i], x), axis=1)
            key_values = torch.cat((prev, x), dim=1)
        # state[6][self.i] = key_values
        memories[self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = x.shape
            dec_valid_lens = torch.arange(1, num_steps + 1, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # 1. self attention
        x2 = self.masked_self_attention(x, key_values, key_values, dec_valid_lens)
        y = self.add_norm1(x, x2)
        # 2. cross attention
        # 跨注意力: 代码 vs (代码+CFG) vs 关键词
        y2_code = self.cross_attention_code(y, source_code_enc, source_code_enc, source_code_len)
        if self.use_cfg and self.fusion_mode == 'gate':
            y2_cfg = self.cross_attention_cfg(y, cfg_enc, cfg_enc, cfg_len)
            w_cg = torch.sigmoid(self.gate_cg(torch.cat([y2_code, y2_cfg], dim=-1)))
            y2_cg = w_cg * y2_code + (1 - w_cg) * y2_cfg
            y2_code = y2_cg
            y2_keyword = self.cross_attention_keywords(y, keywords_enc, keywords_enc, keywords_len)
            gate_weight = torch.sigmoid(self.gate_ck(torch.cat([y2_code, y2_keyword], dim=-1)))
            y2 = gate_weight * y2_code + (1 - gate_weight) * y2_keyword
        else:
            y2_keyword = self.cross_attention_keywords(y, keywords_enc, keywords_enc, keywords_len)
            gate_weight = torch.sigmoid(self.gate(torch.cat([y2_code, y2_keyword], dim=-1)))
            y2 = gate_weight * y2_code + (1. - gate_weight) * y2_keyword
        z = self.add_norm2(y, y2)
        # 3. cross attention
        z2 = self.cross_attention_template(z, template_enc, template_enc, template_len)
        # z2 = z2_keywords + z2_template
        z_end = self.add_norm3(z, z2)
        return self.add_norm4(z_end, self.feedForward(z_end)), state


class DecoderWithKeywords(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, head_num, N=6, dropout=0.1, use_cfg=False, fusion_mode= 'concat'):
        super(DecoderWithKeywords, self).__init__()
        self.use_cfg = use_cfg
        self.fusion_mode = fusion_mode
        self.num_layers = N
        self.d_model = d_model
        self.layers = nn.ModuleList(
            [DecoderBlockWithKeywords(i, d_model, d_ff, head_num, dropout, use_cfg, fusion_mode) for i in range(self.num_layers)])
        self.dense = nn.Linear(d_model, vocab_size)

    def init_state(self, source_code_enc, source_code_len, template_enc, template_len,
                   keywords_enc, keywords_len):
        state = [source_code_enc, source_code_len, template_enc, template_len, keywords_enc, keywords_len]
        if self.use_cfg and self.fusion_mode == 'gate':
            state += [cfg_enc, cfg_len]
        state += [[None] * self.num_layers]
        return state

    def forward(self, x, state):
        for layer in self.layers:
            x, state = layer(x, state)
        return self.dense(x), state


class CodeEncoder(nn.Module):
    def __init__(self, code_embedding, d_model, d_ff, head_num, encoder_layer_num, clipping_distance, dropout=0.1):
        super(CodeEncoder, self).__init__()
        self.code_embedding = code_embedding
        self.code_encoder = EncoderWithRPR(d_model, d_ff, head_num, clipping_distance, encoder_layer_num, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source_code, source_code_len):
        source_code_embed = self.dropout(self.code_embedding(source_code))
        source_code_enc = self.code_encoder(source_code_embed, source_code_len)

        return source_code_enc, source_code_len


class KeywordsEncoder(nn.Module):
    def __init__(self, comment_embedding, d_model, d_ff, head_num, encoder_layer_num, dropout=0.1):
        super(KeywordsEncoder, self).__init__()
        self.comment_embedding = comment_embedding
        self.keywords_encoder = Encoder(d_model, d_ff, head_num, encoder_layer_num, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, keywords, keywords_len):
        keywords_embed = self.dropout(self.comment_embedding(keywords))
        keywords_enc = self.keywords_encoder(keywords_embed, keywords_len)

        return keywords_enc, keywords_len


class TemplateEncoder(nn.Module):
    def __init__(self, comment_embedding, pos_encoding, d_model, d_ff, head_num, encoder_layer_num, dropout=0.1):
        super(TemplateEncoder, self).__init__()
        self.comment_embedding = comment_embedding
        self.pos_encoding = pos_encoding
        self.template_encoder = Encoder(d_model, d_ff, head_num, encoder_layer_num, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, template, template_len):
        b_, seq_template_num = template.size()
        template_pos = torch.arange(seq_template_num, device=template.device).repeat(b_, 1)
        template_embed = self.pos_encoding(self.comment_embedding(template), template_pos)
        template_enc = self.template_encoder(template_embed, template_len)

        return template_enc, template_len


class Evaluator(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(Evaluator, self).__init__()
        self.linear_proj1 = nn.Linear(d_model, d_ff)
        self.linear_proj2 = nn.Linear(d_ff, d_model)
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source_code_enc, source_code_len, comment_enc, comment_len, template_enc, template_len,
                best_result=None, cur_result=None):
        b_ = source_code_enc.size(0)
        # global mean pooling
        source_code_vec = torch.cumsum(source_code_enc, dim=1)[torch.arange(b_), source_code_len - 1]
        source_code_vec = torch.div(source_code_vec.T, source_code_len).T
        source_code_vec = self.linear_proj2(self.dropout(F.relu(self.linear_proj1(source_code_vec))))

        comment_vec = torch.cumsum(comment_enc, dim=1)[torch.arange(b_), comment_len - 1]
        comment_vec = torch.div(comment_vec.T, comment_len).T
        comment_vec = self.linear_proj2(self.dropout(F.relu(self.linear_proj1(comment_vec))))

        template_vec = torch.cumsum(template_enc, dim=1)[torch.arange(b_), template_len - 1]
        template_vec = torch.div(template_vec.T, template_len).T
        template_vec = self.linear_proj2(self.dropout(F.relu(self.linear_proj1(template_vec))))

        if self.training:
            return source_code_vec, comment_vec, template_vec
        else:
            best_enc = [x for x in template_enc]
            best_len = template_len

            assert best_result is not None
            assert cur_result is not None
            pos_sim = self.cos_sim(source_code_vec, comment_vec)
            neg_sim = self.cos_sim(source_code_vec, template_vec)
            better_index = (pos_sim > neg_sim).nonzero(as_tuple=True)[0]
            for ix in better_index:
                best_result[ix] = cur_result[ix]
                best_enc[ix] = comment_enc[ix]
                best_len[ix] = comment_len[ix]

            return best_result, pad_sequence(best_enc, batch_first=True), best_len


class KeywordsGuidedDecoder(nn.Module):
    def __init__(self, comment_embedding, pos_encoding, d_model, d_ff, head_num, decoder_layer_num, comment_vocab_size,
                 bos_token, eos_token, max_comment_len, dropout=0.1, use_cfg=False, fusion_mode='concat', beam_width=4):
        super(KeywordsGuidedDecoder, self).__init__()
        self.comment_embedding = comment_embedding
        self.pos_encoding = pos_encoding
        self.use_cfg = use_cfg
        self.fusion_mode =fusion_mode
        use_cfg_for_decoder = True if (use_cfg and fusion_mode == 'gate') else False
        self.comment_decoder = DecoderWithKeywords(comment_vocab_size, d_model, d_ff, head_num, decoder_layer_num, dropout, use_cfg=self.use_cfg, fusion_mode=self.fusion_mode)

        self.max_comment_len = max_comment_len
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.d_model = d_model
        self.beam_width = beam_width
        self.num_layers = decoder_layer_num
        
        
    def init_state(self,
                       source_code_enc, source_code_len,
                       template_enc, template_len,
                       keywords_enc, keywords_len,
                       cfg_enc=None, cfg_len=None):
        state = [source_code_enc, source_code_len, template_enc, template_len, keywords_enc, keywords_len]
        if self.use_cfg and self.fusion_mode == 'gate':
            state += [cfg_enc, cfg_len]
        state += [[None] * self.num_layers]
        return state

    def forward(self, source_code_enc, comment, template_enc, keywords_enc,
                source_code_len, template_len, keywords_len, cfg_enc=None, cfg_len=None):
        b_, seq_comment_num = comment.size()
        if self.use_cfg and self.fusion_mode == 'gate':
            dec_state = self.comment_decoder.init_state(
                source_code_enc, source_code_len,
                template_enc, template_len,
                keywords_enc, keywords_len,
                cfg_enc, cfg_len)
        else:
            dec_state = self.comment_decoder.init_state(source_code_enc, source_code_len,
                                                    template_enc, template_len,
                                                    keywords_enc, keywords_len)

        if self.training:
            comment_pos = torch.arange(seq_comment_num, device=comment.device).repeat(b_, 1)
            comment_embed = self.pos_encoding(self.comment_embedding(comment), comment_pos)

            comment_pred = self.comment_decoder(comment_embed, dec_state)[0]
#             print(f"[CHECK] beam/greeed search 输出：{comment_pred}")
            return comment_pred
        else:
            if self.beam_width:
                return self.beam_search(b_, comment, dec_state, self.beam_width)
            else:
                return self.greed_search(b_, comment, dec_state)

    def greed_search(self, batch_size, comment, dec_state):
        comment_pred = [[self.bos_token] for _ in range(batch_size)]
        for pos_idx in range(self.max_comment_len):
            comment_pos = torch.arange(pos_idx, pos_idx + 1, device=comment.device).repeat(batch_size, 1)
            comment_embed = self.pos_encoding(self.comment_embedding(comment), comment_pos)
            comment, dec_state = self.comment_decoder(comment_embed, dec_state)
            comment = torch.argmax(comment, -1).detach()
            for i in range(batch_size):
                if comment_pred[i][-1] != self.eos_token:
                    comment_pred[i].append(int(comment[i]))

        comment_pred = [x[1:-1] if x[-1] == self.eos_token and len(x) > 2 else x[1:]
                        for x in comment_pred]
        return comment_pred

    def beam_search(self, batch_size, comment, dec_state, beam_width):
        # comment -> batch * 1
        # first node
        node_list = []
        batchNode_dict = {i: None for i in range(beam_width)}  # 每个时间步都只保留beam_width个node
        # initialization
        for batch_idx in range(batch_size):
            node_comment = comment[batch_idx].unsqueeze(0)
            if self.use_cfg and self.fusion_mode == 'gate':
                node_dec_state = [dec_state[0][batch_idx].unsqueeze(0), dec_state[1][batch_idx].unsqueeze(0),
                                  dec_state[2][batch_idx].unsqueeze(0), dec_state[3][batch_idx].unsqueeze(0),
                                  dec_state[4][batch_idx].unsqueeze(0), dec_state[5][batch_idx].unsqueeze(0),
                                  dec_state[6][batch_idx].unsqueeze(0),  # cfg_enc for this sample
                                  dec_state[7][batch_idx].unsqueeze(0),  # cfg_len for this sample
                                  [None] * self.num_layers
                                 ]
            else:
                node_dec_state = [dec_state[0][batch_idx].unsqueeze(0), dec_state[1][batch_idx].unsqueeze(0),
                                  dec_state[2][batch_idx].unsqueeze(0), dec_state[3][batch_idx].unsqueeze(0),
                                  dec_state[4][batch_idx].unsqueeze(0), dec_state[5][batch_idx].unsqueeze(0),
                                  [None] * self.num_layers]
            node_list.append(BeamSearchNode(node_dec_state, None, node_comment, 0, 0))
        batchNode_dict[0] = BatchNodeWithKeywords(node_list)

        # start beam search
        pos_idx = 0
        while pos_idx < self.max_comment_len:
            beamNode_dict = {i: PriorityQueue() for i in range(batch_size)}
            count = 0
            for idx in range(beam_width):
                if batchNode_dict[idx] is None:
                    continue

                batchNode = batchNode_dict[idx]
                # comment -> batch * 1
                comment = batchNode.get_comment()
                dec_state = batchNode.get_dec_state()

                # decode for one step using decoder
                pos = torch.arange(pos_idx, pos_idx + 1, device=comment.device).repeat(batch_size, 1)
                # comment -> batch * d_model
                comment = self.pos_encoding(self.comment_embedding(comment), pos)
                tensor, dec_state = self.comment_decoder(comment, dec_state)
                tensor = F.log_softmax(tensor.squeeze(1), -1).detach()
                # PUT HERE REAL BEAM SEARCH OF TOP
                # log_prob, comment_candidates -> batch * beam_width
                log_prob, comment_candidates = torch.topk(tensor, beam_width, dim=-1)

                for batch_idx in range(batch_size):
                    pre_node = batchNode.list_node[batch_idx]
                    if self.use_cfg and self.fusion_mode == 'gate':
                        node_dec_state = [dec_state[0][batch_idx].unsqueeze(0), dec_state[1][batch_idx].unsqueeze(0),
                                          dec_state[2][batch_idx].unsqueeze(0), dec_state[3][batch_idx].unsqueeze(0),
                                          dec_state[4][batch_idx].unsqueeze(0), dec_state[5][batch_idx].unsqueeze(0),
                                          dec_state[6][batch_idx].unsqueeze(0), dec_state[7][batch_idx].unsqueeze(0),
                                          [l[batch_idx].unsqueeze(0) for l in dec_state[6]]]
                    else:
                        node_dec_state = [dec_state[0][batch_idx].unsqueeze(0), dec_state[1][batch_idx].unsqueeze(0),
                                          dec_state[2][batch_idx].unsqueeze(0), dec_state[3][batch_idx].unsqueeze(0),
                                          dec_state[4][batch_idx].unsqueeze(0), dec_state[5][batch_idx].unsqueeze(0),
                                          [l[batch_idx].unsqueeze(0) for l in dec_state[6]]]
                    if pre_node.history_word[-1] == self.eos_token:
                        new_node = BeamSearchNode(node_dec_state, pre_node.prevNode, pre_node.commentID,
                                                  pre_node.logp, pre_node.leng)
                        # check
                        assert new_node.score == pre_node.score
                        assert new_node.history_word == pre_node.history_word
                        beamNode_dict[batch_idx].put((-new_node.score, count, new_node))
                        count += 1
                        continue

                    for beam_idx in range(beam_width):
                        node_comment = comment_candidates[batch_idx][beam_idx].view(1, -1)
                        node_log_prob = float(log_prob[batch_idx][beam_idx])
                        new_node = BeamSearchNode(node_dec_state, pre_node, node_comment, pre_node.logp + node_log_prob,
                                                  pre_node.leng + 1)
                        beamNode_dict[batch_idx].put((-new_node.score, count, new_node))
                        count += 1

            for beam_idx in range(beam_width):
                node_list = [beamNode_dict[batch_idx].get()[-1] for batch_idx in range(batch_size)]
                batchNode_dict[beam_idx] = BatchNodeWithKeywords(node_list)

            pos_idx += 1
        # the first batchNode in batchNode_dict is the best node
        best_node = batchNode_dict[0]
        comment_pred = []
        for batch_idx in range(batch_size):
            history_word = best_node.list_node[batch_idx].history_word
            if history_word[-1] == self.eos_token and len(history_word) > 2:
                comment_pred.append(history_word[1:-1])
            else:
                comment_pred.append(history_word[1:])

        return comment_pred

'''wj: 新增基于图注意力网络的CFG编码器--开始'''
# -----------------------
# 新增: CFGEncoder (基于GAT)
# -----------------------
class CFGEncoder(nn.Module):
    def __init__(self, code_embedding, d_model, num_heads=4, num_layers=2, dropout=0.1):
        """
        基于图注意力网络的CFG编码器。
        :param code_embedding: 用于将节点token映射为embedding的共享嵌入层
        :param d_model: 节点特征和输出向量的维度
        :param num_heads: 多头注意力的头数
        :param num_layers: 图注意力层数
        :param dropout: 丢弃率
        """
        super(CFGEncoder, self).__init__()
        self.code_embedding = code_embedding  # 共享代码嵌入层
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        # 图注意力层的参数：每一层为多头注意力，以下简化实现为单层，多头通过参数重复实现
        # 为每个注意力头定义线性变换和注意力系数参数
        self.W = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(d_model, d_model // num_heads, bias=False)
                for _ in range(num_heads)
            ])
            for _ in range(num_layers)
        ])
        self.attn_a_src = nn.ParameterList([nn.Parameter(torch.zeros(size=(d_model // num_heads, 1)))
                                            for _ in range(num_heads * num_layers)])
        self.attn_a_dst = nn.ParameterList([nn.Parameter(torch.zeros(size=(d_model // num_heads, 1)))
                                            for _ in range(num_heads * num_layers)])

        # 3. 多层残差后归一化
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])
        # 4. Dropout 用于 attention 权重和特征输出
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # 参数初始化
        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier 初始化每个头的线性映射
        for layer_w in self.W:
            for head_lin in layer_w:
                nn.init.xavier_uniform_(head_lin.weight)
        # 初始化注意力向量
        for param in self.attn_a_src:
            nn.init.xavier_uniform_(param.data)
        for param in self.attn_a_dst:
            nn.init.xavier_uniform_(param.data)

    def forward(self, cfg_adj, cfg_nodes, cfg_len):
        """
        :param cfg_adj: 张量(batch_size, N, N)，CFG的邻接矩阵，1表示存在边（控制流），0表示无边
        :param cfg_nodes: 张量(batch_size, N, L)，每个节点的token序列ID（已填充PAD到长度L）
        :param cfg_len: 张量(batch_size)，每个样本的节点数量（不含PAD的节点）
        :return: cfg_enc: 张量(batch_size, N, d_model)，每个节点的编码表示
        """
        # 1. 将节点的token序列转换为初始节点向量表示
        # 使用共享的代码嵌入层对节点tokens编码，然后对节点内tokens取平均得到节点初始特征
        batch_size, max_nodes, max_token_len = cfg_nodes.size()
        # shape (batch_size, max_nodes, max_token_len, d_model)
        node_token_embed = self.code_embedding(cfg_nodes)
        # 对每个节点的token维度取平均，得到节点向量 (batch_size, max_nodes, d_model)
        node_feat = torch.mean(node_token_embed, dim=2)
        node_feat = self.dropout(node_feat)  # dropout防止过拟合

        # 2. 图注意力层迭代更新节点表示
        # 为简洁起见，这里将num_layers层的GAT展开在一起，每层num_heads注意力头
        # cfg_enc = node_feat  # 初始节点表示
        # for layer in range(self.num_layers):
        #     # 本层的输出初始化
        #     new_node_feats = []
        #     # 对于每个注意力头，独立计算新的节点特征，然后在最后Concat
        #     for head in range(self.num_heads):
        #         head_index = layer * self.num_heads + head
        #         Wh = self.W[head_index](cfg_enc)  # 对节点特征做线性变换 (batch, N, d_model/num_heads)
        #         # 计算注意力系数 e_ij = LeakyReLU(a^T [Wh_i || Wh_j]) 对于每条边(i,j)
        #         # 先计算 a_src^T * Wh_i 和 a_dst^T * Wh_j
        #         a_src = self.attn_a_src[head_index]  # (d_model/num_heads, 1)
        #         a_dst = self.attn_a_dst[head_index]  # (d_model/num_heads, 1)
        #         # e_i = a_src^T * Wh  -> shape (batch, N, 1)
        #         e_i = torch.matmul(Wh, a_src)
        #         # e_j = a_dst^T * Wh  -> shape (batch, N, 1)
        #         e_j = torch.matmul(Wh, a_dst)
        #         # 利用广播求 e_ij = LeakyReLU(e_i + e_j^T)
        #         # 先将 e_i 扩展为 (batch, N, N), e_j 扩展为 (batch, N, N)
        #         e_i_expand = e_i.expand(-1, -1, cfg_enc.size(1))
        #         e_j_expand = e_j.expand(-1, cfg_enc.size(1), -1)
        #         e = F.leaky_relu(e_i_expand + e_j_expand, negative_slope=0.2)  # (batch, N, N)
        #         # Mask: 对无连接的节点对赋极小值以防影响softmax（邻接矩阵为0的位置）
        #         # 包括超出实际节点数的填充节点也mask掉
        #         # 创建mask: 有边或同一节点(i==j)的为0，否则为-inf
        #         attn_mask = torch.where(cfg_adj > 0, torch.zeros_like(e), torch.full_like(e, float('-inf')))
        #         # (可选)加入自环，即允许节点关注自身: 将对角线位置设为0（确保不被mask）
        #         eye_mask = torch.eye(cfg_enc.size(1), device=cfg_adj.device).unsqueeze(0).expand(batch_size, -1, -1)
        #         attn_mask = torch.where(eye_mask==1, torch.zeros_like(attn_mask), attn_mask)
        #         # 计算注意力权重 α_ij = softmax(e_ij) 在每个节点i的邻居j上
        #         attn_weights = F.softmax(e + attn_mask, dim=-1)  # (batch, N, N)
        #         # 对注意力权重应用dropout
        #         attn_weights = self.dropout(attn_weights)
        #         # 计算加权和: h_i' = sum_j α_ij * Wh_j
        #         h_prime = torch.bmm(attn_weights, Wh)  # (batch, N, d_model/num_heads)
        #         new_node_feats.append(h_prime)
        #     # 多头输出拼接
        #     cfg_enc = torch.cat(new_node_feats, dim=-1)  # 恢复维度 (batch, N, d_model)

        # 把原名映射到内部变量
        adj_matrix = cfg_adj
        node_lens = cfg_len

        h = node_feat  # 初始节点特征

        # 遍历每一层 GAT
        for layer_idx in range(self.num_layers):
            head_outputs = []
            for head_idx in range(self.num_heads):
                # 全局索引：layer * num_heads + head
                idx = layer_idx * self.num_heads + head_idx

                # 线性映射到 head 维度
                Wh = self.W[layer_idx][head_idx](h)  # (batch, N, head_dim)

                # 计算注意力系数 e_ij
                # Broadcasting:  a_src^T Wh_i + a_dst^T Wh_j
                a_src = self.attn_a_src[idx]
                a_dst = self.attn_a_dst[idx]
                # Wh shape: (batch, N, head_dim)
                # e_ij = LeakyReLU(Wh_i @ a_src + Wh_j @ a_dst)
                Wh_i = Wh.unsqueeze(2)  # (batch, N, 1, head_dim)
                Wh_j = Wh.unsqueeze(1)  # (batch, 1, N, head_dim)
                e = F.leaky_relu((Wh_i @ a_src).squeeze(-1)
                                 + (Wh_j @ a_dst).squeeze(-1),
                                 negative_slope=0.2)

                # 用 adj_matrix 屏蔽无边连接
                I = torch.eye(adj_matrix.size(-1), device=adj_matrix.device).unsqueeze(0)
                adj_matrix = (adj_matrix + I).clamp(max=1)
                mask = (adj_matrix == 0)
                e = e.masked_fill(mask, float('-inf'))

                # 软max 得到注意力权重
                alpha = F.softmax(e, dim=-1)
                alpha = self.attn_dropout(alpha)

                # 聚合邻居特征
                h_prime = alpha @ Wh  # (batch, N, head_dim)
                head_outputs.append(h_prime)

            # 拼接所有 heads 输出
            h_cat = torch.cat(head_outputs, dim=-1)  # (batch, N, d_model)
            h_cat = self.out_dropout(h_cat)

            # 残差连接 + LayerNorm
            h = self.norms[layer_idx](h + h_cat)
        return h, node_lens
'''wj: 新增基于图注意力网络的CFG编码器--结束'''

class DECOM(nn.Module):
    def __init__(self, d_model, d_ff, head_num, encoder_layer_num, decoder_layer_num, code_vocab_size,
                 comment_vocab_size, bos_token, eos_token, max_comment_len, clipping_distance, max_iter_num,
                 dropout=0.1, beam_width=4, use_cfg=False, fusion_mode='concat'):
        super(DECOM, self).__init__()

        self.code_embedding = nn.Embedding(code_vocab_size, d_model)
        self.comment_embedding = nn.Embedding(comment_vocab_size, d_model)
        self.pos_encoding = LearnablePositionalEncoding(d_model, dropout, max_comment_len + 2)
        self.code_encoder = CodeEncoder(self.code_embedding, d_model, d_ff, head_num,
                                        encoder_layer_num, clipping_distance, dropout)
        self.keyword_encoder = KeywordsEncoder(self.comment_embedding, d_model, d_ff, head_num,
                                               encoder_layer_num, dropout)
        self.template_encoder = TemplateEncoder(self.comment_embedding, self.pos_encoding, d_model, d_ff, head_num,
                                                encoder_layer_num, dropout)
        # wj:【新增】如果开启CFG功能，初始化CFG编码器
        self.use_cfg = use_cfg
        self.fusion_mode = fusion_mode
        self.bos_token = bos_token
        self.eos_token = eos_token
        if self.use_cfg:
            # 使用共享的code_embedding，将CFG节点编码为与代码相同维度的向量
            self.cfg_encoder = CFGEncoder(self.code_embedding, d_model, num_heads=head_num, num_layers=1,
                                          dropout=dropout)
        # 解码器和评价器初始化保持不变
        self.deliberation_dec = nn.ModuleList(
            [KeywordsGuidedDecoder(self.comment_embedding, self.pos_encoding, d_model, d_ff, head_num,
                                   decoder_layer_num, comment_vocab_size, bos_token, eos_token, max_comment_len,
                                   dropout, self.use_cfg, self.fusion_mode, beam_width) if _ == 0 else
             KeywordsGuidedDecoder(self.comment_embedding,
                                   self.pos_encoding, d_model,
                                   d_ff, head_num,
                                   decoder_layer_num,
                                   comment_vocab_size,
                                   bos_token, eos_token,
                                   max_comment_len,
                                   dropout, self.use_cfg, self.fusion_mode, None)
             for _ in range(max_iter_num)])
        self.evaluator = Evaluator(d_model, d_ff, dropout)
        self.max_iter_num = max_iter_num

    def forward(self, source_code, comment, template, keywords,
                source_code_len, comment_len, template_len, keywords_len, cfg_adj=None, cfg_nodes=None, cfg_len=None):
        # 1. 编码源代码、关键词、相似示例（模板）
        src_enc, src_len = self.code_encoder(source_code, source_code_len)
        kw_enc, kw_len = self.keyword_encoder(keywords, keywords_len)
        tmpl_enc, tmpl_len = self.template_encoder(template, template_len)
        # 2. 如果使用CFG，则编码CFG
        if self.use_cfg:
            # 利用CFGEncoder获取图节点表示
            cfg_enc, cfg_len = self.cfg_encoder(cfg_adj, cfg_nodes, cfg_len)  # shape (batch, N, d_model)
        else:
            cfg_enc, cfg_len = None, None
        # 3. 融合策略
        if self.use_cfg and self.fusion_mode == 'concat':
            src_comb = torch.cat([src_enc, cfg_enc], dim=1)
            len_comb = src_len + cfg_len
            dec_src, dec_src_len = src_comb, len_comb
        elif self.use_cfg and self.fusion_mode == 'cross':
            cross = MultiHeadAttention(d_model, d_model, d_model, d_model, head_num, dropout)
            dec_src = cross(src_enc, cfg_enc, cfg_enc, cfg_len)
            dec_src_len = src_len
        else:
            dec_src, dec_src_len = src_enc, src_len

        # 训练 vs 推理
        batch = comment.size(0)
        bos = torch.full((batch, 1), self.bos_token, dtype=torch.long, device=comment.device)
        # teacher forcing 输入
        comment_in = torch.cat([bos, comment[:, :-1]], dim=1)
        # 4. 解码阶段
        if self.training:
            comment_enc, comment_len = self.template_encoder(comment[:, 1:], comment_len)
            anchor, positive, negative = self.evaluator(dec_src, dec_src_len, comment_enc, comment_len, tmpl_enc, tmpl_len)
            if self.use_cfg and self.fusion_mode == 'gate':
                memory = []
                for iter_idx in range(self.max_iter_num):
                    comment_pred = self.deliberation_dec[iter_idx](dec_src, comment, tmpl_enc,
                                                                   kw_enc, dec_src_len, tmpl_len, kw_len, cfg_enc, cfg_len)
                    memory.append(comment_pred)
                    if iter_idx == self.max_iter_num - 1:
                        return memory, anchor, positive, negative
                    template = torch.argmax(comment_pred.detach(), -1)
                    tmpl_len = comment_len
                    tmpl_enc, tmpl_len = self.template_encoder(template, tmpl_len)
            else:
                memory = []
                for iter_idx in range(self.max_iter_num):
                    comment_pred = self.deliberation_dec[iter_idx](dec_src, comment, tmpl_enc,
                                                                   kw_enc, dec_src_len, tmpl_len, kw_len)
                    memory.append(comment_pred)
                    if iter_idx == self.max_iter_num - 1:
                        return memory, anchor, positive, negative
                    template = torch.argmax(comment_pred.detach(), -1)
                    tmpl_len = comment_len
                    tmpl_enc, tmpl_len = self.template_encoder(template, tmpl_len)
        #推理
        else:
            if self.use_cfg and self.fusion_mode == 'gate':
                memory = []
                best_result = [x.tolist()[:leng] for x, leng in zip(template, tmpl_len)]
                best_enc = tmpl_enc
                best_len = tmpl_len
                for iter_idx in range(self.max_iter_num):
                    comment_pred = self.deliberation_dec[iter_idx](dec_src, comment, tmpl_enc, kw_enc,
                                                               dec_src_len, tmpl_len, kw_len, cfg_enc, cfg_len)
                    memory.append(comment_pred)
                    template = pad_sequence([torch.tensor(x, device=comment.device) for x in comment_pred],
                                            batch_first=True)
                    tmpl_len = torch.tensor([len(x) for x in comment_pred], device=comment.device)
                    tmpl_enc, tmpl_len = self.template_encoder(template, tmpl_len)
                    best_result, best_enc, best_len = self.evaluator(dec_src, dec_src_len, tmpl_enc,
                                                                     tmpl_len, best_enc, best_len, best_result,
                                                                     comment_pred)
                memory.append(best_result)
                assert len(memory) == self.max_iter_num + 1
                return memory
            else:
                memory = []
                best_result = [x.tolist()[:leng] for x, leng in zip(template, tmpl_len)]
                best_enc = tmpl_enc
                best_len = tmpl_len
                for iter_idx in range(self.max_iter_num):
                    comment_pred = self.deliberation_dec[iter_idx](dec_src, comment, tmpl_enc, kw_enc,
                                                               dec_src_len, tmpl_len, kw_len)
                    memory.append(comment_pred)
                    template = pad_sequence([torch.tensor(x, device=comment.device) for x in comment_pred],
                                            batch_first=True)
                    tmpl_len = torch.tensor([len(x) for x in comment_pred], device=comment.device)
                    tmpl_enc, tmpl_len = self.template_encoder(template, tmpl_len)
                    best_result, best_enc, best_len = self.evaluator(dec_src, dec_src_len, tmpl_enc,
                                                                     tmpl_len, best_enc, best_len, best_result,
                                                                     comment_pred)
                memory.append(best_result)
                assert len(memory) == self.max_iter_num + 1
                return memory


class BatchNode(object):
    def __init__(self, list_node):
        self.list_node = list_node

    def get_comment(self):
        comment_list = [node.commentID for node in self.list_node]
        return torch.cat(comment_list, dim=0)

    def get_dec_state(self):
        dec_state_list = [node.dec_state for node in self.list_node]
        batch_dec_state = [torch.cat([batch_state[0] for batch_state in dec_state_list], dim=0),
                           torch.cat([batch_state[1] for batch_state in dec_state_list], dim=0)]
        if dec_state_list[0][2][0] is None:
            batch_dec_state.append(dec_state_list[0][2])
        else:
            state_3 = []
            for i in range(len(dec_state_list[0][2])):
                state_3.append(torch.cat([batch_state[2][i] for batch_state in dec_state_list], dim=0))
            assert len(state_3) == len(dec_state_list[0][2])
            batch_dec_state.append(state_3)
        return batch_dec_state

    def if_allEOS(self, eos_token):
        for node in self.list_node:
            if node.history_word[-1] != eos_token:
                return False
        return True


class BatchNodeWithKeywords(object):
    def __init__(self, list_node):
        self.list_node = list_node

    def get_comment(self):
        comment_list = [node.commentID for node in self.list_node]
        return torch.cat(comment_list, dim=0)

    def get_dec_state(self):
        dec_state_list = [node.dec_state for node in self.list_node]
        # 1) 拼接基础编码部分（前6项总是存在）
        batch_dec_state = [torch.cat([batch_state[0] for batch_state in dec_state_list], dim=0),
                           torch.cat([batch_state[1] for batch_state in dec_state_list], dim=0),
                           torch.cat([batch_state[2] for batch_state in dec_state_list], dim=0),
                           torch.cat([batch_state[3] for batch_state in dec_state_list], dim=0),
                           torch.cat([batch_state[4] for batch_state in dec_state_list], dim=0),
                           torch.cat([batch_state[5] for batch_state in dec_state_list], dim=0)]
        # batch_dec_state = [torch.cat([batch_state[i] for batch_state in dec_state_list], dim=0) for i in range(6)]
         # 2) 如果有CFG部分（state长度大于7），拼接cfg_enc和cfg_len
        if len(dec_state_list[0]) > 7:
            batch_dec_state.append(torch.cat([batch_state[6] for batch_state in dec_state_list], dim=0))  # cfg_encs
            batch_dec_state.append(torch.cat([batch_state[7] for batch_state in dec_state_list], dim=0))  # cfg_lens
            memory_index = 8
        else:
            memory_index = 6
        # 3) 拼接memory列表。注意memory_list本身是一个包含num_layers个张量的list
        if dec_state_list[0][memory_index][0] is None:
            batch_dec_state.append(dec_state_list[0][memory_index][:])
        else:
            state_3 = []
            for i in range(len(dec_state_list[0][memory_index])):
                state_3.append(torch.cat([batch_state[memory_index][i] for batch_state in dec_state_list], dim=0))
            assert len(state_3) == len(dec_state_list[0][memory_index])
            batch_dec_state.append(state_3)
        return batch_dec_state


class BeamSearchNode(object):
    def __init__(self, dec_state, previousNode, commentID, logProb, length, length_penalty=1):
        '''
        :param dec_state:
        :param previousNode:
        :param commentID:
        :param logProb:
        :param length:
        '''
        self.dec_state = dec_state
        self.prevNode = previousNode
        self.commentID = commentID
        self.logp = logProb
        self.leng = length
        self.length_penalty = length_penalty
        if self.prevNode is None:
            self.history_word = [int(commentID)]
            self.score = -100
        else:
            self.history_word = previousNode.history_word + [int(commentID)]
            self.score = self.eval()

    def eval(self):
        return self.logp / self.leng ** self.length_penalty
