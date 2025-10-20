# -*- coding: utf-8 -*-
"""

使用本地部署的 LLaMA 3-8B 模型，对 Java 代码摘要任务结果进行多维度评估。
输入文件：
  - source.code          每行一个 Java 代码片段
  - source.comment       每行一个真实摘要（ground truth）
  - beamSearch_result.3  每行一个模型预测摘要（prediction）
输出文件：
  - evaluation_results.csv  包含 Index、Coherence、Consistency、Fluency、Relevance、Comment
  - detailed_results.log    每条样本的原始模型输出和解析结果

评估维度：
  1. Coherence: 连贯性（确保摘要捕获代码的主要功能和逻辑，而不引入任何附加或不相关的内容）
  2. Consistency: 一致性（摘要是否与代码逻辑自洽）
  3. Fluency: 自然性（语言是否流畅、符合 Java 风格）
  4.Relevance:相关性 专注于代码的业务或功能相关性，确保摘要捕捉到代码在更大的系统或项目中的关键意义。
每个维度分值范围 0-4，并生成简短评语。

依赖： transformers, torch, numpy
"""
import os

# 完全离线模式，必须在导入 transformers/sentence_transformers 之前
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HUGGINGFACE_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import re
import csv
import numpy as np
from transformers import AutoModelForCausalLM
import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from string import Template
# 新增统计库
from scipy.stats import ttest_rel, wilcoxon

# 配置路径
SOURCE_CODE_FILE = './../dataset/JCSD/test/source.code'
GOLD_SUMMARY_FILE = './../dataset/JCSD/test/source.comment'
PRED_SUMMARY_FILE = './../results/JCSD/beamSearch_result.3'
# 基线模型预测摘要文件，用于显著性检验
BASELINE_PRED_FILE = './base_result.3'

# # 配置路径
# SOURCE_CODE_FILE = './../dataset/JCSD/xxx/source.code'
# GOLD_SUMMARY_FILE = './../dataset/JCSD/xxx/source.comment'
# PRED_SUMMARY_FILE = './../dataset/JCSD/xxx/beamSearch_result.3'
# # 基线模型预测摘要文件，用于显著性检验
# BASELINE_PRED_FILE = './../dataset/JCSD/xxx/base_result.3'

OUTPUT_CSV = "OUTPUT_CSV.csv"
BASEOUTPUT_CSV = "BASEOUTPUT_CSV.csv"
DETAILED_LOG = "DETAILED_LOG.log"
BASEDETAILED_LOG = "BASEDETAILED_LOG.log"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 2

# ------------------------- 提示词模板 -------------------------
PROMPT_TEMPLATE = Template('''
Role Image:You are an experienced Java developer and code reviewer.
Task Description: You will be given one summary written for a source code. Your task is to rate the summary from coherence, consistency, fluency and relevance aspects.
Evaluation steps：
1. Read the source code carefully and understand its main intent.
2. Read the code summary and check if it accurately describe the code.
3. Assign a score for coherence, consistency, fluency and relevance on a scale of 0 to 4.

Please rate the predicted summary:
Coherence Dimension (0-4): The summary should be logically orga-nized, with a clear flow of ideas from sentence to sentence, forming a coherent description of the source code.
Consistency Dimension (0-4): The summary must align with the facts within the source code, e.g., specific statements, avoiding unsupported or hallucinated content.
Fluency Dimension (0-4): The summary should be grammatically correct, well-structured, and free from repetition, formatting issues, and capitalization errors that impede readability.
Relevance Dimension (0-4): The summary should capture the essential information from the source code, with penalties for redundancies and excessive details.

Respond strictly in the format:
Coherence: <number ONLY>
Consistency: <number ONLY>
Fluency: <number ONLY>
Relevance: <number ONLY>
Comment: <short evaluation comment>

Here is the sample you need to evaluate. Please strictly follow the output format:
――――――――――――――――――――――――――――
SAMPLE: $idx
Code Snippet:
```java
$code
```
Reference Summary: $gold
Predicted Summary: $pred
――――――――――――――――――――――――――――
''')

SAMPLE_TEMPLATE = '''
SAMPLE: {idx}
Code Snippet:
```java
{code}
```
Reference Summary: {gold}
Predicted Summary: {pred}
'''
Demonstration_BLOCK = '''
Demonstration：
――――――――――――――――――――――――――――
	Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; 
    if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = name ; }
    Reference Summary: creates a new zip entry with the specified name.
    Predicted Summary: creates a new zip entry with the name
    Evaluation Form: 
    Coherence: 4
	Consistency: 4
	Fluency: 4
	Relevance: 4
	Comment: The summary is concise, coherent, fluent, and fully captures the functionality—though you could add “specified” to match the reference more precisely.
――――――――――――――――――――――――――――
	Source Code: public tsactiondelay ( transit section action tsa , int delay ) { tsa = tsa ; delay = delay ; }
    Reference Summary:  a runnable that implements delayed execution of a transitsectionaction
    Predicted Summary:  return a deals which is , a is optionally it not or equal + + equal it not not not not ? a specified , not not . equal can equal ; ; a ; a dispatcher is is , . . . . . 
    Evaluation Form: 
    Coherence: 0
	Consistency: 0
	Fluency: 1
	Relevance: 1
	Comment: The summary is incoherent and inconsistent (word salad), with almost no relevance to the actual constructor; it needs a complete rewrite to state that it initializes a TSActionDelay object by encapsulating the tsa action and the delay parameter.
――――――――――――――――――――――――――――
    Source Code: public process execute async ( final command line command , map < string , string > environment ) throws ioexception { if ( working directory != null && ! working directory exists ( ) ) { throw new ioexception ( working directory + <string> ) ; } return execute internal ( command , environment , working directory , stream handler , null ) ; }
    Reference Summary: methods for starting asynchronous execution .
    Predicted Summary: methods for starting asynchronous execution . process process process process parent parent
    Evaluation Form: 
    Coherence: 2
	Consistency: 2
	Fluency: 2
	Relevance: 3
	Comment: The summary partially captures the idea of starting asynchronous execution but is hampered by repetitive words and omits the working-directory check and exception; it should instead say that it validates the working directory before launching the process asynchronously and returns the corresponding Process object.
――――――――――――――――――――――――――――
'''

def parse_evaluation(text):
    """
    从模型输出文本中提取三个评分和评语。
    返回 (coh, cons, flu, rel, comment)
    """
    def _get(pattern):
        m = re.search(pattern, text)
        return int(m.group(1)) if m and m.group(1).isdigit() else 0  # 默认 0 分

    coh = _get(r"Coherence\s*[:：]\s*([0-4])")
    cons = _get(r"Consistency\s*[:：]\s*([0-4])")
    flu = _get(r"Fluency\s*[:：]\s*([0-4])")
    rel = _get(r"Relevance\s*[:：]\s*([0-4])")
    comment_match = re.search(r"Comment\s*[:：]\s*(.+)", text)
    comment = comment_match.group(1).strip().split('\n')[0] if comment_match else ""

    return coh, cons, flu, rel, comment

def evaluate_predictions(preds, tokenizer, model, logfile, resultfile):
    """
        对一组预测摘要执行批量评估，返回 numpy 数组 shape=(N,4)
        """
    # 加载数据
    with open(SOURCE_CODE_FILE, 'r', encoding='utf-8') as f: codes = [l.rstrip() for l in f]
    with open(GOLD_SUMMARY_FILE, 'r', encoding='utf-8')  as f: golds = [l.rstrip() for l in f]
    total = len(codes)
    results = []
    allresults = []
    with open(resultfile, 'w', newline='', encoding='utf-8') as csvfile, \
        open(logfile, 'w', encoding='utf-8') as logfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['idx', 'Coherence', 'Consistency', 'Fluency', 'Relevance', 'Comment'])
        # 模型和 tokenizer 已在外部加载
        for batch_start in range(0, total, BATCH_SIZE):
            idxs = list(range(batch_start, min(batch_start + BATCH_SIZE, total)))
            prompts = []
            for i in idxs:
                p = PROMPT_TEMPLATE.substitute(
                    idx=i, code=codes[i], gold=golds[i], pred=preds[i]
                )
                if i == 0:
                    p = Demonstration_BLOCK + p
                prompts.append(p)
            # tokenize & record prompt lengths
            inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=4090)
            prompt_lens = [len(ids) for ids in inputs.input_ids]
            inputs = inputs.to(DEVICE)
            # 生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False
                )
            # decode & parse
            for i, plen, out_ids in zip(idxs, prompt_lens, outputs):
                gen_ids = out_ids[plen:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                if not text:
                    logfile.write(f"!!! WARNING: Sample {i} produced EMPTY output\n")
                coh_score, cons_score, flu_score, rel_score, comment = parse_evaluation(text)
                results.append((coh_score, cons_score, flu_score, rel_score))
                allresults.append((i, coh_score, cons_score, flu_score, rel_score, comment))
                csv_writer.writerow([
                    i,
                    coh_score,
                    cons_score,
                    flu_score,
                    rel_score,
                    comment
                ])
                csvfile.flush()
                # 写入详细日志
                logfile.write(f"------------------ 大模型的第{i}次输出 ------------------\n")
                logfile.write(text + "\n\n")
                logfile.flush()

    # ------------------------- 统计分析 -------------------------
    scores = np.array([[r[1], r[2], r[3], r[4]] for r in allresults], dtype=float)
    avg_scores = np.mean(scores, axis=0)
    var_scores = np.var(scores, axis=0)
    print(f"\nOverall Performance:")
    print(f"  Coherence - Mean: {avg_scores[0]:.2f}, Variance: {var_scores[0]:.2f}")
    print(f"  Consistency - Mean: {avg_scores[1]:.2f}, Variance: {var_scores[1]:.2f}")
    print(f"  Fluency - Mean: {avg_scores[2]:.2f}, Variance: {var_scores[2]:.2f}")
    print(f"  Relevance - Mean: {avg_scores[3]:.2f}, Variance: {var_scores[3]:.2f}")

    return np.array(results, dtype=float), allresults

def main():
    # ------------------------- 加载模型 -------------------------
    print("Loading model and tokenizer... This may take a while.")
    # 1. 计算本地模型目录（相对于本脚本 src/ 目录）
    local_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     '../../../../..',
                     'meta-llama',
                     'Llama-3.1-8B-Instruct')
    )
    # 2. 确认目录存在
    assert os.path.isdir(local_dir), f"请检查本地模型目录是否正确：{local_dir}"
    model = LlamaForCausalLM.from_pretrained(local_dir, device_map=None, torch_dtype=torch.float16).to(DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(local_dir, use_fast=True)
    # 1) 明确指定 padding 放左边
    tokenizer.padding_side = 'left'
    # 2) 确保有 pad_token 可用
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        tokenizer.pad_token = '[PAD]'
    print(f"加载模型：{local_dir} 设备：{DEVICE}")

    # 读取两组预测
    with open(PRED_SUMMARY_FILE, 'r', encoding='utf-8') as f:
        preds = [l.rstrip() for l in f]
    with open(BASELINE_PRED_FILE, 'r', encoding='utf-8') as f:
        base_preds = [l.rstrip() for l in f]
    # 评估
    print('Evaluating main model...')
    scores_model, result_model = evaluate_predictions(preds, tokenizer, model, DETAILED_LOG, OUTPUT_CSV)
    print('Evaluating baseline model...')
    scores_base, result_base = evaluate_predictions(base_preds, tokenizer, model, BASEDETAILED_LOG, BASEOUTPUT_CSV)

    # 统计
    dims = ['Coherence','Consistency','Fluency','Relevance']
    print('\nOverall metrics:')
    for j, d in enumerate(dims):
        mean_m = scores_model[:, j].mean()
        mean_b = scores_base[:, j].mean()
        t_stat, p_val = ttest_rel(scores_model[:, j], scores_base[:, j])
        w_stat, w_p = wilcoxon(scores_model[:, j], scores_base[:, j])
        print(f"{d}: Model={mean_m:.2f}, Baseline={mean_b:.2f}, t={t_stat:.3f}, p={p_val:.3e}, w_p={w_p:.3e}")



if __name__ == '__main__':
    main()

