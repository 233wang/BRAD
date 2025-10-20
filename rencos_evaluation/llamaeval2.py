# -*- coding: utf-8 -*-
"""
evaluate_code_summaries.py

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

# 配置路径
SOURCE_CODE_FILE = './../dataset/JCSD/test/source.code'
GOLD_SUMMARY_FILE = './../dataset/JCSD/test/source.comment'
PRED_SUMMARY_FILE = './../results/JCSD/beamSearch_result.3'
OUTPUT_CSV = "evaluation_results.csv"
DETAILED_LOG = "detailed_results.log"
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


    # coh_score = int(coh.group(1)) if coh else None
    # cons_score = int(cons.group(1)) if cons else None
    # flu_score = int(flu.group(1)) if flu else None
    # rel_score = int(rel.group(1)) if rel else None
    # comment_text = comment.group(1).strip() if comment else ""
    return coh, cons, flu, rel, comment


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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"加载模型：{local_dir} 设备：{DEVICE}")

    # 读取所有输入文件
    with open(SOURCE_CODE_FILE,  'r', encoding='utf-8') as f: codes = [l.rstrip() for l in f]
    with open(GOLD_SUMMARY_FILE, 'r', encoding='utf-8') as f: golds = [l.rstrip() for l in f]
    with open(PRED_SUMMARY_FILE, 'r', encoding='utf-8') as f: preds = [l.rstrip() for l in f]
    total = len(codes)
    print(f"共 {total} 个样本，分批次处理，每批 {BATCH_SIZE} 条。")

    # 准备输出
    with open(OUTPUT_CSV,   'w', newline='', encoding='utf-8') as csvfile, \
         open(DETAILED_LOG, 'w', encoding='utf-8') as logfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['idx','Coherence','Consistency','Fluency','Relevance','Comment'])

        # ------------------------- 评估循环 -------------------------
        print("Starting evaluation...")
        results = []

        # 按批次生成
        for batch_start in range(0, total, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total)
            idxs = list(range(batch_start, batch_end))
            print(f"\n>> 处理 batch {batch_start}–{batch_end - 1}")

            # 1) 构造 prompts 列表
            prompts = []
            # prompts.append(PROMPT_TEMPLATE)
            for i in idxs:
                p = PROMPT_TEMPLATE.substitute(
                    idx=i,
                    code=codes[i],
                    gold=golds[i],
                    pred=preds[i],
                )
                # 如果全局索引 i 是 0，就拼一次示例
                if i == 0:
                    p = Demonstration_BLOCK + p
                prompts.append(p)
            # 调试：显示第一个 prompt 开头
            head = prompts[0][:200].replace('\n', '↵')
            print("示例 prompt（前 200 字）：", head + '...')

            # 2) 批量 tokenize
            inputs = tokenizer(
                prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(DEVICE)
            prompt_lens = [len(ids) for ids in inputs.input_ids]
            print("input_ids shape:", inputs.input_ids.shape)

            # 3) 批量生成
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                eos_token_id=tokenizer.eos_token_id,
                temperature=0.2,
                do_sample=False,
            )
            print("生成完成，batch 大小：", len(outputs))

            # 4) 逐条 decode 与解析
            for i, prompt_len, out_ids in zip(idxs, prompt_lens, outputs):
                gen_ids = out_ids[prompt_len:]  # slice 掉 prompt 部分
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                if not text:
                    logfile.write(f"!!! WARNING: Sample {i} produced EMPTY output\n")
                coh_score, cons_score, flu_score, rel_score, comment = parse_evaluation(text)
                results.append((i, coh_score, cons_score, flu_score, rel_score, comment))
                # 写入 CSV
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
                logfile.write(f"--- Sample {i} ---\n")
                logfile.write(text + "\n\n")
                logfile.flush()

        print("\n所有样本处理完成！")
        print("CSV 输出：", OUTPUT_CSV)
        print("日志输出：", DETAILED_LOG)

    # ------------------------- 统计分析 -------------------------
    scores = np.array([[r[1], r[2], r[3], r[4]] for r in results], dtype=float)
    avg_scores = np.mean(scores, axis=0)
    var_scores = np.var(scores, axis=0)
    print(f"\nOverall Performance:")
    print(f"  Coherence - Mean: {avg_scores[0]:.2f}, Variance: {var_scores[0]:.2f}")
    print(f"  Consistency - Mean: {avg_scores[1]:.2f}, Variance: {var_scores[1]:.2f}")
    print(f"  Fluency - Mean: {avg_scores[2]:.2f}, Variance: {var_scores[2]:.2f}")
    print(f"  Relevance - Mean: {avg_scores[3]:.2f}, Variance: {var_scores[3]:.2f}")


if __name__ == '__main__':
    main()

