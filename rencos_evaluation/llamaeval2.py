# -*- coding: utf-8 -*-
import os

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


SOURCE_CODE_FILE = './../dataset/JCSD/test/source.code'
GOLD_SUMMARY_FILE = './../dataset/JCSD/test/source.comment'
PRED_SUMMARY_FILE = './../results/JCSD/beamSearch_result.3'
OUTPUT_CSV = "evaluation_results.csv"
DETAILED_LOG = "detailed_results.log"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 2


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


def main():
    print("Loading model and tokenizer... This may take a while.")

    local_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     '../../../../..',
                     'meta-llama',
                     'Llama-3.1-8B-Instruct')
    )
    
    assert os.path.isdir(local_dir), f"please cheack：{local_dir}"
    model = LlamaForCausalLM.from_pretrained(local_dir, device_map=None, torch_dtype=torch.float16).to(DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(local_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"load model：{local_dir} DEVICE:{DEVICE}")

    
    with open(SOURCE_CODE_FILE,  'r', encoding='utf-8') as f: codes = [l.rstrip() for l in f]
    with open(GOLD_SUMMARY_FILE, 'r', encoding='utf-8') as f: golds = [l.rstrip() for l in f]
    with open(PRED_SUMMARY_FILE, 'r', encoding='utf-8') as f: preds = [l.rstrip() for l in f]
    total = len(codes)

    with open(OUTPUT_CSV,   'w', newline='', encoding='utf-8') as csvfile, \
         open(DETAILED_LOG, 'w', encoding='utf-8') as logfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['idx','Coherence','Consistency','Fluency','Relevance','Comment'])

        print("Starting evaluation...")
        results = []

        for batch_start in range(0, total, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total)
            idxs = list(range(batch_start, batch_end))

            prompts = []
            for i in idxs:
                p = PROMPT_TEMPLATE.substitute(
                    idx=i,
                    code=codes[i],
                    gold=golds[i],
                    pred=preds[i],
                )
                if i == 0:
                    p = Demonstration_BLOCK + p
                prompts.append(p)
            head = prompts[0][:200].replace('\n', '↵')

            inputs = tokenizer(
                prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(DEVICE)
            prompt_lens = [len(ids) for ids in inputs.input_ids]
            print("input_ids shape:", inputs.input_ids.shape)

            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                eos_token_id=tokenizer.eos_token_id,
                temperature=0.2,
                do_sample=False,
            )

            for i, prompt_len, out_ids in zip(idxs, prompt_lens, outputs):
                gen_ids = out_ids[prompt_len:] 
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                if not text:
                    logfile.write(f"!!! WARNING: Sample {i} produced EMPTY output\n")
                coh_score, cons_score, flu_score, rel_score, comment = parse_evaluation(text)
                results.append((i, coh_score, cons_score, flu_score, rel_score, comment))

                csv_writer.writerow([
                    i,
                    coh_score,
                    cons_score,
                    flu_score,
                    rel_score,
                    comment
                ])
                csvfile.flush()

                logfile.write(f"------------------ LLM{i}output： ------------------\n")
                logfile.write(f"--- Sample {i} ---\n")
                logfile.write(text + "\n\n")
                logfile.flush()

        print("\nFinish！")
        print("CSV output：", OUTPUT_CSV)
        print("log output", DETAILED_LOG)


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

