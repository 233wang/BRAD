# -*- coding: utf-8 -*-
from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from meteor.meteor import Meteor
import numpy as np
import sys
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

SCsim_address = './../dataset/JCSD/test/codesimilar.comment'
BYsim_address = './../dataset/JCSD/test/similar.comment'
REF_address = './../dataset/JCSD/test/source.comment'
SC_address = './../dataset/JCSD/test/source.code'
SCsimCode_address = './../dataset/JCSD/test/codesimilar.code'
BYsimCode_address = './../dataset/JCSD/test/bytesimilar.code'

def calculate(hyp, ref, len):
    with open(hyp, 'r') as r:
        hypothesis = r.readlines()
        res = {k: [" ".join(v.strip().lower().split()[:len])] for k, v in enumerate(hypothesis)}
    with open(ref, 'r') as r:
        references = r.readlines()
        gts = {k: [v.strip().lower()] for k, v in enumerate(references)}

    score_Bleu, scores_Bleu = Bleu(4).compute_score(gts, res)
    print("Bleu_1: "), np.mean(scores_Bleu[0])
    print("Bleu_2: "), np.mean(scores_Bleu[1])
    print("Bleu_3: "), np.mean(scores_Bleu[2])
    print("Bleu_4: "), np.mean(scores_Bleu[3])

    score_Meteor, scores_Meteor = Meteor().compute_score(gts, res)
    print("Meteor: "), score_Meteor

    score_Rouge, scores_Rouge = Rouge().compute_score(gts, res)
    print("ROUGe: "), score_Rouge

    score_Cider, scores_Cider = Cider().compute_score(gts, res)
    print("Cider: "), score_Cider

def decide_better_method(metric_name, score_code, score_byte):
    """
    根据指定的指标名称判断哪种检索方式得分更高。
    参数:
        metric_name (str): 指标名，如 "BLEU", "METEOR", "ROUGE", "CIDEr"
        score_code (float): 基于源代码检索的摘要得分
        score_byte (float): 基于字节码检索的摘要得分
    返回:
        better_method (str): "Code", "Bytecode", 或 "Equal"
        score_diff (float): 两者得分差异（取绝对值）
    """
    if score_code > score_byte:
        return "Code", score_code - score_byte
    elif score_code < score_byte:
        return "Bytecode", score_byte - score_code
    else:
        return "Equal", 0.0

def main():
    # 1. 加载输入文件内容
    try:
        with open(SCsim_address, 'r') as f:
            code_sim_lines = [line.strip() for line in f]
        with open(BYsim_address, 'r') as f:
            byte_sim_lines = [line.strip() for line in f]
        with open(REF_address, 'r') as f:
            ref_lines = [line.strip() for line in f]
        with open(SC_address, 'r') as f:
            source_lines = [line.strip() for line in f]
        with open(SCsimCode_address, 'r') as f:
            sc_simcode_lines = [line.strip() for line in f]
        with open(BYsimCode_address, 'r') as f:
            by_simcode_lines = [line.strip() for line in f]
    except IOError as e:
        print("文件未找到: {}".format(e))
        sys.exit(1)
    except Exception as e:
        print("加载文件时出错: {}".format(e))
        sys.exit(1)

    # 确认所有文件的行数相同
    N = len(ref_lines)
    if not (len(code_sim_lines) == len(byte_sim_lines) == len(source_lines) == N):
        print("错误: 输入文件的行数不一致，请检查数据文件。")
        sys.exit(1)

    # 2. 计算每个样本的指标得分（BLEU-4, METEOR, ROUGE-L, CIDEr）
    # 准备参考摘要字典和两个候选摘要字典
    max_len = 50  # 截断候选摘要的最大词数
    gts = {}  # 参考摘要（字典形式）
    res_code = {}  # 源代码检索摘要候选
    res_byte = {}  # 字节码检索摘要候选
    for i in range(N):
        # 转小写、按空格切分并截断词数，然后拼接回字符串
        gts[i] = [ref_lines[i].strip().lower()]
        res_code[i] = [" ".join(code_sim_lines[i].strip().lower().split()[:max_len])]
        res_byte[i] = [" ".join(byte_sim_lines[i].strip().lower().split()[:max_len])]

    # 计算评价指标分数
    # BLEU（同时计算1-4阶，取BLEU-4作为主要指标）
    bleu = Bleu(4)
    score_bleu_code, scores_bleu_code = bleu.compute_score(gts, res_code)
    score_bleu_byte, scores_bleu_byte = bleu.compute_score(gts, res_byte)
    # METEOR
    meteor = Meteor()
    score_meteor_code, scores_meteor_code = meteor.compute_score(gts, res_code)
    score_meteor_byte, scores_meteor_byte = meteor.compute_score(gts, res_byte)
    # ROUGE-L
    rouge = Rouge()
    score_rouge_code, scores_rouge_code = rouge.compute_score(gts, res_code)
    score_rouge_byte, scores_rouge_byte = rouge.compute_score(gts, res_byte)
    # CIDEr
    cider = Cider()
    score_cider_code, scores_cider_code = cider.compute_score(gts, res_code)
    score_cider_byte, scores_cider_byte = cider.compute_score(gts, res_byte)

    # 3. 比较每个样本两种摘要的得分，判断哪种方法更优，并计算得分差异（以 METEOR 分数为标准）
    results = []  # 保存所有样本的评估结果
    better_by_code = []  # 源代码检索摘要更优的样本索引列表
    better_by_byte = []  # 字节码检索摘要更优的样本索引列表

    for i in range(N):
        # 各指标分数（BLEU-4、METEOR、ROUGE-L、CIDEr）
        bleu4_code = scores_bleu_code[3][i]  # BLEU1-4列表中索引3为BLEU-4
        bleu4_byte = scores_bleu_byte[3][i]
        meteor_code = scores_meteor_code[i]
        meteor_byte = scores_meteor_byte[i]
        rouge_code = scores_rouge_code[i]
        rouge_byte = scores_rouge_byte[i]
        cider_code = scores_cider_code[i]
        cider_byte = scores_cider_byte[i]

        better_method, score_diff = decide_better_method("BLEU", bleu4_code, bleu4_byte)
        if better_method == "Code":
            better_by_code.append(i)
        elif better_method == "Bytecode":
            better_by_byte.append(i)

        # 保存该样本的结果（保留四舍五入到4位的小数，以便阅读）
        results.append([
            i + 1,
            source_lines[i],
            ref_lines[i],
            code_sim_lines[i],
            sc_simcode_lines[i],
            byte_sim_lines[i],
            by_simcode_lines[i],
            round(bleu4_code, 4),
            round(meteor_code, 4),
            round(rouge_code, 4),
            round(cider_code, 4),
            round(bleu4_byte, 4),
            round(meteor_byte, 4),
            round(rouge_byte, 4),
            round(cider_byte, 4),
            better_method,
            round(score_diff, 4)
        ])

    # 4. 将上述结果保存为 CSV 文件，并根据更优方法拆分输出
    headers = [
        "ID", "Source Code", "Reference Summary", "Code Retrieval Summary", "Code Retrieval Code",
        "Bytecode Retrieval Summary", "Bytecode Retrieval Code",
        "BLEU-4 (Code)", "METEOR (Code)", "ROUGE-L (Code)", "CIDEr (Code)",
        "BLEU-4 (Bytecode)", "METEOR (Bytecode)", "ROUGE-L (Bytecode)", "CIDEr (Bytecode)",
        "Better Method", "Score Difference (METEOR)"
    ]
    try:
        # 保存所有样本的评估结果
        with open("evaluation_results.csv", "wb") as f:
            writer = csv.writer(f)
            writer.writerow([h.encode("utf-8") for h in headers])
            for row in results:
                writer.writerow([unicode(col).encode("utf-8") if isinstance(col, unicode) else str(col) for col in row])

        # 保存源代码检索更优的样本子集
        if better_by_code:
            with open("better_by_code.csv", "wb") as f:
                writer = csv.writer(f)
                writer.writerow([h.encode("utf-8") for h in headers])
                for idx in better_by_code:
                    row = results[idx]
                    writer.writerow(
                        [unicode(col).encode("utf-8") if isinstance(col, unicode) else str(col) for col in row])

        # 保存字节码检索更优的样本子集
        if better_by_byte:
            with open("better_by_bytecode.csv", "wb") as f:
                writer = csv.writer(f)
                writer.writerow([h.encode("utf-8") for h in headers])
                for idx in better_by_byte:
                    row = results[idx]
                    writer.writerow(
                        [unicode(col).encode("utf-8") if isinstance(col, unicode) else str(col) for col in row])

    except Exception as e:
        print("保存 CSV 文件时出错: {}".format(e))
        # 即使保存失败，仍继续执行后续的统计输出

    # 5. 输出统计信息（平均得分、差异最大样本等）
    # 计算每种方法的平均指标得分
    avg_bleu4_code = score_bleu_code[3] if isinstance(score_bleu_code, (list, tuple)) else score_bleu_code
    avg_bleu4_byte = score_bleu_byte[3] if isinstance(score_bleu_byte, (list, tuple)) else score_bleu_byte
    avg_meteor_code = score_meteor_code
    avg_meteor_byte = score_meteor_byte
    avg_rouge_code = score_rouge_code
    avg_rouge_byte = score_rouge_byte
    avg_cider_code = score_cider_code
    avg_cider_byte = score_cider_byte

    print("评估完毕！各方法的平均指标得分如下：")
    print("源代码检索摘要 – 平均 BLEU-4: {:.4f}, METEOR: {:.4f}, ROUGE-L: {:.4f}, CIDEr: {:.4f}".format(
    avg_bleu4_code, avg_meteor_code, avg_rouge_code, avg_cider_code))
    print("字节码检索摘要 - 平均 BLEU-4: {:.4f}, METEOR: {:.4f}, ROUGE-L: {:.4f}, CIDEr: {:.4f}".format(
    avg_bleu4_byte, avg_meteor_byte, avg_rouge_byte, avg_cider_byte))
    # 查找差异最大的样本（按 METEOR 分数差值）
    if results:
        diffs = [abs(row[-1]) for row in results]  # 取每个样本 METEOR 差值的绝对值
        # 获取差值最大的前5个样本索引
        top_indices = sorted(range(len(diffs)), key=lambda i: diffs[i], reverse=True)[:10]
        print("差异最大的样本（按 METEOR 分数差值排序，前10个）：")
        for j in top_indices:
            bm = results[j][-2]  # Better Method (优胜方法)
            diff_val = diffs[j]  # 差值大小
            print("样本 {}: {} 方法更优，METEOR 差值 = {:.4f}".format(results[j][0], bm, diff_val))

      
    # 6. 可视化对比结果
    # 准备各指标的得分列表（每个样本的源代码法得分列表和字节码法得分列表）
    bleu_scores_code = [scores_bleu_code[3][i] for i in range(N)]
    bleu_scores_byte = [scores_bleu_byte[3][i] for i in range(N)]
    meteor_scores_code = [scores_meteor_code[i] for i in range(N)]
    meteor_scores_byte = [scores_meteor_byte[i] for i in range(N)]
    rouge_scores_code = [scores_rouge_code[i] for i in range(N)]
    rouge_scores_byte = [scores_rouge_byte[i] for i in range(N)]
    # 计算 BLEU-4, METEOR, ROUGE-L 三者的平均值作为综合指标
    avg_scores_code = [(bleu_scores_code[i] + meteor_scores_code[i] + rouge_scores_code[i]) / 3.0 for i in range(N)]
    avg_scores_byte = [(bleu_scores_byte[i] + meteor_scores_byte[i] + rouge_scores_byte[i]) / 3.0 for i in range(N)]

    metrics_data = [
        ("BLEU-4", bleu_scores_code, bleu_scores_byte),
        ("METEOR", meteor_scores_code, meteor_scores_byte),
        ("ROUGE-L", rouge_scores_code, rouge_scores_byte),
        ("Average", avg_scores_code, avg_scores_byte)
    ]

    for metric_name, code_list, byte_list in metrics_data:
        # 统计源代码更优、字节码更优、平局的样本数量
        code_better_count = sum(1 for c, b in zip(code_list, byte_list) if c > b)
        byte_better_count = sum(1 for c, b in zip(code_list, byte_list) if c < b)
        equal_count = N - code_better_count - byte_better_count  # 平局数

        # 绘制维恩图
        plt.figure()
        venn2(subsets=(code_better_count + equal_count,
                       byte_better_count + equal_count,
                       equal_count),
              set_labels=(u'better SourceCode', u'better ByteCode'))
        plt.title(u"{} Comparison".format(metric_name))
        plt.savefig(u"png/venn_{}.png".format(metric_name))
#         plt.show()

        # 绘制箱型图
        plt.figure()
        plt.boxplot([code_list, byte_list], labels=[u'SourceCode', u'ByteCode'])
        plt.title(u"{} score（SourceCode vs ByteCode）".format(metric_name))
        plt.ylabel(u"{} score".format(metric_name))
        plt.savefig(u"png/boxplot_{}.png".format(metric_name))
#         plt.show()

        # 绘制散点图（带45°参考线）
        plt.figure()
        plt.scatter(code_list, byte_list, alpha=0.7)
        # 在图上添加对角线参考线
        max_val = max(max(code_list), max(byte_list))
        min_val = min(min(code_list), min(byte_list))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
        plt.title(u"{} Comparison".format(metric_name))
        plt.xlabel(u"SourceCode {}".format(metric_name))
        plt.ylabel(u"ByteCode {}".format(metric_name))
        plt.savefig(u"png/scatter_{}.png".format(metric_name))
#         plt.show()

if __name__ == '__main__':
    main()


