# -*- coding: utf-8 -*-
"""
脚本：llm_score_compare.py
功能：对比字节码与源代码检索方法在LLM评分下的效果差异，输出结果表和图像。
兼容：Python 2.7
"""

import os
import sys
import csv
import matplotlib
matplotlib.use('Agg')  # 无显示环境下用于保存图像
import matplotlib.pyplot as plt

# 设置 Python 2 中文编码
reload(sys)
sys.setdefaultencoding('utf-8')

def read_scores_from_csv(filepath):
    """读取 CSV 中四个评分字段（idx, Coherence, Consistency, Fluency, Relevance）"""
    scores = []
    with open(filepath, 'rb') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            scores.append([
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4])
            ])
    return scores

def read_source_code(filepath):
    """读取 source.code，每行一个源代码样本"""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f]

def save_result_csv(source_codes, source_comments, base_pres, byte_pres, avg_sc_list, avg_bc_list, diff_list, out_path):
    """保存结果到 CSV 文件"""
    with open(out_path, 'wb') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['SourceCode', 'GroundTruth', 'Src_Prediction', 'Byte_Prediction', 'AvgScore_Source', 'AvgScore_Bytecode', 'Diff'])
        for code, comment, src_pre, byte_pre, avg_sc, avg_bc, diff in zip(source_codes, source_comments, base_pres, byte_pres, avg_sc_list, avg_bc_list, diff_list):
            writer.writerow([code, comment, src_pre, byte_pre, avg_sc, avg_bc, diff])

def draw_plots(avg_sc_list, avg_bc_list, diff_list, out_dir):
    """绘制散点图、直方图、箱线图和Top-N差值柱状图"""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False

    # 散点图
    plt.figure(figsize=(6, 6))
    plt.scatter(avg_sc_list, avg_bc_list, alpha=0.6, color='teal', edgecolor='k')
    plt.title('源代码检索平均分 vs 字节码检索平均分')
    plt.xlabel('源代码检索平均分')
    plt.ylabel('字节码检索平均分')
    plt.plot([0, 5], [0, 5], 'r--')  # 对角线
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'scatter_compare.png'))
    plt.clf()

    # 直方图
    plt.figure(figsize=(6, 4))
    plt.hist(diff_list, bins=30, color='steelblue', edgecolor='black')
    plt.title('平均分差值分布 (字节码 - 源代码)')
    plt.xlabel('平均分差值')
    plt.ylabel('样本数量')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'hist_diff.png'))
    plt.clf()

    # 箱线图
    plt.figure(figsize=(4, 5))
    plt.boxplot(diff_list, vert=True, patch_artist=True, boxprops={'facecolor': 'lightgreen'})
    plt.title('平均分差值箱线图 (字节码 - 源代码)')
    plt.ylabel('平均分差值')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'boxplot_diff.png'))
    plt.clf()

    # Top-N 差值柱状图
    N = 10
    idx_sorted = sorted(range(len(diff_list)), key=lambda i: diff_list[i], reverse=True)
    top_idx = idx_sorted[:N]
    top_diff = [diff_list[i] for i in top_idx]
    labels = [str(i) for i in top_idx]

    plt.figure(figsize=(8, 5))
    plt.bar(range(N), top_diff, color='orange')
    plt.xticks(range(N), labels)
    plt.title('差值最高的{}个样本 (字节码 - 源代码)'.format(N))
    plt.xlabel('样本编号')
    plt.ylabel('平均分差值')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'bar_topN_diff.png'))
    plt.clf()

def main():
    # 文件路径
    path_bytecode = 'OUTPUT_CSV.csv'
    path_source = 'BASEOUTPUT_CSV.csv'
    path_code = './../dataset/JCSD/test/source.code'
    path_source_comment = './../dataset/JCSD/test/source.comment'
    path_base_pres = 'base_result.3'
    path_byte_pres = './../results/JCSD/beamSearch_result.3'
    out_csv = 'llm_png/llmresult.csv'
    out_img_dir = 'llm_png'

    print("读取评分数据中...")
    scores_bc = read_scores_from_csv(path_bytecode)
    scores_sc = read_scores_from_csv(path_source)
    source_codes = read_source_code(path_code)
    source_comments = read_source_code(path_source_comment)
    base_pres = read_source_code(path_base_pres)
    byte_pres = read_source_code(path_byte_pres)

    if len(scores_bc) != len(scores_sc) or len(scores_bc) != len(source_codes):
        print("行数不一致，请检查输入文件。")
        sys.exit(1)

    print("计算平均分与差值中...")
    avg_bc_list = []
    avg_sc_list = []
    diff_list = []

    for bc, sc in zip(scores_bc, scores_sc):
        avg_bc = sum(bc) / len(bc)
        avg_sc = sum(sc) / len(sc)
        avg_bc_list.append(avg_bc)
        avg_sc_list.append(avg_sc)
        diff_list.append(avg_bc - avg_sc)

    print("写入 llmresult.csv...")
    save_result_csv(source_codes, source_comments, base_pres, byte_pres, avg_sc_list, avg_bc_list, diff_list, out_csv)

    print("生成图表...")
    draw_plots(avg_sc_list, avg_bc_list, diff_list, out_img_dir)

    print("完成！输出文件已保存至：")
    print("  表格：{}".format(out_csv))
    print("  图像目录：{}/".format(out_img_dir))


if __name__ == '__main__':
    main()
