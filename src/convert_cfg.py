import os
import json
import argparse

def parse_cfg_line(ir_line, debug=False):
    """
    将一行线性化字节码序列转换为 CFG。返回包含 'nodes' 和 'adj' 的字典。
    debug=True 时，打印中间步骤便于调试。
    """
    # 去除 <BEG> 和 <END>
    content = ir_line.strip()
    if content.startswith("<BEG>") and content.endswith("<END>"):
        content = content[5:-5].strip()
    else:
        # 如果没有 <BEG>/<END>，直接使用整行
        content = content

    # 按 <NEXT> 分割指令
    parts = [p.strip().rstrip(';') for p in content.split("<NEXT>") if p.strip()]
    nodes = []
    label_map = {}
    # 第一轮：识别标签和指令节点
    for part in parts:
        if ':' in part:
            label, sep, rest = part.partition(':')
            label = label.strip()
            rest = rest.strip()
            if not rest:
                # 纯标签
                label_map[label] = len(nodes)
                if debug:
                    print(f"  ↳ 识别纯标签 '{label}' 映射到节点索引 {len(nodes)}")
                continue
            else:
                # 标签 + 指令
                label_map[label] = len(nodes)
                if debug:
                    print(f"  ↳ 识别标签+指令: 标签 '{label}' 映射到节点 {len(nodes)}, 指令='{rest}'")
                nodes.append(rest)
        else:
            # 普通指令
            nodes.append(part)
        if debug and ':' not in part:
            print(f"  ↳ 添加指令节点 '{part}' → 索引 {len(nodes)-1}")

    # 第二轮：构建邻接表
    adj = [[] for _ in range(len(nodes))]
    for i, inst in enumerate(nodes):
        tokens = inst.split()
        if tokens[0] == 'if' and 'goto' in tokens:
            # 条件跳转
            target = tokens[-1]
            tgt_idx = label_map.get(target)
            if tgt_idx is not None:
                adj[i].append(tgt_idx)
            if i + 1 < len(nodes):
                adj[i].append(i + 1)
            if debug:
                print(f"  ↳ 节点{i} ('{inst}') 条件跳转, edges -> {adj[i]}")
        elif tokens[0] == 'goto':
            # 无条件跳转
            target = tokens[-1]
            tgt_idx = label_map.get(target)
            if tgt_idx is not None:
                adj[i].append(tgt_idx)
            if debug:
                print(f"  ↳ 节点{i} ('{inst}') 无条件跳转, edges -> {adj[i]}")
        elif tokens[0] in ('return', 'throw'):
            # 终止指令，无出边
            if debug:
                print(f"  ↳ 节点{i} ('{inst}') 方法终止, 无出边")
        else:
            # 普通顺序流
            if i + 1 < len(nodes):
                adj[i].append(i + 1)
            if debug:
                print(f"  ↳ 节点{i} ('{inst}') 顺序流 -> {adj[i]}")
    return {'nodes': nodes, 'adj': adj}


def main(input_path, output_path, debug=True):
    """
    批量将 input_path 中的字节码序列转换为 CFG，写入 output_path。
    debug=True 时，会在控制台打印调试信息。
    """
    if debug:
        print(f"开始转换：输入文件='{input_path}'，输出文件='{output_path}'")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                if debug:
                    print(f"Line {lineno}: 空行，跳过")
                continue
            try:
                if debug:
                    print(f"\nLine {lineno}: 原始IR='{line}'")
                cfg = parse_cfg_line(line, debug)
                fout.write(json.dumps(cfg, ensure_ascii=False) + "\n")
                if debug:
                    num_nodes = len(cfg['nodes'])
                    num_edges = sum(len(succ) for succ in cfg['adj'])
                    print(f"Line {lineno}: 转换完成 → nodes={num_nodes}, total_edges={num_edges}")
            except Exception as e:
                print(f"❌ Line {lineno}: 转换失败: {e}")
    if debug:
        print("✅ 全量转换完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="线性化字节码序列到CFG的批量转换工具")
    parser.add_argument("--input", "-i", default="dataset/JCSD/train/source.code",
                        help="输入文件路径，每行一条线性化字节码序列")
    parser.add_argument("--output", "-o", default="dataset/JCSD/train/source.cfg",
                        help="输出文件路径，每行一条JSON格式CFG")
    parser.add_argument("--no-debug", dest="debug", action="store_false",
                        help="不输出调试信息")
    args = parser.parse_args()
    main(args.input, args.output, args.debug)

# 示例：在控制台运行
# > python convert_cfg.py -i dataset/JCSD/train/source.code -o dataset/JCSD/train/source.cfg
# 会打印各行的处理进度、节点和边数等调试信息，最后输出 "✅ 全量转换完成！"
