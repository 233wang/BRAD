import os
import json
import argparse

def parse_cfg_line(ir_line, debug=False):

    content = ir_line.strip()
    if content.startswith("<BEG>") and content.endswith("<END>"):
        content = content[5:-5].strip()
    else:
        content = content

    parts = [p.strip().rstrip(';') for p in content.split("<NEXT>") if p.strip()]
    nodes = []
    label_map = {}
    for part in parts:
        if ':' in part:
            label, sep, rest = part.partition(':')
            label = label.strip()
            rest = rest.strip()
            if not rest:
                label_map[label] = len(nodes)
                if debug:
                    print(f"  ↳ label '{label}' nodes {len(nodes)}")
                continue
            else:
                label_map[label] = len(nodes)
                if debug:
                    print(f"  ↳ label '{label}' nodes {len(nodes)}, rest='{rest}'")
                nodes.append(rest)
        else:
            nodes.append(part)
        if debug and ':' not in part:
            print(f"  ↳ part '{part}' → nodes {len(nodes)-1}")

    adj = [[] for _ in range(len(nodes))]
    for i, inst in enumerate(nodes):
        tokens = inst.split()
        if tokens[0] == 'if' and 'goto' in tokens:
            target = tokens[-1]
            tgt_idx = label_map.get(target)
            if tgt_idx is not None:
                adj[i].append(tgt_idx)
            if i + 1 < len(nodes):
                adj[i].append(i + 1)
            if debug:
                print(f"  ↳ nodes{i} ('{inst}') if jump, edges -> {adj[i]}")
        elif tokens[0] == 'goto':
            target = tokens[-1]
            tgt_idx = label_map.get(target)
            if tgt_idx is not None:
                adj[i].append(tgt_idx)
            if debug:
                print(f"  ↳ nodes{i} ('{inst}') no jump, edges -> {adj[i]}")
        elif tokens[0] in ('return', 'throw'):
            if debug:
                print(f"  ↳ nodes{i} ('{inst}') end")
        else:
            if i + 1 < len(nodes):
                adj[i].append(i + 1)
            if debug:
                print(f"  ↳ nodes{i} ('{inst}') flow -> {adj[i]}")
    return {'nodes': nodes, 'adj': adj}


def main(input_path, output_path, debug=True):

    if debug:
        print(f"input_path='{input_path}'output_path='{output_path}'")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                if debug:
                    print(f"Line {lineno}: empty")
                continue
            try:
                if debug:
                    print(f"\nLine {lineno}: rawIR='{line}'")
                cfg = parse_cfg_line(line, debug)
                fout.write(json.dumps(cfg, ensure_ascii=False) + "\n")
                if debug:
                    num_nodes = len(cfg['nodes'])
                    num_edges = sum(len(succ) for succ in cfg['adj'])
                    print(f"Line {lineno}: finish → nodes={num_nodes}, total_edges={num_edges}")
            except Exception as e:
                print(f"❌ Line {lineno}: error: {e}")
    if debug:
        print("✅ all finish！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input", "-i", default="dataset/JCSD/train/source.code")
    parser.add_argument("--output", "-o", default="dataset/JCSD/train/source.cfg")
    parser.add_argument("--no-debug", dest="debug", action="store_false")
    args = parser.parse_args()
    main(args.input, args.output, args.debug)
