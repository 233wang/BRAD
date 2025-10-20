from transformers import LlamaForCausalLM, AutoTokenizer
import os
# 完全离线模式，必须在导入 transformers/sentence_transformers 之前
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HUGGINGFACE_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

if __name__ == '__main__':
    # ----- 离线加载本地 USE 模型目录 ----- #
    # 1. 计算本地模型目录（相对于本脚本 src/ 目录）
    local_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
            '../../../../..',
            'meta-llama',
            'Llama-3.1-8B-Instruct')
            )
    # 2. 确认目录存在
    assert os.path.isdir(local_dir), f"请检查本地模型目录是否正确：{local_dir}"
    #下载好的模型地址
#     model_path = 'xxx/yyy/llama3.1-8b-base'
    model = LlamaForCausalLM.from_pretrained(local_dir, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
 
    prompt = "can you give me a great praise?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
 
    # Generate
    generate_ids = model.generate(inputs.input_ids, max_length=100)
    res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(res)

