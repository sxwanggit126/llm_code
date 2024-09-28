import json
import re

from transformers import AutoTokenizer
import torch

import sys
sys.path.append("../")
from component.utils import ModelUtils


def main():

    # 使用base model和adapter进行推理，无需手动合并权重
    model_name_or_path = '/root/autodl-tmp/llm_code/output/qwen2-7b-Instruct-full-0928'

    # 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
    load_in_4bit = False

    # 生成超参配置
    max_new_tokens = 500
    top_p = 0.9
    temperature = 0.35
    repetition_penalty = 1.0
    device = 'cuda'

    # 加载模型
    model = ModelUtils.load_model(
        model_name_or_path,
        load_in_4bit=load_in_4bit
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )

    tokenizer.padding_side = "left"

    while True:

        text = input('用户：')

        messages = [
            {"role": "user", "content": text}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        text = response.strip(' ').replace('\n', '。')
        text = text.replace(r'<|im_start|>system。You are a helpful assistant.<|im_end|>。<|im_start|>user。', '')
        text = text.replace(r'<|im_end|>。<|im_start|>assistant。', '')
        print('Qwen2：' + text)


if __name__ == '__main__':
    main()
