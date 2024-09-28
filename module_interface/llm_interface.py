#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from common import constants
from transformers import AutoTokenizer
import torch

import sys
sys.path.append("../")
from component.utils import ModelUtils

# 使用base model和adapter进行推理，无需手动合并权重
model_name_or_path = '/root/autodl-tmp/qwen/Qwen2-7B-Instruct'

# 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
load_in_4bit = False

# 指定GPU执行
device = 'cuda'

model = ModelUtils.load_model(
    model_name_or_path,
    load_in_4bit=load_in_4bit
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
)

def get_predict_answer(input_text):
    with torch.autocast("cuda"):
        try:
            messages = [
                {"role": "user", "content": input_text}
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
            response = text.replace(r'<|im_end|>。<|im_start|>assistant。', '')
        except Exception as e:
            print(e)
            print('llm has error')
            response = constants.BACK_AND_FORTH_ANSWER
        return response

def get_llm_answer(query):
    predict_text = get_predict_answer(query)
    if predict_text:
        return True, predict_text
    else:
        return False, None


