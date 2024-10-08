#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import re
import sys
import time
import torch
import argparse
import numpy as np
import pandas as pd
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

sys.path.append("../")
from utils import model_utils
from module_interface import llm_interface


choices = ["A", "B", "C", "D"]


def find_valid_substrings(s):
    # 匹配长度为1到4的、不包含重复字符的子串
    pattern = r'[ABCD]{1,4}'
    substrings = re.findall(pattern, s)
    # 过滤出不包含重复字符的子串
    valid_substrings = [substring for substring in substrings if len(substring) == len(set(substring))]
    return valid_substrings


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        try:
            prompt += "\n{}. {}".format(choices[j],
                                        df.iloc[idx, j + 1].replace(" ．", "").replace("A、", "").replace("B、",
                                                                                                        "").replace(
                                            "C、", "").replace("D、", "").replace("A.", "").replace("B.", "").replace(
                                            "C.", "").replace("D.", "").replace("A", "").replace("B", "").replace("C",
                                                                                                                  "").replace(
                                            "D", "").replace("Ａ、", "").replace("Ｂ、", "").replace("Ｃ、", "").replace("Ｄ、",
                                                                                                                   "").strip())
        except Exception as e:
            prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\n正确答案的序号是："
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


# 生成 prompt
# 不提示专业科目，让模型直接理解题目
def gen_prompt(train_df, k=-1):
    prompt = "请阅读以下选择题并给出正确选项，不要解释原因。请只给出答案的序号。\n"
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)

    return prompt


def plain_chat(
        prompt
):
    # LLM 生成结果
    _, output = llm_interface.get_llm_answer(prompt)
    return output.replace(prompt, "").replace("</s>", "").strip()


def read_file_lines(file):
    with open(file, 'r', errors='ignore', encoding='utf8') as f:
        lines = f.readlines()
    return lines


def writelines_to_file(file, lines):
    with open(file, 'w', encoding='utf8') as f:
        f.writelines(lines)


def writetext_to_file(file, contents):
    with open(file, 'w', encoding='utf8') as f:
        f.write(contents)


def eval(args, subject, dev_df, test_df):
    logfile = "eval_log_0717"
    cors = []
    preds = []
    for i in range(test_df.shape[0]):
        print('i=', i)
        # get prompt and make sure it fits
        k = args.ntrain
        # ============= 题目格式化 ===========
        # 题目： 请阅读以下选择题并给出正确选项，不要解释原因。请只给出答案的序号。
        # 首次急性发作的腰椎间盘突出的治疗方法首选
        # A.绝对卧床休息，3
        # 周后戴腰围下床活动
        # B.卧床休息，可以站立坐起
        # C.皮质类固醇硬膜外注射
        # D.髓核化学溶解
        # 正确答案的序号是：
        # 格式化每个测试样例
        prompt_end = format_example(test_df, i, include_answer=False)
        # 前置 prompt
        train_prompt = gen_prompt(dev_df, k)
        # 组合 prompt
        prompt = train_prompt + prompt_end
        print('prompt length is : ', len(prompt))
        print("题目：", prompt)
        with open(logfile, 'a', encoding='utf8') as f:
            f.write(prompt + "\n")
        try:
            label = test_df.iloc[i, test_df.shape[1] - 1]
            # A B C D 特殊处理  ＡＢＣＤ
            label = label.replace(" ", "").replace("Ａ", "A").replace("Ｂ", "B").replace("Ｃ", "C").replace("Ｄ", "D")
            label = label.replace("\u3000", "").replace(",", "")
            print("正确答案:", label)
            with open(logfile, 'a', encoding='utf8') as f:
                f.write("正确答案:" + label + "\n")
        except Exception as e:
            print(e)
            break

        while True:
            try:
                time.sleep(1)
                # 执行预测
                pred = plain_chat(prompt)
                pred = pred.replace("、", "").replace(".", "").replace(",", "").replace(";", "").replace("，", "")
                try:
                    # 识别答案pattern
                    pred = find_valid_substrings(pred)[0]
                except Exception as e:
                    print(e)
                    pred = "未成功回答"
                print("模型预测答案:", pred)
                with open(logfile, 'a', encoding='utf8') as f:
                    f.write("模型预测答案:" + pred + "\n")
                break
            except Exception as e:
                print(e)
                print("pausing")
                time.sleep(10)
                continue

        try:
            cor = pred == label
            print("是否答对：", cor)
            with open(logfile, 'a', encoding='utf8') as f:
                f.write("是否答对：" + str(cor) + "\n")
            cors.append(cor)
            preds.append(pred + "|||" + label + "|||" + str(cor))

        except Exception as e:
            print(e)

    acc = np.mean(cors)

    acc_info = "Average accuracy {:.3f} - {}".format(acc, subject)
    print(acc_info)

    preds.append(acc_info)
    preds = [x + "\n" for x in preds]

    return preds


def main(args):
    # 按名称进行排序
    subjects = sorted([f.split(".xlsx")[0] for f in os.listdir(os.path.join(args.data_dir, "test"))
                       if ".xlsx" in f and '._' not in f])
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    print("subjects:", subjects)
    print("args", args)

    # 读取excel文件内容
    for subject in subjects:
        # 除医疗外，一个文件只有一个 sheet
        if subject != "医疗":
            dev_df = pd.read_excel(os.path.join(args.data_dir, "dev", subject + ".xlsx"),
                                   header=0, engine='openpyxl')[:args.ntrain]
            test_df = pd.read_excel(os.path.join(args.data_dir, "test", subject + ".xlsx"),
                                    header=0, engine='openpyxl')
            preds = eval(args, subject, dev_df, test_df)
            writelines_to_file(os.path.join(args.save_dir, subject), preds)
        # 医学内部分为多个 sheet，需要逐一处理
        else:
            dev_df = pd.read_excel(os.path.join(args.data_dir, "dev", subject + ".xlsx"),
                                   header=0, engine='openpyxl')[:args.ntrain]
            f = pd.ExcelFile(os.path.join(args.data_dir, "test", subject + ".xlsx"))
            sheet_list = f.sheet_names
            for sheet in sheet_list:
                test_df = pd.read_excel(os.path.join(args.data_dir, "test", subject + ".xlsx"), header=0,
                                        sheet_name=sheet, engine='openpyxl')
                preds = eval(args, sheet, dev_df, test_df)
                writelines_to_file(os.path.join(args.save_dir, sheet), preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    args = parser.parse_args()

    main(args)
