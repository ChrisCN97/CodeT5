import json
import os.path
import random


def solidityProcess():
    root = "/mnt/sda/cn/codet5/data/summarize/smartContract/"
    train_name = root + "train.jsonl"
    examples = []
    with open(train_name, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            code = js['code'].split()
            if len(code) < 25 or len(code) > 200:
                continue
            nl = js['summary'].split()
            if len(nl) < 1 or len(nl) > 100:
                continue
            examples.append({"code_tokens": code, "docstring_tokens": nl})
            if idx + 1 == 26000:
                break
    root = "/mnt/sda/cn/codet5/data/summarize/solidity/"
    with open(root + "train.jsonl", 'w') as f:
        for item in examples[:5000]:
            f.write(json.dumps(item) + "\n")
    with open(root + "valid.jsonl", 'w') as f:
        for item in examples[5000:6000]:
            f.write(json.dumps(item) + "\n")
    with open(root + "test.jsonl", 'w') as f:
        for item in examples[6000:]:
            f.write(json.dumps(item) + "\n")

def summarize2src():
    sum_root = "/mnt/sda/cn/codet5/data/summarize"
    pretrain_root = "/mnt/sda/cn/codet5/data/pretrain/src/"
    langs = ["java", "python", "go", "ruby", "javascript", "php", "solidity"]
    for lang in langs:
        sum_name = os.path.join(sum_root, lang, "train.jsonl")
        with open(sum_name, encoding="utf-8") as f, open(pretrain_root + lang + ".txt", 'w') as f2:
            for idx, line in enumerate(f):
                line = line.strip()
                js = json.loads(line)
                code_tokens = js["code_tokens"]
                for i, token in enumerate(code_tokens):
                    if token == '\n':
                        code_tokens[i] = '\\n'
                    if token == '\r':
                        code_tokens[i] = '\\r'
                f2.write(" ".join(code_tokens) + '\n')

def src2with_lang_v1(trg_folder, train_num, val_num):
    pattern = "output language {} . "
    src_folder = "/mnt/sda/cn/codet5/data/pretrain/src/"
    langs = ["java", "python", "go", "ruby", "javascript", "php", "solidity"]
    codes = []
    for lang in langs:
        with open(src_folder + lang + ".txt") as f:
            for line in f:
                codes.append(pattern.format(lang) + line)
    random.shuffle(codes)
    total = len(codes)
    with open(trg_folder + "train.txt", 'w') as f:
        f.write("".join(codes[total-train_num-val_num:total-val_num]))
    with open(trg_folder + "val.txt", 'w') as f:
        f.write("".join(codes[total-val_num:]))

if __name__ == "__main__":
    trg_folder = "/mnt/sda/cn/codet5/data/pretrain/with_lang/v1/"
    train_num = 100000
    val_num = 5000
    src2with_lang_v1(trg_folder, train_num, val_num)
