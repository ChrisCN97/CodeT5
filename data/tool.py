import json


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

if __name__ == "__main__":
    solidityProcess()