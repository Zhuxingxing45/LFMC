import json
import re

# path
files = {
    "zh_train": "/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/data/logiqa-zh/zh_train.txt",
    "zh_test": "/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/data/logiqa-zh/zh_test.txt",
    "zh_eval": "/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/data/logiqa-zh/zh_eval.txt",
}
json_output = {
    "zh_train": "/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/data/logiqa-zh/zh_train.json",
    "zh_test": "/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/data/logiqa-zh/zh_test.json",
    "zh_eval": "/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/data/logiqa-zh/zh_eval.json",
}

def _process_answer(answer):
    if not any(answer.startswith(x) for x in "ABCD"):
        return answer
    else:
        return answer[1:]

def _process_sentences(text):
    text = text.replace("\n", "")
    sents = text.split(".")
    text = ""
    for sent in sents:
        if len(sent) == 0:
            continue
        if len(text) == 0:
            text += sent
        elif sent[0].isnumeric():
            text += "."+sent
        else:
            text += ". "+sent
    text = text.replace("  ", " ")
    text = text.replace("\\'", "'")
    while text.endswith(" "):
        text = text[:-1]
    if re.match('^[A-Z][\w\s]+[?.!]$', text) is None:
        text += "."
    text = text.replace("?.", "?")
    text = text.replace("!.", "!")
    text = text.replace("..", ".")
    return text

def convert_txt_to_json(file_path, output_path):
    data = []
    with open(file_path, encoding="utf-8") as f:
        logiqa = f.readlines()
        logiqa = [_process_sentences(s) for s in logiqa]

        for key in range(int(len(logiqa)/8)):
            row = 8*key
            correct_answer = logiqa[row+1].replace(".", "")
            context = logiqa[row+2].replace(".", "")
            query = logiqa[row+3].replace(".", "")
            answers = [i.replace(".", "") for i in logiqa[row+4:row+8]]
            entry = {
                "context": context,
                "query": query,
                "options": [_process_answer(answers[i]) for i in range(4)],
                "correct_option": "abcd".index(correct_answer)
            }
            data.append(entry)

    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

# convert
for key in files:
    convert_txt_to_json(files[key], json_output[key])

print("Conversion complete!")
