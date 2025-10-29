import json
import os

# 你的数据文件列表
data_files = [
    "data/LogiQA_v2/LogiQA_fintuing_data_formatted_base.json",
    "data/Reclor/Reclor_fintuing_data_formatted_base.json",
    "data/FOLIO/FOLIO_fintuing_data_formatted_base.json",
    "data/logiqa-zh/logiqa-zh_fintuing_data_formatted_base.json",

    "data/CorrectionData/FOLIO_correct_fintuing_data_formatted.json",
    "data/CorrectionData/LogiQA_correct_fintuing_data_formatted.json",
    "data/CorrectionData/Reclor_correct_fintuing_data_formatted.json",
    "data/CorrectionData/logiqa-zh_correct_fintuing_data_formatted.json",
    "data/CorrectionData/Fintue_data_format/logiqa-zh_correct_fintuing_data_formatted_2.json",

    'data/LogiCoT/mrc_formatted.json',
    'data/LogiCoT/folio2instruction_formatted.json',
    'data/LogiCoT/mrc_zh_formatted.json'

    'data/LFUD/datasets.json', 
]

# 输出目录
output_dir = "data/instruction_format_data"
os.makedirs(output_dir, exist_ok=True)


def convert_file(file_in, file_out):
    with open(file_in, "r", encoding="utf-8") as f:
        raw = json.load(f)

    converted = []
    for item in raw:
        # 如果是 Alpaca 格式
        if "instruction" in item and "output" in item:
            converted.append(item)
        # 如果是 conversation 格式
        elif "conversation" in item:
            conv = item["conversation"][0]
            converted.append({
                "instruction": conv.get("system", ""),
                "input": conv.get("input", ""),
                "output": conv.get("output", "")
            })
        else:
            raise ValueError(f"Unknown format in file {file_in}: {item}")

    with open(file_out, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    for file in data_files:
        filename = os.path.basename(file)
        output_file = os.path.join(output_dir, filename)
        try:
            convert_file(file, output_file)
            print(f"✅ Converted {file} -> {output_file}")
        except Exception as e:
            print(f"❌ Failed to convert {file}: {e}")
