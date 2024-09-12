import json
import os
from tabulate import tabulate


def add_total_time(formal_name: str,
                   prompt_format_type: str,
                   doc_path: str = "MIC電子商務",
                   log_dir: str = "./logs/",
                   score_dir: str = "./scores/"):

    target_json = score_dir + f"{formal_name}-{prompt_format_type}-{doc_path}.json"
    types = [
        "fact-single_choice", "irrelevant-single_choice", "compared-essay",
        "summary-essay"
    ]
    if not os.path.exists(target_json):
        print(f"File not found: {target_json}")
        return
    else:
        eval_json = json.load(open(target_json, 'r', encoding='utf-8'))
    ## Initialize total_time, start to add time
    total_time = 0

    for type_ in types:
        log_file = log_dir + f"{formal_name}-{prompt_format_type}-{doc_path}-{type_}.json"
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'time' in data:
                    total_time += data['time']
                else:
                    print(f"'time' key not found in {log_file}")
        else:
            print(f"File not found: {log_file}")

    eval_json["total_time"] = total_time
    with open(target_json, 'w', encoding='utf-8') as f:
        json.dump(eval_json, f, ensure_ascii=False, indent=4)

    print(f"Total time saved in {target_json}")


#add_total_time("mistral03_7b", "chat_mistral")


def find_data_back():
    target_json = "./scores/qwen2_72b-gpt-MIC電子商務.json"
    types = [
        "fact-single_choice", "irrelevant-single_choice", "compared-essay",
        "summary-essay"
    ]
    ret = {}

    for type_ in types:
        log_file = f"./logs/qwen2_72b-gpt-MIC電子商務-{type_}.json"
        typename = type_.split("-")[0]
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

        ret[typename] = {
            "time": data['time'],
            "doc_length": data['doc_length'],
            "doc_tokens": data['doc_tokens'],
            "score": {}
        }

        if typename in ["fact", "irrelevant"]:
            ret[typename]["avg_score"] = data["correct_rate"]
        else:
            n = len(data["bert"])
            avg_score = {
                "bert": sum(data["bert"]) / n,
                "rouge": sum(data["rouge"]) / n,
                "llm": sum(data["llm_score"]) / n
            }
            ret[typename]["avg_score"] = avg_score

    with open(target_json, 'w', encoding='utf-8') as f:
        json.dump(ret, f, ensure_ascii=False, indent=4)


def show_scores(name_list: list,
                doc_path: str = "MIC電子商務",
                score_dir: str = "./scores/"):

    headers = [
        "Name", "Prompt Format", "Fact", "Irre", "Comp", "Summ", "JSON", "Time"
    ]
    table = []
    flag = True
    for name in name_list:

        formal_name = name.split("-")[0]
        prompt_format_type = name.split("-")[1]
        target_json = score_dir + f"{name}-{doc_path}.json"
        if os.path.exists(target_json):
            with open(target_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if flag:

                total_token = data["fact"]["doc_tokens"] + data["irrelevant"]["doc_tokens"] + \
                    data["compared"]["doc_tokens"] + data["summary"]["doc_tokens"]
                total_length = data["fact"]["doc_length"] + data["irrelevant"]["doc_length"] + \
                    data["compared"]["doc_length"] + data["summary"]["doc_length"]

                print(f"Doc Length: {total_length}, ",
                      f"Doc Tokens: {total_token}")
                flag = False

            row = [
                formal_name, prompt_format_type, data["fact"]["avg_score"],
                data["irrelevant"]["avg_score"],
                data["compared"]["avg_score"]["bert"],
                data["summary"]["avg_score"]["bert"],
                data["format"]["avg_score"], data["total_time"]
            ]
            table.append(row)
        else:
            print(f"File not found: {target_json}")

    print(tabulate(table, headers, tablefmt="grid"))


#### all ####
name_list = ["gpt4o-chat_gpt","gpt35turbo-chat_gpt", "qwen2_72b-chat_gpt", "llama31_70b-chat_gpt", "taiwanllama3_70b-chat_gpt", "gemma2_27b-chat_mistral",\
    "gemma2_9b-chat_mistral", "llama31_8b-chat_gpt", "taiwanllama3_8b-chat_gpt", "taidellama3_8b-chat_gpt", "mistral03_7b-chat_mistral", "breeze10_7b-chat_mistral",\
        "phi35_mini-chat_gpt", "gemma2_2b-chat_mistral"]

### gpt ###
# name_list = [
#     "gpt35turbo-gpt",
#     "gpt35turbo-chat_gpt",
#     "gpt4o-gpt",
#     "gpt4o-chat_gpt",
# ]
# name_list = [
#     "gemma2_27b-chat_mistral",
#     "gemma2_27b-chat_gemma",
#     "taiwanllama3_70b-gpt",
#     "taiwanllama3_70b-chat_gpt",
# ]

show_scores(name_list)
