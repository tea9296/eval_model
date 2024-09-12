from collections import defaultdict
import akasha
import json, os
import akasha.eval as eval
import traceback, re
from typing import Union
import time

EVAL_MODEL = "openai:gpt-4o"


def extract_json(s: str) -> Union[tuple[Union[dict, None], int]]:
    """parse the JSON part of the string

    Args:
        s (str): string that contains JSON part

    Returns:
        Union[dict, None]: return the JSON part of the string, if not found return None
    """
    # Use a regular expression to find the JSON part of the string
    match = re.search(r'\{.*\}', s, re.DOTALL)
    stack = []
    start = 0
    ss = s.replace("`", "").replace("json", "")

    if match is None:
        return None, 0

    s = match.group()
    for i, c in enumerate(s):
        if c == '{':
            stack.append(i)
        elif c == '}':
            if stack:
                start = stack.pop()
                if not stack:
                    try:
                        json_part = s[start:i + 1]
                        json_part = json_part.replace("\n", "")
                        json_part = json_part.replace("'", '"')
                        # Try to parse the JSON part of the string
                        return json.loads(json_part), len(
                            ss.replace(" ", "").replace("\n", "")) - len(
                                json_part.replace(" ", ""))
                    except json.JSONDecodeError:
                        traceback.print_exc()
                        print(s[start:i + 1])
                        print(
                            "The JSON part of the string is not well-formatted"
                        )
                        return None, 0
    return None, 0


def eval_format(model: str, prompt_format_type: str):
    article = """
    全球電動車銷售量 2021 年是 3138000，而 2023 年是 6841000
    以下引用調查公司MarkLines的數據資料，整理電動車市場的主要參與廠商。以銷售量而言，美國Tesla及中國大陸比亞迪等廠商，具有極大的存在感(詳見表一、圖二)。2023年全球電動車銷售量前10名中，有4家是中國大陸廠商。另外，旗下擁有中國上汽通用五菱汽車的GM集團也入列第4名(見附錄註一)。比亞迪2023年的電動車銷售量較2022年大幅成長67%，而2021年未躋身前10大的中國廣州汽車則在2023年排名中一舉上升到第五。


    表一、2021年及2023年全球電動車銷售量前十名


    2021年                                     2023年 
    排名 汽車廠商 銷售量                        排名 汽車廠商 銷售量

    1 美國Tesla  883000                          1   美國Tesla   1753000 

    2 美國GM集團 502000                          2   中國比亞迪  1452000 

    3 德國福斯集團 428000                        3   德國福斯集團 731000 

    4 中國比亞迪 318000                          4   美國GM集團  604000 

    5 韓國現代集團  222000                       5   中國廣州汽車 477000 

    6 法國Renault・Nissan・Mitsubishi  212000    6 中國浙江吉利控股集團  475000  

    7 歐洲Stellantis  182000                     7  韓國現代集團 390000

    8 中國上海汽車（SAIC）137000                  8  德國BMW  365000 

    9 中國長城汽車（Great Wall） 134000           9  中國上海汽車（SAIC）305000 

    10 德國BMW 120000                            10  法國Renault・Nissan・Mitsubishi 289000

    備註：2021至2023年，除比亞迪之外，廣州汽車、浙江吉利控股集團等中

    """

    key1 = ["2021年全球電動車銷售量", "2023年全球電動車銷售量"]
    key2 = [
        "2023年銷售量排名第一汽車廠商",
        "2021年銷售量排名第十汽車廠商",
        "2021年銷售量排名第八汽車廠商",
        "2023年銷售量排名第十汽車廠商",
    ]
    key3 = [
        "2021年美國Tesla銷售量", "2023年美國GM集團銷售量", "2023年韓國現代集團銷售量", "2021年德國福斯集團銷售量"
    ]
    key4 = [
        "2021年銷售量後三名的銷售量list", "2021年銷售量前三名的汽車廠商list", "2023年銷售量前三名的銷售量list",
        "2023年銷售量後三名的汽車廠商list"
    ]
    keys = [key1, key2, key3, key4]
    ans = [{
        "2021年全球電動車銷售量": 3138000,
        "2023年全球電動車銷售量": 6841000
    }, {
        "2023年銷售量排名第一汽車廠商": "美國Tesla",
        "2021年銷售量排名第十汽車廠商": "德國BMW",
        "2021年銷售量排名第八汽車廠商": "中國上海汽車（SAIC）",
        "2023年銷售量排名第十汽車廠商": "法國Renault・Nissan・Mitsubishi"
    }, {
        "2021年美國Tesla銷售量": 883000,
        "2023年美國GM集團銷售量": 604000,
        "2023年韓國現代集團銷售量": 390000,
        "2021年德國福斯集團銷售量": 428000
    }, {
        "2021年銷售量後三名的銷售量list": [137000, 134000, 120000],
        "2021年銷售量前三名的汽車廠商list": ["美國Tesla", "美國GM集團", "德國福斯集團"],
        "2023年銷售量前三名的銷售量list": [1753000, 1452000, 731000],
        "2023年銷售量後三名的汽車廠商list":
        ["德國BMW", "中國上海汽車（SAIC）", "法國Renault・Nissan・Mitsubishi"]
    }]
    model_obj = akasha.handle_model(model, True, 0.0)
    system_prompt = "user will give you a list of Keys and an Article, you need to extract the information from the Article based on the keys and\
        return the extracted information in a json format, do not include redundant information in the response."

    scores = []
    responses = []
    response_dicts = []
    counts = []
    for k in keys:
        score = 0.0
        prompt = "Keys:" + ",".join(k) + "\n\n" + "Article: " + article
        try:
            input_text = akasha.prompts.format_sys_prompt(
                system_prompt, prompt, prompt_format_type)
            response = akasha.call_model(model_obj, input_text)
            response_dict, count = extract_json(response)
        except:
            responses.append(response)
            response_dicts.append(None)
            counts.append(0)
            scores.append(0.0)
            continue
        responses.append(response)
        response_dicts.append(response_dict)
        counts.append(count)
        if response_dict is None:
            scores.append(score)
            continue
        else:
            score += 0.3
            if count <= 20:
                score += 0.2 * (1 - count / 20)

            else_score = 0.5 / (len(ans[keys.index(k)]) * 2)
            for key, value in ans[keys.index(k)].items():
                if key in response_dict:
                    if key is None or response_dict[key] is None:
                        continue
                    score += else_score
                    if type(value) == int:
                        if int(response_dict[key]) == value:
                            score += else_score
                    elif type(value) == list:
                        flag = True
                        for i, v in enumerate(value):
                            if type(v) == int:

                                if int(response_dict[key][i]) != v:
                                    flag = False
                                    break
                            else:
                                if response_dict[key][i] != v:
                                    flag = False
                                    break
                        if flag:
                            score += else_score

                    else:
                        if response_dict[key] == value:
                            score += else_score

        scores.append(score)
    return scores, responses, response_dicts, counts


def eval_qq(model: str,
            qt: str,
            qs: str,
            formal_name: str,
            doc_path: str = "MIC電子商務40",
            prompt_format_type: str = "chat_gpt"):

    #model_name = model.split(':')[-1].replace('-', '')
    ev = eval.Model_Eval(model=model,
                         question_type=qt,
                         question_style=qs,
                         verbose=True,
                         search_type="auto",
                         prompt_format_type=prompt_format_type,
                         threshold=0.0,
                         keep_logs=True)

    ev.auto_evaluation(f"questionset/{doc_path}-{qt}-{qs}.txt",
                       "docs/" + doc_path,
                       eval_model=EVAL_MODEL)

    print(ev.logs[ev.timestamp_list[-1]], "\n\n")

    # Save logs to a JSON file
    with open(
            f"logs/{formal_name}-{prompt_format_type}-{doc_path}-{qt}-{qs}.json",
            "w",
            encoding="utf-8") as f:
        json.dump(ev.logs[ev.timestamp_list[-1]],
                  f,
                  ensure_ascii=False,
                  indent=4)
    a = ev.timestamp_list[-1]
    return ev.logs[a]['time'], ev.logs[a]['doc_length'], ev.logs[a][
        'doc_tokens'], ev.score


def total_eval(model: str,
               doc_path: str,
               model_formal_name: str = "gpt4o",
               prompt_format_type="chat_gpt",
               fact: bool = True,
               irre: bool = True,
               summ: bool = True,
               comp: bool = True,
               format: bool = True):

    model_formal_name = model_formal_name
    chunk_size = 1000
    threshold = 0.0
    max_doc_len = 1500
    search_type = "auto"
    start_time = time.time()
    #### load the json file if exists ###
    output_file_path = f"scores/{model_formal_name}-{prompt_format_type}-{doc_path}.json"
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r', encoding='utf-8') as f:
            eval_json = json.load(f)
    else:
        eval_json = {}
        eval_json["model_formal_name"] = model_formal_name
        eval_json["chunk_size"] = chunk_size
        eval_json["threshold"] = threshold
        eval_json["max_doc_len"] = max_doc_len
        eval_json["search_type"] = search_type
        eval_json["prompt_format_type"] = prompt_format_type

    ### start to test the model ###
    if fact:
        ttime, doc_length, doc_tokens, score = eval_qq(model, "fact",
                                                       "single_choice",
                                                       model_formal_name,
                                                       doc_path,
                                                       prompt_format_type)
        avg_score = score["correct_count"] / 40
        eval_json["fact"] = {
            "time": ttime,
            "doc_length": doc_length,
            "doc_tokens": doc_tokens,
            "score": score,
            "avg_score": avg_score
        }

    if irre:
        ttime, doc_length, doc_tokens, score = eval_qq(model, "irrelevant",
                                                       "single_choice",
                                                       model_formal_name,
                                                       doc_path,
                                                       prompt_format_type)
        avg_score = score["correct_count"] / 40
        eval_json["irrelevant"] = {
            "time": ttime,
            "doc_length": doc_length,
            "doc_tokens": doc_tokens,
            "score": score,
            "avg_score": avg_score
        }

    if summ:
        ttime, doc_length, doc_tokens, score = eval_qq(model, "summary",
                                                       "essay",
                                                       model_formal_name,
                                                       doc_path,
                                                       prompt_format_type)
        n = len(score["bert"])
        avg_score = {
            "bert": sum(score["bert"]) / n,
            "rouge": sum(score["rouge"]) / n,
            "llm": sum(score["llm_score"]) / n
        }
        eval_json["summary"] = {
            "time": ttime,
            "doc_length": doc_length,
            "doc_tokens": doc_tokens,
            "score": score,
            "avg_score": avg_score
        }

    if comp:
        ttime, doc_length, doc_tokens, score = eval_qq(model, "compared",
                                                       "essay",
                                                       model_formal_name,
                                                       doc_path,
                                                       prompt_format_type)
        n = len(score["bert"])
        avg_score = {
            "bert": sum(score["bert"]) / n,
            "rouge": sum(score["rouge"]) / n,
            "llm": sum(score["llm_score"]) / n
        }
        eval_json["compared"] = {
            "time": ttime,
            "doc_length": doc_length,
            "doc_tokens": doc_tokens,
            "score": score,
            "avg_score": avg_score
        }

    if format:
        scores, responses, response_dicts, counts = eval_format(
            model, prompt_format_type)
        eval_json["format"] = {
            "scores": scores,
            "responses": responses,
            "response_dicts": response_dicts,
            "counts": counts,
            "avg_score": sum(scores) / len(scores)
        }
    eval_json["total_time"] = time.time() - start_time
    ## save the json file
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(eval_json, f, ensure_ascii=False, indent=4)

    print(eval_json)


doc_path = "MIC電子商務"

model = "remote:http://140.92.60.189:8601"  #openai:gpt-3.5-turbo
#formal_name = model.split(':')[-1].replace('-', '').replace('.', '')
# model = "openai:gpt-3.5-turbo"  #qwen2_72b  llama31_70b  gemma2_27b  mistral03_7b  breeze10_7b gemma2_2b   gemma2_9b
formal_name = "taidellama3_8b"
#total_eval(model, doc_path, formal_name, "gpt", True, True, True, True, True)
total_eval(model, doc_path, formal_name, "chat_gpt", True, True, True, True,
           True)

# model = "openai:gpt-4o"  remote:http://35.189.188.83:8503
# formal_name = model.split(':')[-1].replace('-', '').replace('.', '')
# total_eval(model, doc_path, formal_name, "gpt", True, True, True, True, True)
# total_eval(model, doc_path, formal_name, "chat_gpt", True, True, True, True,
#            True)
