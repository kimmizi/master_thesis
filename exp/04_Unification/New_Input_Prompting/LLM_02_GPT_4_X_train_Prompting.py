#### LLM: Zero-shot classification through LLMs and prompts ####
#### CHATGPT ####


#### 0 Imports ####

import os
import pandas as pd
import numpy as np
import time
import re
import json
from tqdm import tqdm
from openai import OpenAI
from sklearn.model_selection import train_test_split

data_change = pd.read_csv("../../../dat/dips/DIPS_Data_cleaned_change.csv", sep =",", low_memory = False)

# import prompts for all test data
X_train_simple_prompt = pd.read_csv("X_train_pred/prompts/X_train_simple_prompt.csv", sep =",", index_col = 0)
X_train_class_definitions_prompt = pd.read_csv(
    "X_train_pred/prompts/X_train_class_definitions_prompt.csv", sep =",", index_col = 0)
X_train_profiled_simple_prompt = pd.read_csv(
    "X_train_pred/prompts/X_train_profiled_simple_prompt.csv", sep =",", index_col = 0)
X_train_few_shot_prompt = pd.read_csv("X_train_pred/prompts/X_train_few_shot_prompt.csv", sep =",", index_col = 0)
X_train_vignette_prompt = pd.read_csv("X_train_pred/prompts/X_train_vignette_prompt.csv", sep =",", index_col = 0)
X_train_cot_prompt = pd.read_csv("X_train_pred/prompts/X_train_cot_prompt.csv", sep =",", index_col = 0)

# convert to arrays
X_train_simple_prompt = X_train_simple_prompt.values.flatten()
X_train_class_definitions_prompt = X_train_class_definitions_prompt.values.flatten()
X_train_profiled_simple_prompt = X_train_profiled_simple_prompt.values.flatten()
X_train_few_shot_prompt = X_train_few_shot_prompt.values.flatten()
X_train_vignette_prompt = X_train_vignette_prompt.values.flatten()
X_train_cot_prompt = X_train_cot_prompt.values.flatten()

# import instructions
simple_instruction_df = pd.read_csv("../../../dat/instructions/simple_instruction.csv", sep =",", index_col = 0)
class_definitions_instruction_df = pd.read_csv("../../../dat/instructions/class_definitions_instruction.csv", sep =",", index_col = 0)
profiled_simple_instruction_df = pd.read_csv("../../../dat/instructions/profiled_simple_instruction.csv", sep =",", index_col = 0)
few_shot_instruction_df = pd.read_csv("../../../dat/instructions/few_shot_instruction.csv", sep =",", index_col = 0)
vignette_instruction_df = pd.read_csv("../../../dat/instructions/vignette_instruction.csv", sep =",", index_col = 0)
cot_instruction_df = pd.read_csv("../../../dat/instructions/cot_instruction.csv", sep =",", index_col = 0)

# convert to string
simple_instruction = simple_instruction_df["0"].iloc[0]
class_definitions_instruction = class_definitions_instruction_df["0"].iloc[0]
profiled_simple_instruction = profiled_simple_instruction_df["0"].iloc[0]
few_shot_instruction = few_shot_instruction_df["0"].iloc[0]
vignette_instruction = vignette_instruction_df["0"].iloc[0]
cot_instruction = cot_instruction_df["0"].iloc[0]

# import retry instructions when output format was wrong
retry_instruction_df = pd.read_csv("../../../dat/instructions/retry_instruction.csv", sep =",", index_col = 0)
retry_cot_instruction_df = pd.read_csv("../../../dat/instructions/retry_cot_instruction.csv", sep =",", index_col = 0)

# import instruction for reason of misclassification
instruction_reason_df = pd.read_csv("../../../dat/instructions/instruction_reason.csv", sep=",", index_col = 0)

# convert to string
retry_instruction = retry_instruction_df["0"].iloc[0]
retry_cot_instruction = retry_cot_instruction_df["0"].iloc[0]

instruction_reason = instruction_reason_df["0"].iloc[0]

# predictors
X = data_change
X = X.drop(["hpi"], axis = 1)

# target
y = data_change["hpi"]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

print("LLMs \n",
      "X_train shape: ", X_train.shape, round(X_train.shape[0]/len(X), 2), "\n",
      "X_test shape: ", X_test.shape, round(X_test.shape[0]/len(X), 2),  "\n",
      "y_train shape: ", y_train.shape, round(y_train.shape[0]/len(y), 2), "\n",
      "y_test shape: ", y_test.shape, round(y_test.shape[0]/len(y), 2), "\n")



#### Helper functions ####

def GPT_create_response(prompt, instruction):
    response = client.responses.create(
        model = model_gpt,
        instructions = instruction,
        input = prompt
    )

    if response.output_text.strip() not in ("YES", "NO"):
        print("\n Invalid output. Retry prompting. \n")
        response = client.responses.create(
            model = model_gpt,
            instructions = retry_instruction,
            input = prompt
        )

    return response.output_text.strip()


def save_prompt_to_csv(response_array, filename):
    # value counts for array
    counts = pd.Series(response_array).value_counts()
    print(counts)

    # convert YES to 1 and NO to 0
    response_array = [re.sub(r'^\[|\]$', '', response.strip()) for response in response_array]
    response_array_val = [1 if response == "YES" else 0 if response == "NO" else np.nan for response in response_array]

    # save the array to a csv file
    df = pd.DataFrame({
        "y_pred": response_array_val
    })
    df.to_csv(f"X_train_pred/GPT/X_train_GPT_{filename}.csv", sep = ",", index = False)


def save_prompt_to_csv_cot(response_array, explanation_array, filename):
    # value counts for array
    counts = pd.Series(response_array).value_counts()
    print(counts)

    # convert YES to 1 and NO to 0
    response_array = [re.sub(r'^\[|\]$', '', response.strip()) for response in response_array]
    response_array_val = [1 if response.strip() == "YES" else 0 if response.strip() == "NO" else np.nan for response
                             in response_array]

    # save the array to a csv file
    df = pd.DataFrame({
        "y_pred": response_array_val,
        "explanation": explanation_array
    })
    df.to_csv(f"X_train_pred/GPT/X_train_GPT_{filename}.csv", sep = ",", index = False)


def calc_time(start, end, filename):
    """
    Calculate the time taken for the prompting and save it to a CSV file.
    """
    time_taken = end - start
    print(f"Time taken: {time_taken} seconds")
    # time_df = pd.DataFrame({"time": [time_taken]})
    # time_df.to_csv(f"../exp/times_LLMs/GPT/time_GPT_{filename}.csv", sep = ",", index = False)
    # return time_taken



# #### 1 Testing prompting ####
#
# client = OpenAI(
#     api_key = os.environ.get("OPENAI_API_KEY"),
# )
#
# # testing
# response = client.responses.create(
#     # model = "o3-2025-04-16",
#     model = "o4-mini",
#     reasoning = {
#         "effort": "medium",
#         "summary": "auto"
#     },
#     instructions = simple_instruction,
#     input = X_test_simple_prompt[0],
#     max_output_tokens = 100
# )
#
# print(response.answer.upper())
#
#
# print(json.dumps(response.to_dict(), indent = 2, ensure_ascii = False))
#
#
# # print(response.output_text)
#
#
# # summary_texts = [
# #     summary.text
# #     for item in response.output if hasattr(item, "summary")
# #     for summary in item.summary if hasattr(summary, "text")
# # ]
#
# summary_texts = [
#     " ".join(
#         summary.text
#         for item in response.output if hasattr(item, "summary")
#         for summary in item.summary if hasattr(summary, "text")
#     )
# ]
#
# print(summary_texts[0])



#### 2 Prompting with ChatGPT ####

model_gpt = "gpt-4.1"

client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY"),
)

# #### Simple prompt ####
#
# y_pred_simple_GPT = []
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_train_simple_prompt, desc = "Simple prompting"):
#     response = GPT_create_response(prompt, simple_instruction)
#     y_pred_simple_GPT.append(response)
#     # print(response)
#
#     if len(y_pred_simple_GPT) % 50 == 0 and len(y_pred_simple_GPT) > 0:
#         print(f"\n\nProcessed {len(y_pred_simple_GPT)} prompts.\n")
#         save_prompt_to_csv(y_pred_simple_GPT, "simple_prompt")
#
# end = time.time()
# calc_time(start, end, "simple_prompt")
#
# # save the array to a csv file
# save_prompt_to_csv(y_pred_simple_GPT, "simple_prompt")
#
#
#
# #### Class definition prompt ####
#
# y_pred_class_def_GPT = []
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_train_class_definitions_prompt, desc = "Class definition prompting"):
#     response = GPT_create_response(prompt, class_definitions_instruction)
#     y_pred_class_def_GPT.append(response)
#     # print(response)
#
#     if len(y_pred_class_def_GPT) % 50 == 0 and len(y_pred_class_def_GPT) > 0:
#         print(f"\n\nProcessed {len(y_pred_class_def_GPT)} prompts.\n")
#         save_prompt_to_csv(y_pred_class_def_GPT, "class_definitions_prompt")
#
# end = time.time()
# calc_time(start, end, "class_definitions_prompt")
#
# # save the array to a csv file
# save_prompt_to_csv(y_pred_class_def_GPT, "class_definitions_prompt")
#
#
#
# #### Profiled simple prompt ####
#
# y_pred_profiled_simple_GPT = []
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_train_profiled_simple_prompt, desc = "Profiled simple prompting"):
#     response = GPT_create_response(prompt, profiled_simple_instruction)
#     y_pred_profiled_simple_GPT.append(response)
#     # print(response)
#
#     if len(y_pred_profiled_simple_GPT) % 50 == 0 and len(y_pred_profiled_simple_GPT) > 0:
#         print(f"\n\nProcessed {len(y_pred_profiled_simple_GPT)} prompts.\n")
#         save_prompt_to_csv(y_pred_profiled_simple_GPT, "profiled_simple_prompt")
#
# end = time.time()
# calc_time(start, end, "profiled_simple_prompt")
#
# # save the array to a csv file
# save_prompt_to_csv(y_pred_profiled_simple_GPT, "profiled_simple_prompt")


#### Few shot prompt ####

y_pred_few_shot_GPT = []

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_train_few_shot_prompt, desc = "Few-shot prompting"):
    response = GPT_create_response(prompt, few_shot_instruction)
    y_pred_few_shot_GPT.append(response)
    # print(response)

    if len(y_pred_few_shot_GPT) % 50 == 0 and len(y_pred_few_shot_GPT) > 0:
        print(f"\n\nProcessed {len(y_pred_few_shot_GPT)} prompts.\n")
        save_prompt_to_csv(y_pred_few_shot_GPT, "few_shot_prompt")

end = time.time()
calc_time(start, end, "few_shot_prompt")

# save the array to a csv file
save_prompt_to_csv(y_pred_few_shot_GPT, "few_shot_prompt")



#### Vignette prompt ####

y_pred_vignette_GPT = []

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_train_vignette_prompt, desc = "Vignette prompting"):
    response = GPT_create_response(prompt, vignette_instruction)
    y_pred_vignette_GPT.append(response)
    # print(response)

    if len(y_pred_vignette_GPT) % 50 == 0 and len(y_pred_vignette_GPT) > 0:
        print(f"\n\nProcessed {len(y_pred_vignette_GPT)} prompts.\n")
        save_prompt_to_csv(y_pred_vignette_GPT, "vignette_prompt")

end = time.time()
calc_time(start, end, "vignette_prompt")

save_prompt_to_csv(y_pred_vignette_GPT, "vignette_prompt")



#### Chain-of-thought prompt ####

y_pred_cot_GPT = []
explanation_cot_GPT = []

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_train_cot_prompt, desc = "Chain-of-thought prompting"):
    response = client.responses.create(
        model = model_gpt,
        instructions = cot_instruction,
        input = prompt
    )

    try:
        prediction = re.findall(r'Prediction: (.*)', response.output_text)[0].strip()
        explanation = re.findall(r'Explanation: (.*)', response.output_text)[0].strip()
        y_pred_cot_GPT.append(prediction)
        explanation_cot_GPT.append(explanation)
        # print(prediction)
    except IndexError:
        print("\n IndexError. Retry prompting. \n")
        response = client.responses.create(
            model = model_gpt,
            instructions = cot_instruction,
            input = prompt
        )

        try:
            prediction = re.findall(r'Prediction: (.*)', response.output_text)[0].strip()
            explanation = re.findall(r'Explanation: (.*)', response.output_text)[0].strip()
            y_pred_cot_GPT.append(prediction)
            explanation_cot_GPT.append(explanation)
        except IndexError:
            print("\n Still IndexError. \n")
            y_pred_cot_GPT.append("IndexError")
            explanation_cot_GPT.append("IndexError")

    if len(y_pred_cot_GPT) % 50 == 0 and len(y_pred_cot_GPT) > 0:
        print(f"\n\nProcessed {len(y_pred_cot_GPT)} prompts.\n")
        save_prompt_to_csv_cot(y_pred_cot_GPT, explanation_cot_GPT, "cot_prompt")

end = time.time()
calc_time(start, end, "cot_prompt")

# save the array to a csv file
save_prompt_to_csv_cot(y_pred_cot_GPT, explanation_cot_GPT, "cot_prompt")
