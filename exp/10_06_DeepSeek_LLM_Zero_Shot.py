#### LLM: Zero-shot classification through LLMs and prompts ####
#### DEEPSEEK ####


#### 0 Imports ####
import os
import pandas as pd
import numpy as np
import time
import re
from tqdm import tqdm
from openai import OpenAI
from sklearn.model_selection import train_test_split

data_change = pd.read_csv("../dat/dips/DIPS_Data_cleaned_change.csv", sep = ",", low_memory = False)

# import prompts for all test data
X_test_simple_prompt_df = pd.read_csv("../dat/prompts/X_test_simple_prompt.csv", sep = ",", index_col = 0)
X_test_class_definitions_prompt_df = pd.read_csv("../dat/prompts/X_test_class_definitions_prompt.csv", sep = ",", index_col = 0)
X_test_profiled_simple_prompt_df = pd.read_csv("../dat/prompts/X_test_profiled_simple_prompt.csv", sep = ",", index_col = 0)
X_test_few_shot_prompt_df = pd.read_csv("../dat/prompts/X_test_few_shot_prompt.csv", sep = ",", index_col = 0)
X_test_vignette_prompt_df = pd.read_csv("../dat/prompts/X_test_vignette_prompt.csv", sep = ",", index_col = 0)
X_test_cot_prompt_df = pd.read_csv("../dat/prompts/X_test_cot_prompt.csv", sep = ",", index_col = 0)

# convert to arrays
X_test_simple_prompt = X_test_simple_prompt_df.values.flatten()
X_test_class_definitions_prompt = X_test_class_definitions_prompt_df.values.flatten()
X_test_profiled_simple_prompt = X_test_profiled_simple_prompt_df.values.flatten()
X_test_few_shot_prompt = X_test_few_shot_prompt_df.values.flatten()
X_test_vignette_prompt = X_test_vignette_prompt_df.values.flatten()
X_test_cot_prompt = X_test_cot_prompt_df.values.flatten()

# import instructions
simple_instruction_df = pd.read_csv("../dat/instructions/simple_instruction.csv", sep = ",", index_col = 0)
class_definitions_instruction_df = pd.read_csv("../dat/instructions/class_definitions_instruction.csv", sep = ",", index_col = 0)
profiled_simple_instruction_df = pd.read_csv("../dat/instructions/profiled_simple_instruction.csv", sep = ",", index_col = 0)
few_shot_instruction_df = pd.read_csv("../dat/instructions/few_shot_instruction.csv", sep = ",", index_col = 0)
vignette_instruction_df = pd.read_csv("../dat/instructions/vignette_instruction.csv", sep = ",", index_col = 0)
cot_instruction_df = pd.read_csv("../dat/instructions/cot_instruction.csv", sep = ",", index_col = 0)

# convert to string
simple_instruction = simple_instruction_df["0"].iloc[0]
class_definitions_instruction = class_definitions_instruction_df["0"].iloc[0]
profiled_simple_instruction = profiled_simple_instruction_df["0"].iloc[0]
few_shot_instruction = few_shot_instruction_df["0"].iloc[0]
vignette_instruction = vignette_instruction_df["0"].iloc[0]
cot_instruction = cot_instruction_df["0"].iloc[0]

# import retry instructions when output format was wrong
retry_instruction_df = pd.read_csv("../dat/instructions/retry_instruction.csv", sep = ",", index_col = 0)
retry_cot_instruction_df = pd.read_csv("../dat/instructions/retry_cot_instruction.csv", sep = ",", index_col = 0)

# import instruction for reason of misclassification
instruction_reason_df = pd.read_csv("../dat/instructions/instruction_reason.csv", sep=",", index_col = 0)

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

def DeepSeek_create_response(prompt, instruction):
    response = client.chat.completions.create(
        model = model_deeps,
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ],
        stream = False
    )

    if response.choices[0].message.content.strip() not in ("YES", "NO"):
        print("\n Invalid output. Retry prompting. \n")
        response = client.chat.completions.create(
            model = model_deeps,
            messages = [
                {"role": "system", "content": retry_instruction},
                {"role": "user", "content": prompt},
            ],
            stream = False
        )

    resp = response.choices[0].message.content.strip()
    thinking = response.choices[0].message.reasoning_content

    return resp, thinking


def save_prompt_to_csv(response_array, thinking_array, filename):
    # value counts for array
    counts = pd.Series(response_array).value_counts()
    print(counts)

    # convert YES to 1 and NO to 0
    response_array = [re.sub(r'^\[|\]$', '', response.strip()) for response in response_array]
    response_array_val = [1 if response == "YES" else 0 if response == "NO" else np.nan for response in response_array]

    # save the array to a csv file
    df = pd.DataFrame({
        "y_pred": response_array_val,
        "thinking": thinking_array
    })
    df.to_csv(f"../exp/y_pred_LLMs/DeepSeek/y_pred_deeps_{filename}.csv", sep = ",", index = False)


def save_prompt_to_csv_cot(response_array, thinking_array, explanation_array, filename):
    # value counts for array
    counts = pd.Series(response_array).value_counts()
    print(counts)

    # convert YES to 1 and NO to 0
    response_array = [re.sub(r'^\[|\]$', '', response.strip()) for response in response_array]
    response_array_val = [1 if response == "YES" else 0 if response == "NO" else np.nan for response in response_array]

    # save the array to a csv file
    df = pd.DataFrame({
        "y_pred": response_array_val,
        "thinking": thinking_array,
        "explanation": explanation_array
    })
    df.to_csv(f"../exp/y_pred_LLMs/DeepSeek/y_pred_deeps_{filename}.csv", sep = ",", index = False)


def calc_time(start, end, filename):
    """
    Calculate the time taken for the prompting and save it to a CSV file.
    """
    time_taken = end - start
    print(f"Time taken: {time_taken} seconds")
    time_df = pd.DataFrame({"time": [time_taken]})
    time_df.to_csv(f"../exp/times_LLMs/DeepSeek/time_deeps_{filename}.csv", sep = ",", index = False)
    return time_taken



# **Off-Peak Discounts**：DeepSeek-R1 with 75% off at off-peak hours (16:30-00:30 UTC daily)

#### 1 Testing prompting ####

# client = OpenAI(api_key = os.environ.get("DeepSeek_API_Key"), base_url = "https://api.deepseek.com")
#
# response = client.chat.completions.create(
#     model = "deepseek-reasoner",
#     messages = [
#         {"role": "system", "content": simple_instruction},
#         {"role": "user", "content": X_test_simple_prompt[0]},
#     ],
#     stream = False
# )
#
# print(response.choices[0].message.content)
#
# response.choices[0].message.reasoning_content



#### 2 Prompting with DeepSeek Reasoning R1 ####

model_deeps = "deepseek-reasoner"

client = OpenAI(
    api_key = os.environ.get("DeepSeek_API_Key"),
    base_url = "https://api.deepseek.com"
)


# #### Simple prompt ####
#
# y_pred_simple_deeps = []
# thinking_simple_deeps = []
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_test_simple_prompt, desc = "Simple prompting", unit = "prompt"):
#     response, thinking = DeepSeek_create_response(prompt, simple_instruction)
#     y_pred_simple_deeps.append(response)
#     thinking_simple_deeps.append(thinking)
#     # print(response)
#
#     if len(y_pred_simple_deeps) % 50 == 0 and len(y_pred_simple_deeps) > 0:
#         print(f"\n\nProcessed {len(y_pred_simple_deeps)} prompts.\n")
#         save_prompt_to_csv(y_pred_simple_deeps, thinking_simple_deeps, "simple_prompt")
#
# end = time.time()
# calc_time(start, end, "simple_prompt")
#
# # save the array to a csv file
# save_prompt_to_csv(y_pred_simple_deeps, thinking_simple_deeps, "simple_prompt")
#
#
#
# #### Class definition prompt ####
#
# y_pred_class_def_deeps = []
# thinking_class_def_deeps = []
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_test_class_definitions_prompt, desc = "Class definitions prompting", unit = "prompt"):
#     response, thinking = DeepSeek_create_response(prompt, class_definitions_instruction)
#     y_pred_class_def_deeps.append(response)
#     thinking_class_def_deeps.append(thinking)
#     # print(response)
#
#     if len(y_pred_class_def_deeps) % 50 == 0 and len(y_pred_class_def_deeps) > 0:
#         print(f"\n\nProcessed {len(y_pred_class_def_deeps)} prompts.\n")
#         save_prompt_to_csv(y_pred_class_def_deeps, thinking_class_def_deeps, "class_definitions_prompt")
#
# end = time.time()
# calc_time(start, end, "class_definitions_prompt")
#
# # save the array to a csv file
# save_prompt_to_csv(y_pred_class_def_deeps, thinking_class_def_deeps, "class_definitions_prompt")
#
#
#
# #### Profiled simple prompt ####
#
# y_pred_profiled_simple_deeps = []
# thinking_profiled_simple_deeps = []
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_test_profiled_simple_prompt, desc = "Profiled simple prompting", unit = "prompt"):
#     response, thinking = DeepSeek_create_response(prompt, profiled_simple_instruction)
#     y_pred_profiled_simple_deeps.append(response)
#     thinking_profiled_simple_deeps.append(thinking)
#     # print(response)
#
#     if len(y_pred_profiled_simple_deeps) % 50 == 0 and len(y_pred_profiled_simple_deeps) > 0:
#         print(f"\n\nProcessed {len(y_pred_profiled_simple_deeps)} prompts.\n")
#         save_prompt_to_csv(y_pred_profiled_simple_deeps, thinking_profiled_simple_deeps, "profiled_simple_prompt")
#
# end = time.time()
# calc_time(start, end, "profiled_simple_prompt")
#
# # save the array to a csv file
# save_prompt_to_csv(y_pred_profiled_simple_deeps, thinking_profiled_simple_deeps, "profiled_simple_prompt")
#
#
#
# #### Few shot prompt ####
#
# y_pred_few_shot_deeps = []
# thinking_few_shot_deeps = []
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_test_few_shot_prompt[100:], desc = "Few shot prompting", unit = "prompt"):
#     response, thinking = DeepSeek_create_response(prompt, few_shot_instruction)
#     y_pred_few_shot_deeps.append(response)
#     thinking_few_shot_deeps.append(thinking)
#     # print(response)
#
#     if len(y_pred_few_shot_deeps) % 50 == 0 and len(y_pred_few_shot_deeps) > 0:
#         print(f"\n\nProcessed {len(y_pred_few_shot_deeps)} prompts.\n")
#         save_prompt_to_csv(y_pred_few_shot_deeps, thinking_few_shot_deeps, "few_shot_prompt")
#
# end = time.time()
# calc_time(start, end, "few_shot_prompt")
#
# # save the array to a csv file
# save_prompt_to_csv(y_pred_few_shot_deeps, thinking_few_shot_deeps, "few_shot_prompt")
#
#
#
# #### Vignette prompt ####
#
# y_pred_vignette_deeps = []
# thinking_vignette_deeps = []
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_test_vignette_prompt, desc = "Vignette prompting", unit = "prompt"):
#     response, thinking = DeepSeek_create_response(prompt, vignette_instruction)
#     y_pred_vignette_deeps.append(response)
#     thinking_vignette_deeps.append(thinking)
#     # print(response)
#
#     if len(y_pred_vignette_deeps) % 50 == 0 and len(y_pred_vignette_deeps) > 0:
#         print(f"\n\nProcessed {len(y_pred_vignette_deeps)} prompts.\n")
#         save_prompt_to_csv(y_pred_vignette_deeps, thinking_vignette_deeps, "vignette_prompt_new")
#
# end = time.time()
# calc_time(start, end, "vignette_prompt_new")
#
# # save the array to a csv file
# save_prompt_to_csv(y_pred_vignette_deeps, thinking_vignette_deeps, "vignette_prompt_new")
#
#
#
# #### Chain-of-thought prompt ####
#
# y_pred_cot_deeps = []
# explanation_cot_deeps = []
# thinking_cot_deeps = []
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_test_cot_prompt, desc = "Chain-of-thought prompting", unit = "prompt"):
#     response = client.chat.completions.create(
#         model = "deepseek-reasoner",
#         messages = [
#             {"role": "system", "content": cot_instruction},
#             {"role": "user", "content": prompt},
#         ],
#         stream = False
#     )
#
#     try:
#         prediction = re.findall(r'Prediction: (.*)', response.choices[0].message.content)[0].strip()
#         explanation = re.findall(r'Explanation: (.*)', response.choices[0].message.content)[0].strip()
#         y_pred_cot_deeps.append(prediction)
#         explanation_cot_deeps.append(explanation)
#         thinking_cot_deeps.append(response.choices[0].message.reasoning_content)
#         # print(prediction)
#     except IndexError:
#         print("IndexError")
#         y_pred_cot_deeps.append("IndexError")
#         explanation_cot_deeps.append("IndexError")
#         thinking_cot_deeps.append("IndexError")
#
#     if len(y_pred_cot_deeps) % 50 == 0 and len(y_pred_cot_deeps) > 0:
#         print(f"\n\nProcessed {len(y_pred_cot_deeps)} prompts.\n")
#         save_prompt_to_csv_cot(y_pred_cot_deeps, thinking_cot_deeps, explanation_cot_deeps, "cot_prompt")
#
# end = time.time()
# calc_time(start, end, "cot_prompt")
#
# # save the array to a csv file
# save_prompt_to_csv_cot(y_pred_cot_deeps, thinking_cot_deeps, explanation_cot_deeps, "cot_prompt")