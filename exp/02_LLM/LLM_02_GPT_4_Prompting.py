#### LLM: Zero-shot classification through LLMs and prompts ####
#### CHATGPT ####


#### 0 Imports ####

import os
import pandas as pd
import numpy as np
import time
import re
from tqdm import tqdm
from openai import OpenAI

# import prompts for all test data
X_test_simple_prompt_df = pd.read_csv("../../dat/prompts/X_test_simple_prompt.csv", sep =",", index_col = 0)
X_test_class_definitions_prompt_df = pd.read_csv("../../dat/prompts/X_test_class_definitions_prompt.csv", sep =",", index_col = 0)
X_test_profiled_simple_prompt_df = pd.read_csv("../../dat/prompts/X_test_profiled_simple_prompt.csv", sep =",", index_col = 0)
X_test_few_shot_prompt_df = pd.read_csv("../../dat/prompts/X_test_few_shot_prompt.csv", sep =",", index_col = 0)
X_test_few_shot_prompt_100_df = pd.read_csv("../../dat/prompts/X_test_few_shot_prompt_100.csv", sep =",", index_col = 0)
X_test_few_shot_prompt_200_df = pd.read_csv("../../dat/prompts/X_test_few_shot_prompt_200.csv", sep =",", index_col = 0)
X_test_vignette_prompt_df = pd.read_csv("../../dat/prompts/X_test_vignette_prompt.csv", sep =",", index_col = 0)
X_test_cot_prompt_df = pd.read_csv("../../dat/prompts/X_test_cot_prompt.csv", sep =",", index_col = 0)

# convert to arrays
X_test_simple_prompt = X_test_simple_prompt_df.values.flatten()
X_test_class_definitions_prompt = X_test_class_definitions_prompt_df.values.flatten()
X_test_profiled_simple_prompt = X_test_profiled_simple_prompt_df.values.flatten()
X_test_few_shot_prompt = X_test_few_shot_prompt_df.values.flatten()
X_test_few_shot_prompt_100 = X_test_few_shot_prompt_df.values.flatten()
X_test_few_shot_prompt_200 = X_test_few_shot_prompt_df.values.flatten()
X_test_vignette_prompt = X_test_vignette_prompt_df.values.flatten()
X_test_cot_prompt = X_test_cot_prompt_df.values.flatten()

# import instructions
simple_instruction_df = pd.read_csv("../../dat/instructions/simple_instruction.csv", sep =",", index_col = 0)
class_definitions_instruction_df = pd.read_csv("../../dat/instructions/class_definitions_instruction.csv", sep =",", index_col = 0)
profiled_simple_instruction_df = pd.read_csv("../../dat/instructions/profiled_simple_instruction.csv", sep =",", index_col = 0)
few_shot_instruction_df = pd.read_csv("../../dat/instructions/few_shot_instruction.csv", sep =",", index_col = 0)
vignette_instruction_df = pd.read_csv("../../dat/instructions/vignette_instruction.csv", sep =",", index_col = 0)
cot_instruction_df = pd.read_csv("../../dat/instructions/cot_instruction.csv", sep =",", index_col = 0)

# convert to string
simple_instruction = simple_instruction_df["0"].iloc[0]
class_definitions_instruction = class_definitions_instruction_df["0"].iloc[0]
profiled_simple_instruction = profiled_simple_instruction_df["0"].iloc[0]
few_shot_instruction = few_shot_instruction_df["0"].iloc[0]
vignette_instruction = vignette_instruction_df["0"].iloc[0]
cot_instruction = cot_instruction_df["0"].iloc[0]

# import retry instructions when output format was wrong
retry_instruction_df = pd.read_csv("../../dat/instructions/retry_instruction.csv", sep =",", index_col = 0)
retry_cot_instruction_df = pd.read_csv("../../dat/instructions/retry_cot_instruction.csv", sep =",", index_col = 0)

# import instruction for reason of misclassification
instruction_reason_df = pd.read_csv("../../dat/instructions/instruction_reason.csv", sep=",", index_col = 0)

# convert to string
retry_instruction = retry_instruction_df["0"].iloc[0]
retry_cot_instruction = retry_cot_instruction_df["0"].iloc[0]

instruction_reason = instruction_reason_df["0"].iloc[0]



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
    df.to_csv(f"y_pred_LLMs/GPT/y_pred_GPT_{filename}.csv", sep = ",", index = False)


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
    df.to_csv(f"y_pred_LLMs/GPT/y_pred_GPT_{filename}.csv", sep = ",", index = False)


def calc_time(start, end, filename):
    """
    Calculate the time taken for the prompting and save it to a CSV file.
    """
    time_taken = end - start
    print(f"Time taken: {time_taken} seconds")
    time_df = pd.DataFrame({"time": [time_taken]})
    time_df.to_csv(f"times_LLMs/GPT/time_GPT_{filename}.csv", sep = ",", index = False)
    return time_taken



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
# for prompt in tqdm(X_test_simple_prompt, desc = "Simple prompting"):
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
# for prompt in tqdm(X_test_class_definitions_prompt, desc = "Class definition prompting"):
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
# for prompt in tqdm(X_test_profiled_simple_prompt, desc = "Profiled simple prompting"):
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
for prompt in tqdm(X_test_few_shot_prompt_100, desc = "Few-shot prompting"):
    response = GPT_create_response(prompt, few_shot_instruction)
    y_pred_few_shot_GPT.append(response)
    # print(response)

    if len(y_pred_few_shot_GPT) % 50 == 0 and len(y_pred_few_shot_GPT) > 0:
        print(f"\n\nProcessed {len(y_pred_few_shot_GPT)} prompts.\n")
        save_prompt_to_csv(y_pred_few_shot_GPT, "few_shot_prompt_100")

end = time.time()
calc_time(start, end, "few_shot_prompt_100")

# save the array to a csv file
save_prompt_to_csv(y_pred_few_shot_GPT, "few_shot_prompt_100")



# #### Vignette prompt ####
#
# y_pred_vignette_GPT = []
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_test_vignette_prompt, desc = "Vignette prompting"):
#     response = GPT_create_response(prompt, vignette_instruction)
#     y_pred_vignette_GPT.append(response)
#     # print(response)
#
#     if len(y_pred_vignette_GPT) % 50 == 0 and len(y_pred_vignette_GPT) > 0:
#         print(f"\n\nProcessed {len(y_pred_vignette_GPT)} prompts.\n")
#         save_prompt_to_csv(y_pred_vignette_GPT, "vignette_prompt")
#
# end = time.time()
# calc_time(start, end, "vignette_prompt")
#
# save_prompt_to_csv(y_pred_vignette_GPT, "vignette_prompt")
#
#
#
# #### Chain-of-thought prompt ####
#
# y_pred_cot_GPT = []
# explanation_cot_GPT = []
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_test_cot_prompt, desc = "Chain-of-thought prompting"):
#     response = client.responses.create(
#         model = model_gpt,
#         instructions = cot_instruction,
#         input = prompt
#     )
#
#     try:
#         prediction = re.findall(r'Prediction: (.*)', response.output_text)[0].strip()
#         explanation = re.findall(r'Explanation: (.*)', response.output_text)[0].strip()
#         y_pred_cot_GPT.append(prediction)
#         explanation_cot_GPT.append(explanation)
#         # print(prediction)
#     except IndexError:
#         print("IndexError")
#         y_pred_cot_GPT.append("IndexError")
#         explanation_cot_GPT.append("IndexError")
#
#     if len(y_pred_cot_GPT) % 50 == 0 and len(y_pred_cot_GPT) > 0:
#         print(f"\n\nProcessed {len(y_pred_cot_GPT)} prompts.\n")
#         save_prompt_to_csv_cot(y_pred_cot_GPT, explanation_cot_GPT, "cot_prompt")
#
# end = time.time()
# calc_time(start, end, "cot_prompt")
#
# # save the array to a csv file
# save_prompt_to_csv_cot(y_pred_cot_GPT, explanation_cot_GPT, "cot_prompt")