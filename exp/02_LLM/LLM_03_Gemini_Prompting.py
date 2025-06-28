#### LLM: Zero-shot classification through LLMs and prompts ####
#### GEMINI ####


#### 0 Imports ####

import os
import pandas as pd
import numpy as np
import time
import re
from tqdm import tqdm
from google import genai
from google.genai import types
from sklearn.model_selection import train_test_split

data_change = pd.read_csv("../../dat/dips/DIPS_Data_cleaned_change.csv", sep =",", low_memory = False)

# import prompts for all test data
X_test_simple_prompt_df = pd.read_csv("../../dat/prompts/X_test_simple_prompt.csv", sep =",", index_col = 0)
X_test_class_definitions_prompt_df = pd.read_csv("../../dat/prompts/X_test_class_definitions_prompt.csv", sep =",", index_col = 0)
X_test_profiled_simple_prompt_df = pd.read_csv("../../dat/prompts/X_test_profiled_simple_prompt.csv", sep =",", index_col = 0)
X_test_few_shot_prompt_df = pd.read_csv("../../dat/prompts/X_test_few_shot_prompt.csv", sep =",", index_col = 0)
X_test_few_shot_prompt_20_df = pd.read_csv("../../dat/prompts/X_test_few_shot_prompt_20.csv", sep=",", index_col=0)
X_test_few_shot_prompt_50_df = pd.read_csv("../../dat/prompts/X_test_few_shot_prompt_50.csv", sep=",", index_col=0)
X_test_few_shot_prompt_100_df = pd.read_csv("../../dat/prompts/X_test_few_shot_prompt_100.csv", sep=",", index_col=0)
X_test_few_shot_prompt_200_df = pd.read_csv("../../dat/prompts/X_test_few_shot_prompt_200.csv", sep=",", index_col=0)
X_test_vignette_prompt_df = pd.read_csv("../../dat/prompts/X_test_vignette_prompt.csv", sep =",", index_col = 0)
X_test_cot_prompt_df = pd.read_csv("../../dat/prompts/X_test_cot_prompt.csv", sep =",", index_col = 0)

# convert to arrays
X_test_simple_prompt = X_test_simple_prompt_df.values.flatten()
X_test_class_definitions_prompt = X_test_class_definitions_prompt_df.values.flatten()
X_test_profiled_simple_prompt = X_test_profiled_simple_prompt_df.values.flatten()
X_test_few_shot_prompt = X_test_few_shot_prompt_df.values.flatten()
X_test_few_shot_prompt_20 = X_test_few_shot_prompt_20_df.values.flatten()
X_test_few_shot_prompt_50 = X_test_few_shot_prompt_50_df.values.flatten()
X_test_few_shot_prompt_100 = X_test_few_shot_prompt_100_df.values.flatten()
X_test_few_shot_prompt_200 = X_test_few_shot_prompt_200_df.values.flatten()
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

def Gemini_create_response(prompt, instruction):
    response = client.models.generate_content(
        model = model_gemini,
        config = types.GenerateContentConfig(
            system_instruction = instruction,
            thinking_config = types.ThinkingConfig(
                include_thoughts = True
            )
        ),
        contents = prompt,
    )

    if response.text.strip() not in ("YES", "NO"):
        print("\n Invalid output. Retry prompting. \n")
        response = client.models.generate_content(
            model = model_gemini,
            config = types.GenerateContentConfig(
                system_instruction = retry_instruction,
                thinking_config = types.ThinkingConfig(
                    include_thoughts = True
                )
            ),
            contents = prompt,
        )

    for part in response.candidates[0].content.parts:
        if not part.text:
            continue
        if part.thought:
            thinking = part.text

    return response.text.strip(), thinking


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
    df.to_csv(f"y_pred_LLMs/Gemini/y_pred_gemini_{filename}.csv", sep = ",", index = False)


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
    df.to_csv(f"y_pred_LLMs/Gemini/y_pred_gemini_{filename}.csv", sep = ",", index = False)


def calc_time(start, end, filename):
    """
    Calculate the time taken for the prompting and save it to a CSV file.
    """
    time_taken = end - start
    print(f"Time taken: {time_taken} seconds")
    time_df = pd.DataFrame({"time": [time_taken]})
    time_df.to_csv(f"times_LLMs/Gemini/time_gemini_{filename}.csv", sep = ",", index = False)
    return time_taken



### 2 Prompting with Gemini 2.5 Pro ####

# model_gemini = "gemini-2.5-pro-preview-05-06"
model_gemini = "gemini-2.5-pro"

client = genai.Client(
    api_key = os.environ.get("GEMINI_API_KEY")
)


# #### Simple prompt ####
#
# y_pred_simple_gemini = []
# thinking_simple_gemini = []
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_test_simple_prompt, desc = "Simple prompting"):
#     response, thinking = Gemini_create_response(prompt, simple_instruction)
#     y_pred_simple_gemini.append(response)
#     thinking_simple_gemini.append(thinking)
#     # print(response)
#
#     if len(y_pred_simple_gemini) % 10 == 0 and len(y_pred_simple_gemini) > 0:
#         print(f"\n\nProcessed {len(y_pred_simple_gemini)} prompts.\n")
#         save_prompt_to_csv(y_pred_simple_gemini, thinking_simple_gemini, "simple_prompt")
#
# end = time.time()
# calc_time(start, end, "simple_prompt")
#
# # save the array to a csv file
# save_prompt_to_csv(y_pred_simple_gemini, thinking_simple_gemini, "simple_prompt")
#
#
#
# #### Class definition prompt ####
#
# y_pred_class_def_gemini = []
# thinking_class_def_gemini = []
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_test_class_definitions_prompt, desc = "Class definitions prompting"):
#     response, thinking = Gemini_create_response(prompt, class_definitions_instruction)
#     y_pred_class_def_gemini.append(response)
#     thinking_class_def_gemini.append(thinking)
#     # print(response)
#
#     if len(y_pred_class_def_gemini) % 10 == 0 and len(y_pred_class_def_gemini) > 0:
#         print(f"\n\nProcessed {len(y_pred_class_def_gemini)} prompts.\n")
#         save_prompt_to_csv(y_pred_class_def_gemini, thinking_class_def_gemini, "class_definitions_prompt")
#
# end = time.time()
# calc_time(start, end, "class_definitions_prompt")
#
# # save the array to a csv file
# save_prompt_to_csv(y_pred_class_def_gemini, thinking_class_def_gemini, "class_definitions_prompt")
#
#
#
# #### Profiled simple prompt ####
#
# y_pred_profiled_simple_gemini = []
# thinking_profiled_simple_gemini = []
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_test_profiled_simple_prompt, desc = "Profiled simple prompting"):
#     response, thinking = Gemini_create_response(prompt, simple_instruction)
#     y_pred_profiled_simple_gemini.append(response)
#     thinking_profiled_simple_gemini.append(thinking)
#     # print(response)
#
#     if len(y_pred_profiled_simple_gemini) % 10 == 0 and len(y_pred_profiled_simple_gemini) > 0:
#         print(f"\n\nProcessed {len(y_pred_profiled_simple_gemini)} prompts.\n")
#         save_prompt_to_csv(y_pred_profiled_simple_gemini, thinking_profiled_simple_gemini, "profiled_simple_prompt")
#
# end = time.time()
# calc_time(start, end, "profiled_simple_prompt")
#
# # save the array to a csv file
# save_prompt_to_csv(y_pred_profiled_simple_gemini, thinking_profiled_simple_gemini, "profiled_simple_prompt")




#### Few shot prompt ####

y_pred_few_shot_gemini = []
thinking_few_shot_gemini = []

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_few_shot_prompt_50[190:], desc = "Few-shot prompting"):
    response, thinking = Gemini_create_response(prompt, few_shot_instruction)
    y_pred_few_shot_gemini.append(response)
    thinking_few_shot_gemini.append(thinking)
    # print(response)

    if len(y_pred_few_shot_gemini) % 10 == 0 and len(y_pred_few_shot_gemini) > 0:
        print(f"\n\nProcessed {len(y_pred_few_shot_gemini)} prompts.\n")
        save_prompt_to_csv(y_pred_few_shot_gemini, thinking_few_shot_gemini, "few_shot_prompt_50_2")

end = time.time()
calc_time(start, end, "few_shot_prompt_50_2")

# save the array to a csv file
save_prompt_to_csv(y_pred_few_shot_gemini, thinking_few_shot_gemini, "few_shot_prompt_50_2")



# #### Vignette prompt ####
#
# y_pred_vignette_gemini = []
# thinking_vignette_gemini = []
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_test_vignette_prompt, desc = "Vignette prompting"):
#     response, thinking = Gemini_create_response(prompt, vignette_instruction)
#     y_pred_vignette_gemini.append(response)
#     thinking_vignette_gemini.append(thinking)
#     # print(response)
#
#     if len(y_pred_vignette_gemini) % 10 == 0 and len(y_pred_vignette_gemini) > 0:
#         print(f"\n\nProcessed {len(y_pred_vignette_gemini)} prompts.\n")
#         save_prompt_to_csv(y_pred_vignette_gemini, thinking_vignette_gemini, "vignette_prompt")
#
# end = time.time()
# calc_time(start, end, "vignette_prompt")
#
# # save the array to a csv file
# save_prompt_to_csv(y_pred_vignette_gemini, thinking_vignette_gemini, "vignette_prompt")
#
#
#
# #### Chain-of-thought prompt ####
#
# y_pred_cot_gemini = []
# thinking_cot_gemini = []
# explanation_cot_gemini = []
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_test_cot_prompt, desc = "Chain-of-thought prompting"):
#     response = client.models.generate_content(
#         model = "gemini-2.5-pro-preview-05-06",
#         config = types.GenerateContentConfig(
#             system_instruction = cot_instruction,
#             thinking_config = types.ThinkingConfig(
#                 include_thoughts = True
#             )
#         ),
#         contents = prompt,
#     )
#
#     try:
#         prediction = re.findall(r'Prediction: (.*)', response.text)[0].strip()
#         explanation = re.findall(r'Explanation: (.*)', response.text)[0].strip()
#         for part in response.candidates[0].content.parts:
#             if not part.text:
#                 continue
#             if part.thought:
#                 thinking_cot_gemini.append(part.text)
#         y_pred_cot_gemini.append(prediction)
#         explanation_cot_gemini.append(explanation)
#         # print(prediction)
#
#     except IndexError:
#         print("\n IndexError. Retry prompting. \n")
#         response = client.models.generate_content(
#             model = "gemini-2.5-pro-preview-05-06",
#             config = types.GenerateContentConfig(
#                 system_instruction = cot_instruction,
#                 thinking_config = types.ThinkingConfig(
#                     include_thoughts = True
#                 )
#             ),
#             contents = prompt,
#         )
#
#         try:
#             prediction = re.findall(r'Prediction: (.*)', response.text)[0].strip()
#             explanation = re.findall(r'Explanation: (.*)', response.text)[0].strip()
#             for part in response.candidates[0].content.parts:
#                 if not part.text:
#                     continue
#                 if part.thought:
#                     thinking_cot_gemini.append(part.text)
#             y_pred_cot_gemini.append(prediction)
#             explanation_cot_gemini.append(explanation)
#
#         except IndexError:
#             print("\n Still IndexError. Don't retry prompting. \n")
#             y_pred_cot_gemini.append("IndexError")
#             explanation_cot_gemini.append("IndexError")
#
#     if len(y_pred_cot_gemini) % 10 == 0 and len(y_pred_cot_gemini) > 0:
#         print(f"\n\nProcessed {len(y_pred_cot_gemini)} prompts.\n")
#         save_prompt_to_csv_cot(y_pred_cot_gemini, thinking_cot_gemini, explanation_cot_gemini, "cot_prompt")
#
# end = time.time()
# calc_time(start, end, "cot_prompt")
#
# # save the array to a csv file
# save_prompt_to_csv_cot(y_pred_cot_gemini, thinking_cot_gemini, explanation_cot_gemini, "cot_prompt")
