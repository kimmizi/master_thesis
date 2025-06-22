#### LLM: Zero-shot classification through LLMs and prompts ####
#### GEMMA ####


#### 0 Imports ####
import os
import pandas as pd
import anthropic
import numpy as np
import time
import re
from openai import OpenAI
from tqdm import tqdm
from google import genai
from google.genai import types
from sklearn.model_selection import train_test_split

# import prompts for all train data
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

# convert to string
retry_instruction = retry_instruction_df["0"].iloc[0]
retry_cot_instruction = retry_cot_instruction_df["0"].iloc[0]



#### Helper functions ####

def Gemma_create_response(prompt, instruction):
    time.sleep(15)  # sleep for few seconds to avoid rate limiting
    response = client.models.generate_content(
        model = model_gemma,
        contents = [instruction, prompt]
    )

    if response.text.strip() not in ("YES", "NO"):
        print("\n Invalid output. Retry prompting. \n")
        response = client.models.generate_content(
            model = model_gemma,
            contents = [retry_instruction, prompt]
        )

    return response.text.strip()

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
    df.to_csv(f"X_train_pred/Gemma/X_train_gemma_{filename}.csv", sep = ",", index = False)


def calc_time(start, end, filename):
    """
    Calculate the time taken for the prompting and save it to a CSV file.
    """
    time_taken = end - start
    print(f"Time taken: {time_taken} seconds")


### 2 Prompting with Gemma 3 ####


model_gemma = "gemma-3-27b-it"

client = genai.Client(
    api_key = os.environ.get("GEMINI_API_KEY")
)

#
# #### Simple prompt ####
#
# y_pred_simple_gemma = []
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_train_simple_prompt, desc = "Simple prompting"):
#     response = Gemma_create_response(prompt, simple_instruction)
#     y_pred_simple_gemma.append(response)
#
#     if len(y_pred_simple_gemma) % 50 == 0 and len(y_pred_simple_gemma) > 0:
#         print(f"\n\nProcessed {len(y_pred_simple_gemma)} prompts.\n")
#         save_prompt_to_csv(y_pred_simple_gemma, "simple_prompt")
#
# end = time.time()
# calc_time(start, end, "simple_prompt")
#
# # save the array to a csv file
# save_prompt_to_csv(y_pred_simple_gemma, "simple_prompt")
#
#
#
# #### Class definition prompt ####
#
# y_pred_class_def_gemma = []
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_train_class_definitions_prompt, desc = "Class definition prompting"):
#     response = Gemma_create_response(prompt, class_definitions_instruction)
#     y_pred_class_def_gemma.append(response)
#
#     if len(y_pred_class_def_gemma) % 50 == 0 and len(y_pred_class_def_gemma) > 0:
#         print(f"\n\nProcessed {len(y_pred_class_def_gemma)} prompts.\n")
#         save_prompt_to_csv(y_pred_class_def_gemma, "class_definitions_prompt_2")
#
# end = time.time()
# calc_time(start, end, "class_definitions_prompt_2")
#
# # save the array to a csv file
# save_prompt_to_csv(y_pred_class_def_gemma, "class_definitions_prompt_2")
#
#
#
# #### Profiled simple prompt ####
#
# y_pred_profiled_simple_gemma = []
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_train_profiled_simple_prompt, desc = "Profiled simple prompting"):
#     response = Gemma_create_response(prompt, profiled_simple_instruction)
#     y_pred_profiled_simple_gemma.append(response)
#
#     if len(y_pred_profiled_simple_gemma) % 50 == 0 and len(y_pred_profiled_simple_gemma) > 0:
#         print(f"\n\nProcessed {len(y_pred_profiled_simple_gemma)} prompts.\n")
#         save_prompt_to_csv(y_pred_profiled_simple_gemma, "profiled_simple_prompt")
#
# end = time.time()
# calc_time(start, end, "profiled_simple_prompt")
#
# # save the array to a csv file
# save_prompt_to_csv(y_pred_profiled_simple_gemma, "profiled_simple_prompt")
#
#
#
#### Few shot prompt ####

y_pred_few_shot_gemma = []

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_train_few_shot_prompt, desc = "Few shot prompting"):
    response = Gemma_create_response(prompt, few_shot_instruction)
    y_pred_few_shot_gemma.append(response)

    if len(y_pred_few_shot_gemma) % 50 == 0 and len(y_pred_few_shot_gemma) > 0:
        print(f"\n\nProcessed {len(y_pred_few_shot_gemma)} prompts.\n")
        save_prompt_to_csv(y_pred_few_shot_gemma, "few_shot_prompt")

end = time.time()
calc_time(start, end, "few_shot_prompt")

# save the array to a csv file
save_prompt_to_csv(y_pred_few_shot_gemma, "few_shot_prompt")
#
#
#
# #### Vignette prompt ####
#
# y_pred_vignette_gemma = []
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_train_vignette_prompt, desc = "Vignette prompting"):
#     response = Gemma_create_response(prompt, vignette_instruction)
#     y_pred_vignette_gemma.append(response)
#
#     if len(y_pred_vignette_gemma) % 50 == 0 and len(y_pred_vignette_gemma) > 0:
#         print(f"\n\nProcessed {len(y_pred_vignette_gemma)} prompts.\n")
#         save_prompt_to_csv(y_pred_vignette_gemma, "vignette_prompt")
#
# end = time.time()
# calc_time(start, end, "vignette_prompt")
#
# # save the array to a csv file
# save_prompt_to_csv(y_pred_vignette_gemma, "vignette_prompt")
#
#
#
# #### Chain-of-thought prompt ####
#
# y_pred_cot_gemma = []
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_train_cot_prompt, desc = "Chain-of-thought prompting"):
#     response = Gemma_create_response(prompt, simple_instruction)
#     y_pred_cot_gemma.append(response)
#
#     if len(y_pred_cot_gemma) % 50 == 0 and len(y_pred_cot_gemma) > 0:
#         print(f"\n\nProcessed {len(y_pred_cot_gemma)} prompts.\n")
#         save_prompt_to_csv(y_pred_cot_gemma, "cot_prompt")
#
# end = time.time()
# calc_time(start, end, "cot_prompt")
#
# # save the array to a csv file
# save_prompt_to_csv(y_pred_cot_gemma, "cot_prompt")