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



#### 1 Testing prompting ####

simple_instruction = "Respond only with YES or NO."
# simple_instruction = "You are an expert psychologist tasked with predicting whether an individual will develop a psychological disorder between two time points (T1 and T2) based on various psychological measures and demographic information. Your goal is to provide an accurate YES or NO prediction."


from pydantic import BaseModel, Field

# Step 1: Define schema using Pydantic
class YesNoResponse(BaseModel):
    answer: str = Field(..., description="Must be either YES or NO")


client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY"),
)

# testing
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

response = client.beta.chat.completions.parse(
    model="o4-mini",
    messages=[
        {"role": "system", "content": simple_instruction},
        {"role": "user", "content": X_test_simple_prompt[0]}
    ],
    response_format=YesNoResponse
    # max_tokens=20
)

print(response.answer.upper())


print(json.dumps(response.to_dict(), indent = 2, ensure_ascii = False))


# print(response.output_text)


# summary_texts = [
#     summary.text
#     for item in response.output if hasattr(item, "summary")
#     for summary in item.summary if hasattr(summary, "text")
# ]

summary_texts = [
    " ".join(
        summary.text
        for item in response.output if hasattr(item, "summary")
        for summary in item.summary if hasattr(summary, "text")
    )
]

print(summary_texts[0])



#### 2 Prompting with ChatGPT-o3 ####



#### Simple prompt ####

model_gpt = "o3-2025-04-16" # "gpt-4.1"

# y_pred_simple_GPT = []
#
# client = OpenAI(
#     api_key = os.environ.get("OPENAI_API_KEY"),
# )
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_test_simple_prompt, desc = "Simple prompting"):
#     response = client.responses.create(
#         model = model_gpt,
#         reasoning = {"effort": "medium"},
#         instructions = simple_instruction,
#         input = prompt,
#         max_output_tokens = 10
#     )
#
#     if response.output_text.strip() not in ("YES", "NO"):
#         print("\n Invalid output. Retry prompting. \n")
#         response = client.responses.create(
#             model = model_gpt,
#             instructions = retry_instruction,
#             input = prompt
#         )
#
#     y_pred_simple_GPT.append(response.output_text)
#     # print(response.output_text)
#
#     # save responses to csv after every 50th prompt
#     if len(y_pred_simple_GPT) % 50 == 0:
#         # print("\n\n prompt", len(y_pred_simple_GPT))
#         # # value counts for array
#         # counts_simple_GPT = pd.Series(y_pred_simple_GPT).value_counts()
#         # print(counts_simple_GPT)
#
#         # convert YES to 1 and NO to 0
#         y_pred_simple_GPT = [1 if response == "YES" else 0 if response == "NO" else np.nan for response in y_pred_simple_GPT]
#
#         # save the array to a csv file
#         simple_df_GPT = pd.DataFrame(y_pred_simple_GPT, columns = ["y_pred"])
#         simple_df_GPT.to_csv("../exp/y_pred_LLMs/GPT/y_pred_GPT_simple_prompt.csv", sep = ",", index = False)
#         print("\n\n csv saved \n\n")
#
# end = time.time()
# print(f"Time taken: {end - start} seconds")
# time_GPT_simple_prompt = end - start
# time_GPT_simple_df = pd.DataFrame({"time": [time_GPT_simple_prompt]})
# time_GPT_simple_df.to_csv("../exp/times_LLMs/GPT/time_GPT_simple_prompt.csv", sep = ",", index = False)
#
# # value counts for array
# counts_simple_GPT = pd.Series(y_pred_simple_GPT).value_counts()
# print(counts_simple_GPT)
#
# # convert YES to 1 and NO to 0
# y_pred_simple_GPT = [1 if response == "YES" else 0 if response == "NO" else np.nan for response in y_pred_simple_GPT]
#
# # save the array to a csv file
# simple_df_GPT = pd.DataFrame(y_pred_simple_GPT, columns = ["y_pred"])
# simple_df_GPT.to_csv("../exp/y_pred_LLMs/GPT/y_pred_GPT_simple_prompt.csv", sep = ",", index = False)



#### Class definition prompt ####

# y_pred_class_def_GPT = []
#
# client = OpenAI(
#     api_key = os.environ.get("OPENAI_API_KEY"),
# )
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_test_class_definitions_prompt, desc = "Class definition prompting"):
#     response = client.responses.create(
#         model = model_gpt,
#         instructions = class_definitions_instruction,
#         input = prompt
#     )
#
#     if response.output_text.strip() not in ("YES", "NO"):
#         print("\n Invalid output. Retry prompting. \n")
#         response = client.responses.create(
#             model = model_gpt,
#             instructions = retry_instruction,
#             input = prompt
#         )
#
#     y_pred_class_def_GPT.append(response.output_text)
#     print(response.output_text)
#
# end = time.time()
# print(f"Time taken: {end - start} seconds")
# time_GPT_class_definitions = end - start
# time_GPT_class_definitions_df = pd.DataFrame({"time": [time_GPT_class_definitions]})
# time_GPT_class_definitions_df.to_csv("../exp/times_LLMs/GPT/time_GPT_class_definitions_prompt.csv", sep = ",", index = False)
#
# # value counts for array
# counts_class_def_GPT = pd.Series(y_pred_class_def_GPT).value_counts()
# print(counts_class_def_GPT)
#
# # convert YES to 1 and NO to 0
# y_pred_class_def_GPT = [1 if response == "YES" else 0 for response in y_pred_class_def_GPT]
#
# # save the array to a csv file
# class_def_df_GPT = pd.DataFrame(y_pred_class_def_GPT, columns = ["y_pred"])
# class_def_df_GPT.to_csv("../exp/y_pred_LLMs/GPT/y_pred_GPT_class_definitions_prompt.csv", sep = ",", index = False)
#
#
#
# #### Profiled simple prompt ####
#
# y_pred_profiled_simple_GPT = []
#
# client = OpenAI(
#     api_key = os.environ.get("OPENAI_API_KEY"),
# )
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_test_profiled_simple_prompt, desc = "Profiled simple prompting"):
#     response = client.responses.create(
#         model = model_gpt,
#         instructions = profiled_simple_instruction,
#         input = prompt
#     )
#
#     if response.output_text.strip() not in ("YES", "NO"):
#         print("\n Invalid output. Retry prompting. \n")
#         response = client.responses.create(
#             model = model_gpt,
#             instructions = retry_instruction,
#             input = prompt
#         )
#
#     y_pred_profiled_simple_GPT.append(response.output_text)
#     print(response.output_text)
#
# end = time.time()
# print(f"Time taken: {end - start} seconds")
# time_GPT_profiled_simple = end - start
# time_GPT_profiled_simple_df = pd.DataFrame({"time": [time_GPT_profiled_simple]})
# time_GPT_profiled_simple_df.to_csv("../exp/times_LLMs/GPT/time_GPT_profiled_simple_prompt.csv", sep = ",", index = False)
#
# # value counts for array
# counts_profiled_simple_GPT = pd.Series(y_pred_profiled_simple_GPT).value_counts()
# print(counts_profiled_simple_GPT)
#
# # convert YES to 1 and NO to 0
# y_pred_profiled_simple_GPT_val = [1 if response == "YES" else 0 for response in y_pred_profiled_simple_GPT]
#
# # save the array to a csv file
# profiled_simple_df_GPT = pd.DataFrame(y_pred_profiled_simple_GPT_val, columns = ["y_pred"])
# profiled_simple_df_GPT.to_csv("../exp/y_pred_LLMs/GPT/y_pred_GPT_profiled_simple_prompt.csv", sep = ",", index = False)
#
#
#
# #### Few shot prompt ####
#
# y_pred_few_shot_GPT = []
#
# client = OpenAI(
#     api_key = os.environ.get("OPENAI_API_KEY"),
# )
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_test_few_shot_prompt, desc = "Few-shot prompting"):
#     response = client.responses.create(
#         model = model_gpt,
#         instructions = few_shot_instruction,
#         input = prompt
#     )
#
#     if response.output_text.strip() not in ("YES", "NO"):
#         print("\n Invalid output. Retry prompting. \n")
#         response = client.responses.create(
#             model = model_gpt,
#             instructions = retry_instruction,
#             input = prompt
#         )
#
#     y_pred_few_shot_GPT.append(response.output_text)
#     print(response.output_text)
#
# end = time.time()
# print(f"Time taken: {end - start} seconds")
# time_GPT_few_shot = end - start
# time_GPT_few_shot_df = pd.DataFrame({"time": [time_GPT_few_shot]})
# time_GPT_few_shot_df.to_csv("../exp/times_LLMs/GPT/time_GPT_few_shot_prompt.csv", sep = ",", index = False)
#
# # value counts for array
# counts_few_shot_GPT = pd.Series(y_pred_few_shot_GPT).value_counts()
# print(counts_few_shot_GPT)
#
# # convert YES to 1 and NO to 0
# y_pred_few_shot_GPT_val = [1 if response == "YES" else 0 for response in y_pred_few_shot_GPT]
#
# # save the array to a csv file
# few_shot_df_GPT = pd.DataFrame(y_pred_few_shot_GPT_val, columns = ["y_pred"])
# few_shot_df_GPT.to_csv("../exp/y_pred_LLMs/GPT/y_pred_GPT_few_shot_prompt.csv", sep = ",", index = False)
#
#
#
# #### Vignette prompt ####
#
# y_pred_vignette_GPT = []
#
# client = OpenAI(
#     api_key = os.environ.get("OPENAI_API_KEY"),
# )
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_test_vignette_prompt, desc = "Vignette prompting"):
#     response = client.responses.create(
#         model = model_gpt,
#         instructions = vignette_instruction,
#         input = prompt
#     )
#
#     if response.output_text.strip() not in ("YES", "NO"):
#         print("\n Invalid output. Retry prompting. \n")
#         response = client.responses.create(
#             model = model_gpt,
#             instructions = retry_instruction,
#             input = prompt
#         )
#
#     y_pred_vignette_GPT.append(response.output_text)
#     print(response.output_text)
#
# end = time.time()
# print(f"Time taken: {end - start} seconds")
# time_GPT_vignette = end - start
# time_GPT_vignette_df = pd.DataFrame({"time": [time_GPT_vignette]})
# time_GPT_vignette_df.to_csv("../exp/times_LLMs/GPT/time_GPT_vignette_prompt.csv", sep = ",", index = False)
#
# # value counts for array
# counts_vignette_GPT = pd.Series(y_pred_vignette_GPT).value_counts()
# print(counts_vignette_GPT)
#
# # convert YES to 1 and NO to 0
# y_pred_vignette_GPT_val = [1 if response == "YES" else 0 for response in y_pred_vignette_GPT]
#
# # save the array to a csv file
# vignette_df_GPT = pd.DataFrame(y_pred_vignette_GPT_val, columns = ["y_pred"])
# vignette_df_GPT.to_csv("../exp/y_pred_LLMs/GPT/y_pred_GPT_vignette_prompt.csv", sep = ",", index = False)
#
#
#
# #### Chain-of-thought prompt ####
#
# y_pred_cot_GPT = []
# explanation_cot_GPT = []
#
# client = OpenAI(
#     api_key = os.environ.get("OPENAI_API_KEY"),
# )
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
#         print(prediction)
#     except IndexError:
#         print("IndexError")
#         y_pred_cot_GPT.append("IndexError")
#         explanation_cot_GPT.append("IndexError")
#
# end = time.time()
# print(f"Time taken: {end - start} seconds")
# time_GPT_cot = end - start
# time_GPT_cot_df = pd.DataFrame({"time": [time_GPT_cot]})
# time_GPT_cot_df.to_csv("../exp/times_LLMs/GPT/time_GPT_cot_prompt.csv", sep = ",", index = False)
#
# # value counts for array
# counts_cot_GPT = pd.Series(y_pred_cot_GPT).value_counts()
# print(counts_cot_GPT)
#
# # convert YES to 1 and NO to 0
# y_pred_cot_GPT_val = [1 if response == "YES" else 0 for response in y_pred_cot_GPT]
#
# # save the array to a csv file
# cot_df_GPT = pd.DataFrame(y_pred_cot_GPT_val, columns = ["y_pred"])
# cot_df_GPT.to_csv("../exp/y_pred_LLMs/GPT/y_pred_GPT_cot_prompt.csv", sep = ",", index = False)
#
# cot_df_explanation_GPT = pd.DataFrame(explanation_cot_GPT, columns = ["cot"])
# cot_df_explanation_GPT.to_csv("../exp/y_pred_LLMs/GPT/explanation_GPT_cot_prompt.csv", sep = ",", index = False)