#### LLM: Zero-shot classification through LLMs and prompts ####
#### GROK ####


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



#### 1 Testing prompting ####

# client = OpenAI(
#     api_key = os.environ.get("XAI_API_KEY"),
#     base_url = "https://api.x.ai/v1",
# )
#
# completion = client.chat.completions.create(
#     model = "grok-3-beta",
#     # model = "grok-3-mini-beta",
#     messages = [
#         {"role": "system", "content": simple_instruction},
#         {"role": "user", "content": X_test_simple_prompt[0]},
#     ],
#     # reasoning_effort = "high"
# )
# print(completion.choices[0].message)

# completion.choices[0].message.content



#### 2 Prompting with Grok 3 Beta ####



# #### Simple prompt ####
#
# y_pred_simple_grok = []
#
# client = OpenAI(
#     api_key = os.environ.get("XAI_API_KEY"),
#     base_url = "https://api.x.ai/v1",
# )
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_test_simple_prompt, desc = "Simple prompting"):
#     completion = client.chat.completions.create(
#         model = "grok-3-beta",
#         messages = [
#             {"role": "system", "content": simple_instruction},
#             {"role": "user", "content": prompt},
#         ],
#     )
#
#     if completion.choices[0].message.content.strip() not in ("YES", "NO"):
#         print("\n Invalid output. Retry prompting. \n")
#         completion = client.chat.completions.create(
#             model = "grok-3-beta",
#             messages = [
#                 {"role": "system", "content": retry_instruction},
#                 {"role": "user", "content": prompt},
#             ],
#         )
#
#     if len(y_pred_simple_grok) % 50 == 0 and len(y_pred_simple_grok) > 0:
#         print(f"\n\nProcessed {len(y_pred_simple_grok)} prompts.\n")
#         counts_simple = pd.Series(y_pred_simple_grok).value_counts()
#         print(counts_simple, "\n")
#
#     y_pred_simple_grok.append(completion.choices[0].message.content)
#     # print(completion.choices[0].message.content)
#
# end = time.time()
# print(f"Time taken: {end - start} seconds")
# time_grok_simple_prompt = end - start
# time_grok_simple_df = pd.DataFrame({"time": [time_grok_simple_prompt]})
# time_grok_simple_df.to_csv("../exp/times_LLMs/Grok/time_grok_simple_prompt.csv", sep = ",", index = False)
#
# # value counts for array
# counts_simple_grok = pd.Series(y_pred_simple_grok).value_counts()
# print(counts_simple_grok)
#
# # convert YES to 1 and NO to 0
# y_pred_simple_grok = [re.sub(r'^\[|\]$', '', response.strip()) for response in y_pred_simple_grok]
# y_pred_simple_grok = [1 if response.strip() == "YES" else 0 if response.strip() == "NO" else np.nan for response in y_pred_simple_grok]
#
# # save the array to a csv file
# simple_df_grok = pd.DataFrame(y_pred_simple_grok, columns = ["y_pred"])
# simple_df_grok.to_csv("../exp/y_pred_LLMs/Grok/y_pred_grok_simple_prompt.csv", sep = ",", index = False)


#### Class definition prompt ####

y_pred_class_def_grok = []

client = OpenAI(
    api_key = os.environ.get("XAI_API_KEY"),
    base_url = "https://api.x.ai/v1",
)

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_class_definitions_prompt, desc = "Class definition prompting"):
    completion = client.chat.completions.create(
        model = "grok-3-beta",
        messages = [
            {"role": "system", "content": class_definitions_instruction},
            {"role": "user", "content": prompt},
        ],
    )

    if completion.choices[0].message.content.strip() not in ("YES", "NO"):
        print("\n Invalid output. Retry prompting. \n")
        completion = client.chat.completions.create(
            model = "grok-3-beta",
            messages = [
                {"role": "system", "content": retry_instruction},
                {"role": "user", "content": prompt},
            ],
        )

    y_pred_class_def_grok.append(completion.choices[0].message.content)
    # print(completion.choices[0].message.content)

    if len(y_pred_class_def_grok) % 50 == 0 and len(y_pred_class_def_grok) > 0:
        print(f"\n\nProcessed {len(y_pred_class_def_grok)} prompts.\n")
        counts_class_def_grok = pd.Series(y_pred_class_def_grok).value_counts()
        print(counts_class_def_grok, "\n")

end = time.time()
print(f"Time taken: {end - start} seconds")
time_grok_class_definitions = end - start
time_grok_class_definitions_df = pd.DataFrame({"time": [time_grok_class_definitions]})
time_grok_class_definitions_df.to_csv("../exp/times_LLMs/Grok/time_grok_class_definitions_prompt.csv", sep = ",", index = False)

# value counts for array
counts_class_def_grok = pd.Series(y_pred_class_def_grok).value_counts()
print(counts_class_def_grok)


# convert YES to 1 and NO to 0
y_pred_class_def_grok = [re.sub(r'^\[|\]$', '', response.strip()) for response in y_pred_class_def_grok]
y_pred_class_def_grok = [1 if response.strip() == "YES" else 0 if response.strip() == "NO" else np.nan for response in y_pred_class_def_grok]

# save the array to a csv file
class_def_df_grok = pd.DataFrame(y_pred_class_def_grok, columns = ["y_pred"])
class_def_df_grok.to_csv("../exp/y_pred_LLMs/Grok/y_pred_grok_class_definitions_prompt.csv", sep = ",", index = False)



# #### Profiled simple prompt ####
#
# y_pred_profiled_simple_grok = []
#
# client = OpenAI(
#     api_key = os.environ.get("XAI_API_KEY"),
#     base_url = "https://api.x.ai/v1",
# )
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in tqdm(X_test_profiled_simple_prompt, desc = "Profiled simple prompting"):
#     completion = client.chat.completions.create(
#         model = "grok-3-beta",
#         messages = [
#             {"role": "system", "content": profiled_simple_instruction},
#             {"role": "user", "content": prompt},
#         ],
#     )
#
#     if completion.choices[0].message.content.strip() not in ("YES", "NO"):
#         print("\n Invalid output. Retry prompting. \n")
#         completion = client.chat.completions.create(
#             model = "grok-3-beta",
#             messages = [
#                 {"role": "system", "content": retry_instruction},
#                 {"role": "user", "content": prompt},
#             ],
#         )
#
#     y_pred_profiled_simple_grok.append(completion.choices[0].message.content)
#     # print(completion.choices[0].message.content)
#
#     if len(y_pred_profiled_simple_grok) % 50 == 0 and len(y_pred_profiled_simple_grok) > 0:
#         print(f"\n\nProcessed {len(y_pred_profiled_simple_grok)} prompts.\n")
#         counts_profiled_simple_grok = pd.Series(y_pred_profiled_simple_grok).value_counts()
#         print(counts_profiled_simple_grok, "\n")
#
# end = time.time()
# print(f"Time taken: {end - start} seconds")
# time_grok_profiled_simple = end - start
# time_grok_profiled_simple_df = pd.DataFrame({"time": [time_grok_profiled_simple]})
# time_grok_profiled_simple_df.to_csv("../exp/times_LLMs/Grok/time_grok_profiled_simple_prompt.csv", sep = ",", index = False)
#
# # value counts for array
# counts_profiled_simple_grok = pd.Series(y_pred_profiled_simple_grok).value_counts()
# print(counts_profiled_simple_grok)
#
# # convert YES to 1 and NO to 0
# y_pred_profiled_simple_grok = [re.sub(r'^\[|\]$', '', response.strip()) for response in y_pred_profiled_simple_grok]
# y_pred_profiled_simple_grok_val = [1 if response.strip() == "YES" else 0 if response.strip() == "NO" else np.nan for response in y_pred_profiled_simple_grok]
#
# # save the array to a csv file
# profiled_simple_df_grok = pd.DataFrame(y_pred_profiled_simple_grok_val, columns = ["y_pred"])
# profiled_simple_df_grok.to_csv("../exp/y_pred_LLMs/Grok/y_pred_grok_profiled_simple_prompt.csv", sep = ",", index = False)
#
#
#
# #### Few shot prompt ####
#
# y_pred_few_shot_grok = []
#
# client = OpenAI(
#     api_key = os.environ.get("XAI_API_KEY"),
#     base_url = "https://api.x.ai/v1",
# )
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in X_test_few_shot_prompt:
#     completion = client.chat.completions.create(
#         model = "grok-3-beta",
#         messages = [
#             {"role": "system", "content": few_shot_instruction},
#             {"role": "user", "content": prompt},
#         ],
#     )
#
#     if completion.choices[0].message.content.strip() not in ("YES", "NO"):
#         print("\n Invalid output. Retry prompting. \n")
#         completion = client.chat.completions.create(
#             model = "grok-3-beta",
#             messages = [
#                 {"role": "system", "content": retry_instruction},
#                 {"role": "user", "content": prompt},
#             ],
#         )
#
#     y_pred_few_shot_grok.append(completion.choices[0].message.content)
#     # print(completion.choices[0].message.content)
#
#     if len(y_pred_few_shot_grok) % 50 == 0 and len(y_pred_few_shot_grok) > 0:
#         print(f"\n\nProcessed {len(y_pred_few_shot_grok)} prompts.\n")
#         counts_few_shot_grok = pd.Series(y_pred_few_shot_grok).value_counts()
#         print(counts_few_shot_grok, "\n")
#
# end = time.time()
# print(f"Time taken: {end - start} seconds")
# time_grok_few_shot = end - start
# time_grok_few_shot_df = pd.DataFrame({"time": [time_grok_few_shot]})
# time_grok_few_shot_df.to_csv("../exp/times_LLMs/Grok/time_grok_few_shot_prompt.csv", sep = ",", index = False)
#
# # value counts for array
# counts_few_shot_grok = pd.Series(y_pred_few_shot_grok).value_counts()
# print(counts_few_shot_grok)
#
# # convert YES to 1 and NO to 0
# y_pred_few_shot_grok = [re.sub(r'^\[|\]$', '', response.strip()) for response in y_pred_few_shot_grok]
# y_pred_few_shot_grok_val = [1 if response.strip() == "YES" else 0 if response.strip() == "NO" else np.nan for response in y_pred_few_shot_grok]
#
# # save the array to a csv file
# few_shot_df_grok = pd.DataFrame(y_pred_few_shot_grok_val, columns = ["y_pred"])
# few_shot_df_grok.to_csv("../exp/y_pred_LLMs/Grok/y_pred_grok_few_shot_prompt.csv", sep = ",", index = False)
#
#
#
# #### Vignette prompt ####
#
# y_pred_vignette_grok = []
#
# client = OpenAI(
#     api_key = os.environ.get("XAI_API_KEY"),
#     base_url = "https://api.x.ai/v1",
# )
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in X_test_vignette_prompt:
#     completion = client.chat.completions.create(
#         model = "grok-3-beta",
#         messages = [
#             {"role": "system", "content": vignette_instruction},
#             {"role": "user", "content": prompt},
#         ],
#     )
#
#     if completion.choices[0].message.content.strip() not in ("YES", "NO"):
#         print("\n Invalid output. Retry prompting. \n")
#         completion = client.chat.completions.create(
#             model = "grok-3-beta",
#             messages = [
#                 {"role": "system", "content": retry_instruction},
#                 {"role": "user", "content": prompt},
#             ],
#         )
#
#     y_pred_vignette_grok.append(completion.choices[0].message.content)
#     # print(completion.choices[0].message.content)
#
#     if len(y_pred_vignette_grok) % 50 == 0 and len(y_pred_vignette_grok) > 0:
#         print(f"\n\nProcessed {len(y_pred_vignette_grok)} prompts.\n")
#         counts_vignette_grok = pd.Series(y_pred_vignette_grok).value_counts()
#         print(counts_vignette_grok, "\n")
#
# end = time.time()
# print(f"Time taken: {end - start} seconds")
# time_grok_vignette = end - start
# time_grok_vignette_df = pd.DataFrame({"time": [time_grok_vignette]})
# time_grok_vignette_df.to_csv("../exp/times_LLMs/Grok/time_grok_vignette_prompt.csv", sep = ",", index = False)
#
# # value counts for array
# counts_vignette_grok = pd.Series(y_pred_vignette_grok).value_counts()
# print(counts_vignette_grok)
#
# # convert YES to 1 and NO to 0
# y_pred_vignette_grok = [re.sub(r'^\[|\]$', '', response.strip()) for response in y_pred_vignette_grok]
# y_pred_vignette_grok_val = [1 if response.strip() == "YES" else 0 if response.strip() == "NO" else np.nan for response in y_pred_vignette_grok]
#
# # save the array to a csv file
# vignette_df_grok = pd.DataFrame(y_pred_vignette_grok_val, columns = ["y_pred"])
# vignette_df_grok.to_csv("../exp/y_pred_LLMs/Grok/y_pred_grok_vignette_prompt.csv", sep = ",", index = False)
#
#
#
# #### Chain-of-thought prompt ####
#
# y_pred_cot_grok = []
# explanation_cot_grok = []
#
# client = OpenAI(
#     api_key = os.environ.get("XAI_API_KEY"),
#     base_url = "https://api.x.ai/v1",
# )
#
# # measure time in seconds
# start = time.time()
#
# # iterate over the test set and save the response for each prompt in an array
# for prompt in X_test_cot_prompt:
#     completion = client.chat.completions.create(
#         model = "grok-3-beta",
#         messages = [
#             {"role": "system", "content": cot_instruction},
#             {"role": "user", "content": prompt},
#         ],
#     )
#     try:
#         prediction = re.findall(r'Prediction: (.*)', completion.choices[0].message.content)[0].strip()
#         explanation = re.findall(r'Explanation: (.*)', completion.choices[0].message.content)[0].strip()
#         y_pred_cot_grok.append(prediction)
#         explanation_cot_grok.append(explanation)
#         # print(prediction)
#     except IndexError:
#         print("IndexError")
#         y_pred_cot_grok.append("IndexError")
#         explanation_cot_grok.append("IndexError")
#
#     if len(y_pred_cot_grok) % 50 == 0 and len(y_pred_cot_grok) > 0:
#         print(f"\n\nProcessed {len(y_pred_cot_grok)} prompts.\n")
#         counts_cot_grok = pd.Series(y_pred_cot_grok).value_counts()
#         print(counts_cot_grok, "\n")
#
# end = time.time()
# print(f"Time taken: {end - start} seconds")
# time_grok_cot_prompt = end - start
# time_grok_cot_df = pd.DataFrame({"time": [time_grok_cot_prompt]})
# time_grok_cot_df.to_csv("../exp/times_LLMs/Grok/time_grok_cot_prompt.csv", sep = ",", index = False)
#
# # value counts for array
# counts_cot_grok = pd.Series(y_pred_cot_grok).value_counts()
# print(counts_cot_grok)
#
# # convert YES to 1 and NO to 0
# y_pred_cot_grok = [re.sub(r'^\[|\]$', '', response.strip()) for response in y_pred_cot_grok]
# y_pred_cot_grok_val = [1 if response.strip() == "YES" else 0 if response.strip() == "NO" else np.nan for response in y_pred_cot_grok]
#
# # save the array to a csv file
# cot_df_grok = pd.DataFrame({
#     "y_pred": y_pred_cot_grok_val,
#     "explanation": explanation_cot_grok
# })
# cot_df_grok.to_csv("../exp/y_pred_LLMs/Grok/y_pred_grok_cot_prompt.csv", sep = ",", index = False)
#
