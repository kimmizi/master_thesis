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

def Grok_create_completion(prompt, instruction):
    completion = client.chat.completions.create(
        model = model_grok,
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ],
    )

    if completion.choices[0].message.content.strip() not in ("YES", "NO"):
        print("\n Invalid output. Retry prompting. \n")
        completion = client.chat.completions.create(
            model = model_grok,
            messages = [
                {"role": "system", "content": retry_instruction},
                {"role": "user", "content": prompt},
            ],
        )

    return completion.choices[0].message.content.strip()


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
    df.to_csv(f"X_train_pred/Grok/X_train_grok_{filename}.csv", sep = ",", index = False)


def calc_time(start, end, filename):
    """
    Calculate the time taken for the prompting and save it to a CSV file.
    """
    time_taken = end - start
    print(f"Time taken: {time_taken} seconds")



#### 2 Prompting with Grok 3 Beta ####

model_grok = "grok-3-beta"

client = OpenAI(
    api_key = os.environ.get("XAI_API_KEY"),
    base_url = "https://api.x.ai/v1",
)


#### Simple prompt ####

y_pred_simple_grok = []

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_train_simple_prompt, desc = "Simple prompting"):
    completion = Grok_create_completion(prompt, simple_instruction)
    y_pred_simple_grok.append(completion)

    if len(y_pred_simple_grok) % 50 == 0 and len(y_pred_simple_grok) > 0:
        print(f"\n\nProcessed {len(y_pred_simple_grok)} prompts.\n")
        save_prompt_to_csv(y_pred_simple_grok, "simple_prompt")

end = time.time()
calc_time(start, end, "simple_prompt")

# save the array to a csv file
save_prompt_to_csv(y_pred_simple_grok, "simple_prompt")



#### Class definition prompt ####

y_pred_class_def_grok = []

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_train_class_definitions_prompt, desc = "Class definition prompting"):
    completion = Grok_create_completion(prompt, class_definitions_instruction)
    y_pred_class_def_grok.append(completion)

    if len(y_pred_class_def_grok) % 50 == 0 and len(y_pred_class_def_grok) > 0:
        print(f"\n\nProcessed {len(y_pred_class_def_grok)} prompts.\n")
        save_prompt_to_csv(y_pred_class_def_grok, "class_definitions_prompt")

end = time.time()
calc_time(start, end, "class_definitions_prompt")

# save the array to a csv file
save_prompt_to_csv(y_pred_class_def_grok, "class_definitions_prompt")


#### Profiled simple prompt ####

y_pred_profiled_simple_grok = []

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_train_profiled_simple_prompt, desc = "Profiled simple prompting"):
    completion = Grok_create_completion(prompt, profiled_simple_instruction)
    y_pred_profiled_simple_grok.append(completion)

    if len(y_pred_profiled_simple_grok) % 50 == 0 and len(y_pred_profiled_simple_grok) > 0:
        print(f"\n\nProcessed {len(y_pred_profiled_simple_grok)} prompts.\n")
        save_prompt_to_csv(y_pred_profiled_simple_grok, "profiled_simple_prompt")

end = time.time()
calc_time(start, end, "profiled_simple_prompt")

# save the array to a csv file
save_prompt_to_csv(y_pred_profiled_simple_grok, "profiled_simple_prompt")


#### Few shot prompt ####

y_pred_few_shot_grok = []

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_train_few_shot_prompt, desc = "Few shot prompting"):
    completion = Grok_create_completion(prompt, few_shot_instruction)
    y_pred_few_shot_grok.append(completion)

    if len(y_pred_few_shot_grok) % 50 == 0 and len(y_pred_few_shot_grok) > 0:
        print(f"\n\nProcessed {len(y_pred_few_shot_grok)} prompts.\n")
        save_prompt_to_csv(y_pred_few_shot_grok, "few_shot_prompt")

end = time.time()
calc_time(start, end, "few_shot_prompt")

# save the array to a csv file
save_prompt_to_csv(y_pred_few_shot_grok, "few_shot_prompt")



#### Vignette prompt ####

y_pred_vignette_grok = []

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_train_vignette_prompt, desc = "Vignette prompting"):
    completion = Grok_create_completion(prompt, vignette_instruction)
    y_pred_vignette_grok.append(completion)

    if len(y_pred_vignette_grok) % 50 == 0 and len(y_pred_vignette_grok) > 0:
        print(f"\n\nProcessed {len(y_pred_vignette_grok)} prompts.\n")
        save_prompt_to_csv(y_pred_vignette_grok, "vignette_prompt")

end = time.time()
calc_time(start, end, "vignette_prompt")

# save the array to a csv file
save_prompt_to_csv(y_pred_vignette_grok, "vignette_prompt")



### Chain-of-thought prompt ####

y_pred_cot_grok = []
explanation_cot_grok = []

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_train_cot_prompt, desc = "Chain-of-thought prompting"):
    completion = Grok_create_completion(prompt, simple_instruction)
    y_pred_cot_grok.append(completion)

    if len(y_pred_cot_grok) % 50 == 0 and len(y_pred_cot_grok) > 0:
        print(f"\n\nProcessed {len(y_pred_cot_grok)} prompts.\n")
        save_prompt_to_csv(y_pred_cot_grok, "cot_prompt")

end = time.time()
calc_time(start, end, "cot_prompt")

# save the array to a csv file
save_prompt_to_csv(y_pred_cot_grok, "cot_prompt")
