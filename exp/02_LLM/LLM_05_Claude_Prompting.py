#### LLM: Zero-shot classification through LLMs and prompts ####
#### CLAUDE ####


#### 0 Imports ####
import os
import pandas as pd
import anthropic
import numpy as np
import time
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split

data_change = pd.read_csv("../../dat/dips/DIPS_Data_cleaned_change.csv", sep =",", low_memory = False)

# import prompts for all test data
X_test_simple_prompt_df = pd.read_csv("../../dat/prompts/X_test_simple_prompt.csv", sep =",", index_col = 0)
X_test_class_definitions_prompt_df = pd.read_csv("../../dat/prompts/X_test_class_definitions_prompt.csv", sep =",", index_col = 0)
X_test_profiled_simple_prompt_df = pd.read_csv("../../dat/prompts/X_test_profiled_simple_prompt.csv", sep =",", index_col = 0)
X_test_few_shot_prompt_df = pd.read_csv("../../dat/prompts/X_test_few_shot_prompt.csv", sep =",", index_col = 0)
X_test_vignette_prompt_df = pd.read_csv("../../dat/prompts/X_test_vignette_prompt.csv", sep =",", index_col = 0)
X_test_cot_prompt_df = pd.read_csv("../../dat/prompts/X_test_cot_prompt.csv", sep =",", index_col = 0)

# convert to arrays
X_test_simple_prompt = X_test_simple_prompt_df.values.flatten()
X_test_class_definitions_prompt = X_test_class_definitions_prompt_df.values.flatten()
X_test_profiled_simple_prompt = X_test_profiled_simple_prompt_df.values.flatten()
X_test_few_shot_prompt = X_test_few_shot_prompt_df.values.flatten()
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

def Claude_create_message(prompt, instruction):
    message = client.messages.create(
        model = model_claude,
        max_tokens = 10000,
        thinking = {
            "type": "enabled",
            "budget_tokens": 2000
        },
        system = instruction,
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )

    if message.content[1].text.strip() not in ("YES", "NO"):
        print("\n Invalid output. Retry prompting. \n")
        message = client.messages.create(
            model = model_claude,
            max_tokens = 10000,
            thinking = {
                "type": "enabled",
                "budget_tokens": 2000
            },
            system = retry_instruction,
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )

    return message.content[1].text.strip(), message.content[0].thinking


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
    df.to_csv(f"../exp/y_pred_LLMs/Claude/y_pred_claude_{filename}.csv", sep = ",", index = False)


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
        "cot": explanation_array
    })
    df.to_csv(f"../exp/y_pred_LLMs/Claude/y_pred_claude_{filename}.csv", sep = ",", index = False)


def calc_time(start, end, filename):
    """
    Calculate the time taken for the prompting and save it to a CSV file.
    """
    time_taken = end - start
    print(f"Time taken: {time_taken} seconds")
    time_df = pd.DataFrame({"time": [time_taken]})
    time_df.to_csv(f"../exp/times_LLMs/Claude/time_claude_{filename}.csv", sep = ",", index = False)
    return time_taken



### 1 Testing prompting ####

# client = anthropic.Anthropic(
#     api_key = os.environ.get("ANTHROPIC_API_KEY")
# )
#
# message = client.messages.create(
#     model = "claude-sonnet-4-20250514",
#     max_tokens = 20000,
#     temperature = 1,
#     thinking = {
#         "type": "enabled",
#         "budget_tokens": 16000
#     },
#     system = claude_instruction,
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": X_test_claude_prompt[0]
#                 }
#             ]
#         }
#     ]
# )
# print(message.content)

# print(message.content[0].thinking)

# prediction = re.findall(r'Prediction: (.*)', message.content[1].text)
# prediction[0]

# # extract what comes after Explanation:
# explanation = re.findall(r'Explanation: (.*)', message.content[1].text)
# explanation[0]



### 2 Prompting with Claude 3.7 Sonnet ####

# model_claude = "claude-3-7-sonnet-20250219"
model_claude = "claude-sonnet-4-20250514"

client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)


#### Simple prompt ####

y_pred_simple_claude = []
thinking_simple_claude = []

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_simple_prompt, desc = "Simple Prompting"):
    response, thinking = Claude_create_message(prompt, simple_instruction)
    y_pred_simple_claude.append(response)
    thinking_simple_claude.append(thinking)
    # print(response)

    if len(y_pred_simple_claude) % 50 == 0 and len(y_pred_simple_claude) > 0:
        print(f"\n\nProcessed {len(y_pred_simple_claude)} prompts.\n")
        save_prompt_to_csv(y_pred_simple_claude, thinking_simple_claude, "simple_prompt")

end = time.time()
calc_time(start, end, "simple_prompt")

# save the array to a csv file
save_prompt_to_csv(y_pred_simple_claude, thinking_simple_claude, "simple_prompt")



#### Class definition prompt ####

y_pred_class_def_claude = []
thinking_class_def_claude = []

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_class_definitions_prompt, desc = "Class Definitions Prompting"):
    response, thinking = Claude_create_message(prompt, class_definitions_instruction)
    y_pred_class_def_claude.append(response)
    thinking_class_def_claude.append(thinking)
    # print(response)

    if len(y_pred_class_def_claude) % 50 == 0 and len(y_pred_class_def_claude) > 0:
        print(f"\n\nProcessed {len(y_pred_class_def_claude)} prompts.\n")
        save_prompt_to_csv(y_pred_class_def_claude, thinking_class_def_claude, "class_definitions_prompt")

end = time.time()
calc_time(start, end, "class_definitions_prompt")

# save the array to a csv file
save_prompt_to_csv(y_pred_class_def_claude, thinking_class_def_claude, "class_definitions_prompt")



#### Profiled simple prompt ####

y_pred_profiled_simple_claude = []
thinking_profiled_simple_claude = []

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_profiled_simple_prompt, desc = "Profiled Simple Prompting"):
    response, thinking = Claude_create_message(prompt, profiled_simple_instruction)
    y_pred_profiled_simple_claude.append(response)
    thinking_profiled_simple_claude.append(thinking)
    # print(response)

    if len(y_pred_profiled_simple_claude) % 50 == 0 and len(y_pred_profiled_simple_claude) > 0:
        print(f"\n\nProcessed {len(y_pred_profiled_simple_claude)} prompts.\n")
        save_prompt_to_csv(y_pred_profiled_simple_claude, thinking_profiled_simple_claude, "profiled_simple_prompt")

end = time.time()
calc_time(start, end, "profiled_simple_prompt")

# save the array to a csv file
save_prompt_to_csv(y_pred_profiled_simple_claude, thinking_profiled_simple_claude, "profiled_simple_prompt")



#### Few shot prompt ####

y_pred_few_shot_claude = []
thinking_few_shot_claude = []

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_few_shot_prompt, desc = "Few Shot Prompting"):
    response, thinking = Claude_create_message(prompt, few_shot_instruction)
    y_pred_few_shot_claude.append(response)
    thinking_few_shot_claude.append(thinking)
    # print(response)

    if len(y_pred_few_shot_claude) % 50 == 0 and len(y_pred_few_shot_claude) > 0:
        print(f"\n\nProcessed {len(y_pred_few_shot_claude)} prompts.\n")
        save_prompt_to_csv(y_pred_few_shot_claude, thinking_few_shot_claude, "few_shot_prompt")

end = time.time()
calc_time(start, end, "few_shot_prompt")

# save the array to a csv file
save_prompt_to_csv(y_pred_few_shot_claude, thinking_few_shot_claude, "few_shot_prompt")



#### Vignette prompt ####

y_pred_vignette_claude = []
thinking_vignette_claude = []

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_vignette_prompt, desc = "Vignette Prompting"):
    response, thinking = Claude_create_message(prompt, vignette_instruction)
    y_pred_vignette_claude.append(response)
    thinking_vignette_claude.append(thinking)
    # print(response)

    if len(y_pred_vignette_claude) % 50 == 0 and len(y_pred_vignette_claude) > 0:
        print(f"\n\nProcessed {len(y_pred_vignette_claude)} prompts.\n")
        save_prompt_to_csv(y_pred_vignette_claude, thinking_vignette_claude, "vignette_prompt")

end = time.time()
calc_time(start, end, "vignette_prompt")

# save the array to a csv file
save_prompt_to_csv(y_pred_vignette_claude, thinking_vignette_claude, "vignette_prompt")



### Chain-of-thought prompt ####

y_pred_cot_claude = []
explanation_cot_claude = []
thinking_cot_claude = []

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_cot_prompt, desc = "Chain-of-Thought Prompting"):
    message = client.messages.create(
        model = "claude-sonnet-4-20250514",
        max_tokens = 20000,
        temperature = 1,
        thinking = {
            "type": "enabled",
            "budget_tokens": 16000
        },
        system = cot_instruction,
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )

    try:
        prediction = re.findall(r'Prediction: (.*)', message.content[1].text)[0].strip()
        explanation = re.findall(r'Explanation: (.*)', message.content[1].text)[0].strip()
        y_pred_cot_claude.append(prediction)
        explanation_cot_claude.append(explanation)
        thinking_cot_claude.append(message.content[0].thinking)
        # print(prediction)

    except IndexError:
        print("\n IndexError. Retry prompting. \n")
        message = client.messages.create(
            model = model_claude,
            max_tokens = 20000,
            temperature = 1,
            thinking = {
                "type": "enabled",
                "budget_tokens": 16000
            },
            system = cot_instruction,
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        try:
            prediction = re.findall(r'Prediction: (.*)', message.content[1].text)[0].strip()
            explanation = re.findall(r'Explanation: (.*)', message.content[1].text)[0].strip()
            y_pred_cot_claude.append(prediction)
            explanation_cot_claude.append(explanation)
            thinking_cot_claude.append(message.content[0].thinking)

        except IndexError:
            print("\n Still IndexError. Don't retry prompting. \n")
            y_pred_cot_claude.append("IndexError")
            explanation_cot_claude.append("IndexError")
            thinking_cot_claude.append("IndexError")

    if len(y_pred_cot_claude) % 50 == 0 and len(y_pred_cot_claude) > 0:
        print(f"\n\nProcessed {len(y_pred_cot_claude)} prompts.\n")
        save_prompt_to_csv_cot(y_pred_cot_claude, thinking_cot_claude, explanation_cot_claude, "cot_prompt")

end = time.time()
calc_time(start, end, "cot_prompt")

# save the array to a csv file
save_prompt_to_csv_cot(y_pred_cot_claude, thinking_cot_claude, explanation_cot_claude, "cot_prompt")