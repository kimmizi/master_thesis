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
from openai import OpenAI
from google import genai
from google.genai import types
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



#### Simple prompt ####

y_pred_simple_claude = []
thinking_simple_claude = []

client = anthropic.Anthropic(
    api_key = os.environ.get("ANTHROPIC_API_KEY")
)

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_simple_prompt, desc = "Simple Prompting"):
    message = client.messages.create(
        model = "claude-sonnet-4-20250514",
        max_tokens = 20000,
        temperature = 1,
        thinking = {
            "type": "enabled",
            "budget_tokens": 16000
        },
        system = simple_instruction,
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
            model = "claude-sonnet-4-20250514",
            max_tokens = 20000,
            temperature = 1,
            thinking = {
                "type": "enabled",
                "budget_tokens": 16000
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

    if len(y_pred_simple_claude) % 50 == 0 and len(y_pred_simple_claude) > 0:
        print(f"\n\nProcessed {len(y_pred_simple_claude)} prompts.\n")
        counts_profiled_simple_claude = pd.Series(y_pred_simple_claude).value_counts()
        print(counts_profiled_simple_claude, "\n")

        y_pred_simple_claude_val = [1 if response.strip() == "YES" else 0 if response.strip() == "NO" else np.nan for response in y_pred_simple_claude]

        # save as df
        simple_df_claude = pd.DataFrame({
            "y_pred": y_pred_simple_claude_val,
            "thinking": y_pred_simple_claude
        })
        simple_df_claude.to_csv("../exp/y_pred_LLMs/Claude/y_pred_claude_simple_prompt.csv", sep=",", index=False)
        print("Saved df")


    y_pred_simple_claude.append(message.content[1].text)
    thinking_simple_claude.append(message.content[0].thinking)
    # print(message.content[1].text)

end = time.time()
print(f"Time taken: {end - start} seconds")
time_claude_simple_prompt = end - start
time_claude_simple_df = pd.DataFrame({"time": [time_claude_simple_prompt]})
time_claude_simple_df.to_csv("../exp/times_LLMs/Claude/time_claude_simple_prompt.csv", sep = ",", index = False)

# value counts for array
counts_simple_claude = pd.Series(y_pred_simple_claude).value_counts()
print(counts_simple_claude)

# convert YES to 1 and NO to 0
y_pred_simple_claude = [1 if response == "YES" else 0 if response == "NO" else np.nan for response in y_pred_simple_claude]

# save the array to a csv file
simple_df_claude = pd.DataFrame({
    "y_pred": y_pred_simple_claude,
    "thinking": thinking_simple_claude
})
simple_df_claude.to_csv("../exp/y_pred_LLMs/Claude/y_pred_claude_simple_prompt.csv", sep = ",", index = False)



#### Class definition prompt ####

y_pred_class_def_claude = []
thinking_class_def_claude = []

client = anthropic.Anthropic(
    api_key = os.environ.get("ANTHROPIC_API_KEY")
)

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_class_definitions_prompt, desc = "Class Definitions Prompting"):
    message = client.messages.create(
        model = "claude-sonnet-4-20250514",
        max_tokens = 20000,
        temperature = 1,
        thinking = {
            "type": "enabled",
            "budget_tokens": 16000
        },
        system = class_definitions_instruction,
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
            model = "claude-sonnet-4-20250514",
            max_tokens = 20000,
            temperature = 1,
            thinking = {
                "type": "enabled",
                "budget_tokens": 16000
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

    if len(y_pred_class_def_claude) % 50 == 0 and len(y_pred_class_def_claude) > 0:
        print(f"\n\nProcessed {len(y_pred_class_def_claude)} prompts.\n")
        counts_class_def_claude = pd.Series(y_pred_class_def_claude).value_counts()
        print(counts_class_def_claude, "\n")

        y_pred_class_def_claude_val = [1 if response.strip() == "YES" else 0 if response.strip() == "NO" else np.nan for response in y_pred_class_def_claude]

        # save as df
        class_def_df_claude = pd.DataFrame({
            "y_pred": y_pred_class_def_claude_val,
            "thinking": thinking_class_def_claude
        })
        class_def_df_claude.to_csv("../exp/y_pred_LLMs/Claude/y_pred_claude_class_definitions_prompt.csv", sep=",", index=False)
        print("Saved df")

    y_pred_class_def_claude.append(message.content[1].text)
    thinking_class_def_claude.append(message.content[0].thinking)
    # print(message.content[1].text)

end = time.time()
print(f"Time taken: {end - start} seconds")
time_claude_class_definitions = end - start
time_claude_class_definitions_df = pd.DataFrame({"time": [time_claude_class_definitions]})
time_claude_class_definitions_df.to_csv("../exp/times_LLMs/Claude/time_claude_class_definitions_prompt.csv", sep = ",", index = False)

# value counts for array
counts_class_def_claude = pd.Series(y_pred_class_def_claude).value_counts()
print(counts_class_def_claude)

# convert YES to 1 and NO to 0
y_pred_class_def_claude = [1 if response == "YES" else 0 for response in y_pred_class_def_claude]

# save the array to a csv file
class_def_df_claude = pd.DataFrame({
    "y_pred": y_pred_class_def_claude,
    "thinking": thinking_class_def_claude
})
class_def_df_claude.to_csv("../exp/y_pred_LLMs/Claude/y_pred_claude_class_definitions_prompt.csv", sep = ",", index = False)



#### Profiled simple prompt ####

y_pred_profiled_simple_claude = []
thinking_profiled_simple_claude = []

client = anthropic.Anthropic(
    api_key = os.environ.get("ANTHROPIC_API_KEY")
)

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_profiled_simple_prompt, desc = "Profiled Simple Prompting"):
    message = client.messages.create(
        model = "claude-sonnet-4-20250514",
        max_tokens = 20000,
        temperature = 1,
        thinking = {
            "type": "enabled",
            "budget_tokens": 16000
        },
        system = profiled_simple_instruction,
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
            model = "claude-sonnet-4-20250514",
            max_tokens = 20000,
            temperature = 1,
            thinking = {
                "type": "enabled",
                "budget_tokens": 16000
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

    if len(y_pred_profiled_simple_claude) % 50 == 0 and len(y_pred_profiled_simple_claude) > 0:
        print(f"\n\nProcessed {len(y_pred_profiled_simple_claude)} prompts.\n")
        counts_profiled_simple_claude = pd.Series(y_pred_profiled_simple_claude).value_counts()
        print(counts_profiled_simple_claude, "\n")

        y_pred_profiled_simple_claude_val = [1 if response.strip() == "YES" else 0 if response.strip() == "NO" else np.nan for response in y_pred_profiled_simple_claude]

        # save as df
        profiled_simple_df_claude = pd.DataFrame({
            "y_pred": y_pred_profiled_simple_claude_val,
            "thinking": thinking_profiled_simple_claude
        })
        profiled_simple_df_claude.to_csv("../exp/y_pred_LLMs/Claude/y_pred_claude_profiled_simple_prompt.csv", sep=",", index=False)
        print("Saved df")

    y_pred_profiled_simple_claude.append(message.content[1].text)
    thinking_profiled_simple_claude.append(message.content[0].thinking)
    # print(message.content[1].text)

end = time.time()
print(f"Time taken: {end - start} seconds")
time_claude_profiled_simple = end - start
time_claude_profiled_simple_df = pd.DataFrame({"time": [time_claude_profiled_simple]})
time_claude_profiled_simple_df.to_csv("../exp/times_LLMs/Claude/time_claude_profiled_simple_prompt.csv", sep = ",", index = False)

# value counts for array
counts_profiled_simple_claude = pd.Series(y_pred_profiled_simple_claude).value_counts()
print(counts_profiled_simple_claude)

# convert YES to 1 and NO to 0
y_pred_profiled_simple_claude_val = [1 if response == "YES" else 0 for response in y_pred_profiled_simple_claude]

# save the array to a csv file
profiled_simple_df_claude = pd.DataFrame({
    "y_pred": y_pred_profiled_simple_claude_val,
    "thinking": thinking_profiled_simple_claude
})
profiled_simple_df_claude.to_csv("../exp/y_pred_LLMs/Claude/y_pred_claude_profiled_simple_prompt.csv", sep = ",", index = False)



#### Few shot prompt ####

y_pred_few_shot_claude = []
thinking_few_shot_claude = []

client = anthropic.Anthropic(
    api_key = os.environ.get("ANTHROPIC_API_KEY")
)

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_few_shot_prompt, desc = "Few Shot Prompting"):
    message = client.messages.create(
        model = "claude-sonnet-4-20250514",
        max_tokens = 20000,
        temperature = 1,
        thinking = {
            "type": "enabled",
            "budget_tokens": 16000
        },
        system = few_shot_instruction,
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
            model = "claude-sonnet-4-20250514",
            max_tokens = 20000,
            temperature = 1,
            thinking = {
                "type": "enabled",
                "budget_tokens": 16000
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

    if len(y_pred_few_shot_claude) % 50 == 0 and len(y_pred_few_shot_claude) > 0:
        print(f"\n\nProcessed {len(y_pred_few_shot_claude)} prompts.\n")
        counts_few_shot_claude = pd.Series(y_pred_few_shot_claude).value_counts()
        print(counts_few_shot_claude, "\n")

        y_pred_few_shot_claude_val = [1 if response.strip() == "YES" else 0 if response.strip() == "NO" else np.nan for response in y_pred_few_shot_claude]

        # save as df
        few_shot_df_claude = pd.DataFrame({
            "y_pred": y_pred_few_shot_claude_val,
            "thinking": thinking_few_shot_claude
        })
        few_shot_df_claude.to_csv("../exp/y_pred_LLMs/Claude/y_pred_claude_few_shot_prompt.csv", sep=",", index=False)
        print("Saved df")

    y_pred_few_shot_claude.append(message.content[1].text)
    thinking_few_shot_claude.append(message.content[0].thinking)
    # print(message.content[1].text)

end = time.time()
print(f"Time taken: {end - start} seconds")
time_claude_few_shot = end - start
time_claude_few_shot_df = pd.DataFrame({"time": [time_claude_few_shot]})
time_claude_few_shot_df.to_csv("../exp/times_LLMs/Claude/time_claude_few_shot_prompt.csv", sep = ",", index = False)

# value counts for array
counts_few_shot_claude = pd.Series(y_pred_few_shot_claude).value_counts()
print(counts_few_shot_claude)

# convert YES to 1 and NO to 0
y_pred_few_shot_claude_val = [1 if response == "YES" else 0 for response in y_pred_few_shot_claude]

# save the array to a csv file
few_shot_df_claude = pd.DataFrame({
    "y_pred": y_pred_few_shot_claude_val,
    "thinking": thinking_few_shot_claude
})
few_shot_df_claude.to_csv("../exp/y_pred_LLMs/Claude/y_pred_claude_few_shot_prompt.csv", sep = ",", index = False)



#### Vignette prompt ####

y_pred_vignette_claude = []
thinking_vignette_claude = []

client = anthropic.Anthropic(
    api_key = os.environ.get("ANTHROPIC_API_KEY")
)

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_vignette_prompt, desc = "Vignette Prompting"):
    message = client.messages.create(
        model = "claude-sonnet-4-20250514",
        max_tokens = 20000,
        temperature = 1,
        thinking = {
            "type": "enabled",
            "budget_tokens": 16000
        },
        system = vignette_instruction,
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
            model = "claude-sonnet-4-20250514",
            max_tokens = 20000,
            temperature = 1,
            thinking = {
                "type": "enabled",
                "budget_tokens": 16000
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

    if len(y_pred_vignette_claude) % 50 == 0 and len(y_pred_vignette_claude) > 0:
        print(f"\n\nProcessed {len(y_pred_vignette_claude)} prompts.\n")
        counts_vignette_claude = pd.Series(y_pred_vignette_claude).value_counts()
        print(counts_vignette_claude, "\n")

        y_pred_vignette_claude_val = [1 if response.strip() == "YES" else 0 if response.strip() == "NO" else np.nan for response in y_pred_vignette_claude]

        # save as df
        vignette_df_claude = pd.DataFrame({
            "y_pred": y_pred_vignette_claude_val,
            "thinking": thinking_vignette_claude
        })
        vignette_df_claude.to_csv("../exp/y_pred_LLMs/Claude/y_pred_claude_vignette_prompt.csv", sep=",", index=False)
        print("Saved df")

    y_pred_vignette_claude.append(message.content[1].text)
    thinking_vignette_claude.append(message.content[0].thinking)
    # print(message.content[1].text)

end = time.time()
print(f"Time taken: {end - start} seconds")
time_claude_vignette = end - start
time_claude_vignette_df = pd.DataFrame({"time": [time_claude_vignette]})
time_claude_vignette_df.to_csv("../exp/times_LLMs/Claude/time_claude_vignette_prompt.csv", sep = ",", index = False)

# value counts for array
counts_vignette_claude = pd.Series(y_pred_vignette_claude).value_counts()
print(counts_vignette_claude)

# convert YES to 1 and NO to 0
y_pred_vignette_claude_val = [1 if response == "YES" else 0 for response in y_pred_vignette_claude]

# save the array to a csv file
vignette_df_claude = pd.DataFrame({
    "y_pred": y_pred_vignette_claude_val,
    "thinking": thinking_vignette_claude
})
vignette_df_claude.to_csv("../exp/y_pred_LLMs/Claude/y_pred_claude_vignette_prompt.csv", sep = ",", index = False)



#### Chain-of-thought prompt ####

y_pred_cot_claude = []
explanation_cot_claude = []
thinking_cot_claude = []

client = anthropic.Anthropic(
    api_key = os.environ.get("ANTHROPIC_API_KEY")
)

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

    if len(y_pred_cot_claude) % 50 == 0 and len(y_pred_cot_claude) > 0:
        print(f"\n\nProcessed {len(y_pred_cot_claude)} prompts.\n")
        counts_cot_claude = pd.Series(y_pred_cot_claude).value_counts()
        print(counts_cot_claude, "\n")

        y_pred_cot_claude_val = [1 if response.strip() == "YES" else 0 if response.strip() == "NO" else np.nan for response in y_pred_cot_claude]

        # save as df
        cot_df_claude = pd.DataFrame({
            "y_pred": y_pred_cot_claude_val,
            "thinking": thinking_cot_claude,
            "cot": explanation_cot_claude
        })
        cot_df_claude.to_csv("../exp/y_pred_LLMs/Claude/y_pred_claude_cot_prompt.csv", sep=",", index=False)
        print("Saved df")

    try:
        prediction = re.findall(r'Prediction: (.*)', message.content[1].text)[0].strip()
        explanation = re.findall(r'Explanation: (.*)', message.content[1].text)[0].strip()
        y_pred_cot_claude.append(prediction)
        explanation_cot_claude.append(explanation)
        thinking_cot_claude.append(message.content[0].thinking)
        # print(prediction)
    except IndexError:
        print("IndexError")
        y_pred_cot_claude.append("IndexError")
        explanation_cot_claude.append("IndexError")
        thinking_cot_claude.append("IndexError")

end = time.time()
print(f"Time taken: {end - start} seconds")
time_claude_cot_prompt = end - start
time_claude_cot_df = pd.DataFrame({"time": [time_claude_cot_prompt]})
time_claude_cot_df.to_csv("../exp/times_LLMs/Claude/time_claude_cot_prompt.csv", sep = ",", index = False)

# value counts for array
counts_cot_claude = pd.Series(y_pred_cot_claude).value_counts()
print(counts_cot_claude)

# convert YES to 1 and NO to 0
y_pred_cot_claude_val = [1 if response == "YES" else 0 for response in y_pred_cot_claude]

# save the array to a csv file
cot_df_claude = pd.DataFrame({
    "y_pred": y_pred_cot_claude_val,
    "thinking": thinking_cot_claude,
    "cot": explanation_cot_claude
})
cot_df_claude.to_csv("../exp/y_pred_LLMs/Claude/y_pred_claude_cot_prompt.csv", sep = ",", index = False)
