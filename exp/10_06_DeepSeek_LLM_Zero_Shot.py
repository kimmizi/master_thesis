#### LLM: Zero-shot classification through LLMs and prompts ####
#### DEEPSEEK ####


#### 0 Imports ####
import os
import pandas as pd
import numpy as np
import time
import re
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


# **Off-Peak Discounts**：DeepSeek-R1 with 75% off at off-peak hours (16:30-00:30 UTC daily)

#### 1 Testing prompting ####

client = OpenAI(api_key = os.environ.get("DeepSeek_API_Key"), base_url = "https://api.deepseek.com")

response = client.chat.completions.create(
    model = "deepseek-reasoner",
    messages = [
        {"role": "system", "content": simple_instruction},
        {"role": "user", "content": X_test_simple_prompt[0]},
    ],
    stream = False
)

print(response.choices[0].message.content)

response.choices[0].message.reasoning_content



#### 2 Prompting with DeepSeek Reasoning R1 ####



#### Simple prompt ####

y_pred_simple_deeps = []
thinking_simple_deeps = []

client = OpenAI(api_key = os.environ.get("DeepSeek_API_Key"), base_url = "https://api.deepseek.com")

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in X_test_simple_prompt:
    response = client.chat.completions.create(
        model = "deepseek-reasoner",
        messages = [
            {"role": "system", "content": simple_instruction},
            {"role": "user", "content": prompt},
        ],
        stream = False
    )

    if response.choices[0].message.content.strip() not in ("YES", "NO"):
        print("\n Invalid output. Retry prompting. \n")
        response = client.chat.completions.create(
            model = "deepseek-reasoner",
            messages = [
                {"role": "system", "content": retry_instruction},
                {"role": "user", "content": prompt},
            ],
            stream = False
        )

    y_pred_simple_deeps.append(response.choices[0].message.content)
    thinking_simple_deeps.append(response.choices[0].message.reasoning_content)
    print(response.choices[0].message.content)

end = time.time()
print(f"Time taken: {end - start} seconds")
time_deeps_simple_prompt = end - start
time_deeps_simple_df = pd.DataFrame({"time": [time_deeps_simple_prompt]})
time_deeps_simple_df.to_csv("../exp/times_LLMs/DeepSeek/time_deeps_simple_prompt.csv", sep = ",", index = False)

# value counts for array
counts_simple_deeps = pd.Series(y_pred_simple_deeps).value_counts()
print(counts_simple_deeps)

# convert YES to 1 and NO to 0
y_pred_simple_deeps = [1 if response == "YES" else 0 if response == "NO" else np.nan for response in y_pred_simple_deeps]

# save the array to a csv file
simple_df_deeps = pd.DataFrame(y_pred_simple_deeps, columns = ["y_pred"])
simple_df_deeps.to_csv("../exp/y_pred_LLMs/DeepSeek/y_pred_deeps_simple_prompt.csv", sep = ",", index = False)

simple_df_thinking_deeps = pd.DataFrame(thinking_simple_deeps, columns = ["thinking"])
simple_df_thinking_deeps.to_csv("../exp/y_pred_LLMs/DeepSeek/Thinking/thinking_deeps_simple_prompt.csv", sep = ",", index = False)



#### Class definition prompt ####

y_pred_class_def_deeps = []
thinking_class_def_deeps = []

client = OpenAI(api_key = os.environ.get("DeepSeek_API_Key"), base_url = "https://api.deepseek.com")

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in X_test_class_definitions_prompt:
    response = client.chat.completions.create(
        model = "deepseek-reasoner",
        messages = [
            {"role": "system", "content": class_definitions_instruction},
            {"role": "user", "content": prompt},
        ],
        stream = False
    )

    if response.choices[0].message.content.strip() not in ("YES", "NO"):
        print("\n Invalid output. Retry prompting. \n")
        response = client.chat.completions.create(
            model = "deepseek-reasoner",
            messages = [
                {"role": "system", "content": retry_instruction},
                {"role": "user", "content": prompt},
            ],
            stream = False
        )

    y_pred_class_def_deeps.append(response.choices[0].message.content)
    thinking_class_def_deeps.append(response.choices[0].message.reasoning_content)
    print(response.choices[0].message.content)

end = time.time()
print(f"Time taken: {end - start} seconds")
time_deeps_class_definitions = end - start
time_deeps_class_definitions_df = pd.DataFrame({"time": [time_deeps_class_definitions]})
time_deeps_class_definitions_df.to_csv("../exp/times_LLMs/DeepSeek/time_deeps_class_definitions_prompt.csv", sep = ",", index = False)

# value counts for array
counts_class_def_deeps = pd.Series(y_pred_class_def_deeps).value_counts()
print(counts_class_def_deeps)

# convert YES to 1 and NO to 0
y_pred_class_def_deeps = [1 if response == "YES" else 0 for response in y_pred_class_def_deeps]

# save the array to a csv file
class_def_df_deeps = pd.DataFrame(y_pred_class_def_deeps, columns = ["y_pred"])
class_def_df_deeps.to_csv("../exp/y_pred_LLMs/DeepSeek/y_pred_deeps_class_definitions_prompt.csv", sep = ",", index = False)

class_def_df_thinking_deeps = pd.DataFrame(thinking_class_def_deeps, columns = ["thinking"])
class_def_df_thinking_deeps.to_csv("../exp/y_pred_LLMs/DeepSeek/Thinking/thinking_deeps_class_def_prompt.csv", sep = ",", index = False)



#### Profiled simple prompt ####

y_pred_profiled_simple_deeps = []
thinking_profiled_simple_deeps = []

client = OpenAI(api_key = os.environ.get("DeepSeek_API_Key"), base_url = "https://api.deepseek.com")

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in X_test_profiled_simple_prompt:
    response = client.chat.completions.create(
        model = "deepseek-reasoner",
        messages = [
            {"role": "system", "content": profiled_simple_instruction},
            {"role": "user", "content": prompt},
        ],
        stream = False
    )

    if response.choices[0].message.content.strip() not in ("YES", "NO"):
        print("\n Invalid output. Retry prompting. \n")
        response = client.chat.completions.create(
            model = "deepseek-reasoner",
            messages = [
                {"role": "system", "content": retry_instruction},
                {"role": "user", "content": prompt},
            ],
            stream = False
        )

    y_pred_profiled_simple_deeps.append(response.choices[0].message.content)
    thinking_profiled_simple_deeps.append(response.choices[0].message.reasoning_content)
    print(response.choices[0].message.content)

end = time.time()
print(f"Time taken: {end - start} seconds")
time_deeps_profiled_simple = end - start
time_deeps_profiled_simple_df = pd.DataFrame({"time": [time_deeps_profiled_simple]})
time_deeps_profiled_simple_df.to_csv("../exp/times_LLMs/DeepSeek/time_deeps_profiled_simple_prompt.csv", sep = ",", index = False)

# value counts for array
counts_profiled_simple_deeps = pd.Series(y_pred_profiled_simple_deeps).value_counts()
print(counts_profiled_simple_deeps)

# convert YES to 1 and NO to 0
y_pred_profiled_simple_deeps_val = [1 if response == "YES" else 0 for response in y_pred_profiled_simple_deeps]

# save the array to a csv file
profiled_simple_df_deeps = pd.DataFrame(y_pred_profiled_simple_deeps_val, columns = ["y_pred"])
profiled_simple_df_deeps.to_csv("../exp/y_pred_LLMs/DeepSeek/y_pred_deeps_profiled_simple_prompt.csv", sep = ",", index = False)

profiled_simple_df_thinking_deeps = pd.DataFrame(thinking_profiled_simple_deeps, columns = ["thinking"])
profiled_simple_df_thinking_deeps.to_csv("../exp/y_pred_LLMs/DeepSeek/Thinking/thinking_deeps_profiled_simple_prompt.csv", sep = ",", index = False)



#### Few shot prompt ####

y_pred_few_shot_deeps = []
thinking_few_shot_deeps = []

client = OpenAI(api_key = os.environ.get("DeepSeek_API_Key"), base_url = "https://api.deepseek.com")

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in X_test_few_shot_prompt:
    response = client.chat.completions.create(
        model = "deepseek-reasoner",
        messages = [
            {"role": "system", "content": few_shot_instruction},
            {"role": "user", "content": prompt},
        ],
        stream = False
    )

    if response.choices[0].message.content.strip() not in ("YES", "NO"):
        print("\n Invalid output. Retry prompting. \n")
        response = client.chat.completions.create(
            model = "deepseek-reasoner",
            messages = [
                {"role": "system", "content": retry_instruction},
                {"role": "user", "content": prompt},
            ],
            stream = False
        )

    y_pred_few_shot_deeps.append(response.choices[0].message.content)
    thinking_few_shot_deeps.append(response.choices[0].message.reasoning_content)
    print(response.choices[0].message.content)

end = time.time()
print(f"Time taken: {end - start} seconds")
time_deeps_few_shot = end - start
time_deeps_few_shot_df = pd.DataFrame({"time": [time_deeps_few_shot]})
time_deeps_few_shot_df.to_csv("../exp/times_LLMs/DeepSeek/time_deeps_few_shot_prompt.csv", sep = ",", index = False)

# value counts for array
counts_few_shot_deeps = pd.Series(y_pred_few_shot_deeps).value_counts()
print(counts_few_shot_deeps)

# convert YES to 1 and NO to 0
y_pred_few_shot_deeps_val = [1 if response == "YES" else 0 for response in y_pred_few_shot_deeps]

# save the array to a csv file
few_shot_df_deeps = pd.DataFrame(y_pred_few_shot_deeps_val, columns = ["y_pred"])
few_shot_df_deeps.to_csv("../exp/y_pred_LLMs/DeepSeek/y_pred_deeps_few_shot_prompt.csv", sep = ",", index = False)

few_shot_df_thinking_deeps = pd.DataFrame(thinking_few_shot_deeps, columns = ["thinking"])
few_shot_df_thinking_deeps.to_csv("../exp/y_pred_LLMs/DeepSeek/Thinking/thinking_deeps_few_shot_prompt.csv", sep = ",", index = False)



#### Vignette prompt ####

y_pred_vignette_deeps = []
thinking_vignette_deeps = []

client = OpenAI(
    api_key = os.environ.get("DeepSeek_API_Key"),
    base_url = "https://api.deepseek.com"
)

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in X_test_vignette_prompt:
    response = client.chat.completions.create(
        model = "deepseek-reasoner",
        messages = [
            {"role": "system", "content": vignette_instruction},
            {"role": "user", "content": prompt},
        ],
        stream = False
    )

    if response.choices[0].message.content.strip() not in ("YES", "NO"):
        print("\n Invalid output. Retry prompting. \n")
        response = client.chat.completions.create(
            model = "deepseek-reasoner",
            messages = [
                {"role": "system", "content": retry_instruction},
                {"role": "user", "content": prompt},
            ],
            stream = False
        )

    y_pred_vignette_deeps.append(response.choices[0].message.content)
    thinking_vignette_deeps.append(response.choices[0].message.reasoning_content)
    print(response.choices[0].message.content)

end = time.time()
print(f"Time taken: {end - start} seconds")
time_deeps_vignette = end - start
time_deeps_vignette_df = pd.DataFrame({"time": [time_deeps_vignette]})
time_deeps_vignette_df.to_csv("../exp/times_LLMs/DeepSeek/time_deeps_vignette_prompt.csv", sep = ",", index = False)

# value counts for array
counts_vignette_deeps = pd.Series(y_pred_vignette_deeps).value_counts()
print(counts_vignette_deeps)

# convert YES to 1 and NO to 0
y_pred_vignette_deeps_val = [1 if response == "YES" else 0 for response in y_pred_vignette_deeps]

# save the array to a csv file
vignette_df_deeps = pd.DataFrame(y_pred_vignette_deeps_val, columns = ["y_pred"])
vignette_df_deeps.to_csv("../exp/y_pred_LLMs/DeepSeek/y_pred_deeps_vignette_prompt.csv", sep = ",", index = False)

vignette_df_thinking_deeps = pd.DataFrame(thinking_vignette_deeps, columns = ["thinking"])
vignette_df_thinking_deeps.to_csv("../exp/y_pred_LLMs/DeepSeek/Thinking/thinking_deeps_vignette_prompt.csv", sep = ",", index = False)



#### Chain-of-thought prompt ####

y_pred_cot_deeps = []
explanation_cot_deeps = []
thinking_cot_deeps = []

client = OpenAI(
    api_key = os.environ.get("DeepSeek_API_Key"),
    base_url = "https://api.deepseek.com"
)

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in X_test_cot_prompt:
    response = client.chat.completions.create(
        model = "deepseek-reasoner",
        messages = [
            {"role": "system", "content": cot_instruction},
            {"role": "user", "content": prompt},
        ],
        stream = False
    )
    try:
        prediction = re.findall(r'Prediction: (.*)', response.choices[0].message.content)[0].strip()
        explanation = re.findall(r'Explanation: (.*)', response.choices[0].message.content)[0].strip()
        y_pred_cot_deeps.append(prediction)
        explanation_cot_deeps.append(explanation)
        thinking_cot_deeps.append(response.choices[0].message.reasoning_content)
        print(prediction)
    except IndexError:
        print("IndexError")
        y_pred_cot_deeps.append("IndexError")
        explanation_cot_deeps.append("IndexError")
        thinking_cot_deeps.append("IndexError")

end = time.time()
print(f"Time taken: {end - start} seconds")
time_deeps_cot_prompt = end - start
time_deeps_cot_df = pd.DataFrame({"time": [time_deeps_cot_prompt]})
time_deeps_cot_df.to_csv("../exp/times_LLMs/DeepSeek/time_deeps_cot_prompt.csv", sep = ",", index = False)

# value counts for array
counts_cot_deeps = pd.Series(y_pred_cot_deeps).value_counts()
print(counts_cot_deeps)

# convert YES to 1 and NO to 0
y_pred_cot_deeps_val = [1 if response == "YES" else 0 for response in y_pred_cot_deeps]

# save the array to a csv file
cot_df_deeps = pd.DataFrame(y_pred_cot_deeps_val, columns = ["y_pred"])
cot_df_deeps.to_csv("../exp/y_pred_LLMs/DeepSeek/y_pred_deeps_cot_prompt.csv", sep = ",", index = False)

cot_df_thinking_deeps = pd.DataFrame(thinking_cot_deeps, columns = ["thinking"])
cot_df_thinking_deeps.to_csv("../exp/y_pred_LLMs/DeepSeek/Thinking/thinking_deeps_cot_prompt.csv", sep = ",", index = False)

cot_df_explanation_deeps = pd.DataFrame(explanation_cot_deeps, columns = ["cot"])
cot_df_explanation_deeps.to_csv("../exp/y_pred_LLMs/DeepSeek/explanation_deeps_cot_prompt.csv", sep = ",", index = False)
