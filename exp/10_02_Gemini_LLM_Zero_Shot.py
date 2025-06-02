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

# client = genai.Client(api_key = os.environ.get("GEMINI_API_KEY"))
#
# response = client.models.generate_content(
#     model = "gemini-2.5-pro-preview-05-06",
#     contents = "Explain how AI works in a few words",
# )
#
# print(response.text)
#
# client = genai.Client(
#     api_key = os.environ.get("GEMINI_API_KEY")
# )
#
# response = client.models.generate_content(
#     model = "gemini-2.5-pro-preview-05-06",
#     config = types.GenerateContentConfig(
#         system_instruction = simple_instruction,
#         thinking_config = types.ThinkingConfig(
#             include_thoughts = True
#         )
#     ),
#     contents = X_test_simple_prompt[0]
# )
#
# print(response.text)
#
# print(response.candidates[0].content.parts)
#
# for part in response.candidates[0].content.parts:
#   if not part.text:
#     continue
#   if part.thought:
#     print("Thought summary:")
#     print(part.text)
#     print()
#   else:
#     print("Answer:")
#     print(part.text)
#     print()



### 2 Prompting with Gemini 2.5 Pro ####



#### Simple prompt ####

y_pred_simple_gemini = []
thinking_simple_gemini = []

client = genai.Client(
    api_key = os.environ.get("GEMINI_API_KEY")
)

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_simple_prompt[:1], desc = "Simple prompting"):
    response = client.models.generate_content(
        model = "gemini-2.5-pro-preview-05-06",
        config = types.GenerateContentConfig(
            system_instruction = simple_instruction,
            thinking_config = types.ThinkingConfig(
                include_thoughts = True
            )
        ),
        contents = prompt,
    )

    if response.text.strip() not in ("YES", "NO"):
        print("\n Invalid output. Retry prompting. \n")
        response = client.models.generate_content(
            model = "gemini-2.5-pro-preview-05-06",
            config = types.GenerateContentConfig(
                system_instruction = retry_instruction,
                thinking_config = types.ThinkingConfig(
                    include_thoughts = True
                )
            ),
            contents = prompt,
        )

    if len(y_pred_simple_gemini) % 50 == 0:
        print(f"\n\nProcessed {len(y_pred_simple_gemini)} prompts.\n")
        counts_profiled_simple_grok = pd.Series(y_pred_simple_gemini).value_counts()
        print(counts_profiled_simple_grok, "\n")

    for part in response.candidates[0].content.parts:
        if not part.text:
            continue
        if part.thought:
            thinking_simple_gemini.append(part.text)

    y_pred_simple_gemini.append(response.text)
    # print(response.text)

end = time.time()
print(f"Time taken: {end - start} seconds")
time_gemini_simple_prompt = end - start
time_gemini_simple_df = pd.DataFrame({"time": [time_gemini_simple_prompt]})
time_gemini_simple_df.to_csv("../exp/times_LLMs/Gemini/time_gemini_simple_prompt.csv", sep = ",", index = False)

# value counts for array
counts_simple_gemini = pd.Series(y_pred_simple_gemini).value_counts()
print(counts_simple_gemini)

# convert YES to 1 and NO to 0
y_pred_simple_gemini_val = [1 if response == "YES" else 0 if response == "NO" else np.nan for response in y_pred_simple_gemini]

# save as df
simple_df_gemini = pd.DataFrame({
    "y_pred": y_pred_simple_gemini_val,
    "thinking": thinking_simple_gemini
})
simple_df_gemini.to_csv("../exp/y_pred_LLMs/Gemini/simple_gemini_prompt.csv", sep = ",", index = False)


#### Class definition prompt ####

y_pred_class_def_gemini = []
thinking_class_def_gemini = []

client = genai.Client(
    api_key = os.environ.get("GEMINI_API_KEY")
)

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_class_definitions_prompt[:1], desc = "Class definitions prompting"):
    response = client.models.generate_content(
        model = "gemini-2.5-pro-preview-05-06",
        config = types.GenerateContentConfig(
            system_instruction = class_definitions_instruction,
            thinking_config = types.ThinkingConfig(
                include_thoughts = True
            )
        ),
        contents = prompt,
    )

    if response.text.strip() not in ("YES", "NO"):
        print("\n Invalid output. Retry prompting. \n")
        response = client.models.generate_content(
            model = "gemini-2.5-pro-preview-05-06",
            config = types.GenerateContentConfig(
                system_instruction = retry_instruction,
                thinking_config = types.ThinkingConfig(
                    include_thoughts = True
                )
            ),
            contents = prompt,
        )

    if len(y_pred_class_def_gemini) % 50 == 0:
        print(f"\n\nProcessed {len(y_pred_class_def_gemini)} prompts.\n")
        counts_class_def_gemini = pd.Series(y_pred_class_def_gemini).value_counts()
        print(counts_class_def_gemini, "\n")

    for part in response.candidates[0].content.parts:
        if not part.text:
            continue
        if part.thought:
            thinking_class_def_gemini.append(part.text)

    y_pred_class_def_gemini.append(response.text)
    # print(response.text)

end = time.time()
print(f"Time taken: {end - start} seconds")
time_gemini_class_def = end - start
time_gemini_class_def_df = pd.DataFrame({"time": [time_gemini_class_def]})
time_gemini_class_def_df.to_csv("../exp/times_LLMs/Gemini/time_gemini_class_definitions_prompt.csv", sep = ",", index = False)

# value counts for array
counts_class_def_gemini = pd.Series(y_pred_class_def_gemini).value_counts()
print(counts_class_def_gemini)

# convert YES to 1 and NO to 0
y_pred_class_def_gemini_val = [1 if response == "YES" else 0 for response in y_pred_class_def_gemini]

# save as df
class_def_df_gemini = pd.DataFrame({
    "y_pred": y_pred_class_def_gemini_val,
    "thinking": thinking_class_def_gemini
})
class_def_df_gemini.to_csv("../exp/y_pred_LLMs/Gemini/class_definitions_gemini_prompt.csv", sep = ",", index = False)


#### Profiled simple prompt ####

y_pred_profiled_simple_gemini = []
thinking_profiled_simple_gemini = []

client = genai.Client(
    api_key = os.environ.get("GEMINI_API_KEY")
)

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_profiled_simple_prompt[:1], desc = "Profiled simple prompting"):
    response = client.models.generate_content(
        model = "gemini-2.5-pro-preview-05-06",
        config = types.GenerateContentConfig(
            system_instruction = profiled_simple_instruction,
            thinking_config = types.ThinkingConfig(
                include_thoughts = True
            )
        ),
        contents = prompt,
    )

    if response.text.strip() not in ("YES", "NO"):
        print("\n Invalid output. Retry prompting. \n")
        response = client.models.generate_content(
            model = "gemini-2.5-pro-preview-05-06",
            config = types.GenerateContentConfig(
                system_instruction = retry_instruction,
                thinking_config = types.ThinkingConfig(
                    include_thoughts = True
                )
            ),
            contents = prompt,
        )

    if len(y_pred_profiled_simple_gemini) % 50 == 0:
        print(f"\n\nProcessed {len(y_pred_profiled_simple_gemini)} prompts.\n")
        counts_profiled_simple_gemini = pd.Series(y_pred_profiled_simple_gemini).value_counts()
        print(counts_profiled_simple_gemini, "\n")

    for part in response.candidates[0].content.parts:
        if not part.text:
            continue
        if part.thought:
            thinking_profiled_simple_gemini.append(part.text)

    y_pred_profiled_simple_gemini.append(response.text)
    # print(response.text)

end = time.time()
print(f"Time taken: {end - start} seconds")
time_gemini_profiled_simple = end - start
time_gemini_profiled_simple_df = pd.DataFrame({"time": [time_gemini_profiled_simple]})
time_gemini_profiled_simple_df.to_csv("../exp/times_LLMs/Gemini/time_gemini_profiled_simple_prompt.csv", sep = ",", index = False)

# value counts for array
counts_profiled_simple_gemini = pd.Series(y_pred_profiled_simple_gemini).value_counts()
print(counts_profiled_simple_gemini)

# convert YES to 1 and NO to 0
y_pred_profiled_simple_gemini_val = [1 if response == "YES" else 0 for response in y_pred_profiled_simple_gemini]

# save as df
profiled_simple_df_gemini = pd.DataFrame({
    "y_pred": y_pred_profiled_simple_gemini_val,
    "thinking": thinking_profiled_simple_gemini
})
profiled_simple_df_gemini.to_csv("../exp/y_pred_LLMs/Gemini/profiled_simple_gemini_prompt.csv", sep = ",", index = False)


#### Few shot prompt ####

y_pred_few_shot_gemini = []
thinking_few_shot_gemini = []

client = genai.Client(
    api_key = os.environ.get("GEMINI_API_KEY")
)

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_few_shot_prompt[:1], desc = "Few-shot prompting"):
    response = client.models.generate_content(
        model = "gemini-2.5-pro-preview-05-06",
        config = types.GenerateContentConfig(
            system_instruction = few_shot_instruction,
            thinking_config = types.ThinkingConfig(
                include_thoughts = True
            )
        ),
        contents = prompt,
    )

    if response.text.strip() not in ("YES", "NO"):
        print("\n Invalid output. Retry prompting. \n")
        response = client.models.generate_content(
            model = "gemini-2.5-pro-preview-05-06",
            config = types.GenerateContentConfig(
                system_instruction = retry_instruction,
                thinking_config = types.ThinkingConfig(
                    include_thoughts = True
                )
            ),
            contents = prompt,
        )

    if len(y_pred_few_shot_gemini) % 50 == 0:
        print(f"\n\nProcessed {len(y_pred_few_shot_gemini)} prompts.\n")
        counts_few_shot_gemini = pd.Series(y_pred_few_shot_gemini).value_counts()
        print(counts_few_shot_gemini, "\n")

    for part in response.candidates[0].content.parts:
        if not part.text:
            continue
        if part.thought:
            thinking_few_shot_gemini.append(part.text)

    y_pred_few_shot_gemini.append(response.text)
    # print(response.text)

end = time.time()
print(f"Time taken: {end - start} seconds")
time_gemini_few_shot = end - start
time_gemini_few_shot_df = pd.DataFrame({"time": [time_gemini_few_shot]})
time_gemini_few_shot_df.to_csv("../exp/times_LLMs/Gemini/time_gemini_few_shot_prompt.csv", sep = ",", index = False)

# value counts for array
counts_few_shot_gemini = pd.Series(y_pred_few_shot_gemini).value_counts()
print(counts_few_shot_gemini)

# convert YES to 1 and NO to 0
y_pred_few_shot_gemini_val = [1 if response == "YES" else 0 for response in y_pred_few_shot_gemini]

# save as df
few_shot_df_gemini = pd.DataFrame({
    "y_pred": y_pred_few_shot_gemini_val,
    "thinking": thinking_few_shot_gemini
})
few_shot_df_gemini.to_csv("../exp/y_pred_LLMs/Gemini/few_shot_gemini_prompt.csv", sep = ",", index = False)


#### Vignette prompt ####

y_pred_vignette_gemini = []
thinking_vignette_gemini = []

client = genai.Client(
    api_key = os.environ.get("GEMINI_API_KEY")
)

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_vignette_prompt[:1], desc = "Vignette prompting"):
    response = client.models.generate_content(
        model = "gemini-2.5-pro-preview-05-06",
        config = types.GenerateContentConfig(
            system_instruction = vignette_instruction,
            thinking_config = types.ThinkingConfig(
                include_thoughts = True
            )
        ),
        contents = prompt,
    )

    if response.text.strip() not in ("YES", "NO"):
        print("\n Invalid output. Retry prompting. \n")
        response = client.models.generate_content(
            model = "gemini-2.5-pro-preview-05-06",
            config = types.GenerateContentConfig(
                system_instruction = retry_instruction,
                thinking_config = types.ThinkingConfig(
                    include_thoughts = True
                )
            ),
            contents = prompt,
        )

    if len(y_pred_vignette_gemini) % 50 == 0:
        print(f"\n\nProcessed {len(y_pred_vignette_gemini)} prompts.\n")
        counts_vignette_gemini = pd.Series(y_pred_vignette_gemini).value_counts()
        print(counts_vignette_gemini, "\n")

    for part in response.candidates[0].content.parts:
        if not part.text:
            continue
        if part.thought:
            thinking_vignette_gemini.append(part.text)

    y_pred_vignette_gemini.append(response.text)
    # print(response.text)

end = time.time()
print(f"Time taken: {end - start} seconds")
time_gemini_vignette = end - start
time_gemini_vignette_df = pd.DataFrame({"time": [time_gemini_vignette]})
time_gemini_vignette_df.to_csv("../exp/times_LLMs/Gemini/time_gemini_vignette_prompt.csv", sep = ",", index = False)

# value counts for array
counts_vignette_gemini = pd.Series(y_pred_vignette_gemini).value_counts()
print(counts_vignette_gemini)

# convert YES to 1 and NO to 0
y_pred_vignette_gemini_val = [1 if response == "YES" else 0 for response in y_pred_vignette_gemini]

# save as df
vignette_df_gemini = pd.DataFrame({
    "y_pred": y_pred_vignette_gemini_val,
    "thinking": thinking_vignette_gemini
})
vignette_df_gemini.to_csv("../exp/y_pred_LLMs/Gemini/vignette_gemini_prompt.csv", sep = ",", index = False)


#### Chain-of-thought prompt ####

y_pred_cot_gemini = []
thinking_cot_gemini = []
explanation_cot_gemini = []

client = genai.Client(
    api_key = os.environ.get("GEMINI_API_KEY")
)

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_cot_prompt[:1], desc = "Chain-of-thought prompting", units = "prompts"):
    response = client.models.generate_content(
        model = "gemini-2.5-pro-preview-05-06",
        config = types.GenerateContentConfig(
            system_instruction = cot_instruction,
            thinking_config = types.ThinkingConfig(
                include_thoughts = True
            )
        ),
        contents = prompt,
    )

    if len(y_pred_cot_gemini) % 50 == 0:
        print(f"\n\nProcessed {len(y_pred_cot_gemini)} prompts.\n")
        counts_cot_gemini = pd.Series(y_pred_cot_gemini).value_counts()
        print(counts_cot_gemini, "\n")


    try:
        prediction = re.findall(r'Prediction: (.*)', response.text)[0].strip()
        explanation = re.findall(r'Explanation: (.*)', response.text)[0].strip()
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                thinking_cot_gemini.append(part.text)
        y_pred_cot_gemini.append(prediction)
        explanation_cot_gemini.append(explanation)
        # print(prediction)
    except IndexError:
        print("IndexError")
        y_pred_cot_gemini.append("IndexError")
        explanation_cot_gemini.append("IndexError")

end = time.time()
print(f"Time taken: {end - start} seconds")
time_gemini_cot = end - start
time_gemini_cot_df = pd.DataFrame({"time": [time_gemini_cot]})
time_gemini_cot_df.to_csv("../exp/times_LLMs/Gemini/time_gemini_cot_prompt.csv", sep = ",", index = False)

# value counts for array
counts_cot_gemini = pd.Series(y_pred_cot_gemini).value_counts()
print(counts_cot_gemini)

# convert YES to 1 and NO to 0
y_pred_cot_gemini_val = [1 if response == "YES" else 0 for response in y_pred_cot_gemini]

# save as df (including explanation)
cot_df_gemini = pd.DataFrame({
    "y_pred": y_pred_cot_gemini_val,
    "thinking": thinking_cot_gemini,
    "explanation": explanation_cot_gemini
})
cot_df_gemini.to_csv("../exp/y_pred_LLMs/Gemini/cot_gemini_prompt.csv", sep = ",", index = False)