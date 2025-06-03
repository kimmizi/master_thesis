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

# client = genai.Client(
#     api_key = os.environ.get("GEMINI_API_KEY")
# )
#
# response = client.models.generate_content(
#     model = "gemma-3-27b-it",
#     contents = "Roses are red...",
# )
#
# print(response.text)

# client = genai.Client(
#     api_key = os.environ.get("GEMINI_API_KEY")
# )
#
# response = client.models.generate_content(
#     model = "gemma-3-27b-it",
#     contents = [simple_instruction, X_test_simple_prompt[0]]
# )
#
# print(response.text)



### 2 Prompting with Gemma 3 ####



#### Simple prompt ####

y_pred_simple_gemma = []

client = genai.Client(
    api_key = os.environ.get("GEMINI_API_KEY")
)

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_simple_prompt, desc = "Simple prompting"):
    response = client.models.generate_content(
        model = "gemma-3-27b-it",
        contents = [simple_instruction, prompt]
    )

    if response.text.strip() not in ("YES", "NO"):
        print("\n Invalid output. Retry prompting. \n")
        response = client.models.generate_content(
            model = "gemma-3-27b-it",
            contents = [retry_instruction, prompt]
        )

    if len(y_pred_simple_gemma) % 50 == 0 and len(y_pred_simple_gemma) > 0:
        print(f"\n\nProcessed {len(y_pred_simple_gemma)} prompts.\n")
        counts_profiled_simple_gemma = pd.Series(y_pred_simple_gemma).value_counts()
        print(counts_profiled_simple_gemma, "\n")

        y_pred_simple_gemma_val = [1 if response == "YES" else 0 if response == "NO" else np.nan for response in y_pred_simple_gemma]

        # save as df
        simple_df_gemma = pd.DataFrame(y_pred_simple_gemma_val, columns=["y_pred"])
        simple_df_gemma.to_csv("../exp/y_pred_LLMs/Gemma/y_pred_gemma_simple_prompt.csv", sep=",", index=False)
        print("Saved df")

    y_pred_simple_gemma.append(response.text)
    # print(response.text)

end = time.time()
print(f"Time taken: {end - start} seconds")
time_gemma_simple_prompt = end - start
time_gemma_simple_df = pd.DataFrame({"time": [time_gemma_simple_prompt]})
time_gemma_simple_df.to_csv("../exp/times_LLMs/Gemma/time_gemma_simple_prompt.csv", sep = ",", index = False)

# value counts for array
counts_simple_gemma = pd.Series(y_pred_simple_gemma).value_counts()
print(counts_simple_gemma)

# convert YES to 1 and NO to 0
y_pred_simple_gemma_val = [1 if response == "YES" else 0 if response == "NO" else np.nan for response in y_pred_simple_gemma]

# save the array to a csv file
simple_df_gemma = pd.DataFrame(y_pred_simple_gemma_val, columns = ["y_pred"])
simple_df_gemma.to_csv("../exp/y_pred_LLMs/Gemma/y_pred_gemma_simple_prompt.csv", sep = ",", index = False)



#### Class definition prompt ####

y_pred_class_def_gemma = []

client = genai.Client(
    api_key = os.environ.get("GEMINI_API_KEY")
)

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in X_test_class_definitions_prompt:
    response = client.models.generate_content(
        model = "gemma-3-27b-it",
        contents = [simple_instruction, prompt]
    )

    if response.text.strip() not in ("YES", "NO"):
        print("\n Invalid output. Retry prompting. \n")
        response = client.models.generate_content(
            model = "gemma-3-27b-it",
            contents = [retry_instruction, prompt]
        )

    if len(y_pred_class_def_gemma) % 50 == 0 and len(y_pred_class_def_gemma) > 0:
        print(f"\n\nProcessed {len(y_pred_class_def_gemma)} prompts.\n")
        counts_class_def_gemma = pd.Series(y_pred_class_def_gemma).value_counts()
        print(counts_class_def_gemma, "\n")

        y_pred_class_def_gemma_val = [1 if response == "YES" else 0 if response == "NO" else np.nan for response in y_pred_class_def_gemma]

        # save as df
        class_def_df_gemma = pd.DataFrame(y_pred_class_def_gemma_val, columns=["y_pred"])
        class_def_df_gemma.to_csv("../exp/y_pred_LLMs/Gemma/y_pred_gemma_class_definitions_prompt.csv", sep=",", index=False)
        print("Saved df")

    y_pred_class_def_gemma.append(response.text)
    # print(response.text)

end = time.time()
print(f"Time taken: {end - start} seconds")
time_gemma_class_def = end - start
time_gemma_class_def_df = pd.DataFrame({"time": [time_gemma_class_def]})
time_gemma_class_def_df.to_csv("../exp/times_LLMs/Gemma/time_gemma_class_definitions_prompt.csv", sep = ",", index = False)

# value counts for array
counts_class_def_gemma = pd.Series(y_pred_class_def_gemma).value_counts()
print(counts_class_def_gemma)

# convert YES to 1 and NO to 0
y_pred_class_def_gemma_val = [1 if response == "YES" else 0 for response in y_pred_class_def_gemma]

# save the array to a csv file
class_def_df_gemma = pd.DataFrame(y_pred_class_def_gemma_val, columns = ["y_pred"])
class_def_df_gemma.to_csv("../exp/y_pred_LLMs/Gemma/y_pred_gemma_class_definitions_prompt.csv", sep = ",", index = False)



#### Profiled simple prompt ####

y_pred_profiled_simple_gemma = []

client = genai.Client(
    api_key = os.environ.get("GEMINI_API_KEY")
)

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_profiled_simple_prompt, desc = "Profiled simple prompting"):
    response = client.models.generate_content(
        model = "gemma-3-27b-it",
        contents = [simple_instruction, prompt]
    )

    if response.text.strip() not in ("YES", "NO"):
        print("\n Invalid output. Retry prompting. \n")
        response = client.models.generate_content(
            model = "gemma-3-27b-it",
            contents = [retry_instruction, prompt]
        )

    if len(y_pred_profiled_simple_gemma) % 50 == 0 and len(y_pred_profiled_simple_gemma) > 0:
        print(f"\n\nProcessed {len(y_pred_profiled_simple_gemma)} prompts.\n")
        counts_profiled_simple_gemma = pd.Series(y_pred_profiled_simple_gemma).value_counts()
        print(counts_profiled_simple_gemma, "\n")

        y_pred_profiled_simple_gemma_val = [1 if response == "YES" else 0 if response == "NO" else np.nan for response in y_pred_profiled_simple_gemma]

        # save as df
        profiled_simple_df_gemma = pd.DataFrame(y_pred_profiled_simple_gemma_val, columns=["y_pred"])
        profiled_simple_df_gemma.to_csv("../exp/y_pred_LLMs/Gemma/y_pred_gemma_profiled_simple_prompt.csv", sep=",", index=False)
        print("Saved df")

    y_pred_profiled_simple_gemma.append(response.text)
    # print(response.text)

end = time.time()
print(f"Time taken: {end - start} seconds")
time_gemma_profiled_simple = end - start
time_gemma_profiled_simple_df = pd.DataFrame({"time": [time_gemma_profiled_simple]})
time_gemma_profiled_simple_df.to_csv("../exp/times_LLMs/Gemma/time_gemma_profiled_simple_prompt.csv", sep = ",", index = False)

# value counts for array
counts_profiled_simple_gemma = pd.Series(y_pred_profiled_simple_gemma).value_counts()
print(counts_profiled_simple_gemma)

# convert YES to 1 and NO to 0
y_pred_profiled_simple_gemma_val = [1 if response == "YES" else 0 for response in y_pred_profiled_simple_gemma]

# save the array to a csv file
profiled_simple_df_gemma = pd.DataFrame(y_pred_profiled_simple_gemma_val, columns = ["y_pred"])
profiled_simple_df_gemma.to_csv("../exp/y_pred_LLMs/Gemma/y_pred_gemma_profiled_simple_prompt.csv", sep = ",", index = False)



#### Few shot prompt ####

y_pred_few_shot_gemma = []

client = genai.Client(
    api_key = os.environ.get("GEMINI_API_KEY")
)

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_few_shot_prompt, desc = "Few shot prompting"):
    response = client.models.generate_content(
        model = "gemma-3-27b-it",
        contents = [simple_instruction, prompt]
    )

    if response.text.strip() not in ("YES", "NO"):
        print("\n Invalid output. Retry prompting. \n")
        response = client.models.generate_content(
            model = "gemma-3-27b-it",
            contents = [retry_instruction, prompt]
        )

    if len(y_pred_few_shot_gemma) % 50 == 0 and len(y_pred_few_shot_gemma) > 0:
        print(f"\n\nProcessed {len(y_pred_few_shot_gemma)} prompts.\n")
        counts_few_shot_gemma = pd.Series(y_pred_few_shot_gemma).value_counts()
        print(counts_few_shot_gemma, "\n")

        y_pred_few_shot_gemma_val = [1 if response == "YES" else 0 if response == "NO" else np.nan for response in y_pred_few_shot_gemma]

        # save as df
        few_shot_df_gemma = pd.DataFrame(y_pred_few_shot_gemma_val, columns=["y_pred"])
        few_shot_df_gemma.to_csv("../exp/y_pred_LLMs/Gemma/y_pred_gemma_few_shot_prompt.csv", sep=",", index=False)
        print("Saved df")

    y_pred_few_shot_gemma.append(response.text)
    # print(response.text)

end = time.time()
print(f"Time taken: {end - start} seconds")
time_gemma_few_shot = end - start
time_gemma_few_shot_df = pd.DataFrame({"time": [time_gemma_few_shot]})
time_gemma_few_shot_df.to_csv("../exp/times_LLMs/Gemma/time_gemma_few_shot_prompt.csv", sep = ",", index = False)

# value counts for array
counts_few_shot_gemma = pd.Series(y_pred_few_shot_gemma).value_counts()
print(counts_few_shot_gemma)

# convert YES to 1 and NO to 0
y_pred_few_shot_gemma_val = [1 if response == "YES" else 0 for response in y_pred_few_shot_gemma]

# save the array to a csv file
few_shot_df_gemma = pd.DataFrame(y_pred_few_shot_gemma_val, columns = ["y_pred"])
few_shot_df_gemma.to_csv("../exp/y_pred_LLMs/Gemma/y_pred_gemma_few_shot_prompt.csv", sep = ",", index = False)



#### Vignette prompt ####

y_pred_vignette_gemma = []

client = genai.Client(
    api_key = os.environ.get("GEMINI_API_KEY")
)

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_vignette_prompt, desc = "Vignette prompting"):
    response = client.models.generate_content(
        model = "gemma-3-27b-it",
        contents = [simple_instruction, prompt]
    )

    if response.text.strip() not in ("YES", "NO"):
        print("\n Invalid output. Retry prompting. \n")
        response = client.models.generate_content(
            model = "gemma-3-27b-it",
            contents = [retry_instruction, prompt]
        )

    if len(y_pred_vignette_gemma) % 50 == 0 and len(y_pred_vignette_gemma) > 0:
        print(f"\n\nProcessed {len(y_pred_vignette_gemma)} prompts.\n")
        counts_vignette_gemma = pd.Series(y_pred_vignette_gemma).value_counts()
        print(counts_vignette_gemma, "\n")

        y_pred_vignette_gemma_val = [1 if response == "YES" else 0 if response == "NO" else np.nan for response in y_pred_vignette_gemma]

        # save as df
        vignette_df_gemma = pd.DataFrame(y_pred_vignette_gemma_val, columns=["y_pred"])
        vignette_df_gemma.to_csv("../exp/y_pred_LLMs/Gemma/y_pred_gemma_vignette_prompt.csv", sep=",", index=False)
        print("Saved df")

    y_pred_vignette_gemma.append(response.text)
    # print(response.text)

end = time.time()
print(f"Time taken: {end - start} seconds")
time_gemma_vignette = end - start
time_gemma_vignette_df = pd.DataFrame({"time": [time_gemma_vignette]})
time_gemma_vignette_df.to_csv("../exp/times_LLMs/Gemma/time_gemma_vignette_prompt.csv", sep = ",", index = False)

# value counts for array
counts_vignette_gemma = pd.Series(y_pred_vignette_gemma).value_counts()
print(counts_vignette_gemma)

# convert YES to 1 and NO to 0
y_pred_vignette_gemma_val = [1 if response == "YES" else 0 for response in y_pred_vignette_gemma]

# save the array to a csv file
vignette_df_gemma = pd.DataFrame(y_pred_vignette_gemma_val, columns = ["y_pred"])
vignette_df_gemma.to_csv("../exp/y_pred_LLMs/Gemma/y_pred_gemma_vignette_prompt.csv", sep = ",", index = False)



#### Chain-of-thought prompt ####

y_pred_cot_gemma = []
explanation_cot_gemma = []

client = genai.Client(
    api_key = os.environ.get("GEMINI_API_KEY")
)

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_cot_prompt, desc = "Chain-of-thought prompting"):
    response = client.models.generate_content(
        model = "gemma-3-27b-it",
        contents = [simple_instruction, prompt]
    )

    if len(y_pred_cot_gemma) % 50 == 0 and len(y_pred_cot_gemma) > 0:
        print(f"\n\nProcessed {len(y_pred_cot_gemma)} prompts.\n")
        counts_cot_gemma = pd.Series(y_pred_cot_gemma).value_counts()
        print(counts_cot_gemma, "\n")

        y_pred_cot_gemma_val = [1 if response == "YES" else 0 if response == "NO" else np.nan for response in y_pred_cot_gemma]

        # save as df
        cot_df_gemma = pd.DataFrame(y_pred_cot_gemma_val, columns=["y_pred"])
        cot_df_gemma.to_csv("../exp/y_pred_LLMs/Gemma/y_pred_gemma_cot_prompt.csv", sep=",", index=False)
        print("Saved df")

    try:
        prediction = re.findall(r'Prediction: (.*)', response.text)[0].strip()
        explanation = re.findall(r'Explanation: (.*)', response.text)[0].strip()
        y_pred_cot_gemma.append(prediction)
        explanation_cot_gemma.append(explanation)
        # print(prediction)
    except IndexError:
        print("IndexError")
        y_pred_cot_gemma.append("IndexError")
        explanation_cot_gemma.append("IndexError")

end = time.time()
print(f"Time taken: {end - start} seconds")
time_gemma_cot = end - start
time_gemma_cot_df = pd.DataFrame({"time": [time_gemma_cot]})
time_gemma_cot_df.to_csv("../exp/times_LLMs/Gemma/time_gemma_cot_prompt.csv", sep = ",", index = False)

# value counts for array
counts_cot_gemma = pd.Series(y_pred_cot_gemma).value_counts()
print(counts_cot_gemma)

# convert YES to 1 and NO to 0
y_pred_cot_gemma_val = [1 if response == "YES" else 0 for response in y_pred_cot_gemma]

# save the array to a csv file
cot_df_gemma = pd.DataFrame({
    "y_pred": y_pred_cot_gemma_val,
    "explanation": explanation_cot_gemma
})
cot_df_gemma.to_csv("../exp/y_pred_LLMs/Gemma/y_pred_gemma_cot_prompt.csv", sep = ",", index = False)
