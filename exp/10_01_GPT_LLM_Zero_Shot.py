#### LLM: Zero-shot classification through LLMs and prompts ####
#### CHATGPT ####


#### 0 Imports ####
import os
import pandas as pd
import anthropic
import numpy as np
import time
import re
from openai import OpenAI
# from mistralai import Mistral
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
X_test_thinking_prompt_df = pd.read_csv("../dat/prompts/X_test_thinking_prompt.csv", sep = ",", index_col = 0)

# convert to arrays
X_test_simple_prompt = X_test_simple_prompt_df.values.flatten()
X_test_class_definitions_prompt = X_test_class_definitions_prompt_df.values.flatten()
X_test_profiled_simple_prompt = X_test_profiled_simple_prompt_df.values.flatten()
X_test_few_shot_prompt = X_test_few_shot_prompt_df.values.flatten()
X_test_vignette_prompt = X_test_vignette_prompt_df.values.flatten()
X_test_thinking_prompt = X_test_thinking_prompt_df.values.flatten()

# import instructions
simple_instruction_df = pd.read_csv("../dat/instructions/simple_instruction.csv", sep = ",", index_col = 0)
class_definitions_instruction_df = pd.read_csv("../dat/instructions/class_definitions_instruction.csv", sep = ",", index_col = 0)
profiled_simple_instruction_df = pd.read_csv("../dat/instructions/profiled_simple_instruction.csv", sep = ",", index_col = 0)
few_shot_instruction_df = pd.read_csv("../dat/instructions/few_shot_instruction.csv", sep = ",", index_col = 0)
vignette_instruction_df = pd.read_csv("../dat/instructions/vignette_instruction.csv", sep = ",", index_col = 0)
thinking_instruction_df = pd.read_csv("../dat/instructions/thinking_instruction.csv", sep = ",", index_col = 0)

# convert to string
simple_instruction = simple_instruction_df["0"].iloc[0]
class_definitions_instruction = class_definitions_instruction_df["0"].iloc[0]
profiled_simple_instruction = profiled_simple_instruction_df["0"].iloc[0]
few_shot_instruction = few_shot_instruction_df["0"].iloc[0]
vignette_instruction = vignette_instruction_df["0"].iloc[0]
thinking_instruction = thinking_instruction_df["0"].iloc[0]

# import retry instructions when output format was wrong
retry_instruction_df = pd.read_csv("../dat/instructions/retry_instruction.csv", sep = ",", index_col = 0)
retry_thinking_instruction_df = pd.read_csv("../dat/instructions/retry_thinking_instruction.csv", sep = ",", index_col = 0)

# import instruction for reason of misclassification
instruction_reason_df = pd.read_csv("../dat/instructions/instruction_reason.csv", sep=",", index_col = 0)

# convert to string
retry_instruction = retry_instruction_df["0"].iloc[0]
retry_thinking_instruction = retry_thinking_instruction_df["0"].iloc[0]

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



#### 1.1 Testing prompting ####

# client = OpenAI(
#     api_key = os.environ.get("OPENAI_API_KEY"),
# )
#
# # testing
# response = client.responses.create(
#     model = "gpt-4o-mini",
#     instructions = "You are a coding assistant that talks like a pirate.",
#     input = "How do I check if a Python object is an instance of a class?",
# )
#
# print(response.output_text)



#### 1.2 Prompting with ChatGPT ####

#### Simple prompt ####

simple_y_pred_GPT = []

client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY"),
)

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in X_test_simple_prompt:
    response = client.responses.create(
        # model = "o3-2025-04-16",
        model = "gpt-4o",
        instructions = simple_instruction,
        input = prompt
    )

    if response.output_text.strip() not in ("YES", "NO"):
        print("\n Invalid output. Retry prompting. \n")
        response = client.responses.create(
            model = "gpt-4o",
            instructions = retry_instruction,
            input = prompt
        )

    simple_y_pred_GPT.append(response.output_text)
    print(response.output_text)

    # save responses to csv after every 50th prompt
    if len(simple_y_pred_GPT) % 50 == 0:
        print("\n \n prompt", len(simple_y_pred_GPT))
        # value counts for array
        counts_simple_GPT = pd.Series(simple_y_pred_GPT).value_counts()
        print(counts_simple_GPT)

        # convert YES to 1 and NO to 0
        simple_y_pred_GPT = [1 if response == "YES" else 0 if response == "NO" else np.nan for response in simple_y_pred_GPT]

        # save the array to a csv file
        simple_prompt_df_GPT = pd.DataFrame(simple_y_pred_GPT, columns = ["y_pred"])
        simple_prompt_df_GPT.to_csv("../exp/y_pred_LLMs/GPT/y_pred_GPT_simple_prompt.csv", sep = ",", index = False)
        print("\n \n csv saved \n \n")

end = time.time()
print(f"Time taken: {end - start} seconds")
time_GPT_simple_prompt = end - start
time_GPT_simple_prompt_df = pd.DataFrame({"time": [time_GPT_simple_prompt]})
time_GPT_simple_prompt_df.to_csv("../exp/times_LLMs/GPT/time_GPT_simple_prompt.csv", sep = ",", index = False)

# value counts for array
counts_simple_GPT = pd.Series(simple_y_pred_GPT).value_counts()
print(counts_simple_GPT)

# convert YES to 1 and NO to 0
simple_y_pred_GPT = [1 if response == "YES" else 0 if response == "NO" else np.nan for response in simple_y_pred_GPT]

# save the array to a csv file
simple_prompt_df_GPT = pd.DataFrame(simple_y_pred_GPT, columns = ["y_pred"])
simple_prompt_df_GPT.to_csv("../exp/y_pred_LLMs/GPT/y_pred_GPT_simple_prompt.csv", sep = ",", index = False)



#### Class definition prompt ####

class_def_y_pred_GPT = []

client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY"),
)

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in X_test_class_definitions_prompt:
    response = client.responses.create(
        model = "o3-2025-04-16",
        instructions = class_definitions_instruction,
        input = prompt
    )

    if response.output_text.strip() not in ("YES", "NO"):
        print("\n Invalid output. Retry prompting. \n")
        response = client.responses.create(
            model = "o3-2025-04-16",
            instructions = retry_instruction,
            input = prompt
        )

    class_def_y_pred_GPT.append(response.output_text)
    print(response.output_text)

end = time.time()
print(f"Time taken: {end - start} seconds")
time_GPT_class_definitions = end - start
time_GPT_class_definitions_df = pd.DataFrame({"time": [time_GPT_class_definitions]})
time_GPT_class_definitions_df.to_csv("../exp/times_LLMs/GPT/time_GPT_class_definitions_prompt.csv", sep = ",", index = False)

# value counts for array
counts_class_def_GPT = pd.Series(class_def_y_pred_GPT).value_counts()
print(counts_class_def_GPT)

# convert YES to 1 and NO to 0
class_def_y_pred_GPT = [1 if response == "YES" else 0 for response in class_def_y_pred_GPT]

# save the array to a csv file
class_def_df_GPT = pd.DataFrame(class_def_y_pred_GPT, columns = ["y_pred"])
class_def_df_GPT.to_csv("../exp/y_pred_LLMs/GPT/y_pred_GPT_class_definitions_prompt.csv", sep = ",", index = False)



#### Profiled simple prompt ####

profiled_simple_y_pred_GPT = []

client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY"),
)

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in X_test_profiled_simple_prompt:
    response = client.responses.create(
        model = "o3-2025-04-16",
        instructions = profiled_simple_instruction,
        input = prompt
    )

    if response.output_text.strip() not in ("YES", "NO"):
        print("\n Invalid output. Retry prompting. \n")
        response = client.responses.create(
            model = "o3-2025-04-16",
            instructions = retry_instruction,
            input = prompt
        )

    profiled_simple_y_pred_GPT.append(response.output_text)
    print(response.output_text)

end = time.time()
print(f"Time taken: {end - start} seconds")
time_GPT_profiled_simple = end - start
time_GPT_profiled_simple_df = pd.DataFrame({"time": [time_GPT_profiled_simple]})
time_GPT_profiled_simple_df.to_csv("../exp/times_LLMs/GPT/time_GPT_profiled_simple_prompt.csv", sep = ",", index = False)

# value counts for array
counts_profiled_simple_GPT = pd.Series(profiled_simple_y_pred_GPT).value_counts()
print(counts_profiled_simple_GPT)

# convert YES to 1 and NO to 0
profiled_simple_y_pred_GPT_val = [1 if response == "YES" else 0 for response in profiled_simple_y_pred_GPT]

# save the array to a csv file
profiled_simple_df_GPT = pd.DataFrame(profiled_simple_y_pred_GPT_val, columns = ["y_pred"])
profiled_simple_df_GPT.to_csv("../exp/y_pred_LLMs/GPT/y_pred_GPT_profiled_simple_prompt.csv", sep = ",", index = False)



#### Few shot prompt ####

few_shot_y_pred_GPT = []

client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY"),
)

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in X_test_few_shot_prompt:
    response = client.responses.create(
        model = "o3-2025-04-16",
        instructions = few_shot_instruction,
        input = prompt
    )

    if response.output_text.strip() not in ("YES", "NO"):
        print("\n Invalid output. Retry prompting. \n")
        response = client.responses.create(
            model = "o3-2025-04-16",
            instructions = retry_instruction,
            input = prompt
        )

    few_shot_y_pred_GPT.append(response.output_text)
    print(response.output_text)

end = time.time()
print(f"Time taken: {end - start} seconds")
time_GPT_few_shot = end - start
time_GPT_few_shot_df = pd.DataFrame({"time": [time_GPT_few_shot]})
time_GPT_few_shot_df.to_csv("../exp/times_LLMs/GPT/time_GPT_few_shot_prompt.csv", sep = ",", index = False)

# value counts for array
counts_few_shot_GPT = pd.Series(few_shot_y_pred_GPT).value_counts()
print(counts_few_shot_GPT)

# convert YES to 1 and NO to 0
few_shot_y_pred_GPT_val = [1 if response == "YES" else 0 for response in few_shot_y_pred_GPT]

# save the array to a csv file
few_shot_df_GPT = pd.DataFrame(few_shot_y_pred_GPT_val, columns = ["y_pred"])
few_shot_df_GPT.to_csv("../exp/y_pred_LLMs/GPT/y_pred_GPT_few_shot_prompt.csv", sep = ",", index = False)



#### Vignette prompt ####

vignette_y_pred_GPT = []

client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY"),
)

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in X_test_vignette_prompt:
    response = client.responses.create(
        model = "o3-2025-04-16",
        instructions = vignette_instruction,
        input = prompt
    )

    if response.output_text.strip() not in ("YES", "NO"):
        print("\n Invalid output. Retry prompting. \n")
        response = client.responses.create(
            model = "o3-2025-04-16",
            instructions = retry_instruction,
            input = prompt
        )

    vignette_y_pred_GPT.append(response.output_text)
    print(response.output_text)

end = time.time()
print(f"Time taken: {end - start} seconds")
time_GPT_vignette = end - start
time_GPT_vignette_df = pd.DataFrame({"time": [time_GPT_vignette]})
time_GPT_vignette_df.to_csv("../exp/times_LLMs/GPT/time_GPT_vignette_prompt.csv", sep = ",", index = False)

# value counts for array
counts_vignette_GPT = pd.Series(vignette_y_pred_GPT).value_counts()
print(counts_vignette_GPT)

# convert YES to 1 and NO to 0
vignette_y_pred_GPT_val = [1 if response == "YES" else 0 for response in vignette_y_pred_GPT]

# save the array to a csv file
vignette_df_GPT = pd.DataFrame(vignette_y_pred_GPT_val, columns = ["y_pred"])
vignette_df_GPT.to_csv("../exp/y_pred_LLMs/GPT/y_pred_GPT_vignette_prompt.csv", sep = ",", index = False)


#### Chain-of-Thought prompt ####

thinking_y_pred_GPT = []
thinking_explanation_GPT = []

client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY"),
)

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in X_test_thinking_prompt:
    response = client.responses.create(
        model = "o3-2025-04-16",
        instructions = thinking_instruction,
        input = prompt
    )

    try:
        prediction = re.findall(r'Prediction: (.*)', response.output_text)[0].strip()
        explanation = re.findall(r'Explanation: (.*)', response.output_text)[0].strip()
        thinking_y_pred_GPT.append(prediction)
        thinking_explanation_GPT.append(explanation)
        print(prediction)
    except IndexError:
        print("IndexError")
        thinking_y_pred_GPT.append("IndexError")
        thinking_explanation_GPT.append("IndexError")

end = time.time()
print(f"Time taken: {end - start} seconds")
time_GPT_thinking = end - start
time_GPT_thinking_df = pd.DataFrame({"time": [time_GPT_thinking]})
time_GPT_thinking_df.to_csv("../exp/times_LLMs/GPT/time_GPT_thinking_prompt.csv", sep = ",", index = False)

# value counts for array
counts_thinking_GPT = pd.Series(thinking_y_pred_GPT).value_counts()
print(counts_thinking_GPT)

# convert YES to 1 and NO to 0
thinking_y_pred_GPT_val = [1 if response == "YES" else 0 for response in thinking_y_pred_GPT]

# save the array to a csv file
thinking_df_GPT = pd.DataFrame(thinking_y_pred_GPT_val, columns = ["y_pred"])
thinking_df_GPT.to_csv("../exp/y_pred_LLMs/GPT/y_pred_GPT_thinking_prompt.csv", sep = ",", index = False)

thinking_df_explanation_GPT = pd.DataFrame(thinking_explanation_GPT, columns = ["thinking"])
thinking_df_explanation_GPT.to_csv("../exp/y_pred_LLMs/GPT/explanation_GPT_thinking_prompt.csv", sep = ",", index = False)

