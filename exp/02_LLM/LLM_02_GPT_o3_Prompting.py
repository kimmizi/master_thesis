#### LLM: Zero-shot classification through LLMs and prompts ####
#### CHATGPT ####


#### 0 Imports ####
import os
import pandas as pd
import numpy as np
import time
import re
from tqdm import tqdm
from openai import OpenAI

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
cot_instruction_df = pd.read_csv("../../dat/instructions/cot_instruction.csv", sep =",", index_col = 0)
retry_instruction_df = pd.read_csv("../../dat/instructions/retry_instruction.csv", sep =",", index_col = 0)
retry_cot_instruction_df = pd.read_csv("../../dat/instructions/retry_cot_instruction.csv", sep =",", index_col = 0)

# convert to string
cot_instruction = cot_instruction_df["0"].iloc[0]
retry_instruction = retry_instruction_df["0"].iloc[0]
retry_cot_instruction = retry_cot_instruction_df["0"].iloc[0]



#### Helper functions ####

def GPT_create_response(prompt, instruction):
    response = client.responses.create(
        model = model_gpt,
        instructions = instruction,
        input = prompt,
        reasoning = {
            "effort": "medium",
            "summary": "auto"
        },
    )

    try:
        prediction = re.findall(r'Prediction: (.*)', response.output_text)[0].strip()
        explanation = re.findall(r'Explanation: (.*)', response.output_text)[0].strip()
    except IndexError:
        print("\n IndexError. Retry prompting. \n")
        response = client.responses.create(
            model = model_gpt,
            instructions = retry_cot_instruction,
            input = prompt,
            reasoning = {
                "effort": "medium",
                "summary": "auto"
            },
        )

        try:
            prediction = re.findall(r'Prediction: (.*)', response.output_text)[0].strip()
            explanation = re.findall(r'Explanation: (.*)', response.output_text)[0].strip()

        except IndexError:
            print("\n Still IndexError. Don't retry prompting. \n")
            prediction = "IndexError"
            explanation = "IndexError"

    summary_texts = [
        " ".join(
            summary.text
            for item in response.output if hasattr(item, "summary")
            for summary in item.summary if hasattr(summary, "text")
        )
    ]
    thinking = summary_texts[0]

    return prediction, explanation, thinking

def save_prompt_to_csv(response_array, explanation_array, thinking_array, filename):
    # value counts for array
    counts = pd.Series(response_array).value_counts()
    print(counts)

    # convert YES to 1 and NO to 0
    response_array = [re.sub(r'^\[|\]$', '', response.strip()) for response in response_array]
    response_array_val = [1 if response == "YES" else 0 if response == "NO" else np.nan for response in response_array]

    # save the array to a csv file
    df = pd.DataFrame({
        "y_pred": response_array_val,
        "explanation": explanation_array,
        "thinking": thinking_array
    })
    df.to_csv(f"y_pred_LLMs/GPT/y_pred_GPT_o3_{filename}.csv", sep = ",", index = False)

def calc_time(start, end, filename):
    """
    Calculate the time taken for the prompting and save it to a CSV file.
    """
    time_taken = end - start
    print(f"Time taken: {time_taken} seconds")
    time_df = pd.DataFrame({"time": [time_taken]})
    # time_df.to_csv(f"times_LLMs/GPT/time_GPT_o3_{filename}.csv", sep = ",", index = False)
    return time_taken



#### 1 Prompting with ChatGPT ####

model_gpt = "o3-2025-04-16"

client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY"),
)

#### Simple prompt ####

y_pred_simple_GPT = []
explanation_simple_GPT = []
thinking_simple_GPT = []

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_simple_prompt, desc = "Simple prompting"):
    response, explanation, thinking = GPT_create_response(prompt, cot_instruction)
    y_pred_simple_GPT.append(response)
    explanation_simple_GPT.append(explanation)
    thinking_simple_GPT.append(thinking)
    # print(response)

    if len(y_pred_simple_GPT) % 50 == 0 and len(y_pred_simple_GPT) > 0:
        print(f"\n\nProcessed {len(y_pred_simple_GPT)} prompts.\n")
        save_prompt_to_csv(y_pred_simple_GPT, explanation_simple_GPT, thinking_simple_GPT, "simple_prompt")

end = time.time()
calc_time(start, end, "simple_prompt")

# save the array to a csv file
save_prompt_to_csv(y_pred_simple_GPT, explanation_simple_GPT, thinking_simple_GPT, "simple_prompt")



### Class definition prompt ####

y_pred_class_def_GPT = []
explanation_class_def_GPT = []
thinking_class_def_GPT = []

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_class_definitions_prompt, desc = "Class definition prompting"):
    response, explanation, thinking = GPT_create_response(prompt, cot_instruction)
    y_pred_class_def_GPT.append(response)
    explanation_class_def_GPT.append(explanation)
    thinking_class_def_GPT.append(thinking)
    # print(response)

    if len(y_pred_class_def_GPT) % 50 == 0 and len(y_pred_class_def_GPT) > 0:
        print(f"\n\nProcessed {len(y_pred_class_def_GPT)} prompts.\n")
        save_prompt_to_csv(y_pred_class_def_GPT, explanation_class_def_GPT, thinking_class_def_GPT, "class_definitions_prompt")

end = time.time()
calc_time(start, end, "class_definitions_prompt")

# save the array to a csv file
save_prompt_to_csv(y_pred_class_def_GPT, explanation_class_def_GPT, thinking_class_def_GPT, "class_definitions_prompt")



#### Profiled simple prompt ####

y_pred_profiled_simple_GPT = []
explanation_profiled_simple_GPT = []
thinking_profiled_simple_GPT = []

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_profiled_simple_prompt, desc = "Profiled simple prompting"):
    response, explanation, thinking = GPT_create_response(prompt, cot_instruction)
    y_pred_profiled_simple_GPT.append(response)
    explanation_profiled_simple_GPT.append(explanation)
    thinking_profiled_simple_GPT.append(thinking)
    # print(response)

    if len(y_pred_profiled_simple_GPT) % 50 == 0 and len(y_pred_profiled_simple_GPT) > 0:
        print(f"\n\nProcessed {len(y_pred_profiled_simple_GPT)} prompts.\n")
        save_prompt_to_csv(y_pred_profiled_simple_GPT, explanation_profiled_simple_GPT, thinking_profiled_simple_GPT, "profiled_simple_prompt")

end = time.time()
calc_time(start, end, "profiled_simple_prompt")

# save the array to a csv file
save_prompt_to_csv(y_pred_profiled_simple_GPT, explanation_profiled_simple_GPT, thinking_profiled_simple_GPT, "profiled_simple_prompt")



#### Few shot prompt ####

y_pred_few_shot_GPT = []
explanation_few_shot_GPT = []
thinking_few_shot_GPT = []

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_few_shot_prompt, desc = "Few-shot prompting"):
    response, explanation, thinking = GPT_create_response(prompt, cot_instruction)
    y_pred_few_shot_GPT.append(response)
    explanation_few_shot_GPT.append(explanation)
    thinking_few_shot_GPT.append(thinking)
    # print(response)

    if len(y_pred_few_shot_GPT) % 50 == 0 and len(y_pred_few_shot_GPT) > 0:
        print(f"\n\nProcessed {len(y_pred_few_shot_GPT)} prompts.\n")
        save_prompt_to_csv(y_pred_few_shot_GPT, explanation_few_shot_GPT, thinking_few_shot_GPT, "few_shot_prompt")

end = time.time()
calc_time(start, end, "few_shot_prompt")

# save the array to a csv file
save_prompt_to_csv(y_pred_few_shot_GPT, explanation_few_shot_GPT, thinking_few_shot_GPT, "few_shot_prompt")



#### Vignette prompt ####

y_pred_vignette_GPT = []
explanation_vignette_GPT = []
thinking_vignette_GPT = []

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_vignette_prompt, desc = "Vignette prompting"):
    response, explanation, thinking = GPT_create_response(prompt, cot_instruction)
    y_pred_vignette_GPT.append(response)
    explanation_vignette_GPT.append(explanation)
    thinking_vignette_GPT.append(thinking)
    # print(response)

    if len(y_pred_vignette_GPT) % 50 == 0 and len(y_pred_vignette_GPT) > 0:
        print(f"\n\nProcessed {len(y_pred_vignette_GPT)} prompts.\n")
        save_prompt_to_csv(y_pred_vignette_GPT, explanation_vignette_GPT, thinking_vignette_GPT, "vignette_prompt")

end = time.time()
calc_time(start, end, "vignette_prompt")

save_prompt_to_csv(y_pred_vignette_GPT, explanation_vignette_GPT, thinking_vignette_GPT, "vignette_prompt")



#### Chain-of-thought prompt ####

y_pred_cot_GPT = []
explanation_cot_GPT = []
thinking_cot_GPT = []

# measure time in seconds
start = time.time()

# iterate over the test set and save the response for each prompt in an array
for prompt in tqdm(X_test_cot_prompt, desc = "Chain-of-thought prompting"):
    response, explanation, thinking = GPT_create_response(prompt, cot_instruction)
    y_pred_cot_GPT.append(response)
    explanation_cot_GPT.append(explanation)
    thinking_cot_GPT.append(thinking)
    # print(response)

    if len(y_pred_cot_GPT) % 50 == 0 and len(y_pred_cot_GPT) > 0:
        print(f"\n\nProcessed {len(y_pred_cot_GPT)} prompts.\n")
        save_prompt_to_csv(y_pred_cot_GPT, explanation_cot_GPT, thinking_cot_GPT, "cot_prompt")

end = time.time()
calc_time(start, end, "cot_prompt")

# save the array to a csv file
save_prompt_to_csv(y_pred_cot_GPT, explanation_cot_GPT, thinking_cot_GPT, "cot_prompt")
