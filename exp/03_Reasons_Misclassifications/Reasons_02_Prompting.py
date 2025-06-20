#### LLM: Zero-shot classification through LLMs and prompts ####
#### CHATGPT ####


#### 0 Imports ####

import os
import pandas as pd
import numpy as np
import anthropic
import time
import re
from tqdm import tqdm
from openai import OpenAI
from sklearn.model_selection import train_test_split
from google import genai
from google.genai import types


data_change = pd.read_csv("../../dat/dips/DIPS_Data_cleaned_change.csv", sep = ",", low_memory = False)

# import prompts for all test data
X_test_simple_prompt_df = pd.read_csv("../../dat/prompts/X_test_simple_prompt.csv", sep = ",", index_col = 0)
X_test_class_definitions_prompt_df = pd.read_csv("../../dat/prompts/X_test_class_definitions_prompt.csv", sep = ",", index_col = 0)
X_test_profiled_simple_prompt_df = pd.read_csv("../../dat/prompts/X_test_profiled_simple_prompt.csv", sep = ",", index_col = 0)
X_test_few_shot_prompt_df = pd.read_csv("../../dat/prompts/X_test_few_shot_prompt.csv", sep = ",", index_col = 0)
X_test_vignette_prompt_df = pd.read_csv("../../dat/prompts/X_test_vignette_prompt.csv", sep = ",", index_col = 0)
X_test_cot_prompt_df = pd.read_csv("../../dat/prompts/X_test_cot_prompt.csv", sep = ",", index_col = 0)

# convert to arrays
X_test_simple_prompt = X_test_simple_prompt_df.values.flatten()
X_test_class_definitions_prompt = X_test_class_definitions_prompt_df.values.flatten()
X_test_profiled_simple_prompt = X_test_profiled_simple_prompt_df.values.flatten()
X_test_few_shot_prompt = X_test_few_shot_prompt_df.values.flatten()
X_test_vignette_prompt = X_test_vignette_prompt_df.values.flatten()
X_test_cot_prompt = X_test_cot_prompt_df.values.flatten()

# import instruction for reason of misclassification
instruction_reason_df = pd.read_csv("../../dat/instructions/instruction_reason.csv", sep=",", index_col = 0)

# convert to string
instruction_reason = instruction_reason_df["0"].iloc[0]

# import all y_pred
y_pred_GPT_4_simple_df = pd.read_csv("../02_LLM/y_pred_LLMs/GPT/y_pred_GPT_simple_prompt.csv", sep = ",")
y_pred_GPT_o3_simple_df = pd.read_csv("../02_LLM/y_pred_LLMs/GPT/y_pred_GPT_o3_simple_prompt.csv", sep = ",")
y_pred_Gemini_simple_df = pd.read_csv("../02_LLM/y_pred_LLMs/Gemini/y_pred_Gemini_simple_prompt.csv", sep = ",")
y_pred_Gemma_simple_df = pd.read_csv("../02_LLM/y_pred_LLMs/Gemma/y_pred_Gemma_simple_prompt.csv", sep = ",")
y_pred_Claude_simple_df = pd.read_csv("../02_LLM/y_pred_LLMs/Claude/y_pred_Claude_4_simple_prompt.csv", sep = ",")
y_pred_DeepSeek_simple_df = pd.read_csv("../02_LLM/y_pred_LLMs/DeepSeek/y_pred_deeps_simple_prompt.csv", sep = ",")
y_pred_Grok_simple_df = pd.read_csv("../02_LLM/y_pred_LLMs/Grok/y_pred_Grok_simple_prompt.csv", sep = ",")

y_pred_GPT_4_class_definitions_df = pd.read_csv("../02_LLM/y_pred_LLMs/GPT/y_pred_GPT_class_definitions_prompt.csv", sep = ",")
y_pred_GPT_o3_class_definitions_df = pd.read_csv("../02_LLM/y_pred_LLMs/GPT/y_pred_GPT_o3_class_definitions_prompt.csv", sep = ",")
y_pred_Gemini_class_definitions_df = pd.read_csv("../02_LLM/y_pred_LLMs/Gemini/y_pred_Gemini_class_definitions_prompt.csv", sep = ",")
y_pred_Gemma_class_definitions_df = pd.read_csv("../02_LLM/y_pred_LLMs/Gemma/y_pred_Gemma_class_definitions_prompt.csv", sep = ",")
y_pred_Claude_class_definitions_df = pd.read_csv("../02_LLM/y_pred_LLMs/Claude/y_pred_Claude_4_class_definitions_prompt.csv", sep = ",")
y_pred_DeepSeek_class_definitions_df = pd.read_csv("../02_LLM/y_pred_LLMs/DeepSeek/y_pred_deeps_class_definitions_prompt.csv", sep = ",")
y_pred_Grok_class_definitions_df = pd.read_csv("../02_LLM/y_pred_LLMs/Grok/y_pred_grok_class_definitions_prompt.csv", sep = ",")

y_pred_GPT_4_profiled_simple_df = pd.read_csv("../02_LLM/y_pred_LLMs/GPT/y_pred_GPT_profiled_simple_prompt.csv", sep = ",")
y_pred_GPT_o3_profiled_simple_df = pd.read_csv("../02_LLM/y_pred_LLMs/GPT/y_pred_GPT_o3_profiled_simple_prompt.csv", sep = ",")
y_pred_Gemini_profiled_simple_df = pd.read_csv("../02_LLM/y_pred_LLMs/Gemini/y_pred_Gemini_profiled_simple_prompt.csv", sep = ",")
y_pred_Gemma_profiled_simple_df = pd.read_csv("../02_LLM/y_pred_LLMs/Gemma/y_pred_Gemma_profiled_simple_prompt.csv", sep = ",")
y_pred_Claude_profiled_simple_df = pd.read_csv("../02_LLM/y_pred_LLMs/Claude/y_pred_Claude_profiled_simple_prompt.csv", sep = ",")
y_pred_DeepSeek_profiled_simple_df = pd.read_csv("../02_LLM/y_pred_LLMs/DeepSeek/y_pred_deeps_profiled_simple_prompt.csv", sep = ",")
y_pred_Grok_profiled_simple_df = pd.read_csv("../02_LLM/y_pred_LLMs/Grok/y_pred_Grok_profiled_simple_prompt.csv", sep = ",")

y_pred_GPT_4_few_shot_df = pd.read_csv("../02_LLM/y_pred_LLMs/GPT/y_pred_GPT_few_shot_prompt.csv", sep = ",")
y_pred_GPT_o3_few_shot_df = pd.read_csv("../02_LLM/y_pred_LLMs/GPT/y_pred_GPT_o3_few_shot_prompt.csv", sep = ",")
y_pred_Gemini_few_shot_df = pd.read_csv("../02_LLM/y_pred_LLMs/Gemini/y_pred_Gemini_few_shot_prompt.csv", sep = ",")
y_pred_Gemma_few_shot_df = pd.read_csv("../02_LLM/y_pred_LLMs/Gemma/y_pred_Gemma_few_shot_prompt.csv", sep = ",")
y_pred_Claude_few_shot_df = pd.read_csv("../02_LLM/y_pred_LLMs/Claude/y_pred_Claude_few_shot_prompt.csv", sep = ",")
y_pred_DeepSeek_few_shot_df = pd.read_csv("../02_LLM/y_pred_LLMs/DeepSeek/y_pred_deeps_few_shot_prompt.csv", sep = ",")
y_pred_Grok_few_shot_df = pd.read_csv("../02_LLM/y_pred_LLMs/Grok/y_pred_Grok_few_shot_prompt.csv", sep = ",")

y_pred_GPT_4_vignette_df = pd.read_csv("../02_LLM/y_pred_LLMs/GPT/y_pred_GPT_vignette_prompt.csv", sep = ",")
y_pred_GPT_o3_vignette_df = pd.read_csv("../02_LLM/y_pred_LLMs/GPT/y_pred_GPT_o3_vignette_prompt.csv", sep = ",")
y_pred_Gemini_vignette_df = pd.read_csv("../02_LLM/y_pred_LLMs/Gemini/y_pred_Gemini_vignette_prompt.csv", sep = ",")
y_pred_Gemma_vignette_df = pd.read_csv("../02_LLM/y_pred_LLMs/Gemma/y_pred_Gemma_vignette_prompt.csv", sep = ",")
y_pred_Claude_vignette_df = pd.read_csv("../02_LLM/y_pred_LLMs/Claude/y_pred_Claude_vignette_prompt.csv", sep = ",")
y_pred_DeepSeek_vignette_df = pd.read_csv("../02_LLM/y_pred_LLMs/DeepSeek/y_pred_deeps_vignette_prompt.csv", sep = ",")
y_pred_Grok_vignette_df = pd.read_csv("../02_LLM/y_pred_LLMs/Grok/y_pred_Grok_vignette_prompt.csv", sep = ",")

y_pred_GPT_4_cot_df = pd.read_csv("../02_LLM/y_pred_LLMs/GPT/y_pred_GPT_cot_prompt.csv", sep = ",")
y_pred_GPT_o3_cot_df = pd.read_csv("../02_LLM/y_pred_LLMs/GPT/y_pred_GPT_o3_cot_prompt.csv", sep = ",")
y_pred_Gemini_cot_df = pd.read_csv("../02_LLM/y_pred_LLMs/Gemini/y_pred_Gemini_cot_prompt.csv", sep = ",")
y_pred_Gemma_cot_df = pd.read_csv("../02_LLM/y_pred_LLMs/Gemma/y_pred_Gemma_cot_prompt.csv", sep = ",")
y_pred_Claude_cot_df = pd.read_csv("../02_LLM/y_pred_LLMs/Claude/y_pred_Claude_cot_prompt.csv", sep = ",")
y_pred_DeepSeek_cot_df = pd.read_csv("../02_LLM/y_pred_LLMs/DeepSeek/y_pred_deeps_cot_prompt.csv", sep = ",")
y_pred_Grok_cot_df = pd.read_csv("../02_LLM/y_pred_LLMs/Grok/y_pred_Grok_cot_prompt.csv", sep = ",")



# convert to arrays
y_pred_GPT_4_simple = y_pred_GPT_4_simple_df["y_pred"].values.flatten()
y_pred_GPT_o3_simple = y_pred_GPT_o3_simple_df["y_pred"].values.flatten()
y_pred_Gemini_simple = y_pred_Gemini_simple_df["y_pred"].values.flatten()
y_pred_Gemma_simple = y_pred_Gemma_simple_df["y_pred"].values.flatten()
y_pred_Claude_simple = y_pred_Claude_simple_df["y_pred"].values.flatten()
y_pred_DeepSeek_simple = y_pred_DeepSeek_simple_df["y_pred"].values.flatten()
y_pred_Grok_simple = y_pred_Grok_simple_df["y_pred"].values.flatten()

y_pred_GPT_4_class_definitions = y_pred_GPT_4_class_definitions_df["y_pred"].values.flatten()
y_pred_GPT_o3_class_definitions = y_pred_GPT_o3_class_definitions_df["y_pred"].values.flatten()
y_pred_Gemini_class_definitions = y_pred_Gemini_class_definitions_df["y_pred"].values.flatten()
y_pred_Gemma_class_definitions = y_pred_Gemma_class_definitions_df["y_pred"].values.flatten()
y_pred_Claude_class_definitions = y_pred_Claude_class_definitions_df["y_pred"].values.flatten()
y_pred_DeepSeek_class_definitions = y_pred_DeepSeek_class_definitions_df["y_pred"].values.flatten()
y_pred_Grok_class_definitions = y_pred_Grok_class_definitions_df["y_pred"].values.flatten()

y_pred_GPT_4_profiled_simple = y_pred_GPT_4_profiled_simple_df["y_pred"].values.flatten()
y_pred_GPT_o3_profiled_simple = y_pred_GPT_o3_profiled_simple_df["y_pred"].values.flatten()
y_pred_Gemini_profiled_simple = y_pred_Gemini_profiled_simple_df["y_pred"].values.flatten()
y_pred_Gemma_profiled_simple = y_pred_Gemma_profiled_simple_df["y_pred"].values.flatten()
y_pred_Claude_profiled_simple = y_pred_Claude_profiled_simple_df["y_pred"].values.flatten()
y_pred_DeepSeek_profiled_simple = y_pred_DeepSeek_profiled_simple_df["y_pred"].values.flatten()
y_pred_Grok_profiled_simple = y_pred_Grok_profiled_simple_df["y_pred"].values.flatten()

y_pred_GPT_4_few_shot = y_pred_GPT_4_few_shot_df["y_pred"].values.flatten()
y_pred_GPT_o3_few_shot = y_pred_GPT_o3_few_shot_df["y_pred"].values.flatten()
y_pred_Gemini_few_shot = y_pred_Gemini_few_shot_df["y_pred"].values.flatten()
y_pred_Gemma_few_shot = y_pred_Gemma_few_shot_df["y_pred"].values.flatten()
y_pred_Claude_few_shot = y_pred_Claude_few_shot_df["y_pred"].values.flatten()
y_pred_DeepSeek_few_shot = y_pred_DeepSeek_few_shot_df["y_pred"].values.flatten()
y_pred_Grok_few_shot = y_pred_Grok_few_shot_df["y_pred"].values.flatten()

y_pred_GPT_4_vignette = y_pred_GPT_4_vignette_df["y_pred"].values.flatten()
y_pred_GPT_o3_vignette = y_pred_GPT_o3_vignette_df["y_pred"].values.flatten()
y_pred_Gemini_vignette = y_pred_Gemini_vignette_df["y_pred"].values.flatten()
y_pred_Gemma_vignette = y_pred_Gemma_vignette_df["y_pred"].values.flatten()
y_pred_Claude_vignette = y_pred_Claude_vignette_df["y_pred"].values.flatten()
y_pred_DeepSeek_vignette = y_pred_DeepSeek_vignette_df["y_pred"].values.flatten()
y_pred_Grok_vignette = y_pred_Grok_vignette_df["y_pred"].values.flatten()

y_pred_GPT_4_cot = y_pred_GPT_4_cot_df["y_pred"].values.flatten()
y_pred_GPT_o3_cot = y_pred_GPT_o3_cot_df["y_pred"].values.flatten()
y_pred_Gemini_cot = y_pred_Gemini_cot_df["y_pred"].values.flatten()
y_pred_Gemma_cot = y_pred_Gemma_cot_df["y_pred"].values.flatten()
y_pred_Claude_cot = y_pred_Claude_cot_df["y_pred"].values.flatten()
y_pred_DeepSeek_cot = y_pred_DeepSeek_cot_df["y_pred"].values.flatten()
y_pred_Grok_cot = y_pred_Grok_cot_df["y_pred"].values.flatten()

thinking_simple_GPT_o3 = y_pred_GPT_o3_simple_df["thinking"].values.flatten()
thinking_simple_Gemini = y_pred_Gemini_simple_df["thinking"].values.flatten()
thinking_simple_Claude = y_pred_Claude_simple_df["thinking"].values.flatten()
thinking_simple_DeepSeek = y_pred_DeepSeek_simple_df["thinking"].values.flatten()

thinking_class_definitions_GPT_o3 = y_pred_GPT_o3_class_definitions_df["thinking"].values.flatten()
thinking_class_definitions_Gemini = y_pred_Gemini_class_definitions_df["thinking"].values.flatten()
thinking_class_definitions_Claude = y_pred_Claude_class_definitions_df["thinking"].values.flatten()
thinking_class_definitions_DeepSeek = y_pred_DeepSeek_class_definitions_df["thinking"].values.flatten()

thinking_profiled_simple_GPT_o3 = y_pred_GPT_o3_profiled_simple_df["thinking"].values.flatten()
thinking_profiled_simple_Gemini = y_pred_Gemini_profiled_simple_df["thinking"].values.flatten()
thinking_profiled_simple_Claude = y_pred_Claude_profiled_simple_df["thinking"].values.flatten()
thinking_profiled_simple_DeepSeek = y_pred_DeepSeek_profiled_simple_df["thinking"].values.flatten()

thinking_few_shot_GPT_o3 = y_pred_GPT_o3_few_shot_df["thinking"].values.flatten()
thinking_few_shot_Gemini = y_pred_Gemini_few_shot_df["thinking"].values.flatten()
thinking_few_shot_Claude = y_pred_Claude_few_shot_df["thinking"].values.flatten()
thinking_few_shot_DeepSeek = y_pred_DeepSeek_few_shot_df["thinking"].values.flatten()

thinking_vignette_GPT_o3 = y_pred_GPT_o3_vignette_df["thinking"].values.flatten()
thinking_vignette_Gemini = y_pred_Gemini_vignette_df["thinking"].values.flatten()
thinking_vignette_Claude = y_pred_Claude_vignette_df["thinking"].values.flatten()
thinking_vignette_DeepSeek = y_pred_DeepSeek_vignette_df["thinking"].values.flatten()

thinking_cot_GPT_o3 = y_pred_GPT_o3_cot_df["thinking"].values.flatten()
thinking_cot_Gemini = y_pred_Gemini_cot_df["thinking"].values.flatten()
thinking_cot_Claude = y_pred_Claude_cot_df["thinking"].values.flatten()
thinking_cot_DeepSeek = y_pred_DeepSeek_cot_df["thinking"].values.flatten()

explanation_cot_GPT_4 = y_pred_GPT_4_cot_df["explanation"].values.flatten()
explanation_cot_GPT_o3 = y_pred_GPT_o3_cot_df["explanation"].values.flatten()
explanation_cot_Gemini = y_pred_Gemini_cot_df["explanation"].values.flatten()
explanation_cot_Gemma = y_pred_Gemma_cot_df["explanation"].values.flatten()
explanation_cot_Claude = y_pred_Claude_cot_df["cot"].values.flatten()
explanation_cot_DeepSeek = y_pred_DeepSeek_cot_df["explanation"].values.flatten()
explanation_cot_Grok = y_pred_Grok_cot_df["explanation"].values.flatten()



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


model_gpt_4 = "gpt-4.1"
model_gpt_o3 = "o3-2025-04-16"
model_gemini = "gemini-2.5-pro-preview-05-06"
model_gemma = "gemma-3-27b-it"
model_claude = "claude-sonnet-4-20250514"
model_deeps = "deepseek-reasoner"
model_grok = "grok-3-beta"



#### Helper functions ####

def GPT_4_create_response(prompt):
    client = OpenAI(
        api_key = os.environ.get("OPENAI_API_KEY"),
    )

    response = client.responses.create(
        model = model_gpt_4,
        instructions = instruction_reason,
        input = prompt
    )

    response = response.output_text

    return response


def GPT_o3_create_response(prompt):
    client = OpenAI(
        api_key = os.environ.get("OPENAI_API_KEY"),
    )

    response = client.responses.create(
        model = model_gpt_o3,
        instructions = instruction_reason,
        input = prompt,
        reasoning = {
            "effort": "medium",
            "summary": "auto"
        },
    )

    response = response.output_text

    return response


def Gemini_create_response(prompt):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY")
    )

    response = client.models.generate_content(
        model = model_gemini,
        config = types.GenerateContentConfig(
            system_instruction = instruction_reason,
            thinking_config = types.ThinkingConfig(
                include_thoughts = True
            )
        ),
        contents = prompt,
    )

    response = response.text

    return response


def Gemma_create_response(prompt):
    client = genai.Client(
        api_key = os.environ.get("GEMINI_API_KEY")
    )

    time.sleep(20)  # sleep for few seconds to avoid rate limiting
    response = client.models.generate_content(
        model = model_gemma,
        contents = [instruction_reason, prompt]
    )

    response = response.text

    return response


def Claude_create_response(prompt):
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )

    message = client.messages.create(
        model = model_claude,
        max_tokens = 1000,
        system = instruction_reason,
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

    # response = message.content
    response = message.content[0].text

    return response


def DeepSeek_create_response(prompt):
    client = OpenAI(
        api_key = os.environ.get("DeepSeek_API_Key"),
        base_url = "https://api.deepseek.com"
    )

    response = client.chat.completions.create(
        model = model_deeps,
        messages = [
            {"role": "system", "content": instruction_reason},
            {"role": "user", "content": prompt},
        ],
        stream = False
    )

    response = response.choices[0].message.content

    return response


def Grok_create_response(prompt):
    client = OpenAI(
        api_key = os.environ.get("XAI_API_KEY"),
        base_url = "https://api.x.ai/v1",
    )

    completion = client.chat.completions.create(
        model = model_grok,
        messages = [
            {"role": "system", "content": instruction_reason},
            {"role": "user", "content": prompt},
        ],
    )

    response = completion.choices[0].message.content

    return response


def extract_reasons(model, X_test, y_pred_model, y_test, thinking, filename):
    """
    Extract reasons for misclassifications from the model's predictions.
    """
    reasons = []
    cleaned_reasons = []
    reasons_dict = {}
    main_reasons_dict = {}

    for i in tqdm(np.where(y_pred_model != y_test)[0], desc = f"Extracting reasons for {model}"):

        if model == "GPT_4" or model == "Gemma" or model == "Grok":
            input = f"Misclassified case {i}: Prompt: {X_test[i]} Response: {y_pred_model[i]} True label: {y_test.iloc[i]}"
        else:
            input = f"Misclassified case {i}: Prompt: {X_test[i]} Response: {y_pred_model[i]} Thinking: {thinking[i]} True label: {y_test.iloc[i]}"

        # input = f"Misclassified case {i}: Prompt: {X_test[i]} Response: {y_pred_model[i]} True label: {y_test.iloc[i]}"

        if model == "GPT_4":
            response = GPT_4_create_response(input)
        elif model == "GPT_O3":
            response = GPT_o3_create_response(input)
        elif model == "Gemini":
            response = Gemini_create_response(input)
        elif model == "Gemma":
            response = Gemma_create_response(input)
        elif model == "Claude":
            response = Claude_create_response(input)
        elif model == "DeepSeek":
            response = DeepSeek_create_response(input)
        elif model == "Grok":
            response = Grok_create_response(input)
        else:
            raise ValueError(f"Model {model} not recognized.")

        reasons.append(response)

    total_cases = len(y_pred_model)
    misclassified_cases = len(reasons)
    correct_cases = total_cases - misclassified_cases

    cases_df = pd.DataFrame({"total": [total_cases], "correct": [correct_cases], "missclassified": [misclassified_cases]})
    cases_df.to_csv(f"../03_Reasons_Misclassifications/reasons/{model}/cases_{model}_{filename}.csv", sep = ",", index = True)

    for reason in reasons:
        reason = reason.split(", ")
        reason = [re.sub(r'[^A-Za-z\s]', '', r).strip() for r in reason]
        cleaned_reasons.append(reason)

    for i in cleaned_reasons:
        for j in i:
            # count the occurrences of each reason
            if j in reasons_dict:
                reasons_dict[j] += 1
            else:
                reasons_dict[j] = 1

        # save main reason for each misclassification (first reason of each answer)
        if i:
            main_reason = i[0]
            if main_reason in main_reasons_dict:
                main_reasons_dict[main_reason] += 1
            else:
                main_reasons_dict[main_reason] = 1

    reasons_df = pd.DataFrame.from_dict(reasons_dict, orient = 'index', columns = ['count'])
    reasons_df.to_csv(f"../03_Reasons_Misclassifications/reasons/{model}/all_reasons_{model}_{filename}.csv", sep = ",", index = True)

    main_reasons_df = pd.DataFrame.from_dict(main_reasons_dict, orient = 'index', columns = ['count'])
    main_reasons_df.to_csv(f"../03_Reasons_Misclassifications/reasons/{model}/main_reasons_{model}_{filename}.csv", sep = ",", index = True)

    return cases_df, reasons_df, main_reasons_df



def extract_reasons_cot(model, X_test, y_pred_model, y_test, explanation, thinking, filename):
    """
    Extract reasons for misclassifications from the model's predictions.
    """
    reasons = []
    cleaned_reasons = []
    reasons_dict = {}
    main_reasons_dict = {}

    for i in tqdm(np.where(y_pred_model != y_test)[0], desc = f"Extracting reasons for {model}"):

        if model == "GPT_4" or model == "Gemma" or model == "Grok":
            input = f"Misclassified case {i}: Prompt: {X_test[i]} Response: {y_pred_model[i]} Explanation: {explanation[i]} True label: {y_test.iloc[i]}"
        else:
            input = f"Misclassified case {i}: Prompt: {X_test[i]} Response: {y_pred_model[i]} Explanation: {explanation[i]} Thinking: {thinking[i]} True label: {y_test.iloc[i]}"

        if model == "GPT_4":
            response = GPT_4_create_response(input)
        elif model == "GPT_O3":
            response = GPT_o3_create_response(input)
        elif model == "Gemini":
            response = Gemini_create_response(input)
        elif model == "Gemma":
            response = Gemma_create_response(input)
        elif model == "Claude":
            response = Claude_create_response(input)
        elif model == "DeepSeek":
            response = DeepSeek_create_response(input)
        elif model == "Grok":
            response = Grok_create_response(input)
        else:
            raise ValueError(f"Model {model} not recognized.")

        reasons.append(response)

    total_cases = len(y_pred_model)
    misclassified_cases = len(reasons)
    correct_cases = total_cases - misclassified_cases

    cases_df = pd.DataFrame({"total": [total_cases], "correct": [correct_cases], "missclassified": [misclassified_cases]})
    cases_df.to_csv(f"../03_Reasons_Misclassifications/reasons/{model}/cases_{model}_{filename}.csv", sep = ",", index = True)

    for reason in reasons:
        reason = reason.split(", ")
        reason = [re.sub(r'[^A-Za-z\s]', '', r).strip() for r in reason]
        cleaned_reasons.append(reason)

    for i in cleaned_reasons:
        for j in i:
            # count the occurrences of each reason
            if j in reasons_dict:
                reasons_dict[j] += 1
            else:
                reasons_dict[j] = 1

        # save main reason for each misclassification (first reason of each answer)
        if i:
            main_reason = i[0]
            if main_reason in main_reasons_dict:
                main_reasons_dict[main_reason] += 1
            else:
                main_reasons_dict[main_reason] = 1

    reasons_df = pd.DataFrame.from_dict(reasons_dict, orient = 'index', columns = ['count'])
    reasons_df.to_csv(f"../03_Reasons_Misclassifications/reasons/{model}/all_reasons_{model}_{filename}.csv", sep = ",", index = True)

    main_reasons_df = pd.DataFrame.from_dict(main_reasons_dict, orient = 'index', columns = ['count'])
    main_reasons_df.to_csv(f"../03_Reasons_Misclassifications/reasons/{model}/main_reasons_{model}_{filename}.csv", sep = ",", index = True)

    # print
    print(f"\n\n{model} cases: \n", cases_df, "\n\n{model} all reasons: \n", reasons_df, "\n\n{model} main reasons: \n", main_reasons_df, "\n\n")

    return cases_df, reasons_df, main_reasons_df


#### 1 Identify misclassified cases ####

# # indentify misclassified cases by comparing y_pred and y_test, save index
# misclassified_indices_GPT = np.where(y_pred_GPT_4_simple != y_test)[0]
# misclassified_indices_GPT_o3 = np.where(y_pred_GPT_o3_simple != y_test)[0]
# misclassified_indices_Gemini = np.where(y_pred_Gemini_simple != y_test)[0]
# misclassified_indices_Gemma = np.where(y_pred_Gemma_simple != y_test)[0]
# misclassified_indices_Claude = np.where(y_pred_Claude_simple != y_test)[0]
# misclassified_indices_DeepSeek = np.where(y_pred_DeepSeek_simple != y_test)[0]
# misclassified_indices_Grok = np.where(y_pred_Grok_simple != y_test)[0]





### 2 Get reasons for misclassifications ###

# # GPT_4_simple_cases_df, GPT_4_simple_all_reasons_df, GPT_4_simple_main_reasons_df = extract_reasons("GPT_4", X_test_simple_prompt, y_pred_GPT_4_simple, y_test, None, "simple")
# # GPT_o3_simple_cases_df, GPT_o3_simple_all_reasons_df, GPT_o3_simple_main_reasons_df = extract_reasons("GPT_O3", X_test_simple_prompt, y_pred_GPT_o3_simple, y_test, thinking_simple_GPT_o3, "simple")
# # Gemini_simple_cases_df, Gemini_simple_all_reasons_df, Gemini_simple_main_reasons_df = extract_reasons("Gemini", X_test_simple_prompt, y_pred_Gemini_simple, y_test, thinking_simple_Gemini, "simple")
# # Gemma_simple_cases_df, Gemma_simple_all_reasons_df, Gemma_simple_main_reasons_df = extract_reasons("Gemma", X_test_simple_prompt, y_pred_Gemma_simple, y_test, None, "simple")
# Claude_simple_cases_df, Claude_simple_all_reasons_df, Claude_simple_main_reasons_df = extract_reasons("Claude", X_test_simple_prompt, y_pred_Claude_simple, y_test, thinking_simple_Claude, "simple")
# DeepSeek_simple_cases_df, DeepSeek_simple_all_reasons_df, DeepSeek_simple_main_reasons_df = extract_reasons("DeepSeek", X_test_simple_prompt, y_pred_DeepSeek_simple, y_test, thinking_simple_DeepSeek, "simple")
# # Grok_simple_cases_df, Grok_simple_all_reasons_df, Grok_simple_main_reasons_df = extract_reasons("Grok", X_test_simple_prompt, y_pred_Grok_simple, y_test, None, "simple")
#
# GPT_4_class_definitions_cases_df, GPT_4_class_definitions_all_reasons_df, GPT_4_class_definitions_main_reasons_df = extract_reasons("GPT_4", X_test_class_definitions_prompt, y_pred_GPT_4_class_definitions, y_test, None, "class_definitions")
# GPT_o3_class_definitions_cases_df, GPT_o3_class_definitions_all_reasons_df, GPT_o3_class_definitions_main_reasons_df = extract_reasons("GPT_O3", X_test_class_definitions_prompt, y_pred_GPT_o3_class_definitions, y_test, thinking_class_definitions_GPT_o3, "class_definitions")
# # Gemini_class_definitions_cases_df, Gemini_class_definitions_all_reasons_df, Gemini_class_definitions_main_reasons_df = extract_reasons("Gemini", X_test_class_definitions_prompt, y_pred_Gemini_class_definitions, y_test, thinking_class_definitions_Gemini, "class_definitions")
# Gemma_class_definitions_cases_df, Gemma_class_definitions_all_reasons_df, Gemma_class_definitions_main_reasons_df = extract_reasons("Gemma", X_test_class_definitions_prompt, y_pred_Gemma_class_definitions, y_test, None, "class_definitions")
# Claude_class_definitions_cases_df, Claude_class_definitions_all_reasons_df, Claude_class_definitions_main_reasons_df = extract_reasons("Claude", X_test_class_definitions_prompt, y_pred_Claude_class_definitions, y_test, thinking_class_definitions_Claude, "class_definitions")
# DeepSeek_class_definitions_cases_df, DeepSeek_class_definitions_all_reasons_df, DeepSeek_class_definitions_main_reasons_df = extract_reasons("DeepSeek", X_test_class_definitions_prompt, y_pred_DeepSeek_class_definitions, y_test, thinking_class_definitions_DeepSeek, "class_definitions")
# Grok_class_definitions_cases_df, Grok_class_definitions_all_reasons_df, Grok_class_definitions_main_reasons_df = extract_reasons("Grok", X_test_class_definitions_prompt, y_pred_Grok_class_definitions, y_test, None, "class_definitions")
#
# GPT_4_profiled_simple_cases_df, GPT_4_profiled_simple_all_reasons_df, GPT_4_profiled_simple_main_reasons_df = extract_reasons("GPT_4", X_test_profiled_simple_prompt, y_pred_GPT_4_profiled_simple, y_test, None, "profiled_simple")
# GPT_o3_profiled_simple_cases_df, GPT_o3_profiled_simple_all_reasons_df, GPT_o3_profiled_simple_main_reasons_df = extract_reasons("GPT_O3", X_test_profiled_simple_prompt, y_pred_GPT_o3_profiled_simple, y_test, thinking_profiled_simple_GPT_o3, "profiled_simple")
# # Gemini_profiled_simple_cases_df, Gemini_profiled_simple_all_reasons_df, Gemini_profiled_simple_main_reasons_df = extract_reasons("Gemini", X_test_profiled_simple_prompt, y_pred_Gemini_profiled_simple, y_test, thinking_profiled_simple_Gemini, "profiled_simple")
# Gemma_profiled_simple_cases_df, Gemma_profiled_simple_all_reasons_df, Gemma_profiled_simple_main_reasons_df = extract_reasons("Gemma", X_test_profiled_simple_prompt, y_pred_Gemma_profiled_simple, y_test, None, "profiled_simple")
# Claude_profiled_simple_cases_df, Claude_profiled_simple_all_reasons_df, Claude_profiled_simple_main_reasons_df = extract_reasons("Claude", X_test_profiled_simple_prompt, y_pred_Claude_profiled_simple, y_test, thinking_profiled_simple_Claude, "profiled_simple")
# DeepSeek_profiled_simple_cases_df, DeepSeek_profiled_simple_all_reasons_df, DeepSeek_profiled_simple_main_reasons_df = extract_reasons("DeepSeek", X_test_profiled_simple_prompt, y_pred_DeepSeek_profiled_simple, y_test, thinking_profiled_simple_DeepSeek, "profiled_simple")
# Grok_profiled_simple_cases_df, Grok_profiled_simple_all_reasons_df, Grok_profiled_simple_main_reasons_df = extract_reasons("Grok", X_test_profiled_simple_prompt, y_pred_Grok_profiled_simple, y_test, None, "profiled_simple")
#
# GPT_4_few_shot_cases_df, GPT_4_few_shot_all_reasons_df, GPT_4_few_shot_main_reasons_df = extract_reasons("GPT_4", X_test_few_shot_prompt, y_pred_GPT_4_few_shot, y_test, None, "few_shot")
# GPT_o3_few_shot_cases_df, GPT_o3_few_shot_all_reasons_df, GPT_o3_few_shot_main_reasons_df = extract_reasons("GPT_O3", X_test_few_shot_prompt, y_pred_GPT_o3_few_shot, y_test, thinking_few_shot_GPT_o3, "few_shot")
# # Gemini_few_shot_cases_df, Gemini_few_shot_all_reasons_df, Gemini_few_shot_main_reasons_df = extract_reasons("Gemini", X_test_few_shot_prompt, y_pred_Gemini_few_shot, y_test, thinking_few_shot_Gemini, "few_shot")
Gemma_few_shot_cases_df, Gemma_few_shot_all_reasons_df, Gemma_few_shot_main_reasons_df = extract_reasons("Gemma", X_test_few_shot_prompt, y_pred_Gemma_few_shot, y_test, None, "few_shot")
# Claude_few_shot_cases_df, Claude_few_shot_all_reasons_df, Claude_few_shot_main_reasons_df = extract_reasons("Claude", X_test_few_shot_prompt, y_pred_Claude_few_shot, y_test, thinking_few_shot_Claude, "few_shot")
DeepSeek_few_shot_cases_df, DeepSeek_few_shot_all_reasons_df, DeepSeek_few_shot_main_reasons_df = extract_reasons("DeepSeek", X_test_few_shot_prompt, y_pred_DeepSeek_few_shot, y_test, thinking_few_shot_DeepSeek, "few_shot")
# Grok_few_shot_cases_df, Grok_few_shot_all_reasons_df, Grok_few_shot_main_reasons_df = extract_reasons("Grok", X_test_few_shot_prompt, y_pred_Grok_few_shot, y_test, None, "few_shot")

# GPT_4_vignette_cases_df, GPT_4_vignette_all_reasons_df, GPT_4_vignette_main_reasons_df = extract_reasons("GPT_4", X_test_vignette_prompt, y_pred_GPT_4_vignette, y_test, None, "vignette")
# GPT_o3_vignette_cases_df, GPT_o3_vignette_all_reasons_df, GPT_o3_vignette_main_reasons_df = extract_reasons("GPT_O3", X_test_vignette_prompt, y_pred_GPT_o3_vignette, y_test, thinking_vignette_GPT_o3, "vignette")
# # Gemini_vignette_cases_df, Gemini_vignette_all_reasons_df, Gemini_vignette_main_reasons_df = extract_reasons("Gemini", X_test_vignette_prompt, y_pred_Gemini_vignette, y_test, thinking_vignette_Gemini, "vignette")
Gemma_vignette_cases_df, Gemma_vignette_all_reasons_df, Gemma_vignette_main_reasons_df = extract_reasons("Gemma", X_test_vignette_prompt, y_pred_Gemma_vignette, y_test, None, "vignette")
# Claude_vignette_cases_df, Claude_vignette_all_reasons_df, Claude_vignette_main_reasons_df = extract_reasons("Claude", X_test_vignette_prompt, y_pred_Claude_vignette, y_test, thinking_vignette_Claude, "vignette")
DeepSeek_vignette_cases_df, DeepSeek_vignette_all_reasons_df, DeepSeek_vignette_main_reasons_df = extract_reasons("DeepSeek", X_test_vignette_prompt, y_pred_DeepSeek_vignette, y_test, thinking_vignette_DeepSeek, "vignette")
# Grok_vignette_cases_df, Grok_vignette_all_reasons_df, Grok_vignette_main_reasons_df = extract_reasons("Grok", X_test_vignette_prompt, y_pred_Grok_vignette, y_test, None, "vignette")

# GPT_4_cot_cases_df, GPT_4_cot_all_reasons_df, GPT_4_cot_main_reasons_df = extract_reasons_cot("GPT_4", X_test_cot_prompt, y_pred_GPT_4_cot, y_test, explanation_cot_GPT_4, None, "cot")
# GPT_o3_cot_cases_df, GPT_o3_cot_all_reasons_df, GPT_o3_cot_main_reasons_df = extract_reasons_cot("GPT_O3", X_test_cot_prompt, y_pred_GPT_o3_cot, y_test, explanation_cot_GPT_o3, thinking_cot_Gemini, "cot")
# Gemini_cot_cases_df, Gemini_cot_all_reasons_df, Gemini_cot_main_reasons_df = extract_reasons_cot("Gemini", X_test_cot_prompt, y_pred_Gemini_cot, y_test, explanation_cot_Gemini, thinking_cot_Gemini, "cot")
# Gemma_cot_cases_df, Gemma_cot_all_reasons_df, Gemma_cot_main_reasons_df = extract_reasons_cot("Gemma", X_test_cot_prompt, y_pred_Gemma_cot, y_test, explanation_cot_Gemma, None, "cot")
# Claude_cot_cases_df, Claude_cot_all_reasons_df, Claude_cot_main_reasons_df = extract_reasons_cot("Claude", X_test_cot_prompt, y_pred_Claude_cot, y_test, explanation_cot_Claude, thinking_cot_Claude, "cot")
# DeepSeek_cot_cases_df, DeepSeek_cot_all_reasons_df, DeepSeek_cot_main_reasons_df = extract_reasons_cot("DeepSeek", X_test_cot_prompt, y_pred_DeepSeek_cot, y_test, explanation_cot_DeepSeek, thinking_cot_DeepSeek, "cot")
# Grok_cot_cases_df, Grok_cot_all_reasons_df, Grok_cot_main_reasons_df = extract_reasons_cot("Grok", X_test_cot_prompt, y_pred_Grok_cot, y_test, explanation_cot_Grok, None, "cot")
