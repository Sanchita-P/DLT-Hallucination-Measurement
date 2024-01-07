import os, torch, logging
import torch
import torch.nn as nn
import bitsandbytes as bnb
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, logging
from datasets import load_dataset, Dataset
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import pandas as pd
import pickle as pkl
import random
import ast
from huggingface_hub import login

login(token = "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")
### model ###
model_name = "meta-llama/Llama-2-7b-chat-hf"
finetuned_model = "saiprasath21/Llama-lora-ft-lamini"

model = AutoModelForCausalLM.from_pretrained(finetuned_model, low_cpu_mem_usage=True, return_dict=True, torch_dtype=torch.float16, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
# You are a domain adapted chatbot learning new information about the Lamini organization and its products. Answer the question based on your knowledge of Lamini, and do not answer questions that you are not confident about. 
def convert2inputformat(text):
    return """<s>[INST] <<SYS>>
                        You are a domain adapted chatbot learning new information about the Lamini organization and its products. Answer the question based on your knowledge of Lamini, and do not answer questions that you are not confident about. 
                        <</SYS>>

                        ### Question: {} [/INST]""".format(text)


### test model ###
logging.set_verbosity(logging.CRITICAL)
while 1:
    prompt = input("Enter the prompt: ")
    prompt = convert2inputformat(prompt)
    result = pipe(prompt, return_full_text = False, max_new_tokens=100, num_beams=3)
    print(result[0]['generated_text'])
    print('*' * 80)

"""
? Ask inverted question to analyze the retrieval ability of the finetuned model
"""

# df = pd.read_csv("summary.csv")
# inverted_questions = list(df["Inverted Questions"])
# expected_answers = list(df["Entity"])
# generated_answers = []

# for exp_ans, question in zip(expected_answers, inverted_questions):
#     print(f"Question: {question}")
#     prompt = convert2inputformat(question)
#     result = pipe(prompt, return_full_text = False, max_new_tokens=100, num_beams=3)
#     generated_answers.append(result[0]['generated_text'])

#     print(f"Expected Answer: {exp_ans}")
#     print(f"Generated Answer: {result[0]['generated_text']}")
#     print('*' * 80)

# pkl.dump(generated_answers, open("inverted_questions_answers.pkl", "wb"))

"""
? Generate summary from Zero-shot finetuned model
"""

# df = pd.read_csv("summary.csv")
# entities = list(df["Entity"])

# generated_summary = []
# for e in entities:
#     prompt = "Generate a description for the " + e +  " function/parameter in 30 words."
#     prompt = convert2inputformat(prompt)
#     result = pipe(prompt, return_full_text = False, max_new_tokens=100, num_beams=3)
#     generated_summary.append(result[0]['generated_text'])
#     print(f"Generated Summary: {result[0]['generated_text']}")
#     print('*' * 80)

# pkl.dump(generated_summary, open("generated_summary.pkl", "wb"))

"""
# ? Stress Test
"""

# df = pd.read_csv("matched_entities.csv")
# questions = list(df["Inverted Questions"])
# options = list(df["Options"])
# answers = list(df["Entities"])

# fct_answers = []

# for i in range(len(questions)):
#     choices = ast.literal_eval(options[i])
#     choices.append(answers[i])
#     random.shuffle(choices)

#     prompt = questions[i] + " Choose one of the following options." + "\n"
#     for j in range(len(choices)):
#         prompt += str(j+1) + ". " + choices[j] + "\n"

#     print(f"Question: {prompt}")
#     prompt = convert2inputformat(prompt)
#     result = pipe(prompt, return_full_text = False, max_new_tokens=100, num_beams=3)
#     fct_answers.append(result[0]['generated_text'])
#     print(f"Answer: {result[0]['generated_text']}")
#     print('*' * 80)

# pkl.dump(fct_answers, open("fct_answers.pkl", "wb"))