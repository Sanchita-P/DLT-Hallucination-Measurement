import os, torch, logging
import torch
import torch.nn as nn
import bitsandbytes as bnb
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, logging
from datasets import load_dataset, Dataset
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import pickle as pkl
import csv
from huggingface_hub import login

login(token = "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")
### model ###
model_name = "meta-llama/Llama-2-7b-chat-hf"


model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, return_dict=True, torch_dtype=torch.float16, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

# The "write_story" function customizes the level of creativity in the generated output by setting the "random" parameter to True or False, resulting in more or less unpredictable and creative output, respectively.
# You are a chatbot that helps analyze the similarity between Lamini products and other products in the market. Lamini is the LLM platform for enterprises and developers to build customized, private models: easier, faster, and higher-performing than any general LLMs. Answer questions only based on the provided context. 
# You are a chatbot that helps summarize the function of a given entity from the provided context. Answer questions only based on the provided context.
def convert2inputformat(text):
    return """<s>[INST] <<SYS>>
                       You are a chatbot that helps retrieve relevant functions from the Microsoft Azure AI platform for a given task. 
                        ### Question: {} [/INST]""".format(text)


### test model ###
logging.set_verbosity(logging.CRITICAL)
while 1:
    prompt = input("Enter the prompt: ")
    prompt = convert2inputformat(prompt)
    result = pipe(prompt, return_full_text = False, max_new_tokens=400, num_beams=3)
    print(result[0]['generated_text'])


# context = pkl.load(open("context.pkl", "rb"))
# summary = []

# for func, _context in context:
#     prompt = _context + " \n" + "Summarize the function" + " \"" + func + "\" in 30 words."
#     print(prompt)
#     prompt = convert2inputformat(prompt)
#     result = pipe(prompt, return_full_text = False, max_new_tokens=100, num_beams=3)
#     _summary = result[0]['generated_text']
#     summary.append([func, _summary])
#     print(result[0]['generated_text'])   

# pkl.dump(summary, open("summary.pkl", "wb"))

# summary = pkl.load(open("summary.pkl", "rb"))

# file_path = 'summary.csv'

# # Writing to a CSV file
# with open(file_path, 'w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     writer.writerow(["Entity", "Summary"])
#     writer.writerows(summary)


# summary = pkl.load(open("summary.pkl", "rb"))
# similar_entities = []

# for func, _summary in summary[0:1]:
#     prompt = "What functions in Google Cloud AI, Microsoft Azure AI, AWS AI service, IBM Watson Studio, and H2O.ai allows you to add training data to the LLM Engine, enabling the model to learn and improve its performance.\nAnswer in the context of Large Language Models and output the answer in the format: <platform>:<function call>"
#     # prompt = "The role of the \"" + func + "\" is : " + _summary + " \n" + "What are some of the functions in Google Cloud AI, Microsoft Azure AI, AWS AI service, IBM Watson Studio, and H2O.ai that have similar role? \nAnswer in the context of Large Language Models and output the answer in the format: <platform>:<function call>"
#     print(prompt)
#     prompt = convert2inputformat(prompt)
#     result = pipe(prompt, return_full_text = False, max_new_tokens=500, num_beams=3)
#     _similar_entities = result[0]['generated_text']
#     similar_entities.append([func, _similar_entities])
#     print(result[0]['generated_text'])   

# pkl.dump(similar_entities, open("similar_entities.pkl", "wb"))
