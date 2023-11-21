import os, torch, logging
import torch
import torch.nn as nn
import bitsandbytes as bnb
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, logging
from datasets import load_dataset, Dataset
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
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