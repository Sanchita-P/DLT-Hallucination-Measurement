import os, torch, logging
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, logging
from datasets import load_dataset, Dataset
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

### model ###
model_name = "meta-llama/Llama-2-7b-chat-hf"
new_model = "llama2_lora_lamni"
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

def convert2inputformat(text):
    return """<s>[INST] <<SYS>>
                        You are a domain adapted chatbot learning new information about the Lamini organization and its products. Answer the question based on your knowledge of Lamini, and do not answer questions that you are not confident about. 
                        <</SYS>>

                        ### Question: {} [/INST]""".format(text)


dataset_name = "lamini/lamini_docs"
test_dataset = load_dataset(dataset_name, split="test")

n = len(test_dataset)
test_dataset = test_dataset.shuffle(seed=42).select(range(int(0.30 * n), n))

# questions = {'text': list(map(convert2inputformat, test_dataset['question']))}
# questions = Dataset.from_dict(questions)
# print(questions)
questions = list(map(convert2inputformat, test_dataset['question']))

generated_output = pipe(questions, batch_size=8, return_full_text = False, max_new_tokens=200, num_beams=3)

print(generated_output)
generated_answers = []
for x in generated_output:
    if 'Answer:' in x[0]['generated_text']:
        clean_output = x[0]['generated_text'].split('Answer:')[1].strip()
    else:
        clean_output = x[0]['generated_text']
        
    if '###' in clean_output:
        clean_output = clean_output.split('###')[0].strip()
    generated_answers.append(clean_output)

# generated_answers = [x[0]['generated_text'].split('Answer:')[1].strip() for x in generated_answers]

import csv

# Open a file for writing
with open('output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(["Question", "Actual Answer", "Generated Answer"])
    
    # Write the content
    for q, actual, generated in zip(test_dataset['question'], test_dataset['answer'], generated_answers):
        writer.writerow([q, actual, generated])


### test model ###
# logging.set_verbosity(logging.CRITICAL)
# while 1:
#     prompt = input("Enter the prompt: ")
#     prompt = convert2inputformat(prompt)
#     result = pipe(prompt, return_full_text = False, max_new_tokens=1000, num_beams=3)
#     print(result[0]['generated_text'])