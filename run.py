import os, torch, logging
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, logging
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

### model ###
model_name = "openlm-research/open_llama_3b_v2"
new_model = "llama2_lora_lamni"
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0}
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

### test model ###
logging.set_verbosity(logging.CRITICAL)
prompt = "Who is the founder of lamini" 
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=100)
result = pipe(f"### Question: {prompt}")
print(result[0]['generated_text'])