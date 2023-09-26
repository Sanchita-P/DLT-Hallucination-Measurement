import os, torch, logging
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

### building dataset to the format: 'text': ### Question: <question> ### Answer: <answer> ###
# reformat the dataset.
def build_text(example):
    example["text"] = "### question: " + str(example["question"])+ "\n### answer: " +str(example["answer"])
    return example

dataset_name = "lamini/lamini_docs"
dataset = load_dataset(dataset_name, split="train")
dataset = dataset.remove_columns(["input_ids", "attention_mask", "labels"])
dataset = dataset.map(build_text)
dataset = dataset.remove_columns(["question", "answer"])
print(dataset[0])
print("==" * 50)
# dataset = dataset.map(lambda x: tokenizer(x["text"]), batched=True)
# print(dataset)


### build the model ###
model_name = "openlm-research/open_llama_3b_v2"
new_model = "llama2_lora_lamni"
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=False,
)

if compute_dtype == torch.float16:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

### lora config ###
peft_config = LoraConfig(
    r=64, #attention heads
    lora_alpha=16, #alpha scaling
    # target_modules=["q_proj", "v_proj"], #if you know the 
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
)


# ### Trainer ###
trainer = SFTTrainer(
    model=model, 
    train_dataset=dataset,
    args=transformers.TrainingArguments(
        output_dir='./results',
        num_train_epochs = 2,
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=1,
        optim = "paged_adamw_32bit",
        warmup_ratio = 0.03,
        save_steps = 0,
        logging_steps=25, 
        gradient_checkpointing = True,
        weight_decay = 0.001,
        group_by_length = True,
        max_grad_norm = 0.3,
        max_steps=-1, 
        learning_rate=2e-4, 
        fp16 = False,
        bf16 = False,
        report_to="none",
        lr_scheduler_type="cosine",
    ),
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    packing=False
)

trainer.train()
trainer.model.save_pretrained(new_model)

print("alles gut")