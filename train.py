import os, torch, logging
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logging.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


### building dataset to the format: 'text': ### Question: <question> ### Answer: <answer> ###
# reformat the dataset.
def build_text(example):
    example["text"] = """<s>[INST] <<SYS>>
                        You are a domain adapted chatbot learning new information about the Lamini organization and its products. Answer the question based on your knowledge of Lamini, and do not answer questions that you are not confident about. 
                        <</SYS>>

                        ### Question: {} [/INST] \n### Answer: {} </s>""".format(example["question"], example["answer"])
    return example

dataset_name = "lamini/lamini_docs"
datasets = load_dataset(dataset_name)

for t in datasets:
    datasets[t] = datasets[t].remove_columns(["input_ids", "attention_mask", "labels"])
    datasets[t] = datasets[t].map(build_text)
    datasets[t] = datasets[t].remove_columns(["question", "answer"])

train_dataset = datasets['train']

n = len(datasets['test'])
val_dataset = datasets['test'].shuffle(seed=42).select(range(0, int(0.30 * n)))
test_dataset = datasets['test'].shuffle(seed=42).select(range(int(0.30 * n), n))

print("***** Loaded Datasets *****")
print(train_dataset[0])
print("==" * 50)
# dataset = dataset.map(lambda x: tokenizer(x["text"]), batched=True)
# print(dataset)


### build the model ###
model_name = "meta-llama/Llama-2-7b-chat-hf"
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
    device_map="auto"
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

print("***** Loaded Models and Tokenizer *****")

### lora config ###
peft_config = LoraConfig(
    r=64, #attention heads
    lora_alpha=16, #alpha scaling
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head'], #if you know the 
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
)


# ### Trainer ###

training_arguments = TrainingArguments(
        output_dir="./results",
        save_strategy="steps",
        save_steps=50,
        save_total_limit=1,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        num_train_epochs=4,
        max_steps = 2000,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        logging_steps=10,
        learning_rate=5e-5,
        weight_decay=0.001,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="linear",
    )

trainer = SFTTrainer(
    model=model, 
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_arguments,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(5, 0.0005)],
)

print("***** Begin Finetuning... *****")
print_trainable_parameters(trainer.model)

trainer.train()
trainer.model.save_pretrained(new_model)

print("***** Finetuning Done *****")