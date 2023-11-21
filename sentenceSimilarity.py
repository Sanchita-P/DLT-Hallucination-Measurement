import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')
dataset_name = "lamini/lamini_docs"
datasets = load_dataset(dataset_name)

example1 = datasets['train'][0]["question"]
example2 = datasets['train'][1]["question"]


def computeSim(example1, example2):
    sentences1_list = sent_tokenize(example1)
    sentences2_list = sent_tokenize(example2)
    # print(len(sentences1_list))
    # print(len(sentences2_list))
    embeddings1 = model.encode(sentences1_list, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2_list, convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(embeddings1, embeddings2).mean().item()
    return round(cosine_sim, 4)

res = computeSim(example1, example2)
print(res)
