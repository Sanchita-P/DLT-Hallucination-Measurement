from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nerExtractor = pipeline("ner", model=model, tokenizer=tokenizer)

dataset_name = "lamini/lamini_docs"
train_dataset = load_dataset(dataset_name, split="train")
datasetText = list(map(lambda x: x['question'] + "\n" + x['answer'], train_dataset))

ners = nerExtractor(datasetText, batch_size=8)


'''
Merge the subtokens to the corresponding word
'''
entities = []
for ner in ners:
    entity = []
    word = None
    for token in ner:
        if word is None and token['word'].startswith("##"):
            continue
        elif word is None:
            word = token['word']
        elif token['word'].startswith("##"):
            word = word + token['word'][2:]
        else:
            entity.append(word)
            word = token['word']
    if word is not None:
        entity.append(word)
    entities.append(entity)


'''
Remove unimportant frequent tokens 
'''
remove_ner = ["Lamini", "lamini", "AI", "LLM", "Lam", "LL", "API"]

for i in range(len(entities)):
    entities[i] = [x for x in entities[i] if x not in remove_ner]


'''
Add functions and variable names present in the test to entitites
'''
for i, text in enumerate(datasetText):
    words = text.split(' ')
    for word in words:
        if "()" in word or "_" in word:
            entities[i].append(word)

import csv

# Open CSV file for writing
with open('train_ners.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    # Write the header
    csvwriter.writerow(['Text', 'NERS'])

    NERList = [', '.join(entity) for entity in entities]

    rows = [[text, ner] for text, ner in zip(datasetText, NERList)]

    csvwriter.writerows(rows)