# Evaluating Hallucinations in Domain Adapted Large Language Models

## Introduction
This study investigates the phenomenon of hallucinations in domain-adapted
Large Language Models (LLMs), focusing on the fine-tuning of the Llama-2
model with the Lamini dataset. Hallucinations, or the generation of nonsensical
or unfaithful content by LLMs, pose a significant challenge, especially when
these models are fine-tuned with domain-specific data. **Our methodology involves a series of experiments testing memorization, recall, and reasoning capabilities of the fine-tuned LLM, comparing its performance on novel question-answer pairs and domain-specific information**. 

We found that:
- While the model shows proficiency in tasks similar to its training data, its capability to accurately reason about and recall new domain-specific information remains limited, leading to instances
of hallucination. 
- The model demonstrates a tendency to provide correct answers with extra information,  suggesting an inclination toward over-generation. 

This research highlights the limitations of current fine-tuning approaches in fully mitigating
hallucinations and underscores the need for more robust methods in adapting
LLMs to specialized domains. The study also provides insights into the varying
performance of LLMs on different types of information, revealing a comparative
weakness in handling domain-specific queries.

## Project Setup
Install required libs: Used for PEFT LoRA finetuning, model quantization, and general analysis. 

```shell
pip3 install accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7
```

First train the model using the following command. The script finetunes a Llama-2-7b-chat-hf (https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) model using the Lamini Docs dataset (https://huggingface.co/datasets/lamini/lamini_docs) using 4 bit quantization.
```shell
python3 trainModel.py
```
The **uploadModel.ipynb** script merges the fine-tuned LoRA adapter weights with the Base Model and stores the final model at: https://huggingface.co/saiprasath21/Llama-lora-ft-lamini. Note that, the repository does **not** contain the adapter weights. 

For running the base model, final model (directly from hf), base model + LoRA adapter, use the respective scripts:
```shell
python3 runBaseModel.py
python3 runModel.py
python3 runAdapterModel.py
```
Each of these scripts also contains the code for peforming the experiments to evaluate the memorization, recall and reasoning capabilities. 

---

For analyzing the memorization, recall, and reasoning abilities of the finetuned model, we use the script **analyze.ipynb**

The **tfidfNER.ipynb** file helps extract the terms introduced by the finetuning dataset (not part of the pretraining dataset). [Refer 4.4.1]

For measuring similarity between questions we use the **Question_Similarity.ipynb**

For comparing the model and ground truth answers, we use GPT4 manually (using ChatGPT). The analysis of the annotations are perfomed using **GPT4_annotations.ipynb**


## Experiments

### Memorization: 

For evaluating the model’s memorization abilities, we focused on analyzing the fine-tuned model’s performance on the test set by prompting it with questions and comparing the generated answers against the expected gold responses. [Refer 4.2]

### Recall: 
For evaluating the recall capabilities of the finetuned model, we creat a set of complex questions that mimic the style of multi-hop question-answering (MHQA). MHQA necessitates that a model retrieves and synthesizes information across multiple textual passages to formulate a coherent response to a query.

We manually selected a subset of question-answer pairs from our test set. These pairs were then
input to GPT-4 for generating MHQA style questions and answers. We refer to the
initial QA pairs as Q1 − A1 and Q2 − A2. We then prompt GPT4 using the prompt for generating
Q3 − A3: “I’ll give you two question-answer pairs, use the information in both q-a pairs and
generate a new question and a new answer. The newly generated question should have 15-30 words.
The generated answer should have less than 80 words”. The newly created question Q3 is designed
to reflect the content of both Q1 and Q2, and the answer A3 merges the information from A1 and
A2. Thus, The fine-tuned model must demonstrate a comprehensive understanding of the original
questions to accurately address Q3. 

To evaluate the recall ability of the finetuned model, we prompt the model with Q3 and compare the generated response with the gold response A3. [Refer 4.3]


### Reasoning:

For studying whether these newly introduced terms (refer 4.4.1) are actually learnt by the model
or whether the model hallucinates and guesses on these terms, we perform a four stage evaluation
process: Understanding, retrieval, stress test, and advanced reasoning (refer 4.4.2). 

## Results

### MEMORIZATION
**Length**: Comparing the length of the generated responses with the expected response, we find that
in 81.6% of the examples the generated response is longer than the expected response. Moreover,
generated responses (= 86.1 words per response) are almost twice as long as the expected responses (= 47.6 words per response). Figure 1 shows the cdf of the expected and the generated responses with respected to the number of words in the response, we can observe that generated responses are much longer on average. Notably, none of the expected reponses contain more than 123 words, however, 18.4% of the generated responses are over 123 words.

**Use of domain specific words**: Comparing the use of domain specific words between the expected
and the generated response, we find that generated responses contain 2.36 domain specific words
per response, whereas, the expected answers contain 1.58 domain specific words per response. The
top-5 domain specific words used by the finetuned model are: {lamini, llms2, chatbots, llamaindex, hyperparameters}. The corresponsing ranks of these entities in the domain-specific entities are: 1, 2, 6, 8, and 10. Refer Section 4.4.1 for collecting domain specific words and their tf-idf ranks.

**Correctness**: We use the annotation system discussed in Section 4.1 for comparing the expected
and generated responses. We find that 43.88% of the time, the model correctly answered the
questions, and in 18.36% of cases, the responses were wrong. Interestingly, 37.75% of the correct
answers included extra details not present in the expected answers. The average BERT similarity between the test question and the (most-similar) train question is 0.8856, which is very high. Therefore, we can observe that the test set questions are very similar to the train set. 

Hence, we find that finetuned domain specific models generate longer responses while memorizing and using important domain specific words. Moreover, the finetuned model correctly answers 80% of the questions, while including extra details in half of those reponses.

### RECALL

Using the manually selected subset of questions-answer pairs from the test dataset, we create 38
MHQA stype question-answer pairs using the method discussed in Section 4.3. Prompting the
finetuned model with Q3, we compare the generated response with the gold response A3.

We fine that most of the trends observed in memorization holds true for recall
as well. The generated responses are almost 1.5 times as long as the gold responses (67.13 words
vs 44.86 words per response), and the generated response uses more domain specific words than the
gold response (2.13 vs 1.32). Even the correctness of the response follows the trend, where 23.68% of the responses are wrong, 28.9% of the responses are correct with extra details, and the remaining 47.36% of the responses are correct.

Hence, we find that the finetuned model can satisfactorily answer complex MHQA style questions,
while generating longer reponses and using important domain specific words more frequently than
expected. Note that, although we create complex questions by combining test set questions, the test set questions used for creating these questions were very similar to the train set questions.


### REASONING

A. Understanding: While 86.95% of common entities pass the understanding test, only
18.75% of the specific entities pass the same test. This shows that the finetuned model cannot learn new entities with uncommon function names.

B. Retrieval: Interestingly, out of the 3 specific entities that pass the understanding test,
none of them pass the retrieval test. Therefore, the model has poor understanding or retrieval
capabilities on newly introduced entities if their function names are not common. However, for the common entities the model achieves a retrieval rate of 45%.

C. Stress Test: The stress test helps in evaluating whether the model actually learns the
common entities or only uses their names to deduce their descriptions and answer questions. We
find that out of the 9 common entities that pass the retrieval test, only one of then passes the stress test. As the model gets easily confused and chooses the wrong function calls, we conclude that even with common entities the model does not actually learn the functions.

D. Advanced Reasoning: As most functions fail the previous tests, we are left with only
one function to perform the reasoning test. The results of the reasoning tests are:

**None of the entities, common or specific, pass all four stages of testing.**