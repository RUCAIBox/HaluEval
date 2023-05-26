# HaluEval: A Hallucination Evaluation Benchmark for LLMs

This is the repo for our paper: [HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models](https://arxiv.org/abs/2305.11747). The repo contains:

- The [35K data](#data-release) used for evaluating the LLM.
- The code for [generating the data](#data-generation-process).
- The code for [evaluating the model](#evaluation).

## Overview

HaluEval includes 5,000 general user queries with ChatGPT responses and  30,000 task-specific examples from three tasks, i.e.,
question answering, knowledge-grounded dialogue, and text summarization. 

For general user queries, we adopt the 52K instruction tuning dataset from [Alpaca](https://github.com/tatsu-lab/stanford_alpaca).
To further screen user queries where LLMs are most likely to produce hallucinations, we use ChatGPT to sample three responses 
for each query and finally retain the queries with low-similarity responses for human labeling.

Furthermore, for the task-specific examples in HaluEval, we design an automatic approach to generate hallucinated samples. 
First, based on existing task datasets (e.g., HotpotQA) as seed data, we design task-specific instructions for ChatGPT
to generate hallucinated samples in two methods, i.e., one-pass and conversational. Second, to select
the most plausible and difficult hallucinated sample for LLMs evaluation, we elaborate the filtering instruction enhanced 
by ground-truth examples and leverage ChatGPT for sample selection.

<a href="https://github.com/RUCAIBox/HaluEval" target="_blank"><img src="assets/pipeline.png" alt="HaluEval" style="width: 90%; min-width: 300px; display: block; margin: auto;"></a>

## Data Release

The directory [`data`](./data) contains 35K generated and human-annotated hallucinated samples we used in our experiments.
There are four JSON files as follows:

- [`qa_data.json`](./data/hotpotqa_data.json): 10K hallucinated samples for QA based on [HotpotQA](https://hotpotqa.github.io/) as seed data. 
For each sample dictionary, the fields `knowledge`, `question`, and `right_answer` refer to the knowledge from Wikipedia, question text, and ground-truth answer collected from HotpotQA. The field `hallucinated_answer` is the generated hallucinated answer correspondingly.
- [`dialogue_data.json`](./data/opendialkg_data.json): 10K hallucinated samples for dialogue based on [OpenDialKG](https://github.com/facebookresearch/opendialkg) as seed data. 
For each sample dictionary, the fields `knowledge`, `dialogue_history`, and `right_response` refer to the knowledge from Wikipedia, dialogue history, and ground-truth response collected from OpenDialKG. The field `hallucinated_response` is the generated hallucinated response correspondingly.
- [`summarization_data.json`](./data/cnndm_data.json): 10K hallucinated samples for summarization based on [CNN/Daily Mail](https://github.com/abisee/cnn-dailymail) as seed data. 
For each sample dictionary, the fields `document` and `right_summary` refer to the document and ground-truth summary collected from CNN/Daily Mail. The field `hallucinated_summary` is the generated hallucinated summary correspondingly.
- [`general_data.json`](./data/general_data.json): 5K human-annotated samples for ChatGPT responses to general user queries from [Alpaca](https://github.com/tatsu-lab/stanford_alpaca).
For each sample dictionary, the fields `user_query`, `chatgpt_response`, and `hallucination_label` refer to the posed user query, ChatGPT response, and hallucination label (Yes/No) annotated by humans.

Based on these data, you can evaluate the ability of LLMs to recognize hallucinations and analyze what type of contents/topics LLMs tend to hallucinate (or fail to recognize the contained hallucination). 

## Data Generation Process

We executed the data generation pipeline via ChatGPT using two different sampling methods, i.e., one-turn and multi-turn.
Here we use the QA task as an example for data generation.

- First, we randomly sample 10K data from the training set of HotpotQA dataset.

```
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
```

- Second, we use the one-turn instruction [`qa_oneturn_instruction.txt`](./generation/qa) to generate hallucinated answers.



