import json
from nltk.tokenize import RegexpTokenizer
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import spacy
from gensim.models import Phrases
from gensim.models import LdaModel
from pprint import pprint
from gensim.corpora import Dictionary
from pyLDAvis import gensim_models
import pyLDAvis
import argparse


def load_qa(filename):
    text = []
    failed_text = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            if "knowledge" in sample:
                text.append(sample["knowledge"] +" "+ sample["question"])
                if sample["ground_truth"] == 'Yes' and sample["judgement"] == 'No':
                    failed_text.append(sample["knowledge"] +" "+ sample["question"])
                
    return text, failed_text

def load_dialog(filename):
    text = []
    failed_text = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            if "knowledge" in sample:
                text.append(sample["knowledge"] +" "+ sample["dialogue_history"])
                if sample["ground_truth"] == 'Yes' and sample["judgement"] == 'No':
                    failed_text.append(sample["knowledge"] +" "+ sample["dialogue_history"])
    
    return text, failed_text

def load_summary(filename):
    text = []
    failed_text = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            if "summary" in sample:
                text.append(sample["summary"])
                if sample['ground_truth'] == 'Yes' and sample['judgement'] == 'No':
                    failed_text.append(sample["summary"])
    
    return text, failed_text

def load_general(filename):
    text = []
    failed_text = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            text.append(sample["user_query"] +" "+ sample["chatgpt_response"])
            if "hallucination" in sample and sample["hallucination"]=="yes":
                failed_text.append(sample["user_query"] +" "+ sample["chatgpt_response"])
    
    return text, failed_text

def process_text(docs):
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()  # Convert to lowercase.
        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 1] for doc in docs]

    # Remove stopwords
    stop_words = stopwords.words('english')
    new_stop_words = ['year', 'new', 'woman', 'man', 'women', 'years', 'time', 'people', 'day', 'days']
    stop_words.extend(new_stop_words)
    docs = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in docs]
    
    # Lemmatize the documents.
    docs = lemmatization(docs, allowed_postags=['NOUN', 'ADJ'])

    # Add bigrams to docs (only ones that appear 20 times or more).
    bigram = Phrases(docs, min_count=20)
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)
    return docs

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ']):
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def lda_model(docs, num_topics):
    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=20, no_above=0.5)

    corpus = [dictionary.doc2bow(doc) for doc in docs]
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    # Set training parameters.
    num_topics = num_topics
    chunksize = 10000
    passes = 20
    iterations = 800
    eval_every = None

    # Make a index to word dictionary.
    temp = dictionary[0]
    id2word = dictionary.id2token

    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )

    top_topics = model.top_topics(corpus, topn=10)

    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)
    pprint(top_topics)

    vis = gensim_models.prepare(model, corpus, dictionary)
    pyLDAvis.show(vis, local=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LDA cluster for topics")

    parser.add_argument("--result", default="", help="the dataset file path")
    parser.add_argument("--task", default="qa", help="qa, dialogue, summarization or general")
    parser.add_argument("--category",default="all", choices=['all', 'failed'], help="all or failed data")
    args = parser.parse_args()

    if args.task == 'qa':
        text, failed_text = load_qa(args.result)
    elif args.task == 'dialogue':
        text, failed_text = load_dialog(args.result)
    elif args.task == 'summarization':
        text, failed_text = load_summary(args.result)
    elif args.task == 'general':
        text, failed_text = load_general(args.result)
    else:
        raise ValueError("The task must be qa, dialogue, summarization, or general!")

    if args.category == 'all':
        docs = process_text(text)
        num_topics = 10
    elif args.category == 'failed':
        docs = process_text(failed_text)
        num_topics = 5
    else:
        raise ValueError("The category must be all or failed!")

    lda_model(docs, num_topics)
