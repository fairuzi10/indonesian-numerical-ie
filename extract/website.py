import os
import re
import json
from nltk.tokenize import sent_tokenize
from model.ner import NERTagger
from model.postag import POSTagger
from model.confidence_value import ConfidenceValueGenerator
import random
from model.tuple import get_predictor
import pandas as pd
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm
import subprocess

DIRPATH = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE_PATH = os.path.join(DIRPATH, 'data', 'website.vert')
OUTPUT_CSV = os.path.join(DIRPATH, 'result', 'website.csv')
NUMERICAL_TAG = ('MON', 'PRC', 'CRD', 'QTY')
EXTRACT_THRESHOLD = 0.5


class TokenLabelOfSentence:

    def __init__(self, char_begin_idx, char_end_idx, token_begin_idx, token_end_idx, label):
        self.char_begin_idx = char_begin_idx
        self.char_end_idx = char_end_idx
        self.token_begin_idx = token_begin_idx
        self.token_end_idx = token_end_idx
        self.label = label

    def __repr__(self):
        return "TokenLabelOfSentence(char_idx {}-{}, token_idx {}-{}, label {})".format(
            self.char_begin_idx, self.char_end_idx, self.token_begin_idx, self.token_end_idx, self.label)

def get_labels(sentence, words, tags):
    result = []
    ptr = 0
    for i in range(len(words)):
        word = words[i]
        tag = tags[i]
        while sentence[ptr].isspace():
            ptr += 1
        if sentence[ptr:ptr+len(word)] != word:
            print('guessed word != word')
            print(sentence)
            print(sentence[ptr:ptr+len(word)])
            print(word)
            return []

        if tag[0] == 'B':
            result.append(TokenLabelOfSentence(ptr, ptr+len(word), i, i, tag[2:]))
        elif tag[0] == 'I':
            if not (result and result[-1].label == tag[2:]):
                print('=== BROKEN TAGS ===')
                print(sentence)
                print(result)
                print(tag)
                ptr += len(word)
                continue
            result[-1].char_end_idx = ptr+len(word)
            result[-1].token_end_idx = i
        ptr += len(word)
    return result

detokenizer = TreebankWordDetokenizer()

to_csv = {
    "id": [],
    "sentence": [],
    "ARG0": [],
    "PRED": [],
    "ARG1": [],
}

data_id = 0
confidence_value_generator = ConfidenceValueGenerator()

def print_sentence_to_file(sentences):
    global data_id
    for words in sentences:
        ner_tagger = NERTagger(POSTagger())
        tags = ner_tagger.tag(words)
        words = ner_tagger.sent_tokenized
        sentence = detokenizer.detokenize(words)
        labels = get_labels(sentence, words, tags)
        filtered_token_labels = list(filter(lambda token_label: token_label.label in NUMERICAL_TAG, labels))
        for numerical_token_label in filtered_token_labels:
            RelPredictor = get_predictor(numerical_token_label.label)
            token_idx_range = range(numerical_token_label.token_begin_idx, numerical_token_label.token_end_idx+1)
            predictor = RelPredictor(sentence, token_idx_range, POSTagger())
            arg0, pred, arg1 = predictor.get_tuple()
            confidence_value = confidence_value_generator.confidence_value(sentence, arg0, pred, arg1)
            if arg0 and pred and arg1 and confidence_value > EXTRACT_THRESHOLD:
                to_csv["id"].append(data_id)
                to_csv["sentence"].append(sentence)
                to_csv["ARG0"].append(arg0)
                to_csv["PRED"].append(pred)
                to_csv["ARG1"].append(arg1)
                data_id += 1


def line_count(filename):
    return int(subprocess.check_output(['wc', '-l', filename]).split()[0])


with open(INPUT_FILE_PATH) as input_file:
    documents = []
    paragraphes = []
    sentences = []
    words = []

    cur_sentence = []
    for i, line in enumerate(tqdm(input_file, total=line_count(INPUT_FILE_PATH))):
        line = line.strip()
        if line[:4] == '<doc':
            documents.append(paragraphes)
            paragraphs = []
        elif line[:2] == '<p':
            sentences = []
        elif line[:2] == '<s':
            words = []
        elif line[:3] == '</s':
            sentences.append(words)
        elif line[:3] == '</p':
            paragraphes.append(sentences)
            print_sentence_to_file(sentences)
        elif line:
            word = line.split()[0]
            words.append(word)


df = pd.DataFrame(to_csv)
df.to_csv(OUTPUT_CSV, header=True, index=False)
