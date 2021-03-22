from transformers import BertForSequenceClassification, BertTokenizer
import torch
import os

MODEL_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_confidence_value_bert')

class ConfidenceValueGenerator():
    def __init__(self):
        self.model = BertForSequenceClassification.from_pretrained(MODEL_FILE_PATH, num_labels=2)
        self.tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')

    # format into [CLS] the original sentence [SEP] ARG0 span [SEP] PRED span [SEP] ARG1 span [SEP] and transform it into BERT token
    def preprocess(self, row):
        sep_token = self.tokenizer.sep_token
        input_repr = row['sentence'] + ' ' + sep_token + ' ' + row['ARG0'] + ' ' + sep_token + ' ' + row['PRED'] + ' ' + sep_token + ' ' + row['ARG1']
        return self.tokenizer(input_repr, padding='max_length', truncation=True, max_length=128)

    def confidence_value(self, sent, arg0, pred, arg1):
        if not sent or not arg0 or not pred or not arg1:
            return 0
        inp = self.preprocess({"sentence": sent, "ARG0": arg0, "PRED": pred, "ARG1": arg1})
        for key in inp:
            inp[key] = torch.unsqueeze(torch.tensor(inp[key]), 0)
        output = self.model(input_ids=inp['input_ids'], attention_mask=inp['attention_mask'])
        output.logits[0]
        score = torch.nn.functional.softmax(output.logits[0], dim=-1)[1].detach().numpy().item()
        return score
