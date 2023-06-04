import json
import math

from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu


class RecMetric:
    def __init__(self, k_list=(1, 10, 50)):
        self.k_list = k_list

        self.metric = {}
        self.reset_metric()

    def evaluate(self, preds, labels):
        for pred_list, label in zip(preds, labels):
            if label == -100:
                continue
            for k in self.k_list:
                self.metric[f'recall@{k}'] += self.compute_recall(pred_list, label, k)
                self.metric[f'ndcg@{k}'] += self.compute_ndcg(pred_list, label, k)
                self.metric[f'mrr@{k}'] += self.compute_mrr(pred_list, label, k)
            self.metric['count'] += 1

    def compute_recall(self, pred_list, label, k):
        return int(label in pred_list[:k])

    def compute_mrr(self, pred_list, label, k):
        if label in pred_list[:k]:
            label_rank = pred_list.index(label)
            return 1 / (label_rank + 1)
        return 0

    def compute_ndcg(self, pred_list, label, k):
        if label in pred_list[:k]:
            label_rank = pred_list.index(label)
            return 1 / math.log2(label_rank + 2)
        return 0

    def reset_metric(self):
        for metric in ['recall', 'ndcg', 'mrr']:
            for k in self.k_list:
                self.metric[f'{metric}@{k}'] = 0
        self.metric['count'] = 0

    def report(self):
        report = {}
        for k, v in self.metric.items():
            if k != 'count':
                report[k] = v / self.metric['count']
        return report


class ConvMetric:
    def __init__(self, tokenizer, log_file_path=None):
        self.tokenizer = tokenizer

        self.metric = {}
        self.sent_cnt = 0
        self.reset_metric()

        self.log_file = None
        if log_file_path:
            self.log_file = open(log_file_path, 'w')

    def log_write_line(self, str):
        self.log_file.write(str + '\n')

    def evaluate(self, preds, labels, log=False):
        if log and self.log_file is not None:
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=False)
            decoded_preds = [pred.replace(f'{self.tokenizer.pad_token}', '').strip() for pred in decoded_preds]
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=False)
            decoded_labels = [label.replace(f'{self.tokenizer.pad_token}', '').strip() for label in decoded_labels]

            for pred, label in zip(decoded_preds, decoded_labels):
                self.log_write_line(json.dumps({
                    'pred': pred,
                    'label': label
                }, ensure_ascii=False))

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=False)
        decoded_preds = [
            pred.replace(f'{self.tokenizer.eos_token}', '').strip()
            for pred in decoded_preds
        ]
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=False)
        decoded_labels = [
            label.replace(f'{self.tokenizer.eos_token}', '').strip()
            for label in decoded_labels
        ]

        self.collect_ngram(decoded_preds)
        self.compute_bleu(decoded_preds, decoded_labels)
        self.sent_cnt += len(decoded_preds)

    def collect_ngram(self, strs):
        for str in strs:
            str = str.split()
            for k in range(1, 5):
                dist_k = f'dist@{k}'
                for token in ngrams(str, k):
                    self.metric[dist_k].add(token)

    def compute_bleu(self, preds, labels):
        for pred, label in zip(preds, labels):
            pred, label = pred.split(), [label.split()]
            for k in range(4):
                weights = [0] * 4
                weights[k] = 1
                self.metric[f'bleu@{k + 1}'] += sentence_bleu(label, pred, weights)

    def report(self):
        report = {}
        for k, v in self.metric.items():
            if self.sent_cnt == 0:
                report[k] = 0
            else:
                if 'dist' in k:
                    v = len(v)
                report[k] = v / self.sent_cnt
        report['sent_cnt'] = self.sent_cnt
        return report

    def reset_metric(self):
        self.metric = {
            'bleu@1': 0,
            'bleu@2': 0,
            'bleu@3': 0,
            'bleu@4': 0,
            'dist@1': set(),
            'dist@2': set(),
            'dist@3': set(),
            'dist@4': set(),
        }
        self.sent_cnt = 0
