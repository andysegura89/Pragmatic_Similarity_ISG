import feature_selection_test as fst
import csv_reader as csvr
import torch
import matplotlib.pyplot as plt


s1_judgement_reader = fst.FeatureSelectionTest('data/judge_features_avg_session1-d.csv')
s1_judgements = s1_judgement_reader.get_judgements()
s2_judgement_reader = fst.FeatureSelectionTest('data/judge_features_avg_session2-d.csv')
s2_judgements = s2_judgement_reader.get_judgements()
s3_judgement_reader = fst.FeatureSelectionTest('data/judge_features_avg_session3-d.csv')
s3_judgements = s3_judgement_reader.get_judgements()

s1_bert_reader = csvr.BertReader('data/BERT_similarity_s1.csv')
s1_bert_judgements = s1_bert_reader.get_judgements()

class JudgementHolder:
    def __init__(self, seed, re, cos_sim, bert_sim):
        self.seed = seed
        self.re = re
        self.cos_sim = cos_sim
        self.bert_sim = bert_sim

judgements = []

count = 0
for judgement in s1_judgements:
    for bert_judgement in s1_bert_judgements:
        if judgement.seed == bert_judgement.seed and '_' + judgement.reenactment == bert_judgement.reenactment:
            cos_sim = torch.nn.functional.cosine_similarity(torch.tensor(judgement.seed_avg), torch.tensor(judgement.re_avg), dim=0)
            plt.scatter(bert_judgement.judge_avg, cos_sim, c="blue", label='cos_sim')  # Blue for cos_sim
            plt.scatter(bert_judgement.judge_avg, bert_judgement.bert_cos, c="red", label='bert_cos')
            plt.title(f"seed = {judgement.seed} : re= {judgement.reenactment} ")
            plt.xlabel(f"cos_sim={cos_sim}")
            plt.ylabel(f"bert_cos ={bert_judgement.bert_cos}")
            plt.xticks([1, 2, 3, 4, 5])
            plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            plt.show()

print(f"num_matches={len(judgements)}")
judgements.append(JudgementHolder(judgement.seed, judgement.reenactment, cos_sim, bert_judgement.bert_cos))
