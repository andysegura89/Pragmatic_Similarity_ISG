import csv
import torch
import itertools
import scipy.stats
import numpy as np
from sklearn.linear_model import LinearRegression
import time

'''
This is the code that was used to perform ten-fold feature selection
you can choose which session to use on line 50. csv files are
stored in the data folder.
'''


class Judgement:
    def __init__(self, question_num, seed_name, reenactment_name, judge_avg, seed_avgs, re_avgs):
        self.question_num = question_num
        self.seed_name = seed_name
        self.reenactment_name = reenactment_name
        self.judge_avg = judge_avg
        self.seed_avgs_list = seed_avgs
        self.re_avgs_list = re_avgs

class HighestCorrelationPerList:
    def __init__(self, correlation, idx_1, idx_2, idx_3=None, idx_4=None):
        self.correlation = correlation
        self.idx_1 = idx_1
        self.idx_2 = idx_2
        self.idx_3 = idx_3
        self.idx_4 = idx_4


class FeatureSelection:
    def __init__(self, num_folds=10, num_features=1024, num_winning_features=100, subset_range=10, subset_size=2):
        self.num_folds = num_folds
        self.num_features = num_features
        self.num_winning_features = num_winning_features
        self.judgements = self.read_judgements()
        self.feature_indexes = [idx for idx in range(0, num_features)]
        self.fold_length = len(self.judgements) // num_folds
        self.all_feature_subsets = self.get_every_subset(subset_range, subset_size)
        self.lr_correlations = []
        self.sub_correlations = []
        self.winning_features_per_fold_lr = []
        self.winning_features_per_fold_sub = []

    def read_judgements(self):
        judgements = []
        with open('data/judge_features_avg_session3.csv', 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            counter = 1
            for row in csv_reader:
                if counter == 1:
                    question_num = row[0]
                    seed_name = row[1]
                    re_name = row[2]
                    judge_avg = float(row[3])
                if counter == 2:
                    seed_avg = [float(x) for x in row]
                if counter == 3:
                    re_avg = [float(x) for x in row]
                    judgements.append(Judgement(question_num, seed_name, re_name, judge_avg, seed_avg, re_avg))
                if counter == 4:
                    counter = 0
                counter += 1
        return judgements

    def get_every_subset(self, subset_range, subset_size):
        all_subsets = []
        for idx in range(0, self.num_features, subset_range):
            subsets = list(itertools.combinations(self.feature_indexes[idx:idx+subset_range], subset_size))
            all_subsets.append(subsets)
        return all_subsets

    def test_all_folds(self):
        for fold_idx in range(self.num_folds):
            train_judgements, test_judgements = self.get_train_test_judgements(fold_idx)
            self.process_fold_sub(train_judgements, test_judgements)
            self.process_fold_lr(train_judgements, test_judgements)
        self.print_final_results()

    def process_fold_sub(self, train_judgements, test_judgements):
        winning_features = self.get_winning_features_sub(train_judgements)
        self.winning_features_per_fold_sub.append(winning_features)
        sub_correlation = self.test_fold(winning_features, test_judgements)
        self.sub_correlations.append(sub_correlation)

    def process_fold_lr(self, train_judgements, test_judgements):
        feature_coefficients = []
        for feature_idx in self.feature_indexes:
            coefficient = self.test_feature(feature_idx, train_judgements)
            feature_coefficients.append((feature_idx, coefficient))
        feature_coefficients_sorted = sorted(feature_coefficients, key=lambda t: t[1], reverse=True)
        winning_features = \
            [feature_idx for feature_idx, _ in feature_coefficients_sorted[0:self.num_winning_features]]
        test_correlation = self.test_fold(winning_features, test_judgements)
        self.lr_correlations.append(test_correlation)
        self.winning_features_per_fold_lr.append(winning_features)
        print(f"cos_correlation={test_correlation}")
        print("-------------------")

    def get_winning_features_sub(self, train_judgements):
        highest_correlations = self.run_all_subsets(train_judgements)
        hcl = sorted(highest_correlations, key=lambda t: t.correlation, reverse=True)
        winning_features_sub = []
        for hc in hcl[0:self.num_winning_features//2]:
            winning_features_sub.append(hc.idx_1)
            winning_features_sub.append(hc.idx_2)
        return winning_features_sub

    def run_all_subsets(self, train):
        highest_correlations = []
        i = 0
        for list_of_subset in self.all_feature_subsets:
            i += 1
            #print(f"subset{i}")
            highest_correlation_per_list = None
            for subset in list_of_subset:
                correlation = self.process_feature_indexes(subset, train)
                if not highest_correlation_per_list or correlation > highest_correlation_per_list.correlation:
                    highest_correlation_per_list = HighestCorrelationPerList(correlation, subset[0], subset[1])
            highest_correlations.append(highest_correlation_per_list)
        return highest_correlations

    def process_feature_indexes(self, subset, train_judgements): #train
        cos_sims = []
        judges_avgs = []
        for judgement in train_judgements:
            seed_avg = judgement.seed_avgs_list
            seed_avg = self.remove_losing_features(seed_avg, subset)
            re_avg = judgement.re_avgs_list
            re_avg = self.remove_losing_features(re_avg, subset)
            judge_avg = judgement.judge_avg
            cos_sim = torch.nn.functional.cosine_similarity(seed_avg,re_avg, dim=0)
            cos_sims.append(cos_sim)
            judges_avgs.append(judge_avg)
        correlation = scipy.stats.pearsonr(cos_sims, judges_avgs)[0]
        return correlation

    def get_train_test_judgements(self, fold_idx):
        test_judgements = []
        train_judgements = []
        fold_start_pos = fold_idx * self.fold_length
        test_indexes = [x for x in range(fold_start_pos, fold_start_pos + self.fold_length)]
        for judgement_idx, judgement in enumerate(self.judgements):
            if judgement_idx in test_indexes:
                test_judgements.append(judgement)
            else:
                train_judgements.append(judgement)
        print(f"fold_{fold_idx + 1}")
        print(f"testing_features = {test_indexes}")
        print(f"len(train) = {len(train_judgements)}")
        print(f"len(test) = {len(test_judgements)}")
        return train_judgements, test_judgements

    def test_feature(self, feature_idx, train_judgements):
        feature_deltas = []
        judgement_avgs = []
        for judgement in train_judgements:
            delta = judgement.re_avgs_list[feature_idx] - judgement.seed_avgs_list[feature_idx]
            feature_deltas.append(delta)
            judgement_avgs.append(judgement.judge_avg)
        feature_deltas = np.array(feature_deltas).reshape((-1, 1))
        judgement_avgs = np.array(judgement_avgs)
        model = LinearRegression()
        model.fit(feature_deltas, judgement_avgs)
        return model.score(feature_deltas, judgement_avgs)  # Returns the coefficient of determination

    def test_fold(self, winning_features, test_judgements):
        cos_similarities = []
        judge_avgs = []
        deltas = []
        for judgement in test_judgements:
            seed_winners = self.remove_losing_features(judgement.seed_avgs_list, winning_features)
            re_winners = self.remove_losing_features(judgement.re_avgs_list, winning_features)
            seed_re_deltas_list = self.get_seed_re_deltas_list(seed_winners, re_winners)
            deltas.append(seed_re_deltas_list)
            cos_sim = torch.nn.functional.cosine_similarity(seed_winners, re_winners, dim=0)
            cos_similarities.append(cos_sim)
            judge_avgs.append(judgement.judge_avg)
        correlation = scipy.stats.pearsonr(cos_similarities, judge_avgs)[0]
        model = LinearRegression()
        model.fit(deltas, judge_avgs)
        print(f"lr_coefficient_determination={model.score(deltas, judge_avgs)}")
        return correlation

    def remove_losing_features(self, feature_averages, winning_feature_indexes):
        winning_features = []
        winning_feature_indexes = sorted(winning_feature_indexes)
        for feature in winning_feature_indexes:
            winning_features.append(feature_averages[feature])
        return torch.tensor(winning_features)

    def get_seed_re_deltas_list(self, seed_avgs_list, re_avgs_list):
        combination = []
        for feature in range(len(seed_avgs_list)):
            combination.append(re_avgs_list[feature] - seed_avgs_list[feature])
        return combination

    def print_final_results(self):
        print(f"avg_correlation_lr = {sum(self.lr_correlations) / len(self.lr_correlations)}\n")
        print(f"avg_correlation_sub = {sum(self.sub_correlations) / len(self.sub_correlations)}")
        print('\nwinning lr features')
        self.get_winning_features_all_folds(self.winning_features_per_fold_lr)
        print('\nwinning sub features')
        self.get_winning_features_all_folds(self.winning_features_per_fold_sub)
        print('\nwinning sub + lr features')
        self.get_winning_features_all_folds(self.winning_features_per_fold_sub + self.winning_features_per_fold_lr)


    def get_winning_features_all_folds(self, winning_features):
        frequency_dictionary = {x: 0 for x in range(self.num_features)}
        for features_list in winning_features:
            for feature in features_list:
                frequency_dictionary[feature] += 1
        sorted_feature_frequency = sorted(frequency_dictionary.items(), key=lambda x: x[1], reverse=True)
        best_features = []
        for feature, frequency in sorted_feature_frequency:
            if frequency >= len(winning_features) // 2:
                best_features.append(feature)
        print(f"best_features = {best_features}")
        print(f"num_best_features = {len(best_features)}")

start = time.time()
FeatureSelection(subset_range=10).test_all_folds()
print(f"time = {time.time()-start}")
