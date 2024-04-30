import torch
import scipy.stats
import matplotlib.pyplot as plt
import csv_reader
import feature_extractor
import csv
import os
import math
import librosa

winning_indexes = [0, 2, 41, 54, 63, 67, 102, 155, 159, 172, 173, 193, 197, 248, 261, 280, 286, 350, 358, 431, 459, 482,
                   488, 528, 603, 616, 618, 707, 708, 715, 717, 731, 792, 799, 804, 809, 824, 828, 870, 878, 880, 900,
                   903, 959, 45, 113, 120, 121, 204, 206, 246, 269, 411, 453, 510, 559, 602, 656, 666, 929, 972, 973,
                   1013, 1016, 13, 17, 105, 134, 136, 185, 188, 474, 578, 622, 651, 882, 925, 162, 410, 577, 628, 750,
                   758, 866, 869, 952, 963, 965, 50, 168, 436, 470, 513, 527, 557, 660, 732, 514, 661, 694, 698, 935,
                   937]

'''
This code was used to see the correlation between the machine learning models
and the judgement sessions scores. 
'''

def remove_losing_features(averages):
    winning_features = []
    for idx in winning_indexes:
        winning_features.append(averages[idx].item())
    return torch.tensor(winning_features)


class SimilarityCalculator:
    def __init__(self, seed_path, reenactment_path, session_csv, bundle, judge_ids, feature_selection=False):
        self.seed_path = seed_path
        self.reenactment_path = reenactment_path
        self.feature_selection = feature_selection
        self.num_layers = 24  # base = 12, large = 24, x-large = 48
        self.num_judges = len(judge_ids)
        self.num_errors = 0

        self.data_reader = csv_reader.JudgementDataReader(session_csv, judge_ids)
        self.judgement_data = self.data_reader.get_judgement_data()
        self.feature_extractor = feature_extractor.FeatureExtractor(bundle)

        self.judge_and_cos_sim = []
        self.highest_wav_vs_human_similarities = []
        self.judge_averages = []
        self.cos_sim_lists = [[] for _ in
                              range(self.num_layers)]  # Each list holds the cosine similarities calculated per layer.
        self.ed_lists = [[] for _ in range(self.num_layers)]
        self.judge_scores_lists = [[] for _ in range(self.num_judges)]
        self.all_errors = []
        self.overall_judge_avg_correlations = []
        self.duration_deltas = []

    def get_features_averages(self, transformation_layer):
        """
        This method converts a 3d tensor to a 2d n*764 tensor where
        n = the number of frames in the audio file and 764 is the number
        of features calculated per frame.
        :param transformation_layer: A 3d tensor representing a transformation layer of an audio file.
        :return: A 1d tensor with a length of 764 that represents the average
        value for each feature in the given audio file.
        """
        transformation_layer = torch.squeeze(transformation_layer, dim=0)
        averages = []
        for col_idx in range(transformation_layer.shape[1]):
            feature_sum = 0
            for row_idx in range(transformation_layer.shape[0]):
                feature_sum += transformation_layer[row_idx, col_idx].item()
            averages.append(feature_sum / transformation_layer.shape[0])
        return torch.tensor(averages)

    def run(self, plot_graphs=False, s3=False):
        for judgement_idx, judgement in enumerate(self.judgement_data):
            question_num, source1, source2, judge_avg, judge_scores = judgement
            if s3:
                source1 = source1[1:]  # Use for spanish data
                source1 = source1[:-1]
                source2 = source2[1:]
            print(f"judge_scores:{judge_scores}")
            print(judgement)
            print(self.seed_path + source1)
            print(self.reenactment_path + source2)
            # try:
            seed_layers = self.feature_extractor.get_transformation_layers(self.seed_path + source1)
            seed_duration = librosa.get_duration(path=self.seed_path + source1)
            print(f"seed_duration={seed_duration}")
            reenactment_layers = self.feature_extractor.get_transformation_layers(self.reenactment_path + source2)
            re_duration = librosa.get_duration(path=self.reenactment_path + source2)
            print(f"reenactment_duration={re_duration}")
            self.duration_deltas.append(abs(seed_duration - re_duration))
            for layer_idx in range(self.num_layers):
                seed_tensor = seed_layers[layer_idx]
                reenactment_tensor = reenactment_layers[layer_idx]
                seed_features_avg = self.get_features_averages(seed_tensor)
                reenactment_features_avg = self.get_features_averages(reenactment_tensor)
                if self.feature_selection:
                    seed_features_avg = remove_losing_features(seed_features_avg)
                    reenactment_features_avg = remove_losing_features(reenactment_features_avg)
                cos_sim = torch.nn.functional.cosine_similarity(seed_features_avg, reenactment_features_avg, dim=0)
                ed = math.dist(seed_features_avg, reenactment_features_avg)
                self.cos_sim_lists[layer_idx].append(cos_sim.item())
                self.ed_lists[layer_idx].append(ed)
                print(f"cos_sim = {cos_sim}")
            # except:
            #     print(f"error for question #{question_num}")
            #     self.num_errors += 1
            #     self.all_errors.append((source1, source2))
            #     continue

            self.judge_averages.append(judge_avg)
            for judge_idx in range(self.num_judges):
                self.judge_scores_lists[judge_idx].append(judge_scores[judge_idx])
            print("--------------------------------\n")

        print(f"num_errors: {self.num_errors}")
        if plot_graphs:
            self.plot_graphs()
        self.get_judge_correlations()
        print(f"all_errors={self.all_errors}")

    def run_duration_correlation(self, s3=False):
        for judgement_idx, judgement in enumerate(self.judgement_data):
            question_num, source1, source2, judge_avg, judge_scores = judgement
            if 'M3' not in source2 and 'M4' not in source2:
                print('discarding')
                continue
            if s3:
                source1 = source1[1:-1]  # Use for spanish data
                source2 = source2[1:]
            print(f"judge_scores:{judge_scores}")
            print(f"judge_avg:{round(judge_avg, 2)}")
            print(judgement)
            print(self.seed_path + source1)
            print(self.reenactment_path + source2)
            seed_duration = librosa.get_duration(path=self.seed_path + source1)
            re_duration = librosa.get_duration(path=self.reenactment_path + source2)
            print(f"seed_duration={seed_duration}")
            print(f"reenactment_duration={re_duration}")
            self.duration_deltas.append(abs(seed_duration - re_duration))
            self.judge_averages.append(round(judge_avg, 2))
            for judge_idx in range(self.num_judges):
                self.judge_scores_lists[judge_idx].append(judge_scores[judge_idx])
            print("--------------------------------\n")
        distance_correlation = scipy.stats.pearsonr(self.judge_averages, self.duration_deltas)
        print(f"distance_correlation={distance_correlation[0]}")

    def save_judges_data(self, s3=False):  # 24th layer only
        for judgement_idx, judgement in enumerate(self.judgement_data):
            question_num, source1, source2, judge_avg, judge_scores = judgement
            if s3:
                source1 = source1[1:]  # Use for spanish data
                source1 = source1[:-1]
                source2 = source2[1:]
            print(f"judge_scores:{judge_scores}")
            print(judgement)
            print(self.seed_path + source1)
            print(os.path.exists(self.seed_path + source1))
            print(self.reenactment_path + source2)
            print(os.path.exists(self.reenactment_path + source2))
            # try:
            seed_layers = self.feature_extractor.get_transformation_layers(self.seed_path + source1)
            reenactment_layers = self.feature_extractor.get_transformation_layers(self.reenactment_path + source2)
            seed_feature_avgs = self.get_features_averages(seed_layers[23])
            seed_feature_avgs = [x.item() for x in seed_feature_avgs]
            re_feature_avgs = self.get_features_averages(reenactment_layers[23])
            re_feature_avgs = [x.item() for x in re_feature_avgs]
            seed_duration = librosa.get_duration(path=self.seed_path + source1)
            re_duration = librosa.get_duration(path=self. reenactment_path + source2)
            self.write_features_csv(question_num, source1, source2, judge_avg, seed_feature_avgs, re_feature_avgs, seed_duration, re_duration)
            # except:
            #     print(f"error for question #{question_num}")
            #     self.num_errors += 1
            #     continue
            print("--------------------------------\n")
        print(f"num_errors: {self.num_errors}")

    def write_features_csv(self, question_num, source1, source2, judge_avg, seed_features, reenactment_features, seed_duration, re_duration):
        info_row = [question_num, source1, source2, judge_avg, seed_duration, re_duration]
        with open('data/judge_features_avg_session3-d.csv', 'a') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows([info_row, seed_features, reenactment_features, []])

    def get_judge_correlations(self):
        judge_correlations = [[] for _ in range(self.num_judges)]
        judge_ed_correlations = [[] for _ in range(self.num_judges)]
        for layer_idx in range(self.num_layers):
            overall_avg_corr = scipy.stats.pearsonr(self.judge_averages, self.cos_sim_lists[layer_idx])[0]
            print(f"layer{layer_idx + 1} overall_avg_correlation{overall_avg_corr}")
            self.overall_judge_avg_correlations.append(overall_avg_corr)
            for judge_idx in range(self.num_judges):
                correlation = scipy.stats.pearsonr(self.judge_scores_lists[judge_idx], self.cos_sim_lists[layer_idx])[0]
                ed_correlation = scipy.stats.pearsonr(self.judge_scores_lists[judge_idx], self.ed_lists[layer_idx])[0]
                judge_correlations[judge_idx].append(correlation)
                judge_ed_correlations[judge_idx].append(ed_correlation)
        print(f"judge_correlations: {judge_correlations}\n")
        print('------')
        for idx, correlations in enumerate(judge_correlations):
            print(f"judge: {idx}: {correlations}")
            print(f"max={max(correlations)}, layer={correlations.index(max(correlations)) + 1}")
            print(f"ed_correlations")
            print(
                f"max={max(judge_ed_correlations[idx])}, layer={judge_ed_correlations[idx].index(max(judge_ed_correlations[idx])) + 1}\n")

        for layer_idx in range(self.num_layers):
            layer_total = 0
            layer_ed_total = 0
            for idx, correlations in enumerate(judge_correlations):
                layer_total += correlations[layer_idx]
                layer_ed_total += judge_ed_correlations[idx][layer_idx]
            print(
                f"layer{layer_idx + 1} - average of correlations with every judges = {layer_total / len(judge_correlations)}")
            print(
                f"layer{layer_idx + 1} - ED average of correlations with every judges = {layer_ed_total / len(judge_correlations)}")

        for layer_idx in range(self.num_layers):
            print(
                f"layer{layer_idx + 1} - overall judge average correlation = {self.overall_judge_avg_correlations[layer_idx]}")

        distance_correlation = scipy.stats.pearsonr(self.judge_averages, self.duration_deltas)
        print(f"distance_correlation={distance_correlation}")

    def plot_graphs(self, plot_layers=True, plot_judges=True):
        judge_avg_correlations = []
        judge_avg_z_correlations = []
        frame_differences_correlations = []
        fd_ja_correlations = []
        if plot_layers:
            for i in range(self.num_layers):
                # cos_similarity compared to human judgement average scatter plot
                correlation = scipy.stats.pearsonr(self.judge_averages, self.cos_sim_lists[i])[0]
                correlation_z = scipy.stats.pearsonr(self.judge_averages_z_norm, self.cos_sim_lists[i])[0]
                correlation_fd = scipy.stats.pearsonr(self.frame_differences, self.cos_sim_lists[i])[0]
                judge_avg_correlations.append(correlation)
                judge_avg_z_correlations.append(correlation_z)
                frame_differences_correlations.append(correlation_fd)
                fd_ja_correlations.append((correlation + correlation_fd) / 2)
                # plt.scatter(self.judge_averages, self.cos_sim_lists[i], c="blue")
                # plt.title(f"Layer: {i + 1} - Cosine Similarity Vs Human Judgement r={round(correlation, 4)}")
                # plt.xlabel("Human Judgement Average Score")
                # plt.ylabel("HuBERT-Large Cosine Similarity")
                # plt.xticks([1, 2, 3, 4, 5])
                # plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                # plt.show()

        if plot_judges:
            # judge_score_correlations = []
            judge_counter = 0
            for judge_idx in range(self.num_judges):
                judge_counter += 1
                if judge_counter == 6:
                    judge_counter += 1
                for layer_idx in range(self.num_layers):
                    correlation = \
                    scipy.stats.pearsonr(self.judge_scores_lists[judge_idx], self.cos_sim_lists[layer_idx])[0]
                    correlation_z = scipy.stats.pearsonr(self.judge_averages_z_norm, self.cos_sim_lists[layer_idx])[0]
                    # judge_score_correlations.append(correlation)
                    # judge_avg_z_correlations.append(correlation_z)
                    plt.scatter(self.judge_averages, self.cos_sim_lists[layer_idx], c="blue")
                    plt.title(
                        f"Layer: {layer_idx + 1} - Cosine Similarity Vs Judge {judge_counter} r={round(correlation, 4)}")
                    plt.xlabel(f"Judge {judge_counter} Score")
                    plt.ylabel("HuBERT- Large Cosine Similarity")
                    plt.xticks([1, 2, 3, 4, 5])
                    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                    plt.show()

        print(f"ac={judge_avg_correlations}")
        print(f"zn={judge_avg_z_correlations}")
        print(f"fd={frame_differences_correlations}")
        print(f"fdja={fd_ja_correlations}")


s1_seed_path = "/Users/andy/Desktop/UTEP/Fall23/Research/similarity_experiment/seeds/"
s1_reenactment_path = "/Users/andy/Desktop/UTEP/Fall23/Research/similarity_experiment/reenactments/_"
s1_csv = "/Users/andy/Desktop/UTEP/Fall23/Research/similarity_experiment/judgment-data.csv"
s1_judge_ids = ['1', '2', '3', '4', '5', '7', '8', '9', '10']

s2_seed_path = "/Users/andy/Desktop/UTEP/Fall23/Research/similarity_session_2/EN-seeds/"
s2_reenactment_path = "/Users/andy/Desktop/UTEP/Fall23/Research/similarity_session_2/EN-reenactments/_"
s2_csv = "/Users/andy/Desktop/UTEP/Fall23/Research/similarity_session_2/EN-judgment-data-session2.csv"
s2_judge_ids = ['2', '3', '4', '5', '7', '8', '11', '12']

s3_seed_path = "/Users/andy/Desktop/UTEP/Fall23/Research/similarity_session_spanish/ES-seeds 2/"
s3_reenactment_path = "/Users/andy/Desktop/UTEP/Fall23/Research/similarity_session_spanish/ES-reenactments 2/"
s3_csv = "/Users/andy/Desktop/UTEP/Fall23/Research/similarity_session_spanish/ES-judgment-data.csv"
s3_judge_ids = ["3", "4", "5", "6", "7", "11"]

w2v_l = "wav2vec_l"
hubert_l = "hubert_l"
wavlm_l = "wavlm_l"
#SimilarityCalculator(s1_seed_path, s1_reenactment_path, s1_csv, hubert_l, s1_judge_ids).save_judges_data()
#SimilarityCalculator(s2_seed_path, s2_reenactment_path, s2_csv, hubert_l, s2_judge_ids).save_judges_data()
SimilarityCalculator(s3_seed_path, s3_reenactment_path, s3_csv, hubert_l, s3_judge_ids).save_judges_data(s3=True)
# /Users/andy/Desktop/UTEP/Fall 2023/Research/similarity_session_spanish/ES-seeds/ES_063l_5.wav
