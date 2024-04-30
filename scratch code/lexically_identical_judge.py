import csv
import feature_extractor as fe
from feature_selector import remove_losing_features
from feature_selector import remove_spanish_losing_features
import torch
import scipy

'''
this was the code used to check the lexically distinct and lexically identical
judgements. e1=english 1 session, e2 = english2 session, s1 = spanish 1 (aka session 3)
I think I might have gotten the criteria for deciding lexically distinct
on line 75 backwards
'''

e1_judge = '/Users/andy/Desktop/UTEP/Fall23/Research/similarity_experiment/judgment-data.csv'
e2_judge = '/Users/andy/Desktop/UTEP/Fall23/Research/similarity_session_2/EN-judgment-data-session2.csv'
s1_judge = '/Users/andy/Desktop/UTEP/Fall23/Research/similarity_session_spanish/ES-judgment-data.csv'

e1_seeds = '/Users/andy/Desktop/UTEP/Fall23/Research/similarity_experiment/seeds/'
e2_seeds = '/Users/andy/Desktop/UTEP/Fall23/Research/similarity_session_2/EN-seeds/'
s1_seeds = '/Users/andy/Desktop/UTEP/Fall23/Research/similarity_session_spanish/ES-seeds/'

e1_res = '/Users/andy/Desktop/UTEP/Fall23/Research/similarity_experiment/reenactments/'
e2_res = '/Users/andy/Desktop/UTEP/Fall23/Research/similarity_session_2/EN-reenactments/'
s1_res = '/Users/andy/Desktop/UTEP/Fall23/Research/similarity_session_spanish/ES-reenactments/'


clip_path_finder = {e1_judge:(e1_seeds, e1_res), e2_judge:(e2_seeds, e2_res), s1_judge:(s1_seeds, s1_res)}


e1_num_judges = 9
e2_num_judges = 8
s1_num_judges = 6

e1_cos_sims = []
e2_cos_sims = []
s1_cos_sims = []

e1_judge_scores = [[] for _ in range(e1_num_judges)]
e2_judge_scores = [[] for _ in range(e2_num_judges)]
s1_judge_scores = [[] for _ in range(s1_num_judges)]

e2_judge_avgs = []

feature_extractor = fe.FeatureExtractor()

def find_cos_sim(seed, re, path):
    seed_path, re_path = clip_path_finder[path]
    # print(seed_path + seed)
    # print(re_path + re)
    # seed = seed[1:-1]
    # re = re[1:]
    re = '_' + re
    seed_features = feature_extractor.get_24th_layer_features_averages(seed_path + seed)
    re_features = feature_extractor.get_24th_layer_features_averages(re_path + re)
    # if path == s1_judge:
    #     seed_features = remove_spanish_losing_features(seed_features)
    #     re_features = remove_spanish_losing_features(re_features)
    # else:
    #     seed_features = remove_losing_features(seed_features)
    #     re_features = remove_losing_features(re_features)
    cos_sim = torch.nn.functional.cosine_similarity(torch.tensor(seed_features), torch.tensor(re_features), dim=0)
    return cos_sim




for path in [e2_judge]:
    with open(path, 'r') as file:
        csv_reader = csv.reader(file)
        for idx, row in enumerate(csv_reader):
            if idx < 2: continue
            seed = row[2]
            re = row[3]
            if 'M3' not in re and 'M4' not in re: continue #lexically distinct
            #if 'M3' in re or 'M4' in re: continue #lexically similar
            print(row)
            cos_sim = find_cos_sim(seed, re, path)
            # if path == e1_judge:
            #     e1_cos_sims.append(cos_sim)
            #     for judge_idx in range(e1_num_judges):
            #         e1_judge_scores[judge_idx].append(float(row[judge_idx + 4]))
            if path == e2_judge:
                e2_cos_sims.append(cos_sim)
                judge_scores = []
                for judge_idx in range(e2_num_judges):
                    e2_judge_scores[judge_idx].append(float(row[judge_idx + 4]))
                    judge_scores.append(float(row[judge_idx + 4]))
                e2_judge_avgs.append(sum(judge_scores)/len(judge_scores))
            # if path == s1_judge:
            #     s1_cos_sims.append(cos_sim)
            #     for judge_idx in range(s1_num_judges):
            #         s1_judge_scores[judge_idx].append(float(row[judge_idx + 4]))
        correlations = []
        # if path == e1_judge:
        #     print("e1 correlations")
        #     for judge_idx, judge_list in enumerate(e1_judge_scores):
        #         correlation = scipy.stats.pearsonr(e1_cos_sims, judge_list)[0]
        #         correlations.append(correlation)
        #         print(f"judge: {judge_idx + 1} : correlation={correlation}")
        #     print(f"average of correlations for session 1 = {sum(correlations)/len(correlations)}")
        if path == e2_judge:
            print("e2 correlations")
            for judge_idx, judge_list in enumerate(e2_judge_scores):
                correlation = scipy.stats.pearsonr(e2_cos_sims, judge_list)[0]
                correlations.append(correlation)
                print(f"judge: {judge_idx + 1} : correlation={correlation}")
            print(f"average of correlations for session 2 = {sum(correlations)/len(correlations)}")
            avg_correlation = scipy.stats.pearsonr(e2_cos_sims, e2_judge_avgs)
            print(f"correlation with average score session 2 = {avg_correlation}")

            all_inter_judge_correlations = []
            for s_idx, judge_list in enumerate(e2_judge_scores):
                for e_idx in range(s_idx + 1, e2_num_judges):
                    print(f"comparing judge {s_idx+1} and judge {e_idx+1}")
                    second_judge_list = e2_judge_scores[e_idx]
                    correlation = scipy.stats.pearsonr(judge_list, second_judge_list)[0]
                    all_inter_judge_correlations.append(correlation)
                    print(f"comparing judge {s_idx+1} and judge {e_idx+1}: correlation = {correlation}")
            print(f"inter judge correlation averages = {sum(all_inter_judge_correlations)/len(all_inter_judge_correlations)}")




        if path == s1_judge:
            print("s1 correlations")
            for judge_idx, judge_list in enumerate(s1_judge_scores):
                correlation = scipy.stats.pearsonr(s1_cos_sims, judge_list)[0]
                correlations.append(correlation)
                print(f"judge: {judge_idx + 1} : correlation={correlation}")
            print(f"average of correlations for session 3 = {sum(correlations)/len(correlations)}")



