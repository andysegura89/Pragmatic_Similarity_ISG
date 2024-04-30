import sys
import csv
import torch
import scipy

'''
This code was used to check if normalizing the judgements would improve the score
as suggested by Dr. Fuentes. 
'''

seen_judgements = set([])


class Judgement:
    def __init__(self, seed_name, reenactment_name, seed_features, reenactment_features, avg_judge_score):
        self.seed_name = seed_name
        self.reenactment_name = reenactment_name
        self.seed_features = seed_features
        self.reenactment_features = reenactment_features
        self.avg_judge_score = avg_judge_score

    def normalize_judgement(self, global_feature_averages):
        for feature_idx, feature_value in enumerate(global_feature_averages):
            self.seed_features[feature_idx] -= feature_value
            self.reenactment_features[feature_idx] -= feature_value


def get_judgements_from_csv(csv_path):
    judgements = []
    with open(csv_path, 'r') as file:
        csv_reader = csv.reader(file)
        for idx, row in enumerate(csv_reader):
            if idx % 4 == 0:
                seed_name = row[1]
                reenactment_name = row[2]
                avg_score = float(row[3])
            if idx % 4 == 1:
                seed_features = [float(num) for num in row]
            if idx % 4 == 2:
                reenactment_features = [float(num) for num in row]
            if idx % 4 == 3:
                new_judgement = Judgement(seed_name, reenactment_name, seed_features, reenactment_features, avg_score)
                judgements.append(new_judgement)
    return judgements


def get_session_csv_path(session_id):
    if session_id == 'english-1':
        csv_path = 'data/judge_features_avg_session1.csv'
    elif session_id == 'english-2':
        csv_path = 'data/judge_features_avg_session2.csv'
    elif session_id == 'spanish-1':
        csv_path = 'data/judge_features_avg_session3.csv'
    else:
        print(f"{session_id} is not a valid session id\n" +
              "One of these values are expected: {english-1, english-2, spanish-1}")
        sys.exit()
    return csv_path


def get_judgements_from_session(session_id):
    csv_path = get_session_csv_path(session_id)
    return get_judgements_from_csv(csv_path)


def get_all_features_from_session(session_id):
    global seen_judgements
    csv_path = get_session_csv_path(session_id)
    judgements = get_judgements_from_csv(csv_path)
    features = []
    for judgement in judgements:
        if judgement.seed_name not in seen_judgements:
            features.append(judgement.seed_features)
            seen_judgements.add(judgement.seed_name)

        if judgement.reenactment_name not in seen_judgements:
            features.append(judgement.reenactment_features)
            seen_judgements.add(judgement.reenactment_name)
    return features


def get_global_features_averages(clips):
    num_clips = len(clips)
    feature_sums = [0 for _ in range(len(clips[0]))]
    for clip in clips:
        for feature_idx, feature_value in enumerate(clip):
            feature_sums[feature_idx] += feature_value
    return [fs/num_clips for fs in feature_sums]


def get_normalized_judgements(session_id):
    judgements = get_judgements_from_session(session_id)
    global_features = get_all_features_from_session(session_id)
    global_features_avgs = get_global_features_averages(global_features)
    for judgement in judgements:
        judgement.normalize_judgement(global_features_avgs)
    return judgements


def check_normalized_judgements_correlation(session_id):
    judgements = get_normalized_judgements(session_id)
    #judgements = get_judgements_from_session(session_id)
    cos_sims = []
    judge_avgs = []
    for judgement in judgements:
        seed = judgement.seed_features
        reenactment = judgement.reenactment_features
        cs = torch.nn.functional.cosine_similarity(torch.tensor(seed), torch.tensor(reenactment), dim=0)
        cos_sims.append(cs.item())
        judge_avgs.append(judgement.avg_judge_score)
    correlation = scipy.stats.pearsonr(cos_sims, judge_avgs)[0]
    print(f"correlation of normalized features for {session_id} session = {correlation}")


check_normalized_judgements_correlation('english-1')
check_normalized_judgements_correlation('english-2')
check_normalized_judgements_correlation('spanish-1')


