import feature_selection_test as fst
from scipy import stats
import torch
from sklearn.linear_model import LinearRegression
import numpy as np
from feature_selector import remove_losing_features
import csv

'''
This code was used to see how linear regression performed with judgement sessions data
'''

s1_judgement_reader = fst.FeatureSelectionTest('data/judge_features_avg_session1-d.csv')
s1_judgements = s1_judgement_reader.get_judgements()
s2_judgement_reader = fst.FeatureSelectionTest('data/judge_features_avg_session2-d.csv')
s2_judgements = s2_judgement_reader.get_judgements()
s3_judgetment_reader = fst.FeatureSelectionTest('data/judge_features_avg_session3-d.csv')
s3_judgements = s3_judgetment_reader.get_judgements()


s1_judge_avgs = []
s1_cos_sims_1024 = []
s1_cos_sims_103 = []
s2_judge_avgs = []
s2_cos_sims_1024 = []
s2_cos_sims_103 = []

dtw_results_s1 = []
dtw_results_s2 = []

with open('/Users/andy/Desktop/UTEP/Fall23/Research/distance_metrics_results/dm_e1.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for idx, row in enumerate(csv_reader):
        if idx == 0: continue
        dtw_results_s1.append(float(row[5]))

with open('/Users/andy/Desktop/UTEP/Fall23/Research/distance_metrics_results/dm_e2.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for idx, row in enumerate(csv_reader):
        if idx == 0: continue
        dtw_results_s2.append(float(row[5]))



for judgement in s1_judgements:
    s1_judge_avgs.append(judgement.judge_avg)
    seed_tensor = torch.tensor(judgement.seed_avg)
    re_tensor = torch.tensor(judgement.re_avg)
    cos_sim = torch.nn.functional.cosine_similarity(seed_tensor, re_tensor, dim=0)
    s1_cos_sims_1024.append(cos_sim.item())
    seed_tensor_103 = torch.tensor(remove_losing_features(judgement.seed_avg))
    re_tensor_103 = torch.tensor(remove_losing_features(judgement.re_avg))
    cos_sim_103 = torch.nn.functional.cosine_similarity(seed_tensor_103, re_tensor_103, dim=0)
    s1_cos_sims_103.append(cos_sim_103)

for judgement in s2_judgements:
    s2_judge_avgs.append(judgement.judge_avg)
    seed_tensor = torch.tensor(judgement.seed_avg)
    re_tensor = torch.tensor(judgement.re_avg)
    cos_sim = torch.nn.functional.cosine_similarity(seed_tensor, re_tensor, dim=0)
    s2_cos_sims_1024.append(cos_sim.item())
    seed_tensor_103 = torch.tensor(remove_losing_features(judgement.seed_avg))
    re_tensor_103 = torch.tensor(remove_losing_features(judgement.re_avg))
    cos_sim_103 = torch.nn.functional.cosine_similarity(seed_tensor_103, re_tensor_103, dim=0)
    s2_cos_sims_103.append(cos_sim_103)



s1_cos_sims_array_1024 = np.array(s1_cos_sims_1024).reshape(-1, 1)  # Reshape to 2D array
s1_cos_sims_array_103 = np.array(s1_cos_sims_103).reshape(-1, 1)
s1_judge_avgs_array = np.array(s1_judge_avgs)
s2_cos_sims_array_1024 = np.array(s2_cos_sims_1024).reshape(-1, 1)  # Reshape to 2D array
s2_cos_sims_array_103 = np.array(s2_cos_sims_103).reshape(-1, 1)
s2_judge_avgs_array = np.array(s2_judge_avgs)

s1_dtw_array = np.array(dtw_results_s1).reshape(-1, 1)
s2_dtw_array = np.array(dtw_results_s2).reshape(-1, 1)



model_1024 = LinearRegression()
model_1024.fit(s1_cos_sims_array_1024, s1_judge_avgs_array)
predictions_1024 = model_1024.predict(s2_cos_sims_array_1024)

model_103 = LinearRegression()
model_103.fit(s1_cos_sims_array_103, s1_judge_avgs_array)
predictions_103 = model_103.predict(s2_cos_sims_array_103)

model_dtw = LinearRegression()
model_dtw.fit(s1_dtw_array, s1_judge_avgs_array)
predictions_dtw = model_dtw.predict(s2_dtw_array)

errors_1024 = []
errors_103 = []
errors_dtw = []
for idx in range(len(predictions_1024)):
    errors_1024.append(abs(predictions_1024[idx]-s2_judge_avgs_array[idx]))
    errors_103.append(abs(predictions_103[idx]-s2_judge_avgs_array[idx]))
    errors_dtw.append(abs(predictions_dtw[idx]-s2_judge_avgs_array[idx]))

# print(f"errors_1024={errors_1024}")
# print(f"errors_103={errors_103}")

print("performing t test with 1024-103 data")

# Perform the matched-pair t-test
t_statistic, two_tail_p_value = stats.ttest_rel(errors_1024, errors_103)

# For a one-sided test, halve the p-value and compare it to your significance level
one_tail_p_value = two_tail_p_value / 2

# Assuming you are testing if mean of errors_array_1 < mean of errors_array_2
if one_tail_p_value < 0.05 and t_statistic < 0:
    print(f'Significant result: t-statistic = {t_statistic}, p-value = {one_tail_p_value}')
else:
    print(f'No significant result: t-statistic = {t_statistic}, p-value = {one_tail_p_value}')

print("performing t test with 103-DTW data")
# Perform the matched-pair t-test
t_statistic, two_tail_p_value = stats.ttest_rel(errors_103, errors_dtw)

# For a one-sided test, halve the p-value and compare it to your significance level
one_tail_p_value = two_tail_p_value / 2

# Assuming you are testing if mean of errors_array_1 < mean of errors_array_2
if one_tail_p_value < 0.05 and t_statistic < 0:
    print(f'Significant result: t-statistic = {t_statistic}, p-value = {one_tail_p_value}')
else:
    print(f'No significant result: t-statistic = {t_statistic}, p-value = {one_tail_p_value}')

print(sum(errors_1024)/len(errors_1024))
print(sum(errors_103)/len(errors_103))
print(sum(errors_dtw)/len(errors_dtw))
