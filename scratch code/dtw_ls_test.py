import csv
import sys
import scipy

'''
Tests the dtw f0 metric on the set of lexically similar judgements
'''

e1_csv = '/Users/andy/Desktop/UTEP/Fall23/Research/distance_metrics_results/dm_e1.csv'
e2_csv = '/Users/andy/Desktop/UTEP/Fall23/Research/distance_metrics_results/dm_e2.csv'
s1_csv = '/Users/andy/Desktop/UTEP/Fall23/Research/distance_metrics_results/dm_s1.csv'

e1_judge = '/Users/andy/Desktop/UTEP/Fall23/Research/similarity_experiment/judgment-data.csv'
e2_judge = '/Users/andy/Desktop/UTEP/Fall23/Research/similarity_session_2/EN-judgment-data-session2.csv'
s1_judge = '/Users/andy/Desktop/UTEP/Fall23/Research/similarity_session_spanish/ES-judgment-data.csv'

e1_num_judges = 9
e2_num_judges = 9
s1_num_judges = 6

def get_judgement_average(stimuli, path, num_judges):
    with open(path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if stimuli == row[1]:
                scores = [float(x) for x in row[4:4+num_judges]]
                return sum(scores)/len(scores)
    print(f'no match found {stimuli}')
    return None

judge_avgs = []
dtws = []
cepstral_distances = []
mfccs = []
mfccdtw = []

with open(e2_csv, 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        re = row[3]
        #if 'M3' in re or 'M4' in re:
        if 'M3' not in re and 'M4' not in re:
            stimuli = row[1]
            if stimuli == 'Stimuli': continue
            judge_avg = get_judgement_average(stimuli, e2_judge, e2_num_judges)
            if judge_avg is None: continue
            dtw = float(row[5])
            cep = float(row[4])
            mfcc = float(row[6])
            mfcc_dtw = float(row[7])
            judge_avgs.append(judge_avg)
            dtws.append(dtw)
            cepstral_distances.append(cep)
            mfccs.append(mfcc)
            mfccdtw.append(mfcc_dtw)


correlation_ceps = scipy.stats.pearsonr(cepstral_distances, judge_avgs)[0]
print(f"correlation cepstral distance = {correlation_ceps}")
correlation_dtw = scipy.stats.pearsonr(dtws, judge_avgs)[0] # change this
print(f"correlation dtw = {correlation_dtw}")
correlation_mfcc = scipy.stats.pearsonr(mfccs, judge_avgs)[0]
print(f"correlation mfcc independent dtw = {correlation_mfcc}")
correlation_mfcc_dtw = scipy.stats.pearsonr(mfccdtw, judge_avgs)[0]
print(f"correlation mfccdtw = {correlation_mfcc_dtw}")



