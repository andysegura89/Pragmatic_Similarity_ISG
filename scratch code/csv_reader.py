import csv
import sys

import numpy as np
import os

'''
This code was used to read different csv files 
'''

class JudgementDataReader:
    def __init__(self, csv_path, judge_ids):
        self.csv_path = csv_path
        self.judge_ids = judge_ids
        self.num_judges = len(self.judge_ids)
        self.labels_to_index = {}
        self.judgements = []
        self.judge_scores = self.get_judge_scores()
        self.judge_means = self.get_judge_means()
        self.judge_sds = self.get_judge_sds()
        print(f"judge_means={self.judge_means}")
        print(f"judge_sds={self.judge_sds}")

    def get_judge_average(self, csv_row):
        total = 0.0
        for judge_idx, judge_id in enumerate(self.judge_ids):
            total += float(csv_row[self.labels_to_index[judge_id]])
        return total / self.num_judges

    def get_judge_scores_per_question (self, csv_row):
        scores = []
        for judge_idx, judge_id in enumerate(self.judge_ids):
            scores.append(float(csv_row[self.labels_to_index[judge_id]]))
        return scores

    def get_judge_scores(self):
        judge_scores = [[] for _ in range(self.num_judges)]
        with open(self.csv_path, 'r') as file:
            csv_reader = csv.reader(file)
            for i, row in enumerate(csv_reader):
                if i < 2:
                    continue
                for j in range(self.num_judges):
                    judge_scores[j].append(float(row[int(j + 4)]))
        return judge_scores

    def get_judge_means(self):
        return np.mean(self.judge_scores, axis=1)

    def get_judge_sds(self):
        return np.std(self.judge_scores, axis=1)

    def get_judgement_data(self):
        with open(self.csv_path, 'r') as file:
            csvreader = csv.reader(file)
            for i, row in enumerate(csvreader):
                if i <= 1:
                    for j, col in enumerate(row):
                        self.labels_to_index[col] = j
                else:
                    # if row[self.labels_to_index['Label']] == 'K':
                    #     continue  \ufeffQuestions'
                    self.judgements.append((row[self.labels_to_index['\ufeffQuestions']],
                                            row[self.labels_to_index['Source1']],
                                            row[self.labels_to_index['Source2']],
                                            self.get_judge_average(row),
                                            self.get_judge_scores_per_question(row)))
        print(f"len: {len(self.judgements)}")
        return self.judgements


class AudioClip:
    def __init__(self, response_id, file_name, file_path, judge_scores):
        self.response_id = response_id
        self.file_name = file_name
        self.file_path = file_path
        self.judge_scores = [int(x) for x in judge_scores]


class SarenneJudgement:
    def __init__(self, clips, golden_key):
        self.clips = clips
        self.golden_key = golden_key


class SarenneReader:
    def __init__(self, csv_path, clips_path):
        self.csv_path = csv_path
        self.clips_path = clips_path
        self.judgements = []

    def get_judgement_data(self):
        with open(self.csv_path, 'r') as file:
            csv_reader = csv.reader(file)
            current_directory = None
            golden_key = None
            for idx, row in enumerate(csv_reader):
                if idx == 0: continue
                if row[0] in ['sw3068_8']:
                    print('discarding directory')
                    continue
                if not row[0] == current_directory:
                    current_directory = row[0]
                    clips = []
                if not os.path.exists(self.clips_path + current_directory):
                    print(current_directory + " does not exist")
                    continue
                response_id = row[1]
                file_path, file_name, is_golden_key = self.find_file(current_directory, response_id)
                if is_golden_key:
                    if golden_key:
                        print('double golden')
                        sys.exit()
                    golden_key = response_id
                row_judgements = self.get_row_judgements(row[2:])
                clip = AudioClip(response_id, file_name, file_path, row_judgements)
                clips.append(clip)
                if len(clips) == 5: # 4 for 2021 data, 5 for 2023
                    if not golden_key:
                        print('no golden key ')
                        sys.exit()
                    judgement = SarenneJudgement(clips, golden_key)
                    self.judgements.append(judgement)
                    golden_key = None

        print(f"number_of_judgements = {len(self.judgements)}")
        return self.judgements

    def get_row_judgements(self, row):
        judgements = []
        for value in row:
            if value:
                judgements.append(value)
        return judgements

    def find_file(self, current_directory, response_id):
        # print(current_directory)
        # print(response_id)
        directory = os.listdir(self.clips_path + current_directory)
        for file_name in directory:
            # first if for 2021, second for 2023
            #if file_name[0] == response_id and 'indv' in file_name:
            if response_id + '_indv' in file_name:

                #golden_key = file_name[2:-10] in current_directory
                golden_key = response_id == '0'
                return self.clips_path + current_directory + '/' + file_name, file_name, golden_key

class BertJudgement:
    def __init__(self, seed, re, judge_avg, bert_cos):
        self.seed = seed
        self.reenactment = re
        self.judge_avg = judge_avg
        self.bert_cos = bert_cos

class BertReader:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.judgements = []
        self.read_judgements()
        print(len(self.judgements))
        print(self.judgements[2].bert_cos)

    def get_judgements(self):
        return self.judgements

    def read_judgements(self):
        with open(self.csv_path, 'r') as file:
            csv_reader = csv.reader(file)
            for idx, row in enumerate(csv_reader):
                if idx == 0: continue
                seed = row[0]
                re = row[1]
                judge_scores = [float(num) for num in row[2:11]]
                avg = sum(judge_scores) / len(judge_scores)
                bert_cos = float(row[11])
                judgement = BertJudgement(seed, re, avg, bert_cos)
                self.judgements.append(judgement)

