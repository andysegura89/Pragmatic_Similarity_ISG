import csv
import torch
from playsound import playsound
import time

'''
Allows user to enter SWBD participant ids and finds typical utterances for them
'''

class Utterance:
    def __init__(self, id, path, features, cos_sim=None):
        self.id = id
        self.path = path
        self.features = features
        self.cos_sim = cos_sim

class TypicalUtteranceFinder:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.participants = {}
        self.participant_builder()
        self.centroids = {}
        self.find_centroids()
        self.print_participants()

    def participant_builder(self):
        with open(self.csv_path) as file:
            csv_reader = csv.reader(file)
            row_iter = 0
            for row in csv_reader:
                if row_iter == 0:
                    participant_id = row[0][72:76]
                    path = row[0]
                elif row_iter == 1:
                    features = [float(x) for x in row]
                elif row_iter == 2:
                    utterance = Utterance(participant_id, path, features)
                    if participant_id in self.participants:
                        self.participants[participant_id].append(utterance)
                    else:
                        self.participants[participant_id] = [utterance]
                    row_iter = -1
                row_iter += 1

    def find_centroids(self):
        for participant in self.participants:
            feature_sums = [0 for _ in range(len(self.participants[participant][0].features))]
            for utterance in self.participants[participant]:
                for feature_idx, feature in enumerate(utterance.features):
                    feature_sums[feature_idx] += feature
            feature_avgs = []
            for feature_sum in feature_sums:
                feature_avgs.append(feature_sum/len(feature_sums))
            self.centroids[participant] = feature_avgs

    def find_typical_utterances(self, particpant_id):
        cos_similarities = []
        centroid = torch.tensor(self.centroids[particpant_id])
        for utterance in self.participants[particpant_id]:
            features = torch.tensor(utterance.features)
            cos_sim = torch.nn.functional.cosine_similarity(centroid, features, dim=0)
            utterance.cos_sim = cos_sim
            cos_similarities.append(utterance)
        cos_similarities_sorted = sorted(cos_similarities, key=lambda utt: utt.cos_sim, reverse=True)
        return cos_similarities_sorted

    def print_participants(self):
        for participant in sorted(self.participants, key=lambda p_id: len((self.participants[p_id]))):
            print(f"{participant} : {len(self.participants[participant])}")

    def run(self):
        while True:
            input_text = input('enter particpant id: ')
            if input_text.lower() == 'exit': return
            if input_text.lower()[-2:] == '-s':
                self.play_participants_clip(input_text[:4])
                continue
            if input_text not in self.participants: continue
            sorted_cos_sims = self.find_typical_utterances(input_text)
            typical = sorted_cos_sims[0]
            least_typical = sorted_cos_sims[-1]
            print(f'playing typical utterance: cos_sim = {typical.cos_sim}')
            playsound(typical.path)
            time.sleep(1)
            print(f'playing least typical utterance {least_typical.cos_sim}')
            playsound(least_typical.path)

    def play_participants_clip(self, p_id):
        print(f'playing {p_id} clips')
        for utterance in self.participants[p_id]:
            playsound(utterance.path)
            will_continue = input('enter any button to continue, enter x to exit: ')
            if will_continue == 'x':
                return

clip_path = 'data/swbd-participant.csv'
tuf = TypicalUtteranceFinder(clip_path)
tuf.run()
