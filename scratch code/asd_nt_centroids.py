import csv
import torch

'''
Classifies ASD/NT based on which one is more similar to the ASD/NT group centroid
'''

def get_participant_id(file_path):
    return file_path.split('/')[9]

class Participant:
    def __init__(self, p_id):
        self.participant_id = p_id
        self.clips = []
        self.centroid = None

    def add_clip(self, clip_features):
        self.clips.append(clip_features)

    def add_centroid(self, centroid):
        self.centroid = centroid


class CentroidFinder:
    def __init__(self):
        self.csv_path = 'data/asdnt-nofs.csv'
        self.participants = {}
        self.asd_centroid = None
        self.nt_centroid = None
        self.global_centroid = None
        self.process_csv()
        self.find_participants_centroids()
        self.find_group_centroids()
        self.test_centroids()

    def process_csv(self):
        with open(self.csv_path, 'r') as file:
            csv_reader = csv.reader(file)
            for idx, row in enumerate(csv_reader):
                if idx % 3 == 0:
                    participant_id = get_participant_id(row[0])
                if idx % 3 == 1:
                    clip_features = [float(x) for x in row]
                if idx % 3 == 2:
                    if participant_id not in self.participants:
                        new_participant = Participant(participant_id)
                        self.participants[participant_id] = new_participant
                    self.participants[participant_id].add_clip(clip_features)

    def find_participants_centroids(self):
        for participant_id in self.participants:
            participant = self.participants[participant_id]
            centroid = self.get_centroid(participant.clips)
            participant.add_centroid(centroid)
            # print(f"participant: {participant_id}")
            # print(f"centroid: {centroid} \n")

    def find_group_centroids(self, exclude_participant=None):
        asd_clips = []
        nt_clips = []
        for participant_id in self.participants:
            if participant_id == exclude_participant: continue
            participant = self.participants[participant_id]
            for clip in participant.clips:
                if 'NT' in participant_id:
                    nt_clips.append(clip)
                else:
                    asd_clips.append(clip)
        self.asd_centroid = self.get_centroid(asd_clips)
        self.nt_centroid = self.get_centroid(nt_clips)
        if exclude_participant is None:
            self.global_centroid = self.get_centroid(asd_clips + nt_clips)

    def get_centroid(self, clips_list):
        feature_sums = [0 for _ in range(len(clips_list[0]))]
        for idx, clip in enumerate(clips_list):
            for feature_idx, feature_value in enumerate(clip):
                feature_sums[feature_idx] += feature_value
        return [x/len(clips_list) for x in feature_sums]

    def test_centroids(self):
        asd_correct = 0
        nt_correct = 0
        for participant_id in self.participants:
            self.find_group_centroids(exclude_participant=participant_id)
            asd_centroid = torch.tensor(self.asd_centroid)
            nt_centroid = torch.tensor(self.nt_centroid)
            centroid = self.participants[participant_id].centroid
            centroid = torch.tensor(centroid)
            asd_cos_sim = torch.nn.functional.cosine_similarity(centroid, asd_centroid, dim=0)
            nt_cos_sim = torch.nn.functional.cosine_similarity(centroid, nt_centroid, dim=0)
            winner = 'ASD' if asd_cos_sim > nt_cos_sim else 'NT'
            if winner in participant_id:
                if winner == 'ASD':
                    asd_correct += 1
                else:
                    nt_correct += 1
            print(f"participant_id = {participant_id}")
            print(f"asd_cos_sim = {asd_cos_sim}")
            print(f"nt_cos_sim = {nt_cos_sim}")
            print(f"winner = {winner} \n")
        print(f"asd_correct = {asd_correct}")
        print(f"nt_correct = {nt_correct}")


CentroidFinder()