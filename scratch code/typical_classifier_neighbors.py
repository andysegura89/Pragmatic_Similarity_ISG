import sys
import json
from typical_atypical_classifier import Participant
import cosine_similarity as cs

'''
This is the code used to classify participants based on the average
distance to 3 typical clips
'''

class TypicalNeighborsClassifier:
    def __init__(self, participants_list, typical_label, atypical_label, threshold, k=3, layer_number='24'):
        self.participants_list = participants_list
        self.typical_participants = [p for p in participants_list if p.group_label == typical_label]
        self.atypical_participants = [p for p in participants_list if p.group_label == atypical_label]
        self.typical_label = typical_label
        self.atypical_label = atypical_label
        self.k = k
        self.layer_number = layer_number
        self.seen_ids = set()
        self.min_distance_sum_typical = 0
        self.min_distance_sum_atypical = 0
        self.min_distance_avg_atypical = 0
        self.min_distance_avg_typical = 0
        self.typical_clips = []
        self.threshold = threshold
        self.num_typical_winners = 0
        self.num_typical_losers = 0
        self.num_atypical_winners = 0
        self.num_atypical_losers = 0

        for p in self.typical_participants:
            self.typical_clips += p.clips

    def process_participants(self):
        for participant in self.atypical_participants + self.typical_participants:
            if participant.participant_id in self.seen_ids:
                print(f"Duplicate ID: {participant.participant_id}")
                sys.exit()
            self.process_single_participant(participant)
            self.seen_ids.add(participant.participant_id)
        avg_total_cos_sim_typical = self.min_distance_sum_typical/len(self.typical_participants)
        avg_total_cos_sim_atypical = self.min_distance_sum_atypical/len(self.atypical_participants)
        print(f"avg_total distance for all {self.typical_label} participants == {avg_total_cos_sim_typical}")
        print(f"avg_total distance for all {self.atypical_label} participants == {avg_total_cos_sim_atypical}")
        print(f"threshold = {self.threshold}")
        print(f"num_{self.typical_label}_winners = {self.num_typical_winners}")
        print(f"num_{self.typical_label}_losers = {self.num_typical_losers}")

        print(f"num_{self.atypical_label}_winners = {self.num_atypical_winners}")
        print(f"num_{self.atypical_label}_losers = {self.num_atypical_losers}")


    def process_single_participant(self, participant):
        k_closest_neighbors = []
        for clip in participant.clips:
            k_closest_neighbors += self.process_single_clip(clip, participant.participant_id, participant.group_label == self.typical_label)
        avg_cos_sim = sum(k_closest_neighbors)/len(k_closest_neighbors)
        if participant.group_label == self.typical_label:
            self.min_distance_sum_typical += avg_cos_sim
            if avg_cos_sim < self.threshold:
                self.num_typical_winners += 1
            else:
                self.num_typical_losers += 1
        else:
            self.min_distance_sum_atypical += avg_cos_sim
            if avg_cos_sim > self.threshold:
                self.num_atypical_winners += 1
            else:
                self.num_atypical_losers += 1
        print(f"participant {participant.participant_id} group {participant.group_label} avg distance of {self.k} neighbors = {avg_cos_sim}")

    def process_single_clip(self, atypical_clip, participant_id, typical=False):
        distance_list = []
        if typical:
            participants_to_compare = [p for p in self.typical_participants if p.participant_id != participant_id]
        else:
            participants_to_compare = self.typical_participants
        clips_to_compare = []
        for p in participants_to_compare:
            clips_to_compare += p.clips
        for typical_clip in clips_to_compare:
            cos_sim = cs.get_cosine_similarity(
                atypical_clip.get_transformation_layer('24'),
                typical_clip.get_transformation_layer('24')
            )
            distance_list.append(1 - cos_sim)
        distance_list.sort()
        return distance_list[:self.k]


typical_label = 'TD'
atypical_label = 'SLI'

with open('data/ENNI-cleaner.json', 'r') as json_file:
    participants_data = json.load(json_file)

participants = [Participant.from_dict(data) for data in participants_data]
typical_participants = [p for p in participants if p.group_label == typical_label]
atypical_participants = [p for p in participants if p.group_label == atypical_label]

# sli_age_frequency = {4: 2, 5: 12, 6: 6, 7: 12, 8: 12, 9: 19, 10: 4}
# td_age_frequency = {4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
# participants = [Participant.from_dict(data) for data in participants_data]

# new_tp = []
# for p in typical_participants:
#     if td_age_frequency[p.age] >= sli_age_frequency[p.age]:
#         continue
#     td_age_frequency[p.age] += 1
#     new_tp.append(p)
# typical_participants = new_tp

print(f"number of {typical_label}_participants = {len(typical_participants)}")
print(f"number of {atypical_label}_participants = {len(atypical_participants)}")
#typical_participants = typical_participants[:len(atypical_participants)]

TypicalNeighborsClassifier(typical_participants + atypical_participants, typical_label, atypical_label, 0.13).process_participants()

