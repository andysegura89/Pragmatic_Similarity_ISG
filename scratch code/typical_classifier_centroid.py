import json
from typical_atypical_classifier import Participant
import cosine_similarity as cs

'''
This is the code that was used to classify participants
based on the similarity to a typical group centroid
'''

layer_number = '24'
typical_label = 'NT'
atypical_label = 'ASD'
threshold = 0.94

with open('data/asd-nt.json', 'r') as json_file:
    participants_data = json.load(json_file)

def get_group_centroid(group):
    group_clips = []
    for p in group:
        group_clips += p.clips
    sums = [0 for _ in range(len(group_clips[0].get_transformation_layer(layer_number)))]
    for clip in group_clips:
        for idx, feature in enumerate(clip.get_transformation_layer(layer_number)):
            sums[idx] += feature
    return [x/len(group_clips) for x in sums]

def get_p_centroid(participant):
    feature_sums = [0 for _ in range(1024)]
    num_clips = 0
    for clip in participant.clips:
        num_clips += 1
        for idx, feature in enumerate(clip.get_transformation_layer(layer_number)):
            feature_sums[idx] += feature
    return [x/num_clips for x in feature_sums]

sli_age_frequency = {4: 2, 5: 12, 6: 6, 7: 12, 8: 12, 9: 19, 10: 4}
td_age_frequency = {4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
participants = [Participant.from_dict(data) for data in participants_data]
typical_participants = [p for p in participants if p.group_label == typical_label]
atypical_participants = [p for p in participants if p.group_label == atypical_label]
# new_tp = []
# for p in typical_participants:
#     if td_age_frequency[p.age] >= sli_age_frequency[p.age]:
#         continue
#     td_age_frequency[p.age] += 1
#     new_tp.append(p)
# typical_participants = new_tp
print(f"number of {typical_label} participants = {len(typical_participants)}")
print(f"number of {atypical_label} participants = {len(atypical_participants)}")

counters = {typical_label: 0, atypical_label: 0}
typical_results = []
atypical_results = []

for p in typical_participants:
    leave_one_out = [p2 for p2 in typical_participants if p2.participant_id != p.participant_id]
    td_centroid = get_group_centroid(leave_one_out)
    player_centroid = get_p_centroid(p)
    cos_sim = cs.get_cosine_similarity(td_centroid, player_centroid)
    counters[typical_label] += cos_sim
    typical_results.append(cos_sim > threshold)


td_centroid = get_group_centroid(typical_participants)

for p in atypical_participants:
    p_centroid = get_p_centroid(p)
    cos_sim = cs.get_cosine_similarity(p_centroid, td_centroid)
    counters[atypical_label] += cos_sim
    atypical_results.append(cos_sim < threshold)

avg_td_cos_sim = counters[typical_label] / len(typical_participants)
avg_sli_cos_sim = counters[atypical_label] / len(atypical_participants)

print(f"avg {typical_label} : {avg_td_cos_sim}")
print(f"avg {atypical_label}: {avg_sli_cos_sim}")

typical_winners = typical_results.count(True)
typical_losers = typical_results.count(False)
atypical_winners = atypical_results.count(True)
atypical_losers = atypical_results.count(False)

print(f"threshold = {threshold}")
print(f"{typical_label}_winners = {typical_winners}")
print(f"{typical_label}_losers = {typical_losers}")
print(f"{atypical_label}_winners = {atypical_winners}")
print(f"{atypical_label}_losers = {atypical_losers}")






