import os
from typical_atypical_classifier import Participant, Clip
from feature_extractor import FeatureExtractor
import json

'''
builds json file for ASD-NT dataset
'''

asd_path = "/Users/andy/Desktop/UTEP/Fall23/Research/phase2/ASD-Mono/"
nt_path = "/Users/andy/Desktop/UTEP/Fall23/Research/phase2/NT-Mono/"
participant_ids = ["CASD001", "CASD003", "CASD005", "CASD006", "CASD008", "CASD009", "CASD011",
                        "CASD012", "CASD014", "CASD015", "CASD018", "CASD019", "CASD020", "CASD021",
                        "CNT001", "CNT004", "CNT008", "CNT012", "CNT014", "CNT017", "CNT018",
                        "CNT022", "CNT025", "CNT026", "CNT027", "CNT028", "CNT031", "CNT033"]
feature_extractor = FeatureExtractor()

def create_participants(group_directory_path):
    """
    Populates the self.participant_data dictionary by first iterating through
    every audio file for the ASD or NT groups. It then gets features averages
    for each audio file. Each audio file has n features averages where n is the
    amount of transformation layers the machine learning model outputs.
    :param group_directory_path: the name of the directory where either the ASD or NT participant
    audio files are stored. The audio files should be stored in the following format.
    group_directory_path/participant_id/audio_clip.wav
    (e.g. NT-Mono/CNT001/CNT001_07192017_S2_1_mono.wav)
    """
    participants_list = []
    print(os.listdir(group_directory_path))
    # iterate through all the participants in either the ASD or NT folders.
    for participant in os.listdir(group_directory_path):
        if participant not in participant_ids:  # in case file is .DS_Store
            continue
        # iterate through each audio file for the participant.
        print(os.listdir(group_directory_path + participant))
        clips_list = []
        for clip in os.listdir(group_directory_path + participant):
            if clip[0] != 'C':
                continue
            clip_path = group_directory_path + participant + '/' + clip
            features_averages = feature_extractor.get_features_averages_from_fp(clip_path)
            clips_list.append(Clip(features_averages, clip))
        participants_list.append(Participant(participant, 'ASD' if 'ASD' in group_directory_path else 'NT', clips_list))
    return participants_list


asd_participants = create_participants(asd_path)
nt_participants = create_participants(nt_path)
all_participants = asd_participants + nt_participants

participants_data = [participant.to_dict() for participant in all_participants]
with open('data/asd-nt.json', 'w') as json_file:
    json.dump(participants_data, json_file, indent=4)



