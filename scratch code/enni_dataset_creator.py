import csv
import os
import feature_extractor as fe
from typical_atypical_classifier import Participant, Clip
from pydub import AudioSegment
import json
import ast

'''
Creates the json files for the ENNI dataset
'''

enni_csv_path = '/Users/andy/Desktop/UTEP/Fall23/Research/enni/talkbank_childes.csv'
enni_clips_path = '/Users/andy/Desktop/UTEP/Fall23/Research/enni/enni_v2/ENNI_post/out/'

feature_extractor = fe.FeatureExtractor()

seen_ids = set()

def get_child_dictionary(data):
    for d in data:
        if d['role'] == 'Target_Child':
            return d
    return None

def convert_stereo_to_mono(input_file_path, new_file_path):
    # Load the stereo audio file
    stereo_audio = AudioSegment.from_file(input_file_path, format="mp3")

    # Convert to mono
    mono_audio = stereo_audio.set_channels(1)

    # Construct the new file path

    # Export the mono audio to the new file
    mono_audio.export(new_file_path, format="mp3")


def create_participant_from_child_id(child_id, group_label, age, gender):
    global feature_extractor, td_count
    directories = os.listdir(enni_clips_path)
    if child_id in directories:
        clips_path = enni_clips_path + child_id + '/'
        clips = os.listdir(clips_path)
        clips = [c for c in clips if c[-3:] == 'mp3' and 'mono_' not in c]
        clips_features = []
        for clip in clips:
            print(f"processing : {clips_path + clip}")
            mono_path = clips_path + 'mono_' + clip
            convert_stereo_to_mono(clips_path + clip, mono_path)
            try:
                tl_averages = feature_extractor.get_features_averages_from_fp(mono_path)
            except:
                print(f'error on clip {clips_path + clip}')
                continue
            tl = Clip(tl_averages, clip)
            clips_features.append(tl)
            # participant_id, group_label, clips, age=None, gender=None
        return Participant(child_id, group_label, clips_features, age=age, gender=gender)
    else:
        return None


sli_age_frequency = {4: 2, 5: 12, 6: 6, 7: 12, 8: 12, 9: 19, 10: 4}
td_age_frequency = {4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
td_participant_counter = 0
participants_data = []
for group_label in ['SLI', 'TD']:
    with open(enni_csv_path, 'r') as file:
        csv_reader = csv.reader(file)
        participants_list = []
        for row in csv_reader:
            if row[7] == group_label:
                metadata = ast.literal_eval(row[-1])
                child_data = get_child_dictionary(metadata)
                if child_data is None:
                    continue
                gender = child_data['sex']
                age_in_days = child_data['age_in_days']
                age = round(age_in_days / 365) if age_in_days is not None else None
                if group_label == 'TD':
                    if age not in sli_age_frequency: continue
                path = row[10]
                slash_split = path.split('/')
                child_id = slash_split[2].split('.')[0]
                if child_id in seen_ids:
                    continue
                else:
                    seen_ids.add(child_id)
                # participant_id, group_label, clips, age = None, gender = None
                participant = create_participant_from_child_id(child_id, group_label, age, gender)
                if participant is not None:
                    participants_list.append(participant)
    participants_data += [participant.to_dict() for participant in participants_list]
with open('data/ENNI-cleaner.json', 'w') as json_file:
    json.dump(participants_data, json_file, indent=4)


