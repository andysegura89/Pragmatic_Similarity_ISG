import csv
import os
import sys

import soundfile as sf

'''
This code was used to oranize the switchboard data to keep track of the
participants that were in the conversations and create clips of them speaking
'''

caller_csv = '/Users/andy/Desktop/UTEP/Fall23/Research/en-swbd/ldc-docs/caller_tab.csv'
call_conversation_csv = '/Users/andy/Desktop/UTEP/Fall23/Research/en-swbd/ldc-docs/call_con_tab.csv'
all_transcripts_path = '/Users/andy/Desktop/UTEP/Fall23/Research/en-swbd-transcripts/'
disk1_clips_path = '/Users/andy/Desktop/UTEP/Fall23/Research/en-swbd/disc1/aufiles/'
disk2_clips_path = '/Users/andy/Desktop/UTEP/Fall23/Research/en-swbd/disc2/aufiles/'
disk3_clips_path = '/Users/andy/Desktop/UTEP/Fall23/Research/en-swbd/disc3/aufiles/'
stereo_clips_directory = '/Users/andy/Desktop/UTEP/Fall23/Research/en-swbd/stereo_clips_participant/'
mono_clips_directory = '/Users/andy/Desktop/UTEP/Fall23/Research/en-swbd/mono_clips_participant/'



class Participant:
    def __init__(self, participant_id, gender):
        self.participant_id = participant_id
        self.gender = gender


class Conversation:
    def __init__(self, conversation_id, participant_a=None, participant_b=None):
        self.conversation_id = conversation_id
        self.participant_a = participant_a
        self.participant_b = participant_b


class SwbdOrganizer:
    def __init__(self):
        self.participants = {}
        self.conversations = {}
        self.clip_counter = 0
        self.read_participants()
        self.participant_counters = {p_id: 0 for p_id in self.participants}
        self.read_conversations()
        self.process_conversations()


    def read_participants(self):
        with open(caller_csv, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                gender = 'f' if 'FEMALE' in row[3] else 'm'
                participant_id = row[0]
                new_participant = Participant(participant_id, gender)
                self.participants[participant_id] = new_participant

    def read_conversations(self):
        with open(call_conversation_csv, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                conversation_id, participant_label, participant_id = self.conversation_row_cleaner(row)
                if conversation_id not in self.conversations:
                    new_conversation = Conversation(conversation_id, participant_a=self.participants[participant_id])
                    self.conversations[conversation_id] = new_conversation
                else:
                    self.conversations[conversation_id].participant_b = self.participants[participant_id]

    def conversation_row_cleaner(self, row):
        conversation_id = row[0]
        participant_label = 'A' if 'A' in row[1] else 'B'
        participant_id = row[2][1:]
        return conversation_id, participant_label, participant_id

    def process_conversations(self):
        for conversation_id in self.conversations:
            clip_path = self.get_clip_path(conversation_id)
            if clip_path is None: continue
            transcript_directory_path = all_transcripts_path + f"{conversation_id[:2]}/{conversation_id}/"
            transcript_a = self.get_transcript_path(transcript_directory_path, conversation_id, 'A')
            transcript_b = self.get_transcript_path(transcript_directory_path, conversation_id, 'B')
            self.chop_transcript_timestamps(transcript_directory_path + transcript_a, clip_path, conversation_id, 'A')
            self.chop_transcript_timestamps(transcript_directory_path + transcript_b, clip_path, conversation_id, 'B')

    def get_clip_path(self, conversation_id):
        for clip_directory in [disk1_clips_path, disk2_clips_path, disk3_clips_path]:
            if f"sw0{conversation_id}.au" in os.listdir(clip_directory):
                return clip_directory + f"sw0{conversation_id}.au"
        return None

    def get_transcript_path(self, path, conversation_id, participant_label):
        directory = os.listdir(path)
        for file in directory:
            if f"sw{conversation_id}{participant_label}" in file and 'trans.text' in file:
                return file
        print(f'transcript file not found{path + participant_label}')
        sys.exit()

    def chop_transcript_timestamps(self, transcript_path, clip_path, conversation_id, participant_label):
        with open(transcript_path) as file:
            for line in file:
                start_time, end_time = self.chop_transcript_line(line)
                if start_time is None: continue
                data, samplerate = sf.read(clip_path)
                start_index = int(start_time * samplerate)
                end_index = int(end_time * samplerate)
                extracted_data = data[start_index:end_index]
                conversation = self.conversations[conversation_id]
                if participant_label == 'A':
                    participant = conversation.participant_a
                else:
                    participant = conversation.participant_b
                p_id = participant.participant_id
                file_name = f'{stereo_clips_directory}{p_id}_{self.participant_counters[p_id]}.wav'
                sf.write(file_name, extracted_data, samplerate)
                self.clip_counter += 1
                self.participant_counters[p_id] += 1


    def chop_transcript_line(self, line):
        split_line = line.split(' ')
        if len(split_line) <= 4:
            return None, None
        return float(split_line[1]), float(split_line[2])


def convert_to_mono():
    stereo_clips = os.listdir(stereo_clips_directory)
    for clip in stereo_clips:
        channel_to_keep = 0 if 'A' in clip else 1
        data, samplerate = sf.read(stereo_clips_directory + clip)
        mono_data = data[:, channel_to_keep]
        sf.write(mono_clips_directory + clip, mono_data, samplerate, subtype='PCM_16')


#SwbdOrganizer()
convert_to_mono()
