import os
import feature_extractor as fe
import csv

'''
This code was used to create .csv files with the feature averages
inside of them to load faster. This created the csv files used in the demo
'''

winning_indexes = [0, 2, 41, 54, 63, 67, 102, 155, 159, 172, 173, 193, 197, 248, 261, 280, 286, 350, 358, 431, 459, 482,
                   488, 528, 603, 616, 618, 707, 708, 715, 717, 731, 792, 799, 804, 809, 824, 828, 870, 878, 880, 900,
                   903, 959, 45, 113, 120, 121, 204, 206, 246, 269, 411, 453, 510, 559, 602, 656, 666, 929, 972, 973,
                   1013, 1016, 13, 17, 105, 134, 136, 185, 188, 474, 578, 622, 651, 882, 925, 162, 410, 577, 628, 750,
                   758, 866, 869, 952, 963, 965, 50, 168, 436, 470, 513, 527, 557, 660, 732, 514, 661, 694, 698, 935,
                   937]


def remove_losing_features(averages):
    global winning_indexes
    winning_features = []
    for idx in winning_indexes:
        winning_features.append(averages[idx])
    return winning_features

class DatasetCreator:
    def __init__(self):
        self.feature_extractor = fe.FeatureExtractor('hubert_l')

    def create_dral_dataset(self, folder_path, csv_path):
        for file in os.listdir(folder_path):
            if file[:2] == 'EN':
                print(f"processing: {file}")
                features = self.feature_extractor.get_24th_layer_features_averages(folder_path+file)
                features = remove_losing_features(features)
                features = [x.item() for x in features]
                self.write_csv(csv_path, folder_path + file, features)

    def create_asdnt_dataset(self, asd_path, nt_path, csv_path):
        for path in [asd_path, nt_path]:
            for participant_id in os.listdir(path):
                if 'DS_Store' in participant_id: continue
                for clip in os.listdir(path + participant_id):
                    if 'DS_Store' in clip: continue
                    print(f'processing clip: {clip}')
                    clip_path = path + participant_id + '/' + clip
                    features = self.feature_extractor.get_24th_layer_features_averages(clip_path)
                    features = [x.item() for x in features]
                    features = remove_losing_features(features)
                    self.write_csv(csv_path, clip_path, features, asd_nt='ASD' if path == asd_path else 'NT')


    def create_swbd_dataset(self, clips_path):
        failed_clips = []
        for clip in os.listdir(clips_path):
            if 'DS_Store' in clip: continue
            print(clips_path + clip)
            try:
                features = self.feature_extractor.get_24th_layer_features_averages(clips_path + clip)
            except:
                failed_clips.append(clips_path + clip)
                continue
            features = remove_losing_features(features)
            features = [x.item() for x in features]
            csv_path = 'data/swbd-male-demo.csv' if 'm.wav' in clip else 'data/swbd-female-demo.csv'
            self.write_csv(csv_path, clips_path + clip, features)
            print(f"processed clip: {clip}")
        print(f"failed = {failed_clips}")

    def create_swbd_participant_Dataset(self, clips_path):
        for clip in os.listdir(clips_path):
            if 'DS_Store' in clip: continue
            features = self.feature_extractor.get_24th_layer_features_averages(clips_path + clip)
            features = remove_losing_features(features)
            csv_path = 'data/swbd-participant.csv'
            self.write_csv(csv_path, clips_path + clip, features)
            print(f"processed clip: {clip}")


    def write_csv(self, csv_path, clips_path, features, asd_nt=None):
        info_row = [clips_path]
        if asd_nt: info_row.append(asd_nt)
        with open(csv_path, 'a') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows([info_row, features, []])


ds = DatasetCreator()
#ds.create_dral_dataset('DRAL-All-Short/', 'data/dral-demo.csv')
#ds.create_asdnt_dataset('ASD-Mono/', 'NT-Mono/', 'data/asdnt-demo.csv')
ds.create_swbd_dataset('swbd_mono/')
#ds.create_swbd_participant_Dataset('/Users/andy/Desktop/UTEP/Fall23/Research/en-swbd/mono_clips_participant/')