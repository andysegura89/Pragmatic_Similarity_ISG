import csv
import torch
from playsound import playsound
import os
import feature_extractor as fe
import feature_selection as fs

s1_winners = [0, 2, 41, 54, 63, 67, 102, 155, 159, 172, 173, 193, 197, 248, 261, 280, 286, 350, 358, 431, 459, 482, 488, 528, 603, 616, 618, 707, 708, 715, 717, 731, 792, 799, 804, 809, 824, 828, 870, 878, 880, 900, 903, 959, 45, 113, 120, 121, 204, 206, 246, 269, 411, 453, 510, 559, 602, 656, 666, 929, 972, 973, 1013, 1016, 13, 17, 105, 134, 136, 185, 188, 474, 578, 622, 651, 882, 925, 162, 410, 577, 628, 750, 758, 866, 869, 952, 963, 965, 50, 168, 436, 470, 513, 527, 557, 660, 732, 514, 661, 694, 698, 935, 937]


class SimilarityFinder:
    def __init__(self, feature_selection=False, directory_path="/Users/andy/Desktop/UTEP/Fall23/Research/similarity_experiment/seeds_reenactments_combined/"):
        self.feature_selection = feature_selection
        self.dral_clips = self.read_clips('data/dral.csv')
        self.asdnt_clips = self.read_clips('data/asdnt.csv')
        self.swbd_male_clips = self.read_clips('data/swbd-male.csv')
        self.swbd_female_clips = self.read_clips('data/swbd-female.csv')
        print(f"number of DRAL clips : {len(self.dral_clips)}")
        print(f"number of ASDNT clips : {len(self.asdnt_clips)}")
        print(f"number of SWBD-Male clips : {len(self.swbd_male_clips)}")
        print(f"number of SWBD-Female clips : {len(self.swbd_female_clips)}")
        self.directory_path = directory_path
        self.times = []
        self.feature_extractor = fe.FeatureExtractor('hubert_l')


    def read_judgements(self):
        with open('/Users/andy/Desktop/UTEP/Fall23/Research/research_code/data/judge_features_avg_session1.csv', 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            counter = 1
            for row in csv_reader:
                if counter == 1:
                    seed_name = row[1]
                    re_name = row[2]
                if counter == 2:
                    seed_avg = [float(x) for x in row]
                if counter == 3:
                    re_avg = [float(x) for x in row]
                    self.dral_clips[seed_name] = seed_avg
                    self.dral_clips[re_name] = re_avg
                if counter == 4:
                    counter = 0
                counter += 1

    def read_clips(self, path):
        clip_dic = {}
        with open(path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            counter = 1
            for row in csv_reader:
                if counter == 1:
                    file_path = row[0]
                if counter == 2:
                    features_avg = [float(x) for x in row]
                    clip_dic[file_path] = features_avg
                if counter == 3:
                    counter = 0
                counter += 1
        return clip_dic


    def find_similar(self, clip_to_find, dataset):
        cos_similarities = []
        clip_to_find_avg = self.feature_extractor.get_24th_layer_features_averages(clip_to_find)
        clip_to_find_avg = fs.remove_losing_features(clip_to_find_avg)

        #clip_to_find_avg = self.remove_losing_features(clip_to_find_avg)
        if dataset == 'ASDNT':
            dataset_to_search = self.asdnt_clips
        elif dataset == 'DRAL':
            dataset_to_search = self.dral_clips
        else:
            if dataset == 'SWBD-MF':
                dataset_to_search = self.swbd_male_clips | self.swbd_female_clips
            elif dataset == 'SWBD-M':
                dataset_to_search = self.swbd_male_clips
            else:
                dataset_to_search = self.swbd_female_clips

        for test_clip in dataset_to_search:
            if test_clip == clip_to_find: continue

            test_clip_avg = dataset_to_search[test_clip]
            cos_sim = torch.nn.functional.cosine_similarity(torch.tensor(clip_to_find_avg), torch.tensor(test_clip_avg), dim=0)
            cos_similarities.append((cos_sim, test_clip))
            # ed = math.dist(clip_to_find_avg, test_clip_avg)
            #euclidean_distances.append((ed, test_clip))
        cos_similarities_sorted = sorted(cos_similarities, key=lambda tup: tup[0], reverse=True)
        return cos_similarities_sorted[0], \
            cos_similarities_sorted[1], \
            cos_similarities_sorted[999], \
            cos_similarities_sorted[1499]


    def play_clip(self, file_name):
        complete_path = self.directory_path + file_name
        if not os.path.exists(complete_path):
            complete_path = self.directory_path + '_' + file_name
        print(complete_path)
        playsound(complete_path)

    def play_original_clip(self, file_name):
        playsound(file_name)

    def get_average_times(self):
        print(f"average_times={sum(self.times)/len(self.times)}")

    def remove_losing_features(self, averages):
        if not self.feature_selection:
            return averages
        winning_features = []
        for idx in s1_winners:
            winning_features.append(averages[idx])
        return torch.tensor(winning_features)
