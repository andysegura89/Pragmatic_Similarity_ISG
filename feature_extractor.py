import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio


class FeatureExtractor:
    """
    Uses pytorch to extract transformation layers from an audio file.
    Utilizes either the HuBERT Large, Wav2Vec2.0 Large, or WavLM Large models
    bundle: String : 'hubert_l' or 'wav2vec_l' or 'wavlm_l' expected
    """
    def __init__(self, bundle='hubert_l'):
        torch.random.manual_seed(0)  # Sets the same random weights everytime the model is run.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CUDA uses GPUs for computations.
        self.bundle = self.get_bundle(bundle)
        self.model = self.bundle.get_model().to(self.device)
        self.print_info()

    def get_bundle(self, bundle):
        if bundle == "hubert_l": return torchaudio.pipelines.HUBERT_LARGE
        if bundle == "wav2vec_l": return torchaudio.pipelines.WAV2VEC2_LARGE
        if bundle == "wavlm_l": return torchaudio.pipelines.WAVLM_LARGE
        print(f"bundle name {bundle} not recognized: 'hubert_l' or 'wav2vec_l' or 'wavlm_l' expected.")
        sys.exit()

    def print_info(self):
        print(f"torch version: {torch.__version__}")
        print(f"torch audio version: {torchaudio.__version__}")
        print(f"device: {self.device}")
        print(f"Sample Rate: {self.bundle.sample_rate}")
        print(f"model class: {self.model.__class__}")

    def get_transformation_layers(self, path, plot_layers=False):
        """
        Passes an audio file to a self-supervised machine learning model.
        :param path: path of the audio file : String
        :param plot_layers: Will visualize the transformation layers if true.
        :return: A list of 24 3d tensors representing the 24 transformation layers
        Tensor Dimensions:
        1st = number of audio files processed at once
        2nd = number of frames per audio file (One frame for every 10ms)
        3rd = number of features extracted per frame (1024)
        """
        if not os.path.exists(path):
            print(f"ERROR: {path} is not a valid file path")
            return
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.to(self.device)
        if sample_rate != self.bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.bundle.sample_rate)
        with torch.inference_mode():  # Disables gradient computation and back propagation.
            features, _ = self.model.extract_features(waveform)
        if plot_layers: self.plot_layers(features)
        return features

    def get_features_averages_from_fp(self, file_path):
        """
        Calculates feature averages from all 24 transformation layers given the file path.
        :param file_path: location where the audio clip is stored.
        :return: A list of size 24 that represents the 24 transformation layers. Each element
        in the list is another list of size 1024 representing the average values of the 1024
        features the SSL models calculate per frame of the audio clip.
        """
        transformation_layers = self.get_transformation_layers(file_path)
        return self.get_features_averages_from_tl(transformation_layers)

    def get_features_averages_from_tl(self, transformation_layers):
        """
        Calculates feature averages of each transformation layer given a list of transformation layers.
        :param transformation_layers: a list of transformation layers.
        :return: A list of size len(transformation_layers). Each element in the
        list is a numpy array of size 1024 representing the average values of the 1024
        features the SSL models calculate per frame of the audio clip.
        """
        all_tl_averages = []
        for layer in transformation_layers:
            layer = torch.squeeze(layer, dim=0) # get rid of first dimension
            tl_averages = [] # will hold averages for this one transformation layer
            num_features = layer.shape[1] # columns
            num_frames = layer.shape[0] # rows
            for feature_idx in range(num_features):
                feature_sum = 0
                for frame_idx in range(num_frames):
                    feature_sum += layer[frame_idx, feature_idx].item()
                tl_averages.append(feature_sum / num_frames)
            all_tl_averages.append(np.array(tl_averages))
        return all_tl_averages

    """
    We found the 24th layer had the highest correlation to our human judgement sessions
    """
    def get_24th_layer_features_averages(self, file_path):
        """
        We found the 24th layer had the highest correlation when compared to human judgements.
        This method takes in a file path and returns the feature averages from the 24th layer only.
        :param file_path: path where the audio clip is stored.
        :return: a 1d numpy array of size 1024 that represent the average value of the features
        calculated in the 24th layer.
        """
        transformation_layers = self.get_transformation_layers(file_path)
        layer = transformation_layers[23] # use only the 24th layer
        return self.get_features_averages_from_tl([layer])[0]

    def plot_layers(self, features):
        """
        Visualizes transformation layers from an audio file.
        :param features: List of 3d tensor objects representing transformation layers.
        :return: None
        """
        fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
        for i, feats in enumerate(features):
            ax[i].imshow(feats[0].cpu(), interpolation="nearest")
            ax[i].set_title(f"Feature from transformer layer {i + 1}")
            ax[i].set_xlabel("Feature dimension")
            ax[i].set_ylabel("Frame (time-axis)")
        plt.tight_layout()
        plt.show()

