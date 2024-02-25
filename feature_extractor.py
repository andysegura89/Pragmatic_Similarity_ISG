import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import os


class FeatureExtractor:
    """
    Uses pytorch to extract transformation layers from an audio file.
    Utilizes either the HuBERT Large, Wav2Vec2.0 Large, or WavLM Large models
    Specify which model you want to use by
    bundle: String : 'hubert_l' or 'wav2vec_l' or 'wavlm_l' expected
    """
    def __init__(self, bundle='hubert_l'):
        torch.random.manual_seed(0)  # Sets the same random weights everytime the model is run.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CUDA uses GPUs for computations.
        self.bundle = self.get_bundle(bundle)
        self.model = self.bundle.get_model().to(self.device)
        self.lengths = []
        self.print_info()

    def get_bundle(self, model):
        if model == "hubert_l": return torchaudio.pipelines.HUBERT_LARGE
        if model == "wav2vec_l": return torchaudio.pipelines.WAV2VEC2_LARGE
        if model == "wavlm_l": return torchaudio.pipelines.WAVLM_LARGE

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
        if plot_layers:
            plot_layers(features)
        return features

    def plot_layers(self, features):
        """
        Visualizes transformation layers from audio file.
        :param features: List of 3d tensor objects representing transformation layers.
        :return: Null
        """
        fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
        for i, feats in enumerate(features):
            ax[i].imshow(feats[0].cpu(), interpolation="nearest")
            ax[i].set_title(f"Feature from transformer layer {i + 1}")
            ax[i].set_xlabel("Feature dimension")
            ax[i].set_ylabel("Frame (time-axis)")
        plt.tight_layout()
        plt.show()

    def get_features_averages_from_fp(self, file_path):
        features_avgs = []
        transformation_layers = self.get_transformation_layers(file_path)
        for layer in transformation_layers:
            layer = torch.squeeze(layer, dim=0)
            averages = []
            num_cols = layer.shape[1]
            num_rows = layer.shape[0]
            # iterate through every feature
            for col_idx in range(num_cols):
                feature_sum = 0
                # iterate through every frame
                for row_idx in range(num_rows):
                    feature_sum += layer[row_idx, col_idx].item()
                averages.append(feature_sum / num_rows)
            features_avgs.append(np.array(averages))
        return features_avgs

    def get_features_averages_from_tl(self, transformation_layers):
        features_avgs = []
        for layer in transformation_layers:
            layer = torch.squeeze(layer, dim=0)
            averages = []
            num_cols = layer.shape[1]
            num_rows = layer.shape[0]
            # iterate through every feature
            for col_idx in range(num_cols):
                feature_sum = 0
                # iterate through every frame
                for row_idx in range(num_rows):
                    feature_sum += layer[row_idx, col_idx].item()
                averages.append(feature_sum / num_rows)
            features_avgs.append(np.array(averages))
        return features_avgs

    """
    We found the 24th layer had the highest correlation to our human judgement sessions
    """
    def get_24th_layer_features_averages(self, file_path):
        transformation_layers = self.get_transformation_layers(file_path)
        layer = transformation_layers[23]
        layer = torch.squeeze(layer, dim=0)
        averages = []
        num_cols = layer.shape[1]
        num_rows = layer.shape[0]
        # iterate through every feature
        for col_idx in range(num_cols):
            feature_sum = 0
            # iterate through every frame
            for row_idx in range(num_rows):
                feature_sum += layer[row_idx, col_idx].item()
            averages.append(feature_sum / num_rows)
        return torch.tensor(averages)
