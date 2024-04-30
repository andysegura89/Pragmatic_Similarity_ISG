from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import feature_selection_test as fst
import scipy
import torch
from feature_selector import remove_losing_features, remove_spanish_losing_features

'''
This was the code I used to see how a basic neural network could predict
the judgement scores.
'''

def get_seed_re_delta(seed, re):
    seed = remove_losing_features(seed)
    re = remove_losing_features(re)
    delta_array = []
    for feature_idx in range(len(seed)):
        delta_array.append(abs(seed[feature_idx]-re[feature_idx]))
    return np.array(delta_array)

keras.utils.set_random_seed(0)

s1_judgement_reader = fst.FeatureSelectionTest('data/judge_features_avg_session1-d.csv')
s1_judgements = s1_judgement_reader.get_judgements()
s2_judgement_reader = fst.FeatureSelectionTest('data/judge_features_avg_session2-d.csv')
s2_judgements = s2_judgement_reader.get_judgements()
s3_judgetment_reader = fst.FeatureSelectionTest('data/judge_features_avg_session3-d.csv')
s3_judgements = s3_judgetment_reader.get_judgements()


s1_delta_arrays = []
s1_judge_avgs = []
s1_cos_sims = []
s1_cats = []
s2_delta_arrays = []
s2_judge_avgs = []
s2_cos_sims = []
s2_cats = []
s3_delta_arrays = []
s3_judge_avgs = []
s3_cos_sims = []
s3_cats = []


for judgement in s1_judgements:
    s1_delta_arrays.append(get_seed_re_delta(judgement.seed_avg, judgement.re_avg))
    s1_judge_avgs.append(judgement.judge_avg)
    seed_tensor = torch.tensor(judgement.seed_avg)
    re_tensor = torch.tensor(judgement.re_avg)
    cos_sim = torch.nn.functional.cosine_similarity(seed_tensor, re_tensor, dim=0)
    s1_cos_sims.append(cos_sim.item())
    seed_re_cat = torch.cat((seed_tensor, re_tensor), 0)
    s1_cats.append(seed_re_cat.numpy())

for judgement in s2_judgements:
    s2_delta_arrays.append(get_seed_re_delta(judgement.seed_avg, judgement.re_avg))
    s2_judge_avgs.append(judgement.judge_avg)
    seed_tensor = torch.tensor(judgement.seed_avg)
    re_tensor = torch.tensor(judgement.re_avg)
    cos_sim = torch.nn.functional.cosine_similarity(seed_tensor, re_tensor, dim=0)
    s2_cos_sims.append(cos_sim.item())
    seed_re_cat = torch.cat((seed_tensor, re_tensor), 0)
    s2_cats.append(seed_re_cat.numpy())

for judgement in s3_judgements:
    s3_delta_arrays.append(get_seed_re_delta(judgement.seed_avg, judgement.re_avg))
    s3_judge_avgs.append(judgement.judge_avg)
    seed_tensor = torch.tensor(judgement.seed_avg)
    re_tensor = torch.tensor(judgement.re_avg)
    cos_sim = torch.nn.functional.cosine_similarity(seed_tensor, re_tensor, dim=0)
    s3_cos_sims.append(cos_sim.item())
    seed_re_cat = torch.cat((seed_tensor, re_tensor), 0)
    s3_cats.append(seed_re_cat.numpy())





X_train = np.array(s1_cats) # arrays of size 206
y_train = np.array(s1_judge_avgs)  # Floating-point numbers

# Creating the model
model = keras.Sequential()
model.add(layers.Dense(1, input_dim=2048, activation='linear'))

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
model.fit(X_train, y_train, epochs=100)

test_array = np.array(s3_cats) #test data
predictions = model.predict(test_array)
print(f"length of predictions = f{predictions.shape}")

all_predictions = []
for prediction in predictions:
    all_predictions.append(prediction[0])

print(all_predictions)
correlation = scipy.stats.pearsonr(all_predictions, s3_judge_avgs)[0] # change this
print(f"correlation = {correlation}")

