# ISG Pragmatic Similarity Code
## Author: Andy Segura

## Feature Extractor
  Uses pytorch to extract transformation layers from an audio file.
  Utilizes either the HuBERT Large, Wav2Vec2.0 Large, or WavLM Large models
  Specify which model you want to use by
  bundle: String : 'hubert_l' or 'wav2vec_l' or 'wavlm_l' expected
  
  Once the features are extracted you can use them all or get them mean
  pooled by this code. 


## Typical Classifier
The typical classifier uses a leave one out strategy along with the
kNN algorithm to try to predict each Participant as belonging to the
typical group or atypical group. Predictions are made by using the features
extracted and mean pooled from the HuBert Large SSL Model. The metric for
kNN is an adapted version of cosine_similarity where we use (1-cosine_similarity)
in order to get a smaller number for clips that are more similar. This will give an
output that resembles Euclidean Distance.


In order to use the classifier you have to pass it a list of 
Participant objects. Each participant object needs a list of clip objects. 
Here is an example of how to create a list of clip objects. 

```
  feature_extractor = fe.FeatureExtractor('hubert_l')
  participant_clips_directory = ... # directory where clips are stored
  clips_list = []
  for clip_path in os.listdir(participant_clips_directory):
      feature_avgs_all_layers = feature_extractor.get_features_averages_from_fp(clip_path)
      new_clip = Clip(feature_avgs_all_layers, clip_path)
      clips_list.append(new_clip)
```

Once you have the clips list complete for a single participant you can now create
a Participant object. Age and gender attributes are optional

```
new_participant = Participant(participant_id, group_label, clips_list, age, gender)
```

Once you have a list of participant objects you can them pass them to the Typical Classifier
to get results. You need to pass the list of participants and the labels for the typical
and atypical groups. Each one of your participants should have one of these labels for 
their group label. 

```
classifier = TypicalClassifier(participants, 'TD', 'SLI')
classifier.run()
classifier.write_all_csv_results('data/', 'ENNI')
```

If you want to print the results for a single participant you can use
the following
```
classifier.write_participant_clip_results_csv(csv_path, participant_id)
```

