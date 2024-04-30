# PSF Demo
## Author: Andy Segura

### Description
The psf demo demonstrates our metric for detecting pragmatic similarity
The way it works is that it records someone speaking and chops up
their utterances into different .wav files. Each utterance is then
compared to all of the clips in one of our datasets. It then shows
clips of differing levels of similarity. 

Here our the datasets that are used:
DRAL (UTEP): A collection of conversations with students from UTEP
SWBD (Texas Instruments): A collection of conversations mostly from people
in the East Texas area. Can choose whether to compare only male or female 
voices for this dataset. 
ASD/NT (NMSU): A collection of conversations with children with autism 
spectrum disorder and neurotypical development. Due to privacy issues
we can't play clips from this dataset, only show you what group the clip
belongs to. 

### How to run

In the psf_ui folder run the following command

```
ng serve
```

In the psf_demo folder, run the following command

```
python rest_api.py
```

Go to this address in your browser
```
locahhost:4200
```




