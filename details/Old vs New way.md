# Old vs New way
Our final implementation of this fall detection service differs to our inital approach.

## Old way - SISFall Dataset

Our initial plan was to build a fall detection model that combined the breadth of the SISFall dataset with our own in-house UWB positional data.\
The idea was that SISFall would provide a large, diverse dataset, while our local data would fine-tune the model for our environment.

The SisFall dataset can be found in ```experiments/sisfall```

### Train the model on the SISFall dataset
1) We analyzed and trained on the SisFall dataset which contains accelerometer and gyroscope readings.
2) Convert our UWB data, as our UWB data is positional 3D data rather than acceleration, we derived the acceleration using differentiation
3) Hoped that the derived acceleration values matched the SISFall dataset values
4) We extracted features of the SISFall dataset
5) Trained the model on the SisFall dataset and our inhouse data and gave our inhouse data more heavier weights
6) Tested

After testing we concluded that this model was unviable having a high rate of false negatives.

Hence we instead decided to train only using our own data.
We concluded that the noise and data mismatch made these two datatypes incompatible.

### Train the model on our dataset
After realizing that combining SISFall with our UWB data produced unreliable results, we decided to train exclusively using our own collected dataset.

1) Data collection, we recorded multiple sessions of fall and non fall events
2) Each trial captured 3d position sampled at 60Hz
3) Data preprocessing, a pipeline was introduced that would clean and standardize the signals
4) Jump Removal -> Downsampling -> Savistzky Golay Smoothing
5) Feature Extraction, we extracted features directly from our position data.
6) Model training

With those steps and a further thresholding step that sees if a drop has occurred first before invoking the model we were able to see a more accurate model.

The verifications and results can be found in the ```details/Verification.md``` file