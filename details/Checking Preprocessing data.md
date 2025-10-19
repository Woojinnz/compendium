# Checking Preprocessing data

We provide ways on plotting the gathered raw data to see if the data is healthy.\
Mainly we show the difference between the raw data and the data after the processing pipeline

These files can be found in ```code/util/visualize_coordinates.py```

### Folder structure
Similar to the train_model.py file the training_data folder and the visualize file must be located on the same level.

The jump filter and saviszky_golay filter is used by this script.

### What it does
1) Load CSVS
2) Randomnly picks one example
4) Preprocess the data using the preprocesing pipeline
5) Plotting - Plot the processed data which should show a smoothening of noise.

Some example outputs are shown below

![alt text](photos/raw.png)

![alt text](photos/smooth.png)