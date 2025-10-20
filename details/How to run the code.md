
## Code 
Under the **codebase** folder you will be able to find the code that is required to train the machine learning model using XGBoost.

First it is recommended to create a *virutal environment* with python version 3.10 and then install the required packages using\
```pip install -r requirements.txt```

You will then be ready to train a new model, the provided preprocessing functions can be found in the ```codebase/util``` folder. Namely the jump_filter and the savistzky_golay filter.

### Training the model
In order to train the model the file requires a strict project layout.\
The falls must be formatted using a .csv format and requires four columns
```ts,x,y,z```, ts= timestamp, (x,y,z) = 3d coordinates

Then these files must be placed under the ```codebase/model_training/training_data``` folder under their respective classification such as adl or falls.

Example of the training data files that we have used to train our model can be found under ```experiments/testing_data```

Once the file strucutre is satisfactory you are able to run the ```train_model.py``` script.

### How does it train?
1) Load and clean each CSV file - get_file_as_dataframe(), removes entries with 0 values
2) Label & Aggregate -> gather() will go through the adl and falls directory and label the csv files to their corresponding type (0,1) and the file path and run them through the preprocessing pipeline
3) Preprocessing pipeline -> jump removal (using z-score method), downsampling to 30Hz, Savistzky-Golay smoothing
4) Feature extraction -> Extract the features (see below)

#### Features
z_std: standard deviation of z.\
fall_duration: time between fall start/end (in seconds from ts).\
fall_height_decrease: vertical drop in z over the detected fall window.\
fall_horizontal_distance: sum of √(Δx²+Δy²) in the window.\
fall_avg_horizontal_speed: fall_horizontal_distance / duration.\
fall_avg_vertical_speed: fall_height_decrease / duration.\
height_change: (95th-percentile pre-fall z) − (95th-percentile post-fall z).

5) Train/Validation split - Split the files to train and to validate against
6) Scaling - StandardScaler fitted
7) Model Training - XGBClassifier is trained on scaled features
8) Evaluation - Prints the classification_report on the evaluation of the model (precision, recall, F1)
9) Print Feature importance - Print the importance of each feature (also saves it as feature_importance.png)
10) Artifacts saved - **model_pos.pkl (Trained XGBoost model), scaler_pos.pkl (Scaler)**

### Integrating the model
Once you have obtained the model_pos.pkl and scaler_pos.pkl files you are ready to use them in the UWB positioning service.\

Now in the main UWB file directory if you go to 
```backend/app/services/fall```\
You will be able to find the ```fall_monitor_service.py``` file.\
We have included a copy of this file under ```codebase/fall_service``` in this compendium

This file is where the backend code for the fall monitor service is.

Simply copy over both .pkl files into this directory such that fall_monitor_service.py and the two .pkl files are located within the same level.

At this point you are now ready to run the UWB service that has our fall detection service included.

### Running the fall detection service
As our code has been fully implemented within the pre-existing UWB positioning service, no new commands are required.\
Simply start the service as you would normally do and the fall detection service will automatically start as well.

#### If you wish to see where the entry point of the fall detection service is read below
In the UWB fall code if you traverse to
```backend/app``` there exists a ```__init__.py``` file.

Within this file there is a method ```create_app()```\
One of the functions is ```start_fall_monitor()``` this function is responsible for the fall detection service, if you wish to **disable** the fall detection service simply comment or remove the start_follow_monitor() line.