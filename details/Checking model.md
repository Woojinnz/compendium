# Checking Model

We also provide ways to check how well the model performs.

```codebase\util\visualise_and classify.py``` contains code that can plot 3d coordinate data, whilst also highlighting in yellow when the model has determined a fall has occured

### Folder Structure
The scripts currently pulls a fall file from the training data folder.
You can change the path to any other csv file that you wish to test.

The jump filter and savitzky-golay filter are used by this script.


### How it works
1) Load the csv file
2) Proprocess (jump, downsample, etc..)
3) Feature extraction
4) Model classifcation
5) Plotting

An example output can found below
![Plot](photos/3dplot.png)