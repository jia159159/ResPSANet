# ResPSANet
We use ResPSANet built by the [PSA module](
https://doi.org/10.48550/arXiv.2105.14447). This network has the ability to extract multi-scale power disturbance abnormal waveform features, and has an attention mechanism that can select important scales for focused feature extraction.Our paper: <<GAF-ResPSA: A Dimension-Increasing Residual Multi-Scale Attention Framework for Identifying Anomalous Waveforms of Fault Recorders>>


## Preconditions:
1. Use PAA for phase adaptive adjustment, which is implemented here using matlab; (code is not provided here)
2. Use GAF to convert the data into two dimensions; (code is not provided here)

## Required libraries
1. Install the required libraries for python, you can view the requirement.txt file here;

## How to use it?
1. Create a folder Dataset in the same directory as main.py to store your data;
2. Check whether the loaded data directory in main.py has the same name as the file you created;
4. Modify the abnormal waveform category, which depends on the number of types of power disturbance abnormal waveform categories you need to identify.
3. Run python main.py in the command line
