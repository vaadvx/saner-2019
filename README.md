Materials for FSE'2018
===============

## Overview
- BiD-TBCNN and D-TBCNN code in Tensorflow : /model
- Baseline code : /baselines
- Data: /data
  - Raw data : /data/code
  - Protobuf : /data/code_pb_slice
  - Pickle :   /data/code_pb_slice_pkl

Since the training data is quite large, we store in on Google Drive.

Download the pretrained embedding, training data and testing data here: https://drive.google.com/open?id=1aA-l31EwaDETdBtFFZ2EXLYkN0Z6zKgT and store into the directory pretrained_embedding/

We use Python with Tensorflow, keras, sklearn to build the model and run the baselines.

## Process

To give an overview of how we process our data, here are the steps:
- Use nicad clone detection tool : https://www.txl.ca/ to remove clones.
- Once the clones are removed, we use the parser from http://www.srcml.org/ to parse the code into Protobuf format.
- From the protobuf format, we dump it into the Python pickle format for training, since our code based is written in Python, from now all, all of the training is based on the pickle files.





