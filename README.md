# emerging-tech-project
![jupyter](https://user-images.githubusercontent.com/14262715/49247722-db8df300-f40f-11e8-9f3b-02d61645b9fa.png)

## Description
This repository is intended for my Emerging Technology module.  It consists of four python notebooks and a python script which re related to a variety of topics related to neural networks

## Contents
### 1. numpy random notebook:
A jupyter notebook explaining the con-cepts behind and the use of the numpy random package, including plots of the various distributions.

### 2. Iris dataset notebook:
A jupyter notebook explaining the famous iris data set including the difficulty in writing an algorithm to separate the three classes of iris based on the variables in the dataset.

### 3. MNIST dataset notebook:
A jupyter notebook explaining how to read the MNIST dataset efficiently into memory in Python.

### 4. Digit recognition script:
A Python script that takes an image file containing a handwritten digit and identifies the digit using a super- vised learning algorithm and the MNIST dataset.

### 5. Digit recognition notebook:
A jupyter notebook explaining how the above Python script works and discussing its performance.

## Documentation

## How to run

### Clone or Download
The first thing you will nee to do is to either clone or download this repository onto your machine.

Click [here] to download Git.

Once it's installed you can either:

1. click on the green "Clone or Download" button to the right and download as a zip file

or...

2. Type this into your terminal / cmd wherever you want hte files to be installed:                          
`git clone git clone https://github.com/sshhane/emerging-tech-project.git`

### Install Python
In order to run both the Jupyter Notebooks and the script you will first need to have python 3 installed.  [Here](https://www.python.org/downloads/) is a link to download python.

### Install Jupyter
Additionally to run the notebooks you must install [Jupyter Notebook](http://jupyter.org/install).

The easiest way to get set up is to use Python’s package manager, [pip](https://pypi.org/project/pip/) to install Jupyter.  [Here](https://pypi.org/project/pip/) is some information on how to install pip if you havent already got it.

Now that we have python and pip, open up a terminal window or command prompt and run:

`python3 -m pip install --upgrade pip`

and...

`python3 -m pip install jupyter`

And finally to run the notebook navigate to where you have downloaded this repo and run:

`jupyter notebook`


## Running the Script
The MNIST digit recognition script takes in different arguments when run.  It will either train a new model and save this new model as a .h5 file or it can be used to classify a given image.

You can also run the script with the -h or --help argument to see what options are available

`python digitRecognition.py -h`

If you want to train a new model you need to run:

`python digitRecognition.py -t`

This will create and train a model off of the MNIST dataset.

To run the script and predict a given digit drag over the digit image that you want to classify into the repo folder on your computer and run:

`python digitRecognition.py -r [filename.png]`

replacing 'filename.png' with the name of the image that you are testing with.

The test image is classified and the version that was scaled and greyscaled is displayed in another window

