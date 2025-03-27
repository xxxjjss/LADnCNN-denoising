LADnCNN: A Lightweight Attention-Based Denoising CNN for Microseismic Signal Processing

Introduction

This repository contains the implementation of five deep-learning models for one-dimensional microseismic (MS) signal denoising, including:

•	AutoEncoder 

•	CNN

•	DnCNN

•	LADnCNN (Lightweight Attention DnCNN)

•	U-Net

Our proposed LADnCNN model integrates a denoising convolutional neural network (DnCNN) with a lightweight attention mechanism (LAM) to enhance the ability to preserve essential features while effectively suppressing noise. Experimental results demonstrate its superior performance over traditional denoising methods and deep-learning models. The dataset consists of field-collected and simulated MS signals, and the best-performing trained models (.pth files) are included in this repository.

Environment Requirements

It is recommended to run the code under Linux. The code is developed using Python 3.9 and relies on the following libraries:

•	PyTorch

•	NumPy

•	SciPy

•	Matplotlib

Folder Structure

•	AutoEncoder/: Implementation of the AutoEncoder-based denoising model.

•	CNN/: Implementation of the CNN-based denoising model.

•	DnCNN/: Implementation of the standard DnCNN model.

•	LADnCNN/: Implementation of our proposed LADnCNN model with lightweight attention.

•	U-Net/: Implementation of the U-Net-based denoising model.

•	datasets/: Scripts for loading and preprocessing MS signal datasets.

models/:

•	AutoEncoder_best.pth

•	CNN_best.pth

•	DnCNN_best.pth

•	LADnCNN_best.pth

•	U-Net_best.pth

(These are the best-performing trained models for each architecture.)
