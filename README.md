# ESC50_multitask
Multitask learning experiment on ESC-50 dataset.

This project makes use of multitask learning (MTL) to try and improve the classification performance on the [ESC-50 dataset](https://github.com/karoldvl/ESC-50). Multitask learning makes use of the inductive transfer of information gleaned from learning related tasks in parallel to the main task using a shared representation. More information on MTL can be found [here](http://reports-archive.adm.cs.cmu.edu/anon/1997/CMU-CS-97-203.pdf).

Raw audio clips from the dataset have been converted into .tif formated spectrograms using two signal processing methods, [Constant-Q Transform](ESC-50-cqt) and [Short-time Fourier Transform](ESC-50-spec). We will be using a convolutional neural network (CNN) for classification.

Currently, we are using the value of the spectral centroids of the audio clips as an additional tasks i.e. secondary classification. To calculate and group the spectral centroids into classes use [Centroid2ndaryClassMaker](Centroid2ndaryClassMaker.ipynb). This will add the new spectral centroid class label onto the filenames.

The main file [ESC50_multitask](ESC50_multitask.ipynb) loads the images into a numpy array and reshapes them for training, which is done using Tensorflow. Images are trained on two separate CNN models for [MTL](model_mtl.py) and single-task learning [(STL)](model_stl.py).
