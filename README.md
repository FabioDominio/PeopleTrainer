# People Trainer
Grid search variation for SVM multi-class classifier designed for gesture recognition and leveraging multi-core architectures

## Problem definition
Grid search is one of the most adopted parameter optimization strategy for Support Vector Machines (SVM) classifiers. Starting from a predefined sampling of the SVM kernel parameter space, often performed with a multidimensional grid (e.g. bidimensional for radial kernels) having logarithmic quantization steps, the classical (naive) implementation of Grid Search trains a different SVM model on the dataset against each parameter tuple and estimates the model performance its recognition accuracy according to k-fold validation. 

The maximum recognition accuracy among all the tested parameter tuples gives an "idea" of the actual recognition capabilities of the model trained with the most performing kernel configuration.

Although k-fold validation is meant to reduce the risk of overfitting on the training set, the obtained model generalization remains poor especially with low cardinality datasets, as those often acquired for gesture recognition purposes. Moreover, since k-fold validation leverages random samples of the dataset, the same feature vectors are likely to end both in training and test sets thus raising the risk of incurring in overfitting.

For this reasons, People Trainer extends the classical Grid Search approach as follows:

1. Assumes the dataset is balanced, that is, the dataset is made of N=P*G*R feature vectors with P the number of people the data refer to, G the number of gesture classes and R the number of instances for each gesture of each class. Note how the only requirement is the uniform dataset composition, not its data semantic, namely all the datasets with such composition are trainable with this approach.

2. Performs P manual partitionings of the dataset into train and test sets, where for the i-th partition the G*R feature vectors referred to the i-th person in the dataset constitute the test set and the remaining (P-1) * G * R the training set.

3. Trains P separate SVM model, one per partition, and consider as the overall recognition accuracy the average of all the single models accuracies.

In particular, 2) aims at modeling the fact that in real gesture recognition applications the gesture recognition model trained previously did not take into account the data coming from a new user. Each training set of (P-1)*G*R feature vectors is then partitioned again (P-1) times, following the same rationale, into a validation set and a training set of G*R and (P-2)*G*R feature vectors respectively, and (P-1) sub-models are trained on each parameter tuple. The maximum average recognition accuracy among all the tested parameter tuples is the estimated accuracy for the i-th model.

## Further readings
People Trainer has been widely discussed and used in publications like:

* Fabio Dominio, "Real-time hand gesture recognition exploiting multiple 2D and 3D cues", 2014
* F. Dominio, G. Marin, M. Piazza and P. Zanuttigh, "Feature descriptors for depth-based hand gesture recognition", Computer Vision and Machine Learning with RGB-D Sensors, Springer 2014

other publications, paper drafts and bibtex ready citations can be found at:

http://www.fabiodominio.com/research.html

Please cite them if you are using this software in your research. Thanks.

## Build instructions
People Trainer is almost self-contained and just requires:

* CMake for automatic project generation for Visual Studio, g++ and other environments;
* OpenCV library as external dependence (unofficial builds, ready for CMake, for Windows and Linux can be found at http://www.fabiodominio.com/tools.html)
* openMP support

Note how in CMake openCV is generally detected when specifying the path of "OpenCVConfig.cmake" file, often provided with any openCV build. Note also how, unless you are linking against a static version of openCV, the program executable also needs the "core" and "ml" dynamic libraries ("opencv_core300.dll" and "opencv_ml300.dll" in Windows, "opencv_core300.so" and "opencv_ml300.so" in Linux).

## Usage
Windows:

`
PeopleTrainer.exe <people_number> <gesture_classes_number> <gesture_repetitions_number> <features_number> <kernel_type> dataset `

Linux:

`
./PeopleTrainer <people_number> <gesture_classes_number> <gesture_repetitions_number> <features_number> <kernel_type> dataset `

With kernel_type:
* 0: linear kernel;
* 1: polynomial kernel;
* 2: radial basis function kernel;
* 3: sigmoidal kernel;
*	4: chi2 kernel;
*	5: histogram intersection kernel.

For kernel details please visit:

http://docs.opencv.org/3.0-beta/modules/ml/doc/support_vector_machines.html

"dataset" indicates, instead, the dataset file name.

The software returns a log file with the performed operations and the computed G*G confusion matrix with G the number of gesture classes. Each confusion matrix entry c(i,j) denotes the fraction (from 0 to 1) of gesture instances of class i assigned to class j. 

