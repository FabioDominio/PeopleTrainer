/**
	People Trainer v. 1.0

	@author Fabio Dominio
	@date 11/29/2015
	
	The MIT License (MIT)

	Copyright (c) 2015 Fabio Dominio

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all
	copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
	SOFTWARE.
*/

// Include standard headers
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <omp.h>

// Include opencv headers
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

// Define used namespaces
using namespace std;
using namespace cv;
using namespace ml;

// Define common variables and data structures
ofstream logFile;

/**
Read and validate dataset

@param dataset file name
@param number of people in dataset
@param number of gesture classes in dataset
@param number of repetitions per gesture in dataset
@param number of features in dataset
@param reference to dataset buffer

@return true in case of success, false otherwise
*/
bool readDataset(string datasetFile, int pn, int gn, int rn, int fn, Mat_<float>& dataset) {
	string line;
	stringstream tokens;
	int row = 0;
	int feat = 0;
	ifstream in;
	in.open(datasetFile);
	if (!in.is_open()) {
		// Can't open dataset
		cerr << "Dataset opening error!" << endl;
		return false;
	}
	dataset = Mat_<float>(pn * gn * rn, fn);
	// Parse file
	float temp;
	while (!in.eof()) {
		// Read line
		getline(in, line);
		// Get tokens
		tokens = stringstream(line);
		feat = 0;
		while (tokens >> temp) {
			if (tokens.fail()) {
				// Read wrong type
				cerr << "Dataset error: read wrong type of feature at line " << row + 1 << " and position " << feat << "; please check dataset!" << endl;
				in.close();
				return false;
			}
			if (feat > fn  && line.size() > 0) {
				cerr << "Dataset error: read wrong number of features at line " << row + 1 << "; please check dataset!" << endl;
				in.close();
				return false;
			}
			dataset(row, feat++) = temp;
		}
		row++;
	}
	if (row != pn*gn*rn + 1) {
		cerr << "Dataset error: read wrong number of feature vectors; please check dataset!";
		return false;
	}
	return true;
}

/**
@brief Train dataset with SVM classifier (parallel version)

@param full dataset (N feature vectors of M features)
@param number of people in the dataset
@param number of gestures in the dataset
@param number of instances per gesture per person
@param number of features
@param kernel type for SVM classification
@param confusion matrix file name
*/
void trainSVMParallel(const Mat_<float>& dataset, int pn, int gn, int rn, int fn, SVM::KernelTypes kernel, string confusionMatrixFilename) {

	// Define required constants
	const int outerTrainLabelsNr = (pn - 1) * gn * rn;
	const int innerTrainLabelsNr = (pn - 2) * gn * rn;
	const int testLabelsNr = 1 * gn * rn;
	const int outerTrainSetSize = (pn - 1) * gn * rn;
	const int innerTrainSetSize = (pn - 2) * gn * rn;
	const int testSetSize = 1 * gn * rn;
	const int expCMin = 0;
	const int expCMax = 10;
	const int degMin = 1;
	const int degMax = (kernel == SVM::KernelTypes::POLY) ? 5 : degMin + 1;
	const int expCoef0Min = -15;
	const int expCoef0Max = (kernel == SVM::KernelTypes::POLY || kernel == SVM::KernelTypes::SIGMOID) ? 10 : expCoef0Min + 1;
	const int expGammaMin = -15;
	const int expGammaMax = (kernel == SVM::KernelTypes::POLY || kernel == SVM::KernelTypes::SIGMOID || kernel == SVM::KernelTypes::RBF || kernel == SVM::KernelTypes::CHI2) ? 10 : expGammaMin + 1;
	const float gridExpStep = 2.0;
	const int threadsNr = omp_get_max_threads();	

	// Define required variables and data structures	
	Mat_<float> results(1, testSetSize, 1);
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(kernel);
	Mat_<int> outerTrainLabels(1, outerTrainLabelsNr);
	Mat_<int> innerTrainLabels(1, innerTrainLabelsNr);
	Mat_<int> testLabels(1, testLabelsNr);
	Mat_<float> bestModelAccuracy(1, pn);
	Mat_<Vec4f> bestParameters(1, pn);
	Mat_<int> trainingTime(1, pn);
	Mat_<float> parameterAccuracy(1, (expCMax - expCMin) * (degMax - degMin) * (expGammaMax - expGammaMin) * (expCoef0Max - expCoef0Min));
	Mat_<Vec4f> parameters(1, parameterAccuracy.cols);
	vector<Mat> innerTestSets(pn - 1);
	vector<Mat> innerTrainSets(pn - 1);	

	// Generate training and test labels	
	for (int i = 0; i < pn - 1; i++)
		for (int j = 0; j < gn; j++)
			for (int k = 0; k < rn; k++)
				outerTrainLabels(rn*gn * i + rn * j + k) = j + 1;

	for (int i = 0; i < pn - 2; i++)
		for (int j = 0; j < gn; j++)
			for (int k = 0; k < rn; k++)
				innerTrainLabels(rn*gn * i + rn * j + k) = j + 1;

	for (int i = 0; i < gn; i++)
		for (int j = 0; j < rn; j++)
			testLabels(rn * i + j) = i + 1;

	// Generate list of parameters couples to test	
	for (int c = expCMin; c < expCMax; ++c)
		for (int g = expGammaMin; g < expGammaMax; ++g)
			for (int d = degMin; d < degMax; ++d)
				for (int f = expCoef0Min; f < expCoef0Max; ++f)
				{
					float C = powf(gridExpStep, c);
					float gamma = powf(gridExpStep, g);
					float deg = d;
					float coef0 = powf(gridExpStep, f);
					//int index = (expGammaMax - expGammaMin) * (degMax - degMin) * (expCoef0Max - expCoef0Min) * (c - expCMin) + (degMax - degMin) * (expCoef0Max - expCoef0Min) * (g - expGammaMin) + (expCoef0Max - expCoef0Min) * (d - degMin) + (f - expCoef0Min);
					parameters((expGammaMax - expGammaMin) * (degMax - degMin) * (expCoef0Max - expCoef0Min) * (c - expCMin) + (degMax - degMin) * (expCoef0Max - expCoef0Min) * (g - expGammaMin) + (expCoef0Max - expCoef0Min) * (d - degMin) + (f - expCoef0Min)) = Vec4f(C, gamma, deg, coef0);
				}

	// Split dataset in outer train/test sets	
	vector<Mat> outerTestSets(pn);
	vector<Mat> outerTrainSets(pn);
	for (int p = 0; p < pn; ++p) {
		// Test set
		outerTestSets[p] = dataset(Rect(0, p * testSetSize, fn, testSetSize));
		// Train set		
		if (p == 0)
			outerTrainSets[p] = dataset(Rect(0, (p + 1)* testSetSize, fn, dataset.rows - (p + 1)* testSetSize));
		else if (p == pn - 1)
			outerTrainSets[p] = dataset(Rect(0, 0, fn, testSetSize * p));
		else {
			vector<Mat> matrices = { dataset(Rect(0, 0, fn, testSetSize * p)), dataset(Rect(0, (p + 1)* testSetSize, fn, dataset.rows - (p + 1)* testSetSize)) };
			vconcat(matrices, outerTrainSets[p]);
		}
	} // End outer dataset split	

	// Write parallelism information
	// TODO: set number of threads to use	
	cout << endl << "Using " << threadsNr << " threads." << endl << endl;
	logFile << endl << "Using " << threadsNr << " threads." << endl << endl;

	// Train models
	for (int p = 0; p < pn; ++p) {
		cout << "Training model: " << p + 1 << "... ";
		logFile << "Training model: " << p + 1 << "... ";
		// Start timer for single model training
		double start = omp_get_wtime();
		// Split p-th model training set in inner train/test sets		
		for (int m = 0; m < pn - 1; ++m) {
			// Test set
			innerTestSets[m] = outerTrainSets[p](Rect(0, m * testSetSize, fn, testSetSize));
			// Train set
			if (m == 0)
				innerTrainSets[m] = outerTrainSets[p](Rect(0, (m + 1)* testSetSize, fn, outerTrainSets[p].rows - (m + 1)* testSetSize));
			else if (m == pn - 2)
				innerTrainSets[m] = outerTrainSets[p](Rect(0, 0, fn, testSetSize * m));
			else {
				vector<Mat> matrices = { outerTrainSets[p](Rect(0, 0, fn, testSetSize * m)), outerTrainSets[p](Rect(0, (m + 1)* testSetSize, fn, outerTrainSets[p].rows - (m + 1)* testSetSize)) };
				vconcat(matrices, innerTrainSets[m]);
			}
		} // End for m < pn	- 1		

	// Perform naive grid search with manual dataset partitioning		
#pragma omp parallel for schedule(dynamic) default(none) private(svm, results) shared(p, cout, kernel, parameters, pn, fn, parameterAccuracy, innerTrainSets, innerTestSets, innerTrainLabels, testLabels) 
		// Test each parameter set in parallel
		for (int k = 0; k < parameters.cols; ++k) {
			float accuracy = 0;
			// Compute average accuracy for each train-test pair
			for (int d = 0; d < pn - 1; ++d) {
				Ptr<SVM> svm = SVM::create();
				svm->setType(SVM::C_SVC);
				svm->setKernel(kernel);
				svm->setC(parameters(k)(0));
				svm->setGamma(parameters(k)(1));
				svm->setDegree(parameters(k)(2));
				svm->setCoef0(parameters(k)(3));
				svm->train(innerTrainSets[d], ROW_SAMPLE, innerTrainLabels);
				svm->predict(innerTestSets[d], results);
				// Add the number of correct matches
				for (int l = 0; l < testSetSize; l++)
					if (testLabels(l) == results(l))
						accuracy += 1;
			}
			accuracy = accuracy / outerTrainSetSize;
			parameterAccuracy(k) = accuracy;
		}// end parallel for
		// Find best parameter accuracy for current model
		double bestAccuracy = -1;
		int bestParameterIdx[2];
		minMaxIdx(parameterAccuracy, 0, &bestAccuracy, NULL, bestParameterIdx);
		bestModelAccuracy(p) = bestAccuracy;
		bestParameters(p) = parameters(bestParameterIdx[1]);

		// Stop timer
		double stop = omp_get_wtime();

		// Print statistics
		trainingTime(p) = (int) floor(stop - start + 0.5);
		cout << " done in " << (int) floor (stop - start + 0.5) << " seconds" << endl;
		logFile << " done in " << (int)floor(stop - start + 0.5) << " seconds" << endl;
		switch (kernel) {
		case SVM::KernelTypes::LINEAR:
			cout << "Model " << p + 1 << ": best Cost = " << bestParameters(p)(0) << "; best model accuracy = " << std::setprecision(3) << bestModelAccuracy(p) * 100 << "%" << endl << endl;
			logFile << "Model " << p + 1 << ": best Cost = " << bestParameters(p)(0) << "; best model accuracy = " << std::setprecision(3) << bestModelAccuracy(p) * 100 << "%" << endl << endl;
			break;
		case SVM::KernelTypes::POLY:
			cout << "Model " << p + 1 << ": best Cost = " << bestParameters(p)(0) << "; best Gamma = " << bestParameters(p)(1) << "; best Degree = " << bestParameters(p)(2) << "; best coef0 = " << bestParameters(p)(3) << "; best model accuracy = " << std::setprecision(3) << bestModelAccuracy(p) * 100 << "%" << endl << endl;
			logFile << "Model " << p + 1 << ": best Cost = " << bestParameters(p)(0) << "; best Gamma = " << bestParameters(p)(1) << "; best Degree = " << bestParameters(p)(2) << "; best coef0 = " << bestParameters(p)(3) << "; best model accuracy = " << std::setprecision(3) << bestModelAccuracy(p) * 100 << "%" << endl << endl;
			break;
		case SVM::KernelTypes::RBF:
			cout << "Model " << p + 1 << ": best Cost = " << bestParameters(p)(0) << "; best Gamma = " << bestParameters(p)(1) << "; best model accuracy = " << std::setprecision(3) << bestModelAccuracy(p) * 100 << "%" << endl << endl;
			logFile << "Model " << p + 1 << ": best Cost = " << bestParameters(p)(0) << "; best Gamma = " << bestParameters(p)(1) << "; best model accuracy = " << std::setprecision(3) << bestModelAccuracy(p) * 100 << "%" << endl << endl;
			break;
		case SVM::KernelTypes::SIGMOID:
			cout << "Model " << p + 1 << ": best Cost = " << bestParameters(p)(0) << "; best Gamma = " << bestParameters(p)(1) << "; best coef0 = " << bestParameters(p)(3) << "; best model accuracy = " << std::setprecision(3) << bestModelAccuracy(p) * 100 << "%" << endl << endl;
			logFile << "Model " << p + 1 << ": best Cost = " << bestParameters(p)(0) << "; best Gamma = " << bestParameters(p)(1) << "; best coef0 = " << bestParameters(p)(3) << "; best model accuracy = " << std::setprecision(3) << bestModelAccuracy(p) * 100 << "%" << endl << endl;
			break;
		case SVM::KernelTypes::CHI2:
			cout << "Model " << p + 1 << ": best Cost = " << bestParameters(p)(0) << "; best Gamma = " << bestParameters(p)(1) << "; best model accuracy = " << std::setprecision(3) << bestModelAccuracy(p) * 100 << "%" << endl << endl;
			logFile << "Model " << p + 1 << ": best Cost = " << bestParameters(p)(0) << "; best Gamma = " << bestParameters(p)(1) << "; best model accuracy = " << std::setprecision(3) << bestModelAccuracy(p) * 100 << "%" << endl << endl;
			break;
		case SVM::KernelTypes::INTER:
			cout << "Model " << p + 1 << ": best Cost = " << bestParameters(p)(0) << "; best model accuracy = " << std::setprecision(3) << bestModelAccuracy(p) * 100 << "%" << endl << endl;
			logFile << "Model " << p + 1 << ": best Cost = " << bestParameters(p)(0) << "; best model accuracy = " << std::setprecision(3) << bestModelAccuracy(p) * 100 << "%" << endl << endl;
			break;
		}
	} // End grid search	

	// Compute global confusion matrix
	Mat_<float> confusionMatrix(gn, gn);
	confusionMatrix.setTo(Scalar(0));

	// Train each model with its best computed parameters and test it
	Mat_<float> modelAccuracy(1, pn);
	modelAccuracy.setTo(Scalar(0));
	for (int p = 0; p < pn; ++p) {
		svm->setC(bestParameters(p)(0));
		svm->setGamma(bestParameters(p)(1));
		svm->setDegree(bestParameters(p)(2));
		svm->setCoef0(bestParameters(p)(3));
		svm->train(outerTrainSets[p], ROW_SAMPLE, outerTrainLabels);
		svm->predict(outerTestSets[p], results);

		// Add the number of correct matches
		for (int l = 0; l < testSetSize; l++){
			if (testLabels(l) == results(l))
				modelAccuracy(p) += 1;
			// Increment counter in confusion matrix
			confusionMatrix((int)testLabels(l) - 1, (int)results(l) - 1) += 1;
		}		
	}
	confusionMatrix /= testSetSize;

	// Print the estimated accuracy for each model
	for (int p = 0; p < pn; ++p) {
		modelAccuracy(p) = modelAccuracy(p) * 100 / testSetSize;
		cout << "Model " << p + 1 << " accuracy = " << std::setprecision(3) << modelAccuracy(p) << endl;
		logFile << "Model " << p + 1 << " accuracy = " << std::setprecision(3) << modelAccuracy(p) << endl;
	}

	// Compute and print overall accuracy
	float overallAccuracy = mean(modelAccuracy)(0);

	cout << endl << "Overall accuracy = " << std::setprecision(3) << overallAccuracy << "%" << endl << endl;
	logFile << endl << "Overall accuracy = " << std::setprecision(3) << overallAccuracy << "%" << endl << endl;

	// Compute and print mean model training time
	int minutes = mean(trainingTime)(0) / 60;
	int seconds = (int) floor(mean(trainingTime)(0)) % 60;
	cout << "Mean model training time: " << minutes << " minutes and " << seconds << " seconds" << endl;
	logFile << "Mean model training time: " << minutes << " minutes and " << seconds << " seconds" << endl;

	// Store confusion matrix
	ofstream confusionMatrixFile;
	confusionMatrixFile.open(confusionMatrixFilename);
	if (confusionMatrixFile.is_open()) {
		for (int i = 0; i < gn; i++)
		{
			for (int j = 0; j < gn; j++)
				confusionMatrixFile << confusionMatrix[i][j] << " ";
			confusionMatrixFile << endl;
		}
		confusionMatrixFile.close();
	}
	else
		cerr << "Unable to open confusion matrix output file!";
}

// Print usage
void printUsage() {
	cout << "People Trainer usage usage: " << endl << endl;
	cout << "PeopleTrainer.exe <people_number> <gesture_classes_number> <gesture_repetitions_number> <features_number> <kernel_type> dataset" << endl << endl;
	cout << "with kernel_type:" << endl;
	cout << "\t 0: linear" << endl;
	cout << "\t 1: polynomial" << endl;
	cout << "\t 2: radial basis function" << endl;
	cout << "\t 3: sigmoidal" << endl;
	cout << "\t 4: chi2" << endl;
	cout << "\t 5: histogram intersection" << endl;
}

// Main method
int main(int argc, char* argv[]) {
	if (argc != 7) {
		// Wrong number of parameters
		cerr << "Wrong number of parameters!" << endl << endl;
		printUsage();
		return -1;
	}

	// Right number of parameters; parse them
	string fileName = "";
	int pn = 0;
	int gn = 0;
	int rn = 0;
	int fn = 0;
	int kernelType = -1;

	try {
		// Split arguments
		fileName = argv[1];
		pn = stoi(argv[2]);
		gn = stoi(argv[3]);
		rn = stoi(argv[4]);
		fn = stoi(argv[5]);
		kernelType = stoi(argv[6]);
	}
	catch (std::invalid_argument e) {
		cerr << "Found wrong parameter!" << endl;
		printUsage();
		return -1;
	}
	catch (std::out_of_range e) {
		cerr << "Found out of range parameter!" << endl;
		printUsage();
		return -1;
	}

	if (pn < 1) {
		cerr << "The number of people must be positive!" << endl;
		printUsage();
		return -1;
	}

	if (gn < 1) {
		cerr << "The number of gesture classes must be positive!" << endl;
		printUsage();
		return -1;
	}

	if (rn < 1) {
		cerr << "The number of repetitions for each gesture must be positive!" << endl;
		printUsage();
		return -1;
	}

	if (fn < 1) {
		cerr << "The number of features must be positive!" << endl;
		printUsage();
		return -1;
	}

	SVM::KernelTypes kernel;
	switch (kernelType) {
	case 0: kernel = SVM::KernelTypes::LINEAR; break;
	case 1: kernel = SVM::KernelTypes::POLY; break;
	case 2: kernel = SVM::KernelTypes::RBF; break;
	case 3: kernel = SVM::KernelTypes::SIGMOID; break;
	case 4: kernel = SVM::KernelTypes::CHI2; break;
	case 5: kernel = SVM::KernelTypes::INTER; break;
	default: cerr << "Invalid kernel type!" << endl;
		printUsage();
		return -1;
	}

	// Open log file	
	logFile.open(fileName + "_log.txt");
	if (!logFile.is_open()) {
		cerr << "Unable to create log file!" << endl;		
		return -1;
	}

	cout << "Dataset file name: " << fileName << endl;
	cout << "Number of people: " << pn << endl;
	cout << "Number of gesture classes: " << gn << endl;
	cout << "Number of repetitions per gesture: " << rn << endl;
	cout << "Number of features: " << fn << endl;
	cout << "Kernel type: " << kernelType << endl;
	logFile << "Dataset file name: " << fileName << endl;
	logFile << "Number of people: " << pn << endl;
	logFile << "Number of gesture classes: " << gn << endl;
	logFile << "Number of repetitions per gesture: " << rn << endl;
	logFile << "Number of features: " << fn << endl;
	logFile << "Kernel type: " << kernelType << endl;

	// Read dataset
	Mat_<float> dataset;
	bool res = readDataset(fileName, pn, gn, rn, fn, dataset);
	if (!res) {
		logFile.close();
		return -1;
	}
	else {
		// Call training method
		trainSVMParallel(dataset, pn, gn, rn, fn, kernel, fileName + "_confusionMatrix.dat");
		logFile.close();
	}	
	return 0;
}
