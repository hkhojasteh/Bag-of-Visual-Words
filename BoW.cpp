//This program created by love by Hadi - Copyright(c) by Hadi Abdi Khojasteh. All right reserved. / Email: hkhojasteh [at] iasbs [dot] ac [dot] ir, info [at] hadiabdikhojasteh [dot] ir / Website: iasbs.ac.ir/~hkhojasteh, hadiabdikhojasteh.ir


#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml/ml.hpp>

//#include <Windows.h>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <omp.h>  

#define NUM_THREADS 8  

using namespace std;
using namespace cv;
using namespace cv::ml;

const string DATASET_PATH = "\\101_ObjectCategories\\";
const string IMAGE_EXT = ".jpg";
const int TESTING_PERCENT_PER = 7;
const int DICT_SIZE = 230;	//80 word per class

inline bool fileExists(const std::string& name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}

Mat allDescriptors;
vector<Mat> allDescPerImg;
vector<int> allClassPerImg;
int allDescPerImgNum = 0;
void readDetectComputeimage(const string& className, int imageNumbers, int classLable) {
#pragma omp parallel
{
#pragma omp for schedule(dynamic) ordered
	for (int i = 1; i <= imageNumbers; i++) {
		//If this image is test not use this in learning
		if (i % TESTING_PERCENT_PER == 0) {
			continue;
		}
		ostringstream ss;
		Mat grayimg;
		Ptr<xfeatures2d::SIFT> siftptr;
		siftptr = xfeatures2d::SIFT::create();

		//Load image, Detect and Describe features
		ss.str("");
		ss << std::setw(4) << std::setfill('0') << i;
		if (fileExists(DATASET_PATH + className + "\\image_" + ss.str() + IMAGE_EXT)) {
			cvtColor(imread(DATASET_PATH + className + "\\image_" + ss.str() + IMAGE_EXT), grayimg, CV_BGR2GRAY);

			vector<KeyPoint> keypoints;
			Mat descriptors;
			siftptr->detectAndCompute(grayimg, noArray(), keypoints, descriptors);
#pragma omp critical
			{
				allDescriptors.push_back(descriptors);
				allDescPerImg.push_back(descriptors);
				allClassPerImg.push_back(classLable);
				allDescPerImgNum++;
			}
			
			/*Mat output;
			drawKeypoints(grayimg, keypoints, output, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
			imshow("Result [" + ss.str() + IMAGE_EXT + "]", output);
			waitKey(10);
			destroyWindow("Result [" + ss.str() + ".jpg]");*/
		}else{
			break;
		}
	}
}
}

Mat kCenters, kLabels;
Mat getDataVector(Mat descriptors) {
	BFMatcher matcher;
	vector<DMatch> matches;
	matcher.match(descriptors, kCenters, matches);

	//Make a Histogram of visual words
	Mat datai = Mat::zeros(1, DICT_SIZE, CV_32F);
	int index = 0;
	for (auto j = matches.begin(); j < matches.end(); j++, index++) {
		datai.at<float>(0, matches.at(index).trainIdx) = datai.at<float>(0, matches.at(index).trainIdx) + 1;
	}
	return datai;
}

Mat inputData;
Mat inputDataLables;
void getHistogram(const string& className, int imageNumbers, int classLable) {
#pragma omp parallel
{
#pragma omp for schedule(dynamic) ordered
	for (int i = 1; i <= imageNumbers; i++) {
		//If this image is test not use this in learning
		if (i % TESTING_PERCENT_PER == 0) {
			continue;
		}
		ostringstream ss;
		Mat grayimg;
		Ptr<xfeatures2d::SIFT> siftptr;
		siftptr = xfeatures2d::SIFT::create();
		//Load image, Detect and Describe features
		ss.str("");
		ss << std::setw(4) << std::setfill('0') << i;
		if (fileExists(DATASET_PATH + className + "\\image_" + ss.str() + IMAGE_EXT)) {
			cvtColor(imread(DATASET_PATH + className + "\\image_" + ss.str() + IMAGE_EXT), grayimg, CV_BGR2GRAY);
			vector<KeyPoint> keypoints;
			Mat descriptors;
			siftptr->detectAndCompute(grayimg, noArray(), keypoints, descriptors);
			
#pragma omp critical
			{
				inputData.push_back(getDataVector(descriptors));
				inputDataLables.push_back(Mat(1, 1, CV_32SC1, classLable));
			}
		}else{
			break;
		}
	}
}
}

void getHistogramFast() {
#pragma omp parallel
{
#pragma omp for schedule(dynamic) ordered
	for (int i = 0; i < allDescPerImgNum; i++) {
		Mat dvec = getDataVector(allDescPerImg[i]);
#pragma omp critical
		{
			inputData.push_back(dvec);
			inputDataLables.push_back(Mat(1, 1, CV_32SC1, allClassPerImg[i]));
		}
	}
}
}

Ptr<SVM> svm;
double testData(const string& className, int imageNumbers, int classLable) {
	int allTests = 0;
	int correctTests = 0;
#pragma omp parallel
{
#pragma omp for schedule(dynamic) ordered
	for (int i = TESTING_PERCENT_PER; i <= imageNumbers; i += TESTING_PERCENT_PER) {
		ostringstream ss;
		Mat grayimg;
		Ptr<xfeatures2d::SIFT> siftptr;
		float r = 0;
		siftptr = xfeatures2d::SIFT::create();
		//Load image, Detect and Describe features
		ss.str("");
		ss << std::setw(4) << std::setfill('0') << i;
		if (fileExists(DATASET_PATH + className + "\\image_" + ss.str() + IMAGE_EXT)) {
			cvtColor(imread(DATASET_PATH + className + "\\image_" + ss.str() + IMAGE_EXT), grayimg, CV_BGR2GRAY);
			vector<KeyPoint> keypoints;
			Mat descriptors;
			siftptr->detectAndCompute(grayimg, noArray(), keypoints, descriptors);
			Mat dvector = getDataVector(descriptors);
#pragma omp critical
			{
				allTests++;
				if (svm->predict(dvector) == classLable) {
					correctTests++;
				}
			}
		}else{
			break;
		}
	}
}
return (double)correctTests / allTests;
}

int main(int argc, char **argv)
{
	cout << "Object detector started." << endl;
	clock_t sTime = clock();
	cout << "Reading inputs..." << endl;
	readDetectComputeimage("starfish", 86, 1);
	readDetectComputeimage("sunflower", 85, 2);
	//readDetectComputeimage("crab", 75, 3);
	//readDetectComputeimage("trilobite", 86, 4);
	cout << "-> Reading, Detect and Describe input in " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " Second(s)." << endl;

	int clusterCount = DICT_SIZE, attempts = 5, iterationNumber = 1e4;
	sTime = clock();
	cout << "Running kmeans..." << endl;
	kmeans(allDescriptors, clusterCount, kLabels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, iterationNumber, 1e-4), attempts, KMEANS_PP_CENTERS, kCenters);
	cout << "-> kmeans run in " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " Second(s)." << endl;

	/*FileStorage storage("kmeans-starfish,sunflower,crab,trilobite.yml", FileStorage::WRITE);
	storage << "kLabels" << kLabels << "kCenters" << kCenters;
	storage.release();*/

	sTime = clock();
	/*cout << "Loading kmeans data..." << endl;
	FileStorage storage("kmeans-starfish,sunflower,crab,trilobite.yml", FileStorage::READ);
	storage["kLabels"] >> kLabels;
	storage["kCenters"] >> kCenters;
	storage.release();
	cout << "-> kmeans data loaded in " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " Second(s)." << endl;*/

	sTime = clock();
	cout << "Finding histograms..." << endl;
	getHistogramFast();
	/*getHistogram("starfish", 86, 1);
	getHistogram("sunflower", 85, 2);
	getHistogram("crab", 75, 3);
	getHistogram("trilobite", 86, 4);*/
	cout << "-> Histograms find in " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " Second(s)." << endl;

	sTime = clock();
	cout << "SVM training..." << endl;
	// Set up SVM's parameters
	svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1e4, 1e-6));
	// Train the SVM with given parameters
	Ptr<TrainData> td = TrainData::create(inputData, ROW_SAMPLE, inputDataLables);
	svm->train(td);
	cout << "-> SVM trained in " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " Second(s)." << endl;

	sTime = clock();
	cout << "Testing images..." << endl;
	cout << "-> " << (float)(testData("starfish", 86, 1) * 100) << "% accuracy in 'starfish' class." << endl;
	cout << "-> " << (float)(testData("sunflower", 85, 2) * 100) << "% accuracy in 'sunflower' class." << endl;
	//cout << "-> " << (float)(testData("crab", 75, 3) * 100) << "% accuracy in 'crab' class." << endl;
	//cout << "-> " << (float)(testData("trilobite", 86, 4) * 100) << "% accuracy in 'trilobite' class." << endl;
	cout << "-> Test completed in " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " Second(s)." << endl;
	
	//imwrite("sift_result.jpg", output);
	cout << "\nObject detector demo ended. press any key for exit.\nFor more information contact hkhojasteh [at] iasbs [dot] ac [dot] ir";
 	cin.get();
	return(0);
}
