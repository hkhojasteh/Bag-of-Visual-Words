//This program created by love by Hadi - Copyright(c) by Hadi Abdi Khojasteh. All right reserved. / Email: hkhojasteh [at] iasbs [dot] ac [dot] ir, info [at] hadiabdikhojasteh [dot] ir / Website: iasbs.ac.ir/~hkhojasteh, hadiabdikhojasteh.ir
<<<<<<< HEAD


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
void readDetectComputeimage(string className, int imageNumbers, int classLable) {
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

	cout << "\nObject detector demo ended. press any key for exit.\nFor more information contact hkhojasteh@iasbs.ac.ir";
 	cin.get();
	return(0);
}
=======
>>>>>>> 9bd6da33561f3c178f29c68ace7c584c55e95022
