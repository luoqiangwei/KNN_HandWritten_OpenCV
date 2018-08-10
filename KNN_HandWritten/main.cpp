#include <iostream>
#include <cstdlib>
#include <ctime>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int fun1();
int readFlippedInteger(FILE *fp);

int main()
{
	fun1();
	return 0;
}

int fun1()
{
	const int K = 14;

	/*FILE *fpi = fopen("train-images.idx3-ubyte", "r");
	FILE *fpl = fopen("train-labels.idx1-ubyte", "r");*/
	FILE *fpi = fopen("train-images.idx3-ubyte", "rb");
	FILE *fpl = fopen("train-labels.idx1-ubyte", "rb");

	if (!fpi || !fpl)
	{
		cout << "Files not Found" << endl;
		return 0;
	}

	int magicNumber = readFlippedInteger(fpi);
	int numImages = readFlippedInteger(fpi);
	int numRows = readFlippedInteger(fpi);
	int numCols = readFlippedInteger(fpi);

	fseek(fpl, 0x08, SEEK_SET);

	int size = numRows * numCols;

	Mat trainingVectors(numImages, size, CV_32FC1);
	Mat trainingLabels(numImages, 1, CV_32FC1);

	//index3 train info
	//cout << magicNumber << " " << numImages << " " << numRows << " " << numCols << " " << size << endl;

	uchar *temp = new uchar[size];

	uchar tempClass = 0;

	Mat img(numRows, numCols, CV_32FC1);
	for (int i = 0; i < numImages; i++)
	{
		fread((void*)temp, size, 1, fpi);
		fread((void*)(&tempClass), sizeof(uchar), 1, fpl);
		trainingLabels.at<float>(i, 0) = tempClass;
		
		for (int k = 0; k < size; k++) 
		{
			trainingVectors.at<float>(i, k) = temp[k];
			//img.at<float>(k / numCols, k % numCols) = temp[k];
		}
		//cout << sizeof(temp) << endl;
		//imshow("data", img);
	}

	Ptr<ml::KNearest> knn = ml::KNearest::create();
	knn->setDefaultK(K);
	knn->setIsClassifier(true);
	knn->setAlgorithmType(ml::KNearest::BRUTE_FORCE);
	knn->train(trainingVectors, ml::ROW_SAMPLE, trainingLabels);
	

	fclose(fpi);
	fclose(fpl);
	delete[] temp;

	fpi = fopen("t10k-images.idx3-ubyte", "rb");
	fpl = fopen("t10k-labels.idx1-ubyte", "rb");

	magicNumber = readFlippedInteger(fpi);
	numImages = readFlippedInteger(fpi);
	numRows = readFlippedInteger(fpi);
	numCols = readFlippedInteger(fpi);

	size = numRows * numCols;

	fseek(fpl, 0x08, SEEK_SET);

	Mat testVectors(numImages, size, CV_32FC1);
	Mat testLabels(numImages, 1, CV_32FC1);
	Mat actualLabels(numImages, 1, CV_32FC1);
	temp = new uchar[size];
	tempClass = 1;
	Mat currentTest(1, size, CV_32FC1);
	Mat currentLabel(1, 1, CV_32FC1);
	int totalCorrect = 0;

	//cout << numImages << endl;

	for (int i = 0; i < numImages; i++)
	{
		fread((void*)temp, size, 1, fpi);
		fread((void*)(&tempClass), sizeof(uchar), 1, fpl);
		actualLabels.at<float>(i) = tempClass;
		for (int k = 0; k < size; k++)
		{
			testVectors.at<float>(i, k) = temp[k];
			currentTest.at<float>(k) = temp[k];
		}
		knn->findNearest(currentTest, K, currentLabel);
		testLabels.at<float>(i) = currentLabel.at<float>(0);
		if (currentLabel.at<float>(0) == actualLabels.at<float>(i)) 
		{
			totalCorrect++;
		}
		cout << "Time: " << i + 1 << " Accuracy: " << totalCorrect << "/" << numImages << endl;
	}
	cout << "KNN K = " << K << "  Accuracy: " << (double)totalCorrect / (double)numImages << endl;

	fclose(fpi);
	fclose(fpl);
	delete[] temp;

	return 0;
}

//THE IDX FILE FORMAT
//the IDX file format is a simple format for vectors and multidimensional matrices of various numerical types.
//The basic format is
//
//magic number
//size in dimension 0
//size in dimension 1
//size in dimension 2
//.....
//size in dimension N
//data
//
//The magic number is an integer(MSB first).The first 2 bytes are always 0.
//
//The third byte codes the type of the data :
//0x08 : unsigned byte
//0x09 : signed byte
//0x0B : short(2 bytes)
//0x0C : int(4 bytes)
//0x0D : float(4 bytes)
//0x0E : double(8 bytes)
//
//The 4 - th byte codes the number of dimensions of the vector / matrix : 1 for vectors, 2 for matrices....
//
//The sizes in each dimension are 4 - byte integers(MSB first, high endian, like in most non - Intel processors).
//
//The data is stored like in a C array, i.e.the index in the last dimension changes the fastest.
int readFlippedInteger(FILE *fp){ //每次读取4个字节 index3 文件前16个字节是说明部分，对之后的数据格式化很有用。index1文件前8个字节是说明部分
	int ret = 0;
	uchar *tmp;
	tmp = (uchar*)(&ret);
	fread(&tmp[3], sizeof(uchar), 1, fp);
	fread(&tmp[2], sizeof(uchar), 1, fp);
	fread(&tmp[1], sizeof(uchar), 1, fp);
	fread(&tmp[0], sizeof(uchar), 1, fp);
	return ret;
}