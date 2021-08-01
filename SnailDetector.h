#pragma once
// **************************************** Snail Detector *****************************************
// By:		Mark Sorvik & Kellan Blake
// Date:	2019
// Purpose: This program detects the presence and location of "the snail" in frames from the series 
//			Adventure Time.
// File:	This is the header file for the main cpp file of the Snail Detector, it describes the 
//			interface and methods of SnailDetector.cpp.
// *************************************************************************************************  
																			  // 100 Char Line >>>>|

#include <opencv2/highgui.hpp>
using namespace cv;
using namespace std;

// This the main method. It reads the starting image files from the sourceImages folder, calls the 
// other methods that smooth, convolve, and otherwise detect the snail, and saves/maybe displays the 
// output images.
// Pre-conditions:
// Post-conditions:
int main();

// this method initializes and returns a smoothing kernel
// Pre-conditions: None
// Post-conditions: Returns a smoothing kernel to be assigned to a Mat
Mat initSmoothingKernel();

// this method will create most of a Gaussian Pyramid and return the images as a vector of paired 
// values containing the image, and how many times it was reduced. This reduction data will be used 
// later to get from the snail location in a reduced image to the location in the original
// Pre-conditions:
// Post-conditions:
void createPyramid(const Mat& originalImage, int minRows, int minCols, vector<pair<Mat, int>> 
	&outputVector);

// testing method that prints the output of createPyramid (saves to folders)
// Pre-conditions: Vector is filled with pyramid of images
// Post-conditions: Outputs pyramid of images as image files in TestOutputImages/PyramidTesting/
void printPyramidTestOutput(vector<pair<Mat, int>>& inputVector, int imageNum);


// this method will rotate the image by the specified amount of degrees and will not cut off any 
// part of the original image
// Pre-conditions:
// Post-conditions:
void rotateImage(Mat& originalImage, double degrees);


// this method will flip the image horizontally
// Pre-conditions:
// Post-conditions:
void flipImageHorizontally(Mat& originalImage);

//Function to detect snail in the image.
Mat templateDetect(const Mat& originalImage, const Mat& snailFilter, double& hitPercent);