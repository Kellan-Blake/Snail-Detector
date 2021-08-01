// **************************************** Snail Detector *****************************************
// By:		Mark Sorvik & Kellan Blake
// Date:	2019
// Purpose: This program detects the presence and location of "the snail" in frames from the series 
//			Adventure Time.
// File:	This is the main cpp file that will handle the IO and image processing of the Snail 
//			Detector program.
// *************************************************************************************************  
																			  // 100 Char Line >>>>|

#include "SnailDetector.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <iostream>
using namespace std;
using namespace cv;

//Threshold value for possible snail detections
const double thresholdValue = 0.82;

//Set to false to only print the output images.
const bool testMode = true;


// This the main method. It reads the starting image files from the sourceImages folder, calls the 
// other methods that smooth, convolve, and otherwise detect the snail, and saves/maybe displays the 
// output images.
// Pre-conditions: threshold value is a constant variable already written, there are input images in
// the folder SourceImages, and snail filter images in the folder FiltersFolder, as well as a folder
// OutputImages to output the results of the program.
// Post-conditions: Any source images are filtered using the snail filters and if the snail is in
// the image it is output to a folder with a rectangle around the snail. If there is no snail in the image
// the original source image is output to the OutputImages folder with no rectangle.
int main() {

	//Print threshold for reference.
	cout << "Threshold value set: " << thresholdValue << endl;

	//Create vectir for the snailFilters.
	vector<Mat> snailFilters;

	//Start at filter number 0
	int filterNum = 0;

	//While there are filters in the folder, read them into the program and increment the filter number.
	while (imread("FiltersFolder/" + std::to_string(filterNum) + ".jpg").data != NULL) {
		Mat thisSnailFilter = imread("FiltersFolder/" + std::to_string(filterNum) + ".jpg");
		snailFilters.push_back(thisSnailFilter);
		filterNum++;
	}
	
	int minCols = 0;
	int minRows = 0;
	
	//For as many snail filters as there are set the minCols and minRows based on the number of rows
	//and columns in the snail filters.
	for (int numFilters = 0; numFilters < snailFilters.size(); numFilters++) {
		if (minCols == 0 || snailFilters.at(numFilters).cols) {
			minCols = snailFilters.at(numFilters).cols;
		}
		if (minRows == 0 || snailFilters.at(numFilters).rows) {
			minRows = snailFilters.at(numFilters).rows;
		}
	}

	//Set the minimum filter dimensions based on the minCols and minRows
	//variables.
	Size minFilterDims = Size(minCols, minRows);

	//Set image number to 1 for reading in images.
	int imageNum = 1;

	//Will loop until the next number of image in SourceImages/# does not exist
	while (imread("SourceImages/" + std::to_string(imageNum) + ".jpg").data != NULL) {
		//Set snail found boolean flag to false.
		bool snailFound = false;

		//Create a vector for the current result images that have been made.
		vector<pair<Mat, double>> CurrentImageResults;

		//Set current image to the source image with number imageNum.
		Mat currentImage = imread("SourceImages/" + std::to_string(imageNum) + ".jpg");

		//If testMode is set to true write the non-blurred image.
		if (testMode) {
			// output the current read image to the test folder
			imwrite("TestOutputImages/BlurTesting/" + std::to_string(imageNum) + ".jpg", currentImage);
		}

		//Blur the image for more accurate snail detection.
		GaussianBlur(currentImage, currentImage, Size(5, 5), 0, 0, BORDER_REPLICATE);

		//If test mode is set to true write the blurred images.
		if (testMode) {
			// output the blurred image to the blurtest folder
			imwrite("TestOutputImages/BlurTesting/" + std::to_string(imageNum) + "BLURRED.jpg", currentImage);
		}
		

		// create pyramid of images section
		vector<std::pair<Mat, int>> pyramid; // holds the pyramid of image sizes for this image
		createPyramid(currentImage, minRows, minCols, pyramid);

		//If testMode is true then write the pyramid of images
		if (testMode) {
			printPyramidTestOutput(pyramid, imageNum); // outputs the pyramid to the test folder

		}
		
		//Output which image and image in the pyramid is being examined.
		cout << pyramid.size() << " rotations to complete image " << imageNum << ":" << endl;

		// loop through each image in the pyramid
		for (int pyramidImageNum = 0; pyramidImageNum < pyramid.size(); pyramidImageNum++) {
			 
			//Cout number of dashes to complete examination of this image due to there
			//being 24 rotations.
			cout << "24 - to complete rotation " << (pyramidImageNum + 1) << ":";
			// rotate the current image in pyramid 24 times in 15 degree increments (360 degrees)
			for (int rotation = 0; rotation < 24; rotation++) {

				//Print a dash for each rotation
				cout << "-";

				// create the rotated image
				Mat currentRotation = pyramid.at(pyramidImageNum).first.clone();
				double currentDegrees = (double)rotation * (double)15;
				rotateImage(currentRotation, currentDegrees);

				if (testMode) {
					// output the rotated image to the test folder
					imwrite("TestOutputImages/RotateTesting/" + std::to_string(imageNum) + "_" +
						std::to_string(pyramidImageNum) + "_" + std::to_string(rotation * 15) +
						".jpg", currentRotation);
				}

				// loop through the current rotated image twice, to facilitate flipping
				for (int flip = 0; flip < 2; flip++) {

					// on the second pass flip this image
					if (flip == 1) { flipImageHorizontally(currentRotation); }

					if (testMode) {
						// output the flipped/not flipped image to the test folder
						imwrite("TestOutputImages/FlipTesting/" + std::to_string(imageNum) + "_" +
							std::to_string(pyramidImageNum) + "_" + std::to_string(rotation * 15) +
							"_" + std::to_string(flip) + ".jpg", currentRotation);
					}

					// THE REST OF THE DETECTOR GOES HERE, perform detection on currentRotation
					
					//For current snail filter size.
					for (int thisFilter = 0; thisFilter < snailFilters.size(); thisFilter++) {
						//If currentRotation is bigger or equal to the filter then try and detect the snail.
						if (currentRotation.rows >= snailFilters.at(thisFilter).rows && currentRotation.cols >= snailFilters.at(thisFilter).cols) {
							double currentResult;
							//Try and detect snail in currentRotation
							Mat templateOutput = templateDetect(currentRotation, snailFilters.at(thisFilter), currentResult);

							
							if (countNonZero(templateOutput != currentRotation) != 0) {
								//Print the percentage match of the snail found in the image and put in the results vector.
								cout << endl << "        Filter: " << thisFilter << " = Match of " << currentResult << "%";
								CurrentImageResults.push_back(pair<Mat, double>(templateOutput, currentResult));

								//If the current result is greater than 93 percent then set variables to where the loop will end.
								if (currentResult > 0.93) {
									thisFilter = snailFilters.size();
									flip = 2;
									rotation = 24;
									pyramidImageNum = pyramid.size();
								}
								//Set snail found to true if result is higher than threshold.
								snailFound = true;
							
							}
						}

					}
					
					
				}
			}
			//Print that this image is complete.
			cout << endl << " >>>>>>>>>>>>>>> Rotation " << (pyramidImageNum + 1) << ": Complete" << endl;
		}
		if (snailFound) {
			//Make sure the best result is saved and check to make sure it is the best result for this image.
			Mat finalResult = CurrentImageResults.at(0).first.clone();
			double bestScore = CurrentImageResults.at(0).second;
			for (int hits = 0; hits < CurrentImageResults.size(); hits++) {
				if (CurrentImageResults.at(hits).second > bestScore) {
					finalResult = CurrentImageResults.at(hits).first.clone();
				}
			}
			//Output the image to a folder.
			imwrite("OutputImages/" + std::to_string(imageNum) + "_" +
				"FOUND.jpg", finalResult);
		}
		
		//Print that the image has been checked for the snail and report
		//the result of whether or not the snail was found and increase image number.
		cout << "Image " << imageNum << " Complete : ";
		if (snailFound) { cout << "Snail FOUND" << endl << endl; }
		else { cout << "Snail NOT FOUND" << endl << endl; }
		imageNum++;
	}
	//Print that program has finished.
	cout << "This program ran to the end!" << endl;
}

//This function tries to detect the snail in the current image with the current snailFilter
//and keeps track of what the best percentage hit is for the snail being in the image.
//Credit to OpenCV for this template matching tutorial where some of the code in this
//function is from pertaining to the matchTemplate function, and drawing the rectangles
//around the match. https://www.docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template_matching/template_matching.html
//Preconditions: originalImage and snailFilter are valid images, hitPercent is a double variable that is valid.
//Postconditions: Snail is either found and an image with a rectangle where the snail is is returned, or the snail
//is not found and the originalImage is returned.
Mat templateDetect(const Mat& originalImage, const Mat& snailFilter, double& hitPercent) {
	
	Mat output = originalImage.clone();
	Mat displayImage = originalImage.clone();

	//Do template matching to find snail in originalImage.
	matchTemplate(originalImage, snailFilter, output, TM_CCOEFF_NORMED);

	double* maxValue = new double;

	//Gets the max element and its location.
	minMaxLoc(output, NULL, maxValue, NULL, NULL);

	//If maxValue is bigger than the threshold
	if (*maxValue >= thresholdValue) {
		//Make the max the hit percent.
		hitPercent = *maxValue;
		delete(maxValue);
		maxValue = NULL;
		//Normalize the output image.
		normalize(output, output, 0, 255, NORM_MINMAX, CV_8U);
		double minVal;
		double maxVal;
		Point minLoc;
		Point maxLoc;
		Point matchLoc;
		//Get the minimum and maximum locations
		minMaxLoc(output, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
		//Set the location of the match to maxLoc
		matchLoc = maxLoc;
		//Put rectangles around the match in both displayImage and output
		rectangle(displayImage, matchLoc, Point(matchLoc.x + snailFilter.cols, matchLoc.y + snailFilter.rows), Scalar::all(0), 2, 8, 0);
		rectangle(output, matchLoc, Point(matchLoc.x + snailFilter.cols, matchLoc.y + snailFilter.rows), Scalar::all(0), 2, 8, 0);
		//Return the colored originalImage with the rectangle from the match
		return displayImage;
	}
	delete(maxValue);
	maxValue = NULL;
	//Return the originalImage if no match is found.
	return originalImage;
}


// this method will create most of a Gaussian Pyramid and return the images as a vector of paired 
// values containing the image, and how many times it was reduced. This reduction data will be used 
// later to get from the snail location in a reduced image to the location in the original
// Pre-conditions: All parameters exist and are valid.
// Post-conditions: The current image has a Gaussian pyramid created from it.
void createPyramid(const Mat &originalImage, int minRows, int minCols , vector<pair<Mat, int>> 
	&outputVector) {
	// loop will continue until image is smaller than the snail (horizontally at least, but we will 
	// assume that the aspect ratio of the image will not be taller than it is wide).

	outputVector.push_back(pair<Mat, int>(originalImage.clone(), 0));
	// loop will continue as long as image is larger than the snail filter both horizongally and
	// vertically
	int colSize = originalImage.cols;
	int rowSize = originalImage.rows;
	int reductionNumber = 1;
	while (rowSize / 2 > minRows && colSize / 2 > minCols) {
		rowSize = rowSize / 2;
		colSize = colSize / 2;
		Mat outputImage = outputVector.at(reductionNumber - 1).first.clone();
		cv::resize(outputImage, outputImage, Size(colSize, rowSize), 1.0, 1.0, INTER_AREA);
		outputVector.push_back(pair<Mat, int>(outputImage, reductionNumber));
		reductionNumber++;

	}
}


// testing method that prints the output of createPyramid (saves to folders)
// Pre-conditions: Vector is filled with pyramid of images
// Post-conditions: Outputs pyramid of images as image files in TestOutputImages/PyramidTesting/
void printPyramidTestOutput(vector<pair<Mat, int>>& inputVector, int imageNum) {
	for (int i = 0; i < inputVector.size(); i++) {
		imwrite("TestOutputImages/PyramidTesting/" + std::to_string(imageNum) + "_" + 
			std::to_string(i) + ".jpg", inputVector.at(i).first);
	}
}


// this method will rotate the image by the specified amount of degrees and will not cut off any 
// part of the original image
// Pre-conditions:
// Post-conditions:
void rotateImage(Mat& originalImage, double degrees) {

	// calculate center of image to rotate around
	int centerCol = (originalImage.cols - 1) / 2;
	int centerRow = (originalImage.rows - 1) / 2;

	// calculate the rotation matrix used to transform the image by the set angle
	Mat rotationMatrix = getRotationMatrix2D(Point(centerCol, centerRow), degrees, 1.0);

	// This function creates a rotated rectangle on a plane, centered around 0.0, rotated by degrees
	RotatedRect boxContainingImage(Point2f(0, 0), originalImage.size(), degrees);
	// This function "returns the minimal up-right integer rectangle containing the rotated 
	// rectangle" (openCV documentation). We can then use the size of this rectangle as the
	// size of the output for our rotated image. We can also use it to determine the translation 
	Rect outsideBorderBoundary = boxContainingImage.boundingRect();

	// adjusting the transformation in the x direction by 1/2 the added width of the outer boundary
	rotationMatrix.at<double>(0, 2) += (outsideBorderBoundary.width / 2.0) - 
		(originalImage.cols / 2.0);
	// adjusting the transformation in the y direction by 1/2 the added length of the outer boundary
	rotationMatrix.at<double>(1, 2) += (outsideBorderBoundary.height / 2.0) - 
		(originalImage.rows / 2.0);

	Mat outputImage;
	// perform the actual rotation using the adjusted rotation matrix, onto an output of the size 
	// determined by boxContainingImage's size
	warpAffine(originalImage, outputImage, rotationMatrix, outsideBorderBoundary.size());

	originalImage = outputImage.clone();
}


// this method will flip the image horizontally
// Pre-conditions: Image is valid and exists
// Post-conditions: Image is flipped horizontally.
void flipImageHorizontally(Mat& originalImage) {
	Mat flippedImage = originalImage.clone();
	for (int row = 0; row < originalImage.rows; row++) {
		for (int col = 0; col < originalImage.cols; col++) {
			flippedImage.at<Vec3b>(row, originalImage.cols - 1 - col) = 
				originalImage.at<Vec3b>(row, col);
		}
	}
	originalImage = flippedImage.clone();
}