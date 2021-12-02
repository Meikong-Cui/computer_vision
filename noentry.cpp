#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <math.h>

using namespace std;
using namespace cv;

void detectAndDisplay( Mat frame );
void sobelXY(Mat &input, int size, Mat &outputX, Mat &outputY);
void houghTransform(int image, Mat &magnitude, Mat &direction);

String cascade_name = "./NoEntrycascade/cascade.xml";
CascadeClassifier cascade;

/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame );

	// 4. Save Result Image
	// imwrite( "detected.jpg", frame );
    // imshow("detected", frame);
	return 0;
}

void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;
    Mat gray;
	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

    cvtColor( frame, gray, CV_BGR2GRAY );
    GaussianBlur(gray, gray, Size(3, 3), 0, 0, BORDER_REPLICATE);

    Mat sobelx;
    Mat sobely;
    sobelXY(gray, 3, sobelx, sobely);

    Mat magnitude;
    magnitude.create(gray.size(), DataType<float>::type);
    cv::magnitude(sobelx, sobely, magnitude);

    Mat direction;
    cv::phase(sobelx, sobely, direction, false);
    // direction = direction * 180 / 3.1416;
    normalize(magnitude, magnitude, 0, 255, NORM_MINMAX, CV_32F);

    // imwrite("magnitude.jpg", magnitude);
    // imwrite("direction.jpg", direction);
}

void houghTransform(int image, Mat &magnitude, Mat &direction) {

}

void sobelXY(cv::Mat &input, int size, cv::Mat &outputX, cv::Mat &outputY) {
    outputX.create(input.size(), DataType<float>::type);
    outputY.create(input.size(), DataType<float>::type);

    Mat kX = getGaussianKernel(size, -1);
    Mat kY = getGaussianKernel(size, -1);
    Mat sobelx = kX * kY.t();
    Mat sobely = kX * kY.t();

    int kernelRadiusX = (sobelx.size[0] - 1) / 2;
	int kernelRadiusY = (sobely.size[1] - 1) / 2;

    int dx[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    int dy[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
    int index = 0;
    for (int a = -kernelRadiusX; a <= kernelRadiusX; a++) {
        for (int b = -kernelRadiusY; b <= kernelRadiusY; b++) {
            sobelx.at<double>(a + kernelRadiusX, b + kernelRadiusY) = dx[index];
            sobely.at<double>(a + kernelRadiusY, b + kernelRadiusX) = dy[index];
            index++;
        }
    }

    cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput, 
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

    for ( int i = 0; i < input.rows; i++ ) {	
		for( int j = 0; j < input.cols; j++ ) {
			double sumX = 0.0;
            double sumY = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ ) {
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ ) {
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					int imageval = (int) paddedInput.at<uchar>( imagex, imagey );
					double kernalXval = sobelx.at<double>( kernelx, kernely );
                    double kernalYval = sobely.at<double>( kernelx, kernely );

					sumX += imageval * kernalXval;
                    sumY += imageval * kernalYval;
				}
			}
			outputX.at<float>(i, j) = (float)sumX;
            outputY.at<float>(i, j) = (float)sumY;
		}
	}
}
