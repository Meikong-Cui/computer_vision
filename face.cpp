/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
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

/** Function Headers */
void beforeTest(Mat frame);
void detectAndDisplay( Mat frame );

int calculateTP(vector<Rect> &faces, Rect groundTruth);

/** Global variables */
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
	beforeTest(frame);
	detectAndDisplay( frame );

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}

int calculateTP(vector<Rect> &faces, Rect groundTruth) {
	int TPcount = 0;
	for(size_t i =0; i < faces.size(); i++) {
		float startX = (faces[i].x < groundTruth.x) ? faces[i].x : groundTruth.x;
		float startY = (faces[i].y < groundTruth.y) ? faces[i].y : groundTruth.y;
		float endX = ((faces[i].x + faces[i].width) > (groundTruth.x + groundTruth.width)) ? (faces[i].x + faces[i].width) : (groundTruth.x + groundTruth.width);
		float endY = ((faces[i].y + faces[i].height) > (groundTruth.y + groundTruth.height)) ? (faces[i].y + faces[i].height) : (groundTruth.y + groundTruth.height);
		int x1 = 0, y1 = 0;
		for(int a = int(startX); a <= int(endX); a++) {
			if(a>faces[i].x && a>groundTruth.x && a<(faces[i].x + faces[i].width) && a<(groundTruth.x + groundTruth.width)) x1++;
		}
		for(int b = int(startY); b <= int(endY); b++) {
			if(b>faces[i].y && b>groundTruth.y && b<(faces[i].y + faces[i].height) && b<(groundTruth.y + groundTruth.height)) y1++;
		}
		float area1 = groundTruth.width * groundTruth.height;
		float area2 = faces[i].width * faces[i].height;
		float IOU = x1*y1 / (area1 + area2 - x1*y1);
		if(IOU > 0.4f) TPcount++;		
	}
	// std::cout << "True Positive rate: " << std::min(TPcount/groundTruth_f, 1) * 100 << '%' << std::endl;
	// std::cout << "F1 score: " << TPcount/(TPcount + 0.5*(faces.size()-TPcount + std::max(0, groundTruth_f - TPcount))) << std::endl;
	return TPcount;
}

void beforeTest(Mat frame) {
	std::vector<Rect> faces;
	Mat frame_gray;
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );
	imwrite("gray.jpg", frame_gray);
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(10, 10), Size(300,300) );

       // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

	int groundTruth_f = 2;
       // 4. Draw box around faces found
	// Rect NoEntry4_1(98, 243, 77, 77);
	// Rect NoEntry4_2(244, 185, 46, 47);
	// Rect NoEntry4_3(300, 120, 24, 39);
	// Rect NoEntry4_4(437, 85, 22, 28);
	// Rect NoEntry4_5(660, 120, 29, 36);
	// Rect NoEntry4_6(819, 252, 74, 76);

	// int groundTruth_f = 1;

	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

	// int TPcount = calculateTP(faces, NoEntry4_1) + calculateTP(faces, NoEntry4_2) + calculateTP(faces, NoEntry4_3) + calculateTP(faces, NoEntry4_4) + calculateTP(faces, NoEntry4_5) + calculateTP(faces, NoEntry4_6);
	// float f1Score = TPcount/(TPcount + 0.5*(faces.size()-TPcount + std::max(0, groundTruth_f - TPcount)));
	// std::cout << TPcount << std::endl;
	// std::cout << f1Score << std::endl;

	// rectangle(frame, Point(NoEntry4_1.x, NoEntry4_1.y), Point(NoEntry4_1.x+NoEntry4_1.width, NoEntry4_1.y+NoEntry4_1.height), Scalar(0, 0, 255), 2);
	// rectangle(frame, Point(NoEntry4_2.x, NoEntry4_2.y), Point(NoEntry4_2.x+NoEntry4_2.width, NoEntry4_2.y+NoEntry4_2.height), Scalar(0, 0, 255), 2);
	// rectangle(frame, Point(NoEntry4_3.x, NoEntry4_3.y), Point(NoEntry4_3.x+NoEntry4_3.width, NoEntry4_3.y+NoEntry4_3.height), Scalar(0, 0, 255), 2);
	// rectangle(frame, Point(NoEntry4_4.x, NoEntry4_4.y), Point(NoEntry4_4.x+NoEntry4_4.width, NoEntry4_4.y+NoEntry4_4.height), Scalar(0, 0, 255), 2);
	// rectangle(frame, Point(NoEntry4_5.x, NoEntry4_5.y), Point(NoEntry4_5.x+NoEntry4_5.width, NoEntry4_5.y+NoEntry4_5.height), Scalar(0, 0, 255), 2);
	// rectangle(frame, Point(NoEntry4_6.x, NoEntry4_6.y), Point(NoEntry4_6.x+NoEntry4_6.width, NoEntry4_6.y+NoEntry4_6.height), Scalar(0, 0, 255), 2);

}
// g++ face.cpp /usr/lib64/libopencv_core.so.2.4 /usr/lib64/libopencv_highgui.so.2.4 /usr/lib64/libopencv_imgproc.so.2.4 /usr/lib64/libopencv_objdetect.so.2.4
// ./opencv_createsamples -img no_entry.jpg -vec no_entry.vec -w 20 -h 20 -num 500 -maxidev 80 -maxxangle 0.8 -maxyangle 0.8 -maxzangle 0.2
// ./opencv_traincascade -data NoEntrycascade -vec no_entry.vec -bg negatives.dat -numPos 500 -numNeg 500 -numStages 3 -maxDepth 1 -w 20 -h 20 -minHitRate 0.999 -maxFalseAlarmRate 0.05 -mode ALL
