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
void detectAndDisplay( Mat frame );

/** Global variables */
String cascade_name = "./frontalface.xml";
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
	imwrite( "detected.jpg", frame );

	return 0;
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

       // 4. Draw box around faces found
	Rect NoEntry1;
	NoEntry1.x = 235;
	NoEntry1.y = 475;
	NoEntry1.width = 80;
	NoEntry1.height = 90;

	int TPcount = 0;
	int groundTruth_f = 1;

	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
		float startX = (faces[i].x < NoEntry1.x) ? faces[i].x : NoEntry1.x;
		float startY = (faces[i].y < NoEntry1.y) ? faces[i].y : NoEntry1.y;
		float endX = ((faces[i].x + faces[i].width) > (NoEntry1.x + NoEntry1.width)) ? (faces[i].x + faces[i].width) : (NoEntry1.x + NoEntry1.width);
		float endY = ((faces[i].y + faces[i].height) > (NoEntry1.y + NoEntry1.height)) ? (faces[i].y + faces[i].height) : (NoEntry1.y + NoEntry1.height);
		int x1 = 0, y1 = 0;
		for(int a = int(startX); a <= int(endX); a++) {
			if(a>faces[i].x && a>NoEntry1.x && a<(faces[i].x + faces[i].width) && a<(NoEntry1.x + NoEntry1.width)) x1++;
		}
		for(int b = int(startY); b <= int(endY); b++) {
			if(b>faces[i].y && b>NoEntry1.y && b<(faces[i].y + faces[i].height) && b<(NoEntry1.y + NoEntry1.height)) y1++;
		}
		float area1 = NoEntry1.width * NoEntry1.height;
		float area2 = faces[i].width * faces[i].height;
		float IOU = x1*y1 / (area1 + area2 - x1*y1);
		if(IOU > 0.5f) TPcount++;
	}

	rectangle(frame, Point(NoEntry1.x, NoEntry1.y), Point(NoEntry1.x+NoEntry1.width, NoEntry1.y+NoEntry1.height), Scalar(0, 0, 255), 2);
	std::cout << "True Positive rate: " << std::min(TPcount/groundTruth_f, 1) * 100 << '%' << std::endl;
	std::cout << "F1 score: " << TPcount/(TPcount + 0.5*(faces.size()-TPcount + std::max(0, groundTruth_f - TPcount))) << std::endl;
}
// g++ face.cpp /usr/lib64/libopencv_core.so.2.4 /usr/lib64/libopencv_highgui.so.2.4 /usr/lib64/libopencv_imgproc.so.2.4 /usr/lib64/libopencv_objdetect.so.2.4
