#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <math.h>
#define pi 3.14159265

using namespace std;
using namespace cv;

int ***malloc3dArray(int dim1, int dim2, int dim3);
int **malloc2dArray(int dim1, int dim2);
void detectAndDisplay( Mat frame );
void sobelXY(Mat &input, int size, Mat &outputX, Mat &outputY);
Mat normalize_and_save(Mat input, string name);
void threshold(Mat &input, int t, string ver);
void houghcircleTrans(int threshold, Mat &magnitude, Mat &direction, vector<Vec3f> &circles);
void houghlineTrans(Mat &input, Mat &direction);
void cutImage(Mat &input, Mat &output, Rect area);

// static int H[1000 + 2 * 200][1000 + 2 * 200][200] = {};
// static int H2[1000 + 2 * 200][1000 + 2 * 200] = {};

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
	imwrite( "detected.jpg", frame );
    imshow("detected", frame);
	return 0;
}

void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;
    Mat gray;
    vector<Vec3f> circles;
	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

    GaussianBlur(frame_gray, gray, Size(9, 9), 2, 2, BORDER_REPLICATE);
    // GaussianBlur(frame_gray, gray, Size(3, 3), 0, 0);

    Mat sobelx;
    Mat sobely;
    sobelXY(gray, 3, sobelx, sobely);

    Mat magnitude;
    magnitude.create(gray.size(), DataType<float>::type);
    cv::magnitude(sobelx, sobely, magnitude);

    Mat direction;
    cv::phase(sobelx, sobely, direction, false);
    // normalize(magnitude, magnitude, 0, 255, NORM_MINMAX, CV_32F);

    normalize(magnitude, magnitude, 0, 255, NORM_MINMAX, CV_8UC1);
    threshold(magnitude, 70, "source");

    houghlineTrans(magnitude, direction);
    houghcircleTrans(70, magnitude, direction, circles);

    cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(10, 10), Size(300,300) );
    
    for( int i = 0; i < faces.size(); i++ ) {
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

    for (size_t i = 0; i < circles.size(); i++) {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);

        // circle( frame, center, 0, Scalar(0, 255, 0), -1, 8, 0 );
        circle( frame, center, radius, Scalar(255, 0, 0), 3, 8, 0 );
    }
    // imshow("dectected", frame);
}

// void houghEllipse(Mat frame) {   // Too hard, 5 dimision
//     Mat colourImage = frame.clone();
//     Mat grayImage;
//     cvtColor(frame, grayImage, CV_BGR2GRAY);
//     GaussianBlur(grayImage, grayImage, Size(3,3), 0, 0);
//     Canny(grayImage, grayImage, 100, 200, 3);
//     imshow("Canny", grayImage);
//     vector< vector<Point> > contours;
//     vector<Vec4i> hierarchy;
//     findContours(grayImage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//     drawContours(colourImage, contours, -1, Scalar(0, 255, 0));
//     imwrite("cour.bmp", colourImage);
// }

// copied from array.c
int ***malloc3dArray(int dim1, int dim2, int dim3) {
    int i, j, k;
    int ***array = (int ***) malloc(dim1 * sizeof(int **));

    for (i = 0; i < dim1; i++) {
        array[i] = (int **) malloc(dim2 * sizeof(int *));
	    for (j = 0; j < dim2; j++) {
  	        array[i][j] = (int *) malloc(dim3 * sizeof(int));
	    }
    }
    return array;
}

int **malloc2dArray(int dim1, int dim2) {
    int i, j;
    int **array = (int **) malloc(dim1 * sizeof(int *));

    for (i = 0; i < dim1; i++) {
        array[i] = (int *) malloc(dim2 * sizeof(int));
    }
    return array;
}

Mat normalize_and_save(Mat input, string name) {
	Mat n_input(input.size(), CV_8UC1, Scalar(0));
	normalize(input, n_input, 0, 255, NORM_MINMAX, CV_8UC1);

	imwrite(name+".jpg", n_input);

	return n_input;
}

void threshold(Mat &input, int t, string ver) {
	for(int i = 0; i < input.rows; i++) {
		for(int j = 0; j < input.cols; j++) {
			int val = (int) input.at<uchar>(i, j);
			if(val > t) {
				input.at<uchar>(i,j) = (uchar) 255;
			} else {
				input.at<uchar>(i,j) = (uchar) 0;
			}
		}
	}
	imwrite("threshold_"+ver+".jpg", input);
}

void houghcircleTrans(int threshold, Mat &magnitude, Mat &direction, vector<Vec3f> &circles) {
    int width = magnitude.rows;
    int height = magnitude.cols;

    int r_min = 0;
    int r_max = 100;

    int ***H = malloc3dArray(width, height, r_max-r_min);
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            for (int r = 0; r < r_max; r++) {
                H[i][j][r] = 0;
            }
        }
    }

    // calculate hough space for every different radius
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            if(magnitude.at<uchar>(x, y) == 0) continue;
                
            for (int r = 0; r < r_max; r++) {
                // int x0, y0;
                // x0 = x + r * int(cos(direction.at<float>(x, y)));
                // y0 = y + r * int(sin(direction.at<float>(x, y)));
                int xc = int(r * sin(direction.at<float>(x,y)));
				int yc = int(r * cos(direction.at<float>(x,y)));
				int x0 = x - xc;
				int y0 = y - yc;

                if(x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
                    H[x0][y0][r] += 1;
                }

                // x0 = x - r * int(cos(direction.at<float>(x, y)));
                // y0 = y - r * int(sin(direction.at<float>(x, y)));
				int x1 = x + xc;
				int y1 = y + yc;
                if(x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
                    H[x1][y1][r] += 1;
                }
            }
        }
    }

    // sum hough space together
    Mat hough_sum(width, height, CV_32FC1);
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int r = r_min; r < r_max; r++) {
                hough_sum.at<float>(x,y) += H[x][y][r];
            }
        }
    }

    imwrite("houghCircle_space.jpg", hough_sum);

	for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
			bool flag = true;
			map<int, int> hashMap;
            for (int r = r_min; r < r_max; r++) {
				if(H[x][y][r] > 15) {
					hashMap[r] = H[x][y][r];
				}
            }
			for(map<int, int>::const_iterator iterator = hashMap.begin(); iterator != hashMap.end(); iterator++) {
				for(int i = 0; i < circles.size(); i++) {
					Vec3f circle = circles[i];
					int r = circle[2];
					if(iterator->first > r-6 && iterator->first < r+6){
						flag = false;
					}
				}
				if(flag == true) {
					Vec3f circle(y, x, iterator->first);
					std::cout << "radius: " << iterator->first << ' ' << x << ' ' << y << endl;
					circles.push_back(circle);
				}
			}
        }
    }
}

void houghlineTrans(Mat &input, Mat &direction) {
	int diag = sqrt(pow(input.rows,2)+pow(input.cols,2));

	int **hough_space = malloc2dArray(diag,360);
    for (int i = 0; i < diag; i++) {
        for (int j = 0; j < 360; j++) {
            hough_space[i][j] = 0;
        }
    }
    for (int x = 0; x < input.rows; x++) {
        for (int y = 0; y < input.cols; y++) {
			if(input.at<uchar>(x,y) == 255) {
				int th = int(direction.at<float>(x,y)*(180/pi)) + 180;
				for(int t = th-5; t <= th+5; t++) {
					int mod_th = (t+360) % 360;
					float t_rad = (mod_th-180)*(pi/180);
					int xc = int(x * sin(t_rad));
					int yc = int(y * cos(t_rad));
					int p = xc + yc;
					if(p >= 0 && p <= diag) {
						hough_space[p][mod_th] += 1;
					}
				}
			}
        }
    }

	Mat hough_output(diag, 360, CV_32FC1, Scalar(0));
 
    for (int p = 0; p < diag; p++) {
        for (int t = 0; t < 360; t++) {
			hough_output.at<float>(p,t) = hough_space[p][t];
        }
    }

	Mat img_threshold = normalize_and_save(hough_output, "rho_theta_space");
	threshold(img_threshold, 10, "rho_theta");

	Mat hough_output_o(input.rows, input.cols, CV_32FC1, Scalar(0));
 
	for(int p = 0; p < hough_output.rows; p++) {
		for(int th = 0; th < hough_output.cols; th++) {
			if(img_threshold.at<uchar>(p,th) == 255) {
				float t_rad = (th-180) * (pi/180);
				for(int x = 0; x < input.cols; x++) {
					int y = ((-cos(t_rad))/sin(t_rad))*x + (p/sin(t_rad));

					if(y >= 0 && y < input.rows) {
						hough_output_o.at<float>(y,x)++;
					}
				}
			}
		}
	}
	Mat img_threshold_o = normalize_and_save(hough_output_o, "hough_space_lines");
	threshold(img_threshold_o, 100, "lines");
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
