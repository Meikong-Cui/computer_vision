/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - RGBtoHSV.cpp
//
// University of Bristol
//
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>

using namespace std;
using namespace cv;

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

void threshold(Mat &input, int t) {
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
}

void houghcircleTrans(int threshold_num, Mat &magnitude, Mat &direction, vector<Vec3f> &circles) {
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
			std::map<int, int> hashMap;
            for (int r = r_min; r < r_max; r++) {
				if(H[x][y][r] > threshold_num) {
					hashMap[r] = H[x][y][r];
				}
            }
			for(std::map<int, int>::const_iterator iterator = hashMap.begin(); iterator != hashMap.end(); iterator++) {
				for(int i = 0; i < circles.size(); i++) {
					Vec3f circle = circles[i];
					int r = circle[2];
					if(iterator->first > r-6 && iterator->first < r+6){
						flag = false;
					}
				}
				if(flag == true) {
					Vec3f circle(y, x, iterator->first);
					// std::cout << x << ' ' << y << << ' ' << "radius: " << iterator->first << endl;
					circles.push_back(circle);
				}
			}
        }
    }    
}

int main( int argc, char** argv ) {

    // LOADING THE IMAGE
    char* imageName = argv[1];

    Mat BGRimage;
    BGRimage = imread( imageName, 1 );

    if( argc != 2 || !BGRimage.data )
    {
        printf( " No image data \n " );
        return -1;
    }

    vector<Vec3f> circles;

    Mat HSVimage;
    cvtColor( BGRimage, HSVimage, CV_BGR2HSV );

    Mat mask, low_mask, high_mask;
    inRange(HSVimage, Scalar(0, 100, 20), Scalar(20, 255, 255), low_mask);
    inRange(HSVimage, Scalar(160, 100, 20), Scalar(179, 255, 255), high_mask);
    mask = low_mask + high_mask;

    Mat gray, result;
    cvtColor(BGRimage, gray, CV_BGR2GRAY);
    equalizeHist( gray, gray );
    bitwise_and(gray, mask, result);
    GaussianBlur(result, result, Size(3,3), 0, 0, BORDER_REPLICATE);

    Mat sobelx;
    Mat sobely;
    sobelXY(result, 3, sobelx, sobely);

    Mat magnitude;
    magnitude.create(result.size(), DataType<float>::type);
    cv::magnitude(sobelx, sobely, magnitude);
    normalize(magnitude, magnitude, 0, 255, NORM_MINMAX, CV_8UC1);

    Mat direction;
    cv::phase(sobelx, sobely, direction, false);
    // normalize(magnitude, magnitude, 0, 255, NORM_MINMAX, CV_32F);
    
    // imwrite("threshold_mag.jpg", magnitude);
    threshold(magnitude, 50);

    houghcircleTrans(8, magnitude, direction, circles);
    for (size_t i = 0; i < circles.size(); i++) {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);

        // circle( frame, center, 0, Scalar(0, 255, 0), -1, 8, 0 );
        circle( BGRimage, center, radius, Scalar(255, 0, 0), 3, 8, 0 );
    }
    imwrite( "hsvfilter.jpg", magnitude);
    imwrite( "test.jpg", BGRimage);

    return 0;
}
