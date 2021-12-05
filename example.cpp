#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
// #include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <stdio.h>
#include <math.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detect_and_display( Mat frame );
void detection_plus(Mat frame, vector<Rect> dart);
void hough_transform (int thresh_image, Mat &magnitude_img, Mat &direction);
void dart_detection(Mat frame);
void sobel(Mat &input, int size, Mat &dx_image, Mat &dy_image);
void draw_corner_point(Mat &image, const vector<Point> &points, Scalar color = Scalar(255, 255, 255), int radius = 3, int thickness = 2);

/** Global variables */
String cascade_name = "./NoEntrycascade/cascade.xml";
CascadeClassifier cascade;
vector<Vec3f> circles;
vector<Rect> dart;
vector<Point> corners;
Mat thresh_mag_img;
static int H[1224 + 2 * 200][1224 + 2 * 200][200] = {};
static int H2[1224 + 2 * 200][1224 + 2 * 200] = {};

/** @function main */
int main( int argc, const char** argv )
{
    // 1. Read Input Image
    Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

    // 2. Load the Strong Classifier in a structure called `Cascade'
    if ( !cascade.load( cascade_name ) ) { printf("--(!)Error loading\n"); return -1; };

    // 3. Detect Dart and Display Result
    detect_and_display( frame );

    // 4. Save Result Image
    imwrite( "detected.jpg", frame );
    imshow("Detected", frame);
    waitKey(0);

    return 0;
}

/** @function detect_and_display */
void detect_and_display( Mat frame )
{
    Mat magnitude_img;
    Mat dx_image;
    Mat dy_image;
    Mat direction;
    Mat frame_gray(frame.rows, frame.cols, CV_8UC1, Scalar(0));
    Mat gray_image;

    // 1. Prepare Image by turning it into Grayscale and normalising lighting
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    cvtColor(frame, gray_image, CV_BGR2GRAY);
    GaussianBlur(gray_image, gray_image, Size(3, 3), 0, 0, BORDER_REPLICATE);

    sobel(gray_image, 3, dx_image, dy_image);
    magnitude_img.create(gray_image.size(), DataType<float>::type);
    magnitude(dx_image, dy_image, magnitude_img);
    phase(dx_image, dy_image, direction, false);
    normalize(magnitude_img, magnitude_img, 0, 255, NORM_MINMAX, CV_32F);

    hough_transform (30, magnitude_img, direction);

    // 2. Perform Viola-Jones Object Detection
    cascade.detectMultiScale( frame_gray, dart, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500) );

    // 3. Draw box around dart found
    detection_plus(frame, dart);

    goodFeaturesToTrack(frame_gray, corners, 200, 0.03, 20);
    draw_corner_point(frame_gray, corners);
    imshow("Corners", frame_gray);

    dart_detection(frame);
}

void detection_plus(Mat frame, vector<Rect> dart)
{
    Mat edge;
    Mat circles_and_rectangles(frame.size(), frame.type());
    circles_and_rectangles = Scalar::all(0);
    Mat edge_gray(frame.rows, frame.cols, CV_8UC1, Scalar(0));

    //Canny(frame, edge, 50, 150, 3);
    Laplacian(frame, edge, frame.depth());
    cvtColor( edge, edge_gray, CV_BGR2GRAY );
    imshow("Edge gray", edge_gray);

    int param2 = 150;
    do {
        circles.clear();

        HoughCircles(edge_gray, circles, CV_HOUGH_GRADIENT, 1.5, 300, 130, param2, 30, 250);//霍夫变换检测圆
        param2 -= 10;
    } while (circles.size() < 1);

    for (size_t i = 0; i < circles.size(); i++) //draw circles
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);

        circle( circles_and_rectangles, center, 0, Scalar(0, 255, 0), -1, 8, 0 );
        circle( circles_and_rectangles, center, radius, Scalar(0, 0, 255), 1, 8, 0 );
    }

    for (int j = 0; j < dart.size(); j++)
        rectangle(circles_and_rectangles, Point(dart[j].x, dart[j].y), Point(dart[j].x + dart[j].width, dart[j].y + dart[j].height), Scalar( 0, 255, 0 ), 2);

    imshow("Circles and Rectangles", circles_and_rectangles);
}

void hough_transform (int thresh_image, Mat &magnitude_img, Mat &direction)
{
    int width = magnitude_img.cols;
    int height = magnitude_img.rows;

    int r_min = 0;
    int r_max = 100;

    thresh_mag_img = magnitude_img > thresh_image;
    imwrite("thresh_mag_img.jpg", thresh_mag_img);

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            for (int r = r_min; r <= r_max; r++)
            {
                if (magnitude_img.at<float>(y, x) > thresh_image )
                {
                    float x0, y0;

                    x0 = x + r * cos(direction.at<float>(y, x));
                    y0 = y + r * sin(direction.at<float>(y, x));
                    H[(int) x0 + r_max ][(int) y0 + r_max ][(int) r - r_min]++;

                    x0 = x - r * cos(direction.at<float>(y, x));
                    y0 = y - r * sin(direction.at<float>(y, x));
                    H[(int) x0 + r_max ][(int) y0 + r_max ][(int) r - r_min]++;
                }
            }
            //cout << "hello" << endl;
        }
    }

    Mat sum_r;
    sum_r.create(1224 + 2 * 200, 1224 + 2 * 200, CV_32SC1);

    for (int r = r_min; r <= r_max; r++)
    {
        for (int x = 0; x < width + 2 * r_max; x++)
        {
            for (int y = 0; y < height + 2 * r_max; y++)
                H2[x][y] += H[x][y][r];
        }
    }

    sum_r = Mat(1224 + 2 * 200, 1224 + 2 * 200, CV_32SC1, &H2);
    Mat hough_image;
    normalize(sum_r, sum_r, 0, 255, NORM_MINMAX, CV_32SC1);
    hough_image = sum_r(Range(0, width + 2 * (r_max - r_min)), Range(0, height + 2 * (r_max - r_min)));
    imwrite("hough space.jpg", hough_image);
}
void dart_detection(Mat frame)
{
    int  rectan_number = 0, rectan_number_2 = 0;
    bool flag = false;
    bool _flag = false;
    int count = 0, count_corner_point = 0, max_corner_number = 0, detection_count = 0;
    double fitness1 = 0.0;
    double fitness2 = 0.0;

    for ( int i = 0; i < circles.size(); i++ )
    {
        for (int j = 0; j < dart.size(); j++)
        {
            //cout << dart[j].width << endl;
            double centre_x_rate = ((double)fabs(dart[j].x + dart[j].width / 2 - circles[i][0]) / (double)frame.rows);
            double centre_y_rate = ((double)fabs(dart[j].y + dart[j].height / 2 - circles[i][1]) / (double)frame.cols);
            //cout << centre_x_rate << endl;
            //cout << centre_y_rate << endl;
            if ( centre_x_rate < 0.076 &&  centre_y_rate < 0.076 )
            {
                flag = true;
                count ++;
                if (count > 1)
                {
                    fitness1 = ((dart[rectan_number].width + dart[rectan_number].height) / 2) / (circles[i][2] * 2);
                    if (fitness1 > 1)
                        fitness1 = 1 / fitness1;
                    fitness2 = ((dart[j].width + dart[j].height) / 2) / (circles[i][2] * 2);
                    if (fitness2 > 1)
                        fitness2 = 1 / fitness2;

                    if (fitness2 > fitness1)
                    {
                        dart[rectan_number].width = 0, dart[rectan_number].height = 0;
                        rectan_number = j;
                    }
                    else
                        dart[j].width = 0, dart[j].height = 0;
                }
                else
                    rectan_number = j;
            }
            else
            {
                for (int h = 0; h < corners.size(); h++)
                {
                    double point_dist_x = (double)fabs(dart[j].x + dart[j].width / 2 - corners[h].x);
                    double point_dist_y = (double)fabs(dart[j].y + dart[j].height / 2 - corners[h].y);

                    if (point_dist_x < (dart[j].width / 2) && point_dist_y < (dart[j].height / 2))
                        count_corner_point ++;
                }
                if (count_corner_point >= 10 && count_corner_point > max_corner_number)
                {
                    _flag = true;
                    rectan_number_2 = j;
                    max_corner_number = count_corner_point;
                }
            }
            count_corner_point = 0;
        }

        double center_dist_x = (double)fabs(dart[rectan_number].x + dart[rectan_number].width / 2 - circles[i][0]);
        double center_dist_y = (double)fabs(dart[rectan_number].y + dart[rectan_number].height / 2 - circles[i][1]);

        if (sqrt(center_dist_x * center_dist_x + center_dist_y * center_dist_y) < circles[i][2])
            detection_count ++;

        cout << detection_count << endl;
        detection_count = 0;
        count = 0;

        double rectan_dist_x = (double)fabs(dart[rectan_number].x + dart[rectan_number].width / 2 - dart[rectan_number_2].x + dart[rectan_number_2].width / 2);
        double rectan_dist_y = (double)fabs(dart[rectan_number].y + dart[rectan_number].height / 2 - dart[rectan_number_2].y + dart[rectan_number_2].height / 2);

        if (rectan_dist_x > (dart[rectan_number].width + dart[rectan_number_2].width) * 2 || rectan_dist_y > (dart[rectan_number].height + dart[rectan_number_2].height) * 2)
            if (_flag)rectangle(frame, Point(dart[rectan_number_2].x, dart[rectan_number_2].y), Point(dart[rectan_number_2].x + dart[rectan_number_2].width, dart[rectan_number_2].y + dart[rectan_number_2].height), Scalar( 255, 0, 255 ), 2);

        if (flag == false)
            if (_flag)rectangle(frame, Point(dart[rectan_number_2].x, dart[rectan_number_2].y), Point(dart[rectan_number_2].x + dart[rectan_number_2].width, dart[rectan_number_2].y + dart[rectan_number_2].height), Scalar( 255, 0, 255 ), 2);

        if (flag)rectangle(frame, Point(dart[rectan_number].x, dart[rectan_number].y), Point(dart[rectan_number].x + dart[rectan_number].width, dart[rectan_number].y + dart[rectan_number].height), Scalar( 0, 255, 0 ), 2);
    }
}

void sobel(Mat &input, int size, Mat &dx_image, Mat &dy_image)
{
    // intialise the output using the input
    dx_image.create(input.size(), DataType<float>::type);
    dy_image.create(input.size(), DataType<float>::type);
    // create the Gaussian kernel in 1D
    Mat kX = getGaussianKernel(size, -1);
    Mat kY = getGaussianKernel(size, -1);

    // make it 2D multiply one by the transpose of the other
    Mat kernel_dx = kX * kY.t();
    Mat kernel_dy = kX * kY.t();

    // we need to create a padded version of the input
    // or there will be border effects
    int kernel_r_x = (kernel_dx.size[0] - 1) / 2;
    int kernel_r_y = (kernel_dx.size[1] - 1) / 2;

    int dx[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    int dy[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
    int count = 0;

    // SET KERNEL VALUES
    for (int m = -kernel_r_x; m <= kernel_r_x; m++)
    {
        for (int n = -kernel_r_y; n <= kernel_r_y; n++)
        {
            kernel_dx.at<double>(m + kernel_r_x, n + kernel_r_y) = dx[count];
            kernel_dy.at<double>(m + kernel_r_x, n + kernel_r_y) = dy[count];
            count++;
        }
    }

    Mat padded_in;
    copyMakeBorder(input, padded_in, kernel_r_x, kernel_r_x, kernel_r_y, kernel_r_y, BORDER_REPLICATE);

    // now we can do the convoltion
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            double sumx = 0.0;
            double sumy = 0.0;
            for (int m = -kernel_r_x; m <= kernel_r_x; m++)
            {
                for (int n = -kernel_r_y; n <= kernel_r_y; n++)
                {
                    // find the correct indices we are using
                    int imagex = i + m + kernel_r_x;
                    int imagey = j + n + kernel_r_y;
                    int kernelx = m + kernel_r_x;
                    int kernely = n + kernel_r_y;

                    // get the values from the padded image and the kernel
                    int imageval = (int)padded_in.at<uchar>(imagex, imagey);
                    double kernalvalx = kernel_dx.at<double>(kernelx, kernely);
                    double kernalvaly = kernel_dy.at<double>(kernelx, kernely);

                    // do the multiplication
                    sumx += imageval * kernalvalx;
                    sumy += imageval * kernalvaly;
                }
            }
            // set the output value as the sum of the convolution
            dx_image.at<float>(i, j) = (float)sumx;
            dy_image.at<float>(i, j) = (float)sumy;
        }
    }

}

void draw_corner_point(Mat &image, const vector<Point> &points, Scalar color , int radius , int thickness)
{
    vector<Point>::const_iterator it = points.begin();
    while (it != points.end())
    {
        circle(image, *it, radius, color, thickness);   // draw corner circles
        ++it;
    }
}