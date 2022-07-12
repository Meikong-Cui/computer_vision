# IPCV Coursework - No Entry sign detection

This is my coursework of Image Processing & Computer Vision. This coursework has four parts:

1. Part 1 is using Viola-Jones face detector of OpenCV, this part does not include too many codes because it used trained models.
2. Part 2 is training a Viola-Jones detector on No Entry sign. This part also does not include many codes, the university provides a package to generate training sets and OpenCV provides the training method.
3. Part 3 is using hough space to detect circles in the picture. The OpenCV function about hough transform is not allowed to use in this part. I implement hough transform with c++ and combine it with Viola-Jones detector trained in part 2. Some image processing techniques is also used, for example, uniform the gray value to make dark pictures more clear.
4. Part 4 is improving the No Entry detector. I write an HSV filter on RGB pictures and a line detector in hough space, which is combined with circle detector to find the No Entry sign.

The results and evaluation is in the 'report.pdf'. The evaluation methods use IOU and F1 score.

This is a screenshot of my detector, green boxes are outputs of Viola-Jones detector, red boxes are the final output of filter + Viola-Jones detector:
![imagetext](https://github.com/Meikong-Cui/computer_vision/blob/main/No_entry/Noentry.png)
