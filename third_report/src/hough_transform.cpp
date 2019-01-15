#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cxcore.h>
#include <highgui.h>

using namespace std;
using namespace cv;

int main() {
    Mat img, grayImage;
    vector<Vec3f> circles;
    
    img = imread("/home/matheus/Dropbox/treinamento-pdi/samples/paint.jpg");

    cvtColor(img, grayImage, CV_RGB2GRAY);

    GaussianBlur(grayImage, grayImage, Size(9, 9), 2, 2);

    HoughCircles(grayImage, circles, CV_HOUGH_GRADIENT, 1, grayImage.rows / 4, 100, 50, 0, 0);

    for (int i = 0; i < circles.size(); i++) {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        circle(img, center, 2, Scalar(0, 255, 0), 2, CV_AA, 0);
        circle(img, center, radius, Scalar(0, 0, 255), 2, CV_AA, 0);
    }

    imshow("image with Hough transform", img);

    imwrite("/home/matheus/Dropbox/treinamento-pdi/results/circle_hough_transform.jpg", img);

    waitKey(0);
    return 0;
}

