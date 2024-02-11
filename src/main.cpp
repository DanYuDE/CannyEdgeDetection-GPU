// includes
#include <stdio.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

// OpenCV header
#include <opencv2/opencv.hpp>

// Undefine Status if defined
#ifdef Status
#undef Status
#endif

// OpenGL and X11 headers
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/glx.h>

#include <CT/DataFiles.hpp>
#include <Core/Assert.hpp>
#include <Core/Image.hpp>
#include <Core/Time.hpp>
#include <OpenCL/Device.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/cl-patched.hpp>

#include <boost/lexical_cast.hpp>

using namespace std;

bool runCpu = true;
bool runGpu = true;
bool displayGpu = true;
bool writeImages = false;

/// change here
//another


int main(){
    cout << "Beginning of the project!" << endl;
    cv::Mat img = cv::imread("../test1.png");
    if (img.empty()) {
        std::cout << "cannot read the image" << std::endl;
        return -1;
    }

    cv::Mat gray;
    cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    cv::Mat blurred;
    GaussianBlur(gray, blurred, cv::Size(5, 5), 1.5);

    cv::Mat canny;
    double low_threshold = 50.0;
    double high_threshold = 150.0;
    Canny(blurred, canny, low_threshold, high_threshold);

    cv::namedWindow("original", cv::WINDOW_AUTOSIZE);
    cv::imshow("original", img);

    cv::namedWindow("Canny Edge Detection", cv::WINDOW_AUTOSIZE);
    cv::imshow("Canny Edge Detection", canny);

    cv::waitKey(0);
    return 0;
}
