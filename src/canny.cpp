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

// OpenGL
void keyboardGL(unsigned char key, int x, int y);
void displayGL();

float alpha = 0; // Current rotation angle
bool animate = true; // Flag to control animation

// Gaussian filter -- Done
cv::Mat applyGaussianBlur(const cv::Mat& src, int kernelSize, double sigma) {
    cv::Mat dst;
    cv::GaussianBlur(src, dst, cv::Size(kernelSize, kernelSize), sigma, sigma);
    return dst;
}

// Compute image gradient -- Done
cv::Mat calculateGradientSobel(const cv::Mat& src, cv::Mat& direction) {
    cv::Mat grad_x, grad_y;
    cv::Sobel(src, grad_x, CV_64F, 1, 0, 3); // originally, using CV_32F
    cv::Sobel(src, grad_y, CV_64F, 0, 1, 3);

    cv::Mat gradient;
    cv::magnitude(grad_x, grad_y, gradient);
    cv::phase(grad_x, grad_y, direction, true);
    return gradient;
}


cv::Mat nonMaximumSuppression(const cv::Mat& gradient, const cv::Mat& direction) {
    cv::Mat nonMaxSuppressed = cv::Mat::zeros(gradient.size(), CV_32F);

    for (int y = 1; y < gradient.rows - 1; y++) {
        for (int x = 1; x < gradient.cols - 1; x++) {
            float angle = direction.at<float>(y, x);
            float neighbor1 = 0.0, neighbor2 = 0.0;

            // Horizontal edge
            if ((angle >= -22.5 && angle <= 22.5) || (angle <= -157.5 || angle >= 157.5)) {
                neighbor1 = gradient.at<float>(y, x - 1);
                neighbor2 = gradient.at<float>(y, x + 1);
            }
            // Diagonal (45 degrees)
            else if ((angle > 22.5 && angle < 67.5) || (angle < -112.5 && angle > -157.5)) {
                neighbor1 = gradient.at<float>(y - 1, x + 1);
                neighbor2 = gradient.at<float>(y + 1, x - 1);
            }
            // Vertical edge
            else if ((angle >= 67.5 && angle <= 112.5) || (angle <= -67.5 && angle >= -112.5)) {
                neighbor1 = gradient.at<float>(y - 1, x);
                neighbor2 = gradient.at<float>(y + 1, x);
            }
            // Diagonal (135 degrees)
            else if ((angle > 112.5 && angle < 157.5) || (angle < -22.5 && angle > -67.5)) {
                neighbor1 = gradient.at<float>(y - 1, x - 1);
                neighbor2 = gradient.at<float>(y + 1, x + 1);
            }

            float magnitude = gradient.at<float>(y, x);
            if (magnitude >= neighbor1 && magnitude >= neighbor2) {
                nonMaxSuppressed.at<float>(y, x) = magnitude;
            } else {
                nonMaxSuppressed.at<float>(y, x) = 0.0;
            }
        }
    }

    return nonMaxSuppressed;
}

cv::Mat doubleThresholdHysteresis(const cv::Mat& src, double lowThresh, double highThresh) {
    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());

    // Define thresholds
    float lowerBound = lowThresh * 255;
    float upperBound = highThresh * 255;

    // Apply thresholds
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            float val = src.at<float>(y, x);
            if (val >= upperBound) {
                dst.at<float>(y, x) = 255;
            } else if (val >= lowerBound) {
                dst.at<float>(y, x) = 125; // Weak edge
            }
        }
    }

    // Edge tracking by hysteresis
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            if (dst.at<float>(y, x) == 125) { // Check for weak edges
                // Check 8-neighbors for strong edges
                if (dst.at<float>(y-1, x-1) == 255 || dst.at<float>(y-1, x) == 255 || dst.at<float>(y-1, x+1) == 255 ||
                    dst.at<float>(y, x-1) == 255 || dst.at<float>(y, x+1) == 255 ||
                    dst.at<float>(y+1, x-1) == 255 || dst.at<float>(y+1, x) == 255 || dst.at<float>(y+1, x+1) == 255) {
                    dst.at<float>(y, x) = 255; // Promote to strong edge
                } else {
                    dst.at<float>(y, x) = 0; // Suppress weak edge
                }
            }
        }
    }

    return dst;
}


//////////////////////////////////////////////////////////////////////////////////////
// CPU Implementation
//////////////////////////////////////////////////////////////////////////////////////
cv::Mat performCannyEdgeDetection(const cv::Mat& inputImage, double gaussianSigma, int kernelSize, double lowThreshold, double highThreshold) {
    cv::Mat imgBlurred, gradient, direction, nonMaxSuppressed, cannyEdges;

    // Convert image to floating-point type
    cv::Mat img;
    inputImage.convertTo(img, CV_32F, 1.0 / 255);
    cv::imshow("inputImage", inputImage);
    // Step 1: Gaussian Blur
    imgBlurred = applyGaussianBlur(img, kernelSize, gaussianSigma);
    cv::imshow("imgBlurred", imgBlurred);

    // Step 2: Gradient Calculation (Sobel Operator)
    gradient = calculateGradientSobel(imgBlurred, direction);
    cv::imshow("gradient", gradient);

    // Step 3: Non-maximum Suppression
    nonMaxSuppressed = nonMaximumSuppression(gradient, direction);
    nonMaxSuppressed.convertTo(nonMaxSuppressed, CV_8U);
    cv::imshow("nonMaxSuppressed", nonMaxSuppressed);

    // Step 4: Double Threshold and Edge Tracking by Hysteresis
    cannyEdges = doubleThresholdHysteresis(nonMaxSuppressed, lowThreshold, highThreshold);
    // cv::imshow("cannyEdges", cannyEdges);

    // Convert results to displayable format
    cannyEdges.convertTo(cannyEdges, CV_8U);
    cv::imshow("cannyEdges", cannyEdges);

    return cannyEdges;
}

int main(int argc, char** argv){
    cout << "------------------------------------------------" << std::endl;
    cout << "OpenCL Exercise 4: Volume Rendering" << std::endl;
    cout << "------------------------------------------------" << std::endl;
    // Initialize GLUT
    // glutInit(&argc, argv);
    // glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    //
    // // Create a window with GLUT
    // glutInitWindowSize(800, 600); // Specify your desired window size
    // glutCreateWindow("Canny Edge Detection");
    //
    // // Initialize GLEW
    // GLenum glewInitResult = glewInit();
    // if (GLEW_OK != glewInitResult) {
    //     cerr << "ERROR: " << glewGetErrorString(glewInitResult) << endl;
    //     return EXIT_FAILURE;
    // }
    // Initialize OpenCL
    // vector<cl::Platform> platforms;
    // cl::Platform::get(&platforms);
    // auto platform = platforms.front();
    // vector<cl::Device> devices;
    // platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    // auto device = devices.front();
    //
    // auto context = cl::Context(device);
    // auto queue = cl::CommandQueue(context, device);
    //
    // // Assume `kernelCode` contains your OpenCL kernel code as a string
    // string kernelCode =
    //     "__kernel void gaussianBlur(__global uchar* inputImage, __global uchar* outputImage) {"
    //     "   // Kernel code here"
    //     "}";
    //
    // // Build kernel
    // cl::Program::Sources sources;
    // sources.push_back({kernelCode.c_str(), kernelCode.length()});
    //
    // cl::Program program(context, sources);
    // auto err = program.build("-cl-std=CL1.2");

    // Load image using OpenCV
    cv::Mat img = cv::imread("../test1.png", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Failed to load image\n";
        return -1;
    }
    cv::imshow("Original", img);
    double gaussianSigma = 1.4;
    int kernelSize = 3;
    double lowThreshold = 0.05;
    double highThreshold = 0.15;

    cv::Mat cannyEdges = performCannyEdgeDetection(img, gaussianSigma, kernelSize, lowThreshold, highThreshold);

    // cv::namedWindow("Canny Edges", cv::WINDOW_AUTOSIZE);
    // cv::imshow("Original", img);
    cv::imshow("Canny Edges", cannyEdges);
    cv::waitKey(0); // Wait for a keystroke in the window

    // // Allocate memory on the device
    // size_t imageBytes = image.total() * image.elemSize();
    // auto d_input = cl::Buffer(context, CL_MEM_READ_ONLY, imageBytes);
    // auto d_output = cl::Buffer(context, CL_MEM_WRITE_ONLY, imageBytes);
    //
    // // Copy data to GPU
    // queue.enqueueWriteBuffer(d_input, CL_TRUE, 0, imageBytes, image.data);
    //
    // // Set kernel arguments and enqueue kernel for execution
    // auto kernel = cl::Kernel(program, "gaussianBlur");
    // kernel.setArg(0, d_input);
    // kernel.setArg(1, d_output);
    //
    // queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image.cols, image.rows));
    // queue.finish();
    //
    // // Read back the result
    // cv::Mat outputImage(image.rows, image.cols, CV_8UC1);
    // queue.enqueueReadBuffer(d_output, CL_TRUE, 0, imageBytes, outputImage.data);

    // Set up GLUT callbacks
    // glutDisplayFunc(displayGL); // Make sure displayGL is prepared to show the processed image
    // glutKeyboardFunc(keyboardGL);
    //
    // // Enter the GLUT main loop
    // glutMainLoop();
    return 0;
}


//////////////////////////////////////////////////////////////////////////////
// OpenGL callbacks
//////////////////////////////////////////////////////////////////////////////
void displayGL() {
    // Clear the window and draw the processed image
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // Code to setup and draw texture from the Canny result, if displaying images with OpenGL
    glutSwapBuffers();
}

void keyboardGL(unsigned char key, int x, int y) {
  (void)x;
  (void)y;

  switch (key) {
    case 'Q':
    case 'q':
    case '\033':  // escape
      glutLeaveMainLoop();
      break;
    default:
      break;
  }
}
