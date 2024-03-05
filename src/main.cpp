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
//#include <GL/glew.h>
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
using namespace cv;

bool runCpu = true;
bool runGpu = true;
bool displayGpu = true;
bool writeImages = false;

// change here

int cpuFunction ( const Mat& src );
Mat GaussianFilter ( const Mat& src );
Mat NonMaximumSuppression ( const Mat& magnitude, const Mat& blurred, const Mat& angle );
void DoubleThresholding ( const Mat& magnitude, Mat& strong, Mat& weak );
Mat Hysteresis ( Mat& strong, const Mat& weak );


int main ( int argc, char* argv[] ) {

    cout << "Beginning of the project!" << endl;

    // GPU setup ------------------------------------------
    // Create Context
    cl::Context context ( CL_DEVICE_TYPE_GPU );
    // Device list
    int deviceNr = argc < 2 ? 1 : atoi ( argv[1] );
    cout << "Using device " << deviceNr << " / "
         << context.getInfo<CL_CONTEXT_DEVICES>().size() << std::endl;
    ASSERT ( deviceNr > 0 );
    ASSERT ( ( size_t ) deviceNr <= context.getInfo<CL_CONTEXT_DEVICES>().size() );
    cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>() [deviceNr - 1];
    std::vector<cl::Device> devices;
    devices.push_back ( device );
    OpenCL::printDeviceInfo ( std::cout, device );

    // Create a command queue
    cl::CommandQueue queue ( context, device, CL_QUEUE_PROFILING_ENABLE );


    // Load the source code
    extern unsigned char CannyEdgeDetection_cl[];
    extern unsigned int CannyEdgeDetection_cl_len;
    cl::Program program ( context,
                          std::string ( ( const char* ) CannyEdgeDetection_cl,
                                        CannyEdgeDetection_cl_len ) );

    OpenCL::buildProgram ( program, devices );

    // ----------------------------------------------------

    // Declare some value for GPU -------------------------
    std::size_t wgSizeX =
        16;  // Number of work items per work group in X direction
    std::size_t wgSizeY = 16;
    std::size_t countX =
        wgSizeX *
        40;  // Overall number of work items in X direction = Number of elements in X direction
    std::size_t countY = wgSizeY * 30;
    //countX *= 3; countY *= 3;
    std::size_t count = countX * countY;       // Overall number of elements
    std::size_t size = count * sizeof ( float ); // Size of data in bytes

    // Allocate space for output data from CPU and GPU on the host
    std::vector<float> h_input ( count );
    std::vector<float> h_outputCpu ( count );
    std::vector<float> h_outputGpu ( count );

    // Allocate space for input and output data on the device
    cl::Buffer d_input ( context, CL_MEM_READ_WRITE, size );
    cl::Image2D d_output ( context, CL_MEM_READ_ONLY,
                          cl::ImageFormat(CL_R, CL_FLOAT), countX, countY );
    // ----------------------------------------------------

    // load image
    string imgName;
    if ( argc > 1 )
        imgName = argv[1];
    else
        imgName = "test1.png";

    string imgPath = "../" + imgName;

    Mat img = imread ( imgPath, IMREAD_GRAYSCALE );
    // Ensure the image is of type CV_32F
    if ( img.type() != CV_32F ) {
        img.convertTo ( img, CV_32F );
        }
    Mat blurred;
    blur ( img, blurred, Size ( 3,3 ) );
    int rows = blurred.rows;
    int cols = blurred.cols;
    std::vector<float> imgVector;
    imgVector.assign ( ( float* ) blurred.datastart, ( float* ) blurred.dataend );
    cout << rows << " " << cols << endl;

    for ( size_t j = 0; j < countY; j++ ) {
        for ( size_t i = 0; i < countX; i++ ) {
            h_input[i + countX * j] = imgVector[ ( i % cols ) + cols * ( j % rows )];
            }
        }
    // for ( size_t k = 0; k < h_input.size(); k++ )
    //     cout << h_input[k] << " ";
    if ( blurred.empty() ) {
        std::cout << imgName + " is not a valid image." << std::endl;
        return 0;
        }

    // imshow ( "img", img );
    Core::TimeSpan cpuStart = Core::getCurrentTime();
    cpuFunction ( img );
    Core::TimeSpan cpuEnd = Core::getCurrentTime();


    // GPU ----------------------------------------------------
    memset ( h_outputGpu.data(), 255, size );
    queue.enqueueWriteBuffer ( d_output, true, 0, size, h_outputGpu.data() );

    cl::Event copy1;
    cl::Image2D image;
    image = cl::Image2D ( context, CL_MEM_READ_ONLY,
                          cl::ImageFormat ( CL_R, CL_FLOAT ), countX, countY );
    cl::size_t<3> origin;
    origin[0] = origin[1] = origin[2] = 0;
    cl::size_t<3> region;
    region[0] = countX;
    region[1] = countY;
    region[2] = 1;
    queue.enqueueWriteImage ( image, true, origin, region,
                              countX * sizeof ( float ), 0, imgVector.data(), NULL,
                              &copy1 );

    // Create a kernel object
    string kernel1 = "NonMaximumSuppression";
    string kernel2 = "DoubleThresholding";
    string kernel3 = "Hysteresis";

    cl::Kernel nmsKernel(program, kernel1.c_str());
    cl::Kernel dtKernel(Program, kernel2.c_str());
    cl::Kernel hKernel(Program, kernel3.c_str());

    cl::Image2D bufferNMStoDT; // Output of the NonMaximumSuppression kernel and input to the DoubleThresholding
    cl::Image2D bufferDTtoH; // Output of the DoubleThresholding kernel and input to the Hysteresis

    // Launch kernel on the device
    cl::Event eventNMS, eventDT, eventH;
    // Set kernel arguments
    nmsKernel.setArg<cl::Image2D>(0, image);
    nmsKernel.setArg<cl::Image2D>(1, bufferNMStoDT); // Output used as input for the next kernel

    dtKernel.setArg<cl::Image2D>(0, bufferNMStoDT); // Output from the previous kernel
    dtKernel.setArg<cl::Image2D>(1, bufferDTtoH); // Output used as input for the next kernel

    hKernel.setArg<cl::Image2D>(0, d_output); // Output from the previous kernel

    queue.enqueueNDRangeKernel(nmsKernel, cl::NullRange,
                               cl::NDRange(countX, countY),
                               cl::NDRange(wgSizeX, wgSizeY), NULL, &eventNMS);
    eventNMS.wait(); // Wait for the NMS kernel to complete

    queue.enqueueNDRangeKernel(dtKernel, cl::NullRange,
                               cl::NDRange(countX, countY),
                               cl::NDRange(wgSizeX, wgSizeY), NULL, &eventDT);
    eventDT.wait(); // Wait for the Double Thresholding kernel to complete

    queue.enqueueNDRangeKernel(hKernel, cl::NullRange,
                               cl::NDRange(countX, countY),
                               cl::NDRange(wgSizeX, wgSizeY), NULL, &eventH);
    eventH.wait(); // Wait for the Hysteresis kernel to complete


    // Copy output data back to host
    cl::Event copy2;
    queue.enqueueReadImage(d_output, true, 0, size, h_outputGpu.data(), NULL, &copy2);

    // --------------------------------------------------------

    // Print performance data
    Core::TimeSpan cpuTime = cpuEnd - cpuStart;
    Core::TimeSpan gpuTime = OpenCL::getElapsedTime(eventNMS) + OpenCL::getElapsedTime(eventDT) + OpenCL::getElapsedTime(eventH);
    Core::TimeSpan copyTime = OpenCL::getElapsedTime(copy1) + OpenCL::getElapsedTime(copy2);
    Core::TimeSpan overallGpuTime = gpuTime + copyTime;

    cout << "CPU Time: " << cpuTime.toString() << ", "
         << ( count / cpuTime.getSeconds() / 1e6 ) << " MPixel/s"
         << endl;
    cout << "Memory copy Time: " << copyTime.toString() << endl;
    cout << "GPU Time w/o memory copy: " << gpuTime.toString()
              << " (speedup = " << (cpuTime.getSeconds() / gpuTime.getSeconds())
              << ", " << (count / gpuTime.getSeconds() / 1e6) << " MPixel/s)"
              << endl;
    cout << "GPU Time with memory copy: " << overallGpuTime.toString()
              << " (speedup = "
              << (cpuTime.getSeconds() / overallGpuTime.getSeconds()) << ", "
              << (count / overallGpuTime.getSeconds() / 1e6) << " MPixel/s)"
              << endl;
    return 0;
    }


Mat GaussianFilter ( const Mat& src ) {
    Mat blurred;
    blur ( src, blurred, Size ( 3,3 ) );
    return blurred;
    }

Mat NonMaximumSuppression ( const Mat& magnitude, const Mat& blurred, const Mat& angle ) {
    Mat result = magnitude.clone();
    int neighbor1X, neighbor1Y, neighbor2X, neighbor2Y;
    float gradientAngle;

    for ( int x = 0; x < blurred.rows; x++ ) {
        for ( int y = 0; y < blurred.cols; y++ ) {
            gradientAngle = angle.at<float> ( x, y );

            // Normalize angle to be in the range [0, 180)
            gradientAngle = fmodf ( fabs ( gradientAngle ), 180.0f );

            // Determine neighbors based on gradient angle
            if ( gradientAngle <= 22.5f || gradientAngle > 157.5f ) {
                neighbor1X = x - 1;
                neighbor1Y = y;
                neighbor2X = x + 1;
                neighbor2Y = y;
                }
            else if ( gradientAngle <= 67.5f ) {
                neighbor1X = x - 1;
                neighbor1Y = y - 1;
                neighbor2X = x + 1;
                neighbor2Y = y + 1;
                }
            else if ( gradientAngle <= 112.5f ) {
                neighbor1X = x;
                neighbor1Y = y - 1;
                neighbor2X = x;
                neighbor2Y = y + 1;
                }
            else {
                neighbor1X = x - 1;
                neighbor1Y = y + 1;
                neighbor2X = x + 1;
                neighbor2Y = y - 1;
                }

            // Check bounds of neighbor1
            if ( neighbor1X >= 0 && neighbor1X < blurred.rows && neighbor1Y >= 0 && neighbor1Y < blurred.cols ) {
                if ( result.at<float> ( x, y ) < result.at<float> ( neighbor1X, neighbor1Y ) ) {
                    result.at<float> ( x, y ) = 0;
                    continue;
                    }
                }

            // Check bounds of neighbor2
            if ( neighbor2X >= 0 && neighbor2X < blurred.rows && neighbor2Y >= 0 && neighbor2Y < blurred.cols ) {
                if ( result.at<float> ( x, y ) < result.at<float> ( neighbor2X, neighbor2Y ) ) {
                    result.at<float> ( x, y ) = 0;
                    continue;
                    }
                }
            }
        }
    return result;
    }

void DoubleThresholding ( const Mat& magnitude, Mat& strong, Mat& weak ) {
    // apply double thresholding
    float magMax = 0.2, magMin = 0.1;
    float gradientMagnitude;
    for ( int x = 0; x < magnitude.rows; x++ ) {
        for ( int y = 0; y < magnitude.cols; y++ ) {
            gradientMagnitude = magnitude.at<float> ( x, y );

            if ( gradientMagnitude > magMax ) {
                strong.at<float> ( x, y ) = gradientMagnitude;
                }
            else if ( gradientMagnitude <= magMax && gradientMagnitude > magMin ) {
                weak.at<float> ( x, y ) = gradientMagnitude;
                };
            }
        }
    }

Mat Hysteresis ( Mat& strong, const Mat& weak ) {
    // imshow ( "strong_test", strong );
    // imshow ( "weak_test", weak );
    for ( int x = 0; x < strong.rows; x++ ) {
        for ( int y = 0; y < strong.cols; y++ ) {
            if ( weak.at<float> ( x, y ) != 0 ) {
                if ( ( x + 1 < strong.rows && strong.at<float> ( x + 1, y ) != 0 ) ||
                        ( x - 1 >= 0 && strong.at<float> ( x - 1, y ) != 0 ) ||
                        ( y + 1 < strong.cols && strong.at<float> ( x, y + 1 ) ) != 0 ||
                        ( y - 1 >= 0 && strong.at<float> ( x, y - 1 ) != 0 ) ||
                        ( x - 1 >= 0 && y - 1 >= 0 && strong.at<float> ( x - 1, y - 1 ) != 0 ) ||
                        ( x + 1 < strong.rows && y - 1 >= 0 && strong.at<float> ( x + 1, y - 1 ) != 0 ) ||
                        ( x - 1 >= 0 && y + 1 < strong.cols && strong.at<float> ( x - 1, y + 1 ) != 0 ) ||
                        ( x + 1 < strong.rows && y + 1 < strong.cols && strong.at<float> ( x + 1, y + 1 ) != 0 ) ) {
                    strong.at<float> ( x, y ) = weak.at<float> ( x, y );
                    }
                }
            }
        }
    return strong;
    }

int cpuFunction ( const Mat& src ) {
    // Apply Gaussian filter
    // Mat blurred = GaussianFilter ( src );
    // imshow ( "blurred", blurred );

    // Compute image gradient
    Mat xComponent, yComponent;
    Sobel ( blurred, xComponent, CV_32F, 1, 0, 3 );
    Sobel ( blurred, yComponent, CV_32F, 0, 1, 3 );

    // Convert to polar coordinates
    Mat magnitude, angle;
    cartToPolar ( xComponent, yComponent, magnitude, angle, true );

    // Normalize values
    normalize ( magnitude, magnitude, 0, 1, NORM_MINMAX );

    // Apply non-maximum suppression
    Mat suppressed = NonMaximumSuppression ( magnitude, blurred, angle );
    imshow ( "non-max suppression", suppressed );

    // Apply double thresholding
    Mat strong = Mat::zeros ( magnitude.rows, magnitude.cols, CV_32F );
    Mat weak = Mat::zeros ( magnitude.rows, magnitude.cols, CV_32F );
    DoubleThresholding ( suppressed, strong, weak );
    imshow ( "strong", strong );
    imshow ( "Weak", weak );

    // Apply hysteresis
    Mat finalEdges = Hysteresis ( strong, weak );
    imshow ( "final edge detection", finalEdges );

    waitKey ( 0 );
    return 0;
    }
