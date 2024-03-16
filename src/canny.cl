#ifndef __OPENCL_VERSION__
#include "../lib/OpenCL/OpenCLKernel.hpp"  // Hack to make syntax highlighting work
#endif

inline float getValueImage(__read_only image2d_t image, int x, int y) {
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | // Non-normalized coordinates
                              CLK_ADDRESS_CLAMP |            // Clamp to edge
                              CLK_FILTER_NEAREST;           // Nearest neighbor interpolation

    int2 coords = (int2)(x, y); // Create a 2D integer vector for coordinates
    float4 pixel = read_imagef(image, sampler, coords); // Read the pixel value
    return pixel.x; // Return the first channel assuming image format is single channel float
}

int getIndexGlobal(size_t countX, int i, int j) { return j * countX + i; }

__kernel void Sobel(__read_only image2d_t blurred,
                           __global float* bufferSBLtoNMS) {
  int i = get_global_id(0);
  int j = get_global_id(1);

  size_t countX = get_global_size(0);
  size_t countY = get_global_size(1);

  float Gmm = getValueImage(blurred, i - 1, j - 1);
  float Gm0 = getValueImage(blurred, i - 1, j);
  float Gmp = getValueImage(blurred, i - 1, j + 1);
  float Gpm = getValueImage(blurred, i + 1, j - 1);
  float Gp0 = getValueImage(blurred, i + 1, j);
  float Gpp = getValueImage(blurred, i + 1, j + 1);
  float G0m = getValueImage(blurred, i, j - 1);
  float G0p = getValueImage(blurred, i, j + 1);

  float Gx = Gmm + 2 * Gm0 + Gmp - Gpm - 2 * Gp0 - Gpp;
  float Gy = Gmm + 2 * G0m + Gpm - Gmp - 2 * G0p - Gpp;
  bufferSBLtoNMS[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);
}

__kernel void cartToPolar(__global const float* x,
                          __global const float* y,
                          __global float* magnitude,
                          __global float* angle,
                          const int angleInDegrees) {
    int idx = get_global_id(0);
    float x_val = x[idx];
    float y_val = y[idx];

    // Calculate magnitude
    magnitude[idx] = hypot(x_val, y_val); // hypot is part of OpenCL's built-in functions

    // Calculate angle
    float ang = atan2(y_val, x_val); // atan2 is also a built-in function in OpenCL

    if (angleInDegrees) {
        ang = ang * (180.0f / M_PI); // Convert radians to degrees
    }

    angle[idx] = ang;
}

__kernel void NonMaximumSuppression(__read_only image2d_t blurred,__read_only image2d_t magnitude,
        __read_only image2d_t angle,
        __write_only image2d_t bufferNMStoDT){

            int2 pos = (int2)(get_global_id(0), get_global_id(1));

        const float pi = 3.14159265358979323846f;
        const float angleStep = pi / 8.0f;

        // Sample magnitudes and angles at the current position
        float magCenter = read_imagef(magnitude, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, pos).x;
        float angleCenter = read_imagef(angle, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, pos).x;

        // Determine angle bin based on the angle
        int angleBin = (int)((angleCenter + pi) / angleStep) % 8;

        // Neighboring positions based on the angle
        int2 pos1, pos2;
        switch (angleBin) {
            case 0:  pos1 = (int2)(-1, 0); pos2 = (int2)(1, 0); break;
            case 1:  pos1 = (int2)(-1, -1); pos2 = (int2)(1, 1); break;
            case 2:  pos1 = (int2)(0, -1); pos2 = (int2)(0, 1); break;
            case 3:  pos1 = (int2)(-1, 1); pos2 = (int2)(1, -1); break;
            case 4:  pos1 = (int2)(-1, 0); pos2 = (int2)(1, 0); break;
            case 5:  pos1 = (int2)(-1, -1); pos2 = (int2)(1, 1); break;
            case 6:  pos1 = (int2)(0, -1); pos2 = (int2)(0, 1); break;
            case 7:  pos1 = (int2)(-1, 1); pos2 = (int2)(1, -1); break;
        }

        // Sample magnitudes at neighboring positions
        float mag1 = read_imagef(magnitude, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, pos + pos1).x;
        float mag2 = read_imagef(magnitude, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, pos + pos2).x;

        // Perform NMS
        float nmsValue = (magCenter >= mag1 && magCenter >= mag2) ? magCenter : 0.0f;

        // Write NMS result to output
        write_imagef(bufferNMStoDT, pos, (float4)(nmsValue, 0.0f, 0.0f, 0.0f));

}

__kernel void DoubleThresholding(
    read_only image2d_t magnitudeImg, // Input magnitude image
    write_only image2d_t strongImg,   // Output image for strong edges
    write_only image2d_t weakImg,     // Output image for weak edges
    const float magMax,               // Upper threshold value
    const float magMin                // Lower threshold value
){
    int2 pos = (int2)(get_global_id(0), get_global_id(1)); // Current position

    // Sampler for reading the magnitude image
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | // Use unnormalized coordinates
                              CLK_ADDRESS_CLAMP_TO_EDGE |    // Clamp to edge addressing mode
                              CLK_FILTER_NEAREST;            // Nearest filtering

    float gradientMagnitude = read_imagef(magnitudeImg, sampler, pos).x;

    // Initial strong and weak values
    float strongVal = 0.0f, weakVal = 0.0f;

    // Apply double thresholding
    if (gradientMagnitude > magMax) {
        strongVal = gradientMagnitude;
    } else if (gradientMagnitude > magMin) {
        weakVal = gradientMagnitude;
    }

    // Write results to the output images
    write_imagef(strongImg, pos, (float4)(strongVal, 0, 0, 0));
    write_imagef(weakImg, pos, (float4)(weakVal, 0, 0, 0));
}

__kernel void Hysteresis(
    __read_only image2d_t strongImg,  // Input image for strong edges
    __read_only image2d_t weakImg,    // Input image for weak edges
    __write_only image2d_t outputImg  // Output image
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

    float strong = read_imagef(strongImg, smp, (int2)(x, y)).x;
    float output = strong; // Default to the strong value

    if (strong == 0.0f) { // If not already a strong edge
        float weak = read_imagef(weakImg, smp, (int2)(x, y)).x;
        if (weak != 0.0f) {
            // Check 8 neighbors to see if any is a strong edge
            bool isNearStrong = false;
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    if (dx == 0 && dy == 0) continue; // Skip self
                    float neighborStrong = read_imagef(strongImg, smp, (int2)(x + dx, y + dy)).x;
                    if (neighborStrong != 0.0f) {
                        isNearStrong = true;
                        break;
                    }
                }
                if (isNearStrong) break;
            }
            if (isNearStrong) {
                output = weak; // Promote weak edge to strong
            }
        }
    }

    // Write the result
    write_imagef(outputImg, (int2)(x, y), (float4)(output, 0, 0, 0));
}
