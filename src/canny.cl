#ifndef __OPENCL_VERSION__
#include "../lib/OpenCL/OpenCLKernel.hpp"  // Hack to make syntax highlighting work
#endif

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

__kernel void NonMaximumSuppression(){

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

__kernel void hysteresis(
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
