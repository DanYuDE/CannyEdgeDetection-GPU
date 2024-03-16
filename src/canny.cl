#ifndef __OPENCL_VERSION__
#include "../lib/OpenCL/OpenCLKernel.hpp"  // Hack to make syntax highlighting work
#endif

inline float getValueImage(__read_only image2d_t image, int x, int y, const sampler_t sampler) {
    int2 coords = (int2)(x, y); // Create a 2D integer vector for coordinates
    float4 pixel = read_imagef(image, sampler, coords); // Read the pixel value using the sampler
    return pixel.x; // Return the first channel assuming image format is single channel float
}

__kernel void Sobel(__read_only image2d_t inputImage,
                           __write_only image2d_t bufferSBLtoNMS) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | // Non-normalized coordinates
                              CLK_ADDRESS_CLAMP |            // Clamp to edge
                              CLK_FILTER_NEAREST;           // Nearest neighbor interpolation

    // Sobel filter in X direction (horizontal edges)
    float Gx = -1 * getValueImage(inputImage, x - 1, y - 1, sampler)
               -2 * getValueImage(inputImage, x - 1, y, sampler)
               -1 * getValueImage(inputImage, x - 1, y + 1, sampler)
               +1 * getValueImage(inputImage, x + 1, y - 1, sampler)
               +2 * getValueImage(inputImage, x + 1, y, sampler)
               +1 * getValueImage(inputImage, x + 1, y + 1, sampler);

    // Sobel filter in Y direction (vertical edges)
    float Gy = -1 * getValueImage(inputImage, x - 1, y - 1, sampler)
               -2 * getValueImage(inputImage, x, y - 1, sampler)
               -1 * getValueImage(inputImage, x + 1, y - 1, sampler)
               +1 * getValueImage(inputImage, x - 1, y + 1, sampler)
               +2 * getValueImage(inputImage, x, y + 1, sampler)
               +1 * getValueImage(inputImage, x + 1, y + 1, sampler);

    // Compute magnitude and angle (in radians for further processing)
    float magnitude = hypot(Gx, Gy);
    float angle = atan2(Gy, Gx);

    // Writing out the magnitude to the output image
    write_imagef(bufferSBLtoNMS, (int2)(x, y), (float4)(magnitude, 0, 0, 0));

}


__kernel void NonMaximumSuppression(__read_only image2d_t blurred,
                                    __write_only image2d_t bufferNMStoDT) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    // Create a sampler object
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | // Non-normalized coordinates
                              CLK_ADDRESS_CLAMP_TO_EDGE |    // Clamp to edge addressing mode
                              CLK_FILTER_NEAREST;            // Nearest neighbor interpolation

    // Corrected usage of read_imagef with a sampler
    float2 gradient = read_imagef(blurred, sampler, pos).xy;

    const float pi = 3.14159265358979323846f;
    const float angleStep = pi / 8.0f;

    // Sample gradients at the current position
    float x_val = gradient.x;
    float y_val = gradient.y;

    // Calculate magnitude
    float magCenter = hypot(x_val, y_val);

    // Calculate angle
    float ang = atan2(y_val, x_val);
    
    // Convert angle to degrees if necessary
    ang = ang * (180.0f / pi);

    // Determine angle bin based on the angle
    int angleBin = (int)((ang + pi) / angleStep) % 8;

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
    float mag1 = read_imagef(blurred, sampler, pos + pos1).x;
    float mag2 = read_imagef(blurred, sampler, pos + pos2).x;

    // Perform NMS
    float nmsValue = (magCenter >= mag1 && magCenter >= mag2) ? magCenter : 0.0f;

    // Write NMS result to output
    write_imagef(bufferNMStoDT, pos, (float4)(nmsValue, ang, 0.0f, 0.0f));
}


__kernel void DoubleThresholding(
    read_only image2d_t magnitudeImg,
    write_only image2d_t strongImg,
    write_only image2d_t weakImg,
    const float magMax,
    const float magMin
){
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP_TO_EDGE |
                              CLK_FILTER_NEAREST;

    float gradientMagnitude = read_imagef(magnitudeImg, sampler, pos).x;

    float strongVal = 0.0f, weakVal = 0.0f;
    if (gradientMagnitude > magMax) {
        strongVal = 1.0f; // Strong edge
    } else if (gradientMagnitude > magMin) {
        weakVal = 1.0f; // Weak edge (candidate for hysteresis)
    }

    write_imagef(strongImg, pos, (float4)(strongVal, 0, 0, 0));
    write_imagef(weakImg, pos, (float4)(weakVal, 0, 0, 0));
}

__kernel void Hysteresis(
    read_only image2d_t strongImg,
    read_only image2d_t weakImg,
    write_only image2d_t outputImg
){
    int x = get_global_id(0);
    int y = get_global_id(1);
    const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE |
                          CLK_ADDRESS_CLAMP |
                          CLK_FILTER_NEAREST;

    float strong = read_imagef(strongImg, smp, (int2)(x, y)).x;
    float weak = read_imagef(weakImg, smp, (int2)(x, y)).x;
    float output = strong; // Start with strong edges

    if (strong == 0.0f && weak != 0.0f) {
        // Examine the 8 neighbors
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                if (dx == 0 && dy == 0) continue; // Skip the center pixel
                float neighborStrong = read_imagef(strongImg, smp, (int2)(x + dx, y + dy)).x;
                if (neighborStrong != 0.0f) {
                    output = 1.0f; // Promote to strong edge
                    break;
                }
            }
        }
    }
    write_imagef(outputImg, (int2)(x, y), (float4)(output, 0, 0, 0));
}

