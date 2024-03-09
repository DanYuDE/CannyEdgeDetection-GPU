#ifndef __OPENCL_VERSION__
#include "../lib/OpenCL/OpenCLKernel.hpp"  // Hack to make syntax highlighting work
#endif

__kernel void sobelKernel4(__read_only image2d_t blurred,
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

