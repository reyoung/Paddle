// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda.h>
#include <chrono>  // NOLINT
#include <cstdio>
#include <ctime>
#include <iostream>
#include <thread>  // NOLINT
#include <vector>

#include "../memory/malloc.h"
#include "../platform/device_context.h"
#include "../platform/place.h"
#include "paddle/fluid/framework/init.h"
#include "paddle/fluid/platform/device_context.h"

typedef std::chrono::high_resolution_clock Clock;

template <class T>
__global__ void im2col(const T *data_im, int num_outs, int im_height,
                       int im_width, int dilation_h, int dilation_w,
                       int filter_height, int filter_width, int stride_height,
                       int stride_width, int padding_height, int padding_width,
                       int col_height, int col_width, T *data_col) {
  const int index =
      (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
  if (index < num_outs) {
    int w_out = index % col_width;
    int h_out = (index / col_width) % col_height;
    int channel_in = index / col_width / col_height;
    int channel_out = channel_in * filter_height * filter_width;
    int h_in = h_out * stride_height - padding_height;
    int w_in = w_out * stride_width - padding_width;

    data_col += (channel_out * col_height + h_out) * col_width + w_out;
    data_im += (channel_in * im_height + h_in) * im_width + w_in;
    for (int i = 0; i < filter_height; ++i) {
      for (int j = 0; j < filter_width; ++j) {
        int rIdx = h_in + i * dilation_h;
        int cIdx = w_in + j * dilation_w;
        *data_col =
            (rIdx >= im_height || rIdx < 0 || cIdx >= im_width || cIdx < 0)
                ? 0
                : data_im[i * dilation_h * im_width + j * dilation_w];
        data_col += col_height * col_width;
      }
    }
  }
}

void task1(float *im_ptr_d, int im_channels, int im_height, int im_width,
           int filter_height, int filter_width, int col_height, int col_width,
           const std::vector<int> &padding, const std::vector<int> &dilation,
           const std::vector<int> &stride, cudaStream_t stream,
           float *col_ptr_d, int dev_id) {
  std::cout << "run..." << std::endl;
  auto t1 = Clock::now();

  typedef std::chrono::high_resolution_clock Clock;
  for (int i = 0; i < 1000; ++i) {
    cudaSetDevice(dev_id);
    int num_outputs = im_channels * col_height * col_width;
    int blocks = (num_outputs + 1024 - 1) / 1024;
    int block_x = 512;
    int block_y = (blocks + 512 - 1) / 512;
    dim3 threads(1024, 1);
    dim3 grid(block_x, block_y);

    im2col<float><<<grid, threads, 0, stream>>>(
        im_ptr_d, num_outputs, im_height, im_width, dilation[0], dilation[1],
        filter_height, filter_width, stride[0], stride[1], padding[0],
        padding[1], col_height, col_width, col_ptr_d);
    std::this_thread::sleep_for(std::chrono::microseconds(300));
  }
  cudaFree(im_ptr_d);
  cudaFree(col_ptr_d);

  auto t2 = Clock::now();
  std::cout
      << "Delta t2-t1: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << " milliseconds" << std::endl;
}

int main() {
  paddle::framework::InitDevices(true);
  // Config
  int im_channels = 256 * 2;
  int im_height = 224 * 2;
  int im_width = 224 * 2;
  int filter_height = 5;
  int filter_width = 5;
  std::vector<int> padding{1, 1, 1, 1};
  std::vector<int> dilation{1, 1};
  std::vector<int> stride{2, 2};

  int col_height = (im_height + padding[0] + padding[2] -
                    (dilation[0] * (filter_height - 1) + 1)) /
                       stride[0] +
                   1;
  int col_width = (im_width + padding[1] + padding[3] -
                   (dilation[1] * (filter_width - 1) + 1)) /
                      stride[1] +
                  1;

  size_t im_size = im_channels * im_height * im_width;
  size_t col_size =
      im_channels * filter_height * filter_width * col_height * col_width;

  const int t_cnt = paddle::platform::GetCUDADeviceCount();
  float *src_h;
  std::vector<float *> src_d(t_cnt);
  std::vector<float *> dst_d(t_cnt);
  std::vector<cudaStream_t> streams(t_cnt);

  // Init
  src_h = new float[im_size];

  for (int i = 0; i < t_cnt; ++i) {
    src_d[i] = reinterpret_cast<float *>(paddle::memory::Alloc(
        paddle::platform::CUDAPlace(i), im_size * sizeof(float)));
    dst_d[i] = reinterpret_cast<float *>(paddle::memory::Alloc(
        paddle::platform::CUDAPlace(i), col_size * sizeof(float)));
    streams[i] = reinterpret_cast<paddle::platform::CUDADeviceContext *>(
                     paddle::platform::DeviceContextPool::Instance().Get(
                         paddle::platform::CUDAPlace(i)))
                     ->stream();
  }

  for (int i = 0; i < im_size; ++i) {
    src_h[i] = i;
  }

  for (int i = 0; i < t_cnt; ++i) {
    cudaSetDevice(i);
    cudaMemcpy(src_d[i], src_h, im_size * sizeof(float),
               cudaMemcpyHostToDevice);
  }

  // Run
  std::vector<std::thread> workers(t_cnt);
  for (size_t i = 0; i < t_cnt; ++i) {
    workers[i] = std::thread([&, i] {
      task1(src_d[i], im_channels, im_height, im_width, filter_height,
            filter_width, col_height, col_width, padding, dilation, stride,
            streams[i], dst_d[i], i);
    });
  }

  // Wait
  for (auto &worker : workers) {
    worker.join();
    std::cout << "over\n";
  }

  // delete
  delete src_h;
}
