/* The copyright in this software is being made available under the BSD
 * Licence, included below.  This software may be subject to other third
 * party and contributor rights, including patent rights, and no such
 * rights are granted under this licence.
 *
 * Copyright (c) 2017-2018, ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 * * Neither the name of the ISO/IEC nor the names of its contributors
 *   may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "TMC3.h"
#include"my_function.h"
#include <memory>
#include "stb_image_write.h"
#include "PCCTMC3Encoder.h"
#include "PCCTMC3Decoder.h"
#include "constants.h"
#include "ply.h"
#include <opencv2/opencv.hpp>
#include "pointset_processing.h"
#include "program_options_lite.h"
#include "io_tlv.h"
#include "version.h"
using namespace std;
using namespace pcc;
using namespace cv;
struct Box {
    Vec3<int> min = std::numeric_limits<int32_t>::max();
    Vec3<int> max= std::numeric_limits<int32_t>::lowest();
};


//============================================================================
class buildingSeg{

public:PCCPointSet3 pointcloud;


      buildingSeg(PCCPointSet3& pointcloud) {
          this->pointcloud = pointcloud;
          
          for (size_t i = 0; i < pointcloud.getPointCount(); ++i) {
              const auto& pt = pointcloud[i];
              for (int k = 0; k < 3; ++k) {
                  if (pt[k] > this->box.max[k]) {
                      this->box.max[k] = pt[k];
                  }
                  if (pt[k] < this->box.min[k]) {
                      this->box.min[k] = pt[k];
                  }
              }
          }

          for (size_t i = 0; i < pointcloud.getPointCount(); ++i) {       //搬移为正数
              this->pointcloud[i] -= this->box.min;
          }


          width = (this->box.max[0] - this->box.min[0]) / this->bin+2;
          height = (this->box.max[1] - this->box.min[1]) / this->bin+2;

          this->image.resize(this->width * this->height* channels, 0);
      }

      void save_image(string savePath) {
          std::vector<uint8_t> image; // 使用8位无符号整数（0-255范围）
          image.resize(width * height*channels, 0);

          double max[3] = { 0 };
          for (int i = 0;i < width;i++)
              for (int j = 0;j < height;j++)
                  for(int channel=0;channel< channels;channel++)
                      if(max[channel]< pixel(i, j, channel))
                            max[channel]=pixel(i, j, channel);
         

          for (int i = 0;i < width;i++)
              for (int j = 0;j < height;j++)
                  for (int channel = 0;channel < 1;channel++)
                      if (max[channel]!=0)
                            image[(i  + j * width) * channels + channel]= 255.0 * (1.0 * pixel(i, j, channel)/ max[channel]);
          stbi_write_png((savePath+"平均高度.png").c_str(), this->width, this->height, this->channels, image.data(), this->width * this->channels);

          
          image.clear();
          image.resize(width * height * channels, 0);
          for (int i = 0;i < width;i++)
              for (int j = 0;j < height;j++)
                  for (int channel = 1;channel < 2;channel++)
                      if (max[channel] != 0)
                          image[(i + j * width) * channels + channel] = 255.0 * (1.0 * pixel(i, j, channel) / max[channel]);
          stbi_write_png((savePath + "像素数量.png").c_str(), this->width, this->height, this->channels, image.data(), this->width * this->channels);



          image.clear();
          image.resize(width * height * channels, 0);
          for (int i = 0;i < width;i++)
              for (int j = 0;j < height;j++)
                  for (int channel = 2;channel < 3;channel++)
                      if (max[channel] != 0)
                          image[(i + j * width) * channels+1] = 255.0 * (1.0 * pixel(i, j, channel) / max[channel]);
          stbi_write_png((savePath + "像素数量+高度.png").c_str(), this->width, this->height, this->channels, image.data(), this->width * this->channels);

      }

      double& pixel(int x, int y,int channel) {
          return this->image[(y * width + x) * channels + channel];
      }

      void compute_gird_picture() {
          auto bin = this->bin;

          double th = groundTH();

          for (size_t i = 0; i < this->pointcloud.getPointCount(); ++i) {
              auto pt = this->pointcloud[i];
              int x = pt[0] / bin;
              int y = pt[1] / bin;
              for(int xi=0;xi<2;xi++)
                  for (int yi = 0;yi < 2;yi++)
                  {
                      if (pt[2] < th)
                          continue;
                      auto w = 1.0*pt[0] / bin-x;
                      auto h = 1.0 * pt[1] / bin - y;
                      double s = ((xi == 1) ? w : (1 - w)) * ((yi == 1) ? h : (1 - h));
                      this->pixel(x+xi, y+yi,1)+=s;                                               //像素数量为g分量
                      this->pixel(x + xi, y + yi, 0) += s*pt[2];                                  //高度图为r分量

                  }
          }


          for (int i = 0;i < width;i++)                                 //计算平均高度，获得清晰图像
              for (int j = 0;j < height;j++) {
                  if (pixel(i, j, 1) != 0) {
                      pixel(i, j, 0) = pixel(i, j, 0) / pixel(i, j, 1);
                  }  
              }

          for (int i = 0;i < width;i++)                                 //像素数量对数化，获得清晰图像
              for (int j = 0;j < height;j++) {
                  pixel(i, j, 1) = std::log(pixel(i, j, 1) + 1);
                  if (pixel(i, j, 1) != 0)                              
                      pixel(i, j, 1) += 20;               
              }


          for (int i = 0;i < width;i++)                                 //计算平均高度，获得清晰图像
              for (int j = 0;j < height;j++) {
                      pixel(i, j, 2) = pixel(i, j, 0) * pixel(i, j, 1);
              }

      }


      
private: Box box;
    int bin=80,bin_height=1000;//分箱长度为1000毫米,高度每段0.1米
    int width, height, channels=3;
    std::vector<double> image;

    double groundTH() {
        std::vector<int> num_heigh;
        num_heigh.resize((box.max[2]-box.min[2])/ bin_height+1, 0);
        int TH = pointcloud.getPointCount() / 2;
        for (size_t i = 0; i < pointcloud.getPointCount(); ++i) {
            const auto& pt = pointcloud[i];
            num_heigh[pt[2] / bin_height]++;
        }

        int total = 0,i;
        for (i = 0;i < num_heigh.size();i++){
            total += num_heigh[i];
            if (total > TH)
                break;
        }

        return i*bin_height;
    }

};

int
main(int argc, char* argv[])
{
  param path = analyse_path(argv);  
  PCCPointSet3 pointCloud;
  double positionScale = 1000;                   //转成毫米       
  if (!ply::read(path.readPath, { "x", "y", "z" }, positionScale, pointCloud)
    || pointCloud.getPointCount() == 0) {
    cout << "Error: can't open "<< path.readPath << endl;
    return -1;
  }                
  buildingSeg seg=buildingSeg(pointCloud);
  seg.compute_gird_picture();
  string base= "C:\\Users\\31046\\Desktop\\city3D\\";
  seg.save_image(base);

  extracted_contour(base+"像素数量.png", base + "extracted_contours.png",base+"extracted_contours_flip.png");

  return 0;
}

//ply::write(encoder_pointCloud, { "x", "y", "z" }, 1.00, { 0, 0, 0 }, path.savePath, false);