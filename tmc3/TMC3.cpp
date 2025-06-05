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
#include"my_function.h"
#include <memory>
#include <iostream>
#include <vector>
#include "stb_image_write.h"
#include <opencv2/opencv.hpp>
#include "PCCPointSet.h"
#include "ply.h"
using namespace open3d;
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

          for (size_t i = 0; i < pointcloud.getPointCount(); ++i) {       //����Ϊ����
              this->pointcloud[i] -= this->box.min;
          }


          width = (this->box.max[0] - this->box.min[0]) / this->bin+2;
          height = (this->box.max[1] - this->box.min[1]) / this->bin+2;

          this->image.resize(this->width * this->height* channels, 0);
      }

      void save_image(string savePath) {
          std::vector<uint8_t> image; // ʹ��8λ�޷���������0-255��Χ��
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
          stbi_write_png((savePath+"ƽ���߶�.png").c_str(), this->width, this->height, this->channels, image.data(), this->width * this->channels);

          
          image.clear();
          image.resize(width * height * channels, 0);
          for (int i = 0;i < width;i++)
              for (int j = 0;j < height;j++)
                  for (int channel = 1;channel < 2;channel++)
                      if (max[channel] != 0)
                          image[(i + j * width) * channels + channel] = 255.0 * (1.0 * pixel(i, j, channel) / max[channel]);
          stbi_write_png((savePath + "��������.png").c_str(), this->width, this->height, this->channels, image.data(), this->width * this->channels);



          image.clear();
          image.resize(width * height * channels, 0);
          for (int i = 0;i < width;i++)
              for (int j = 0;j < height;j++)
                  for (int channel = 2;channel < 3;channel++)
                      if (max[channel] != 0)
                          image[(i + j * width) * channels+1] = 255.0 * (1.0 * pixel(i, j, channel) / max[channel]);
          stbi_write_png((savePath + "��������+�߶�.png").c_str(), this->width, this->height, this->channels, image.data(), this->width * this->channels);

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
                      this->pixel(x+xi, y+yi,1)+=s;                                               //��������Ϊg����
                      this->pixel(x + xi, y + yi, 0) += s*pt[2];                                  //�߶�ͼΪr����

                  }
          }



          for (int i = 0;i < width;i++)                                 //����ƽ���߶ȣ��������ͼ��
              for (int j = 0;j < height;j++) {
                  if (pixel(i, j, 1) != 0) {
                      pixel(i, j, 0) = pixel(i, j, 0) / pixel(i, j, 1);
                  }  
              }

          for (int i = 0;i < width;i++)                                 //�����������������������ͼ��
              for (int j = 0;j < height;j++) {
                  pixel(i, j, 1) = std::log(pixel(i, j, 1) + 1);
                  if (pixel(i, j, 1) != 0)                              
                      pixel(i, j, 1) += 20;               
              }


          //for (int i = 0;i < width;i++)                                 //����ƽ���߶ȣ��������ͼ��
          //    for (int j = 0;j < height;j++) {
          //            pixel(i, j, 2) = pixel(i, j, 0) * pixel(i, j, 1);
          //    }

      }


      
private: Box box;
    int bin=100,bin_height=1000;//���䳤��Ϊ1000����,�߶�ÿ��0.1��
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
int o3d(string path) {
    // 1. ��ȡ����
    geometry::PointCloud pcd;
    if (io::ReadPointCloud(path, pcd)) {
        std::cout << "�ɹ���ȡ���ƣ����� " << pcd.points_.size() << " ����" << std::endl;
    }
    else {
        std::cerr << "��ȡ����ʧ��" << std::endl;
        return -1;
    }

    // 2. ���㷨���� (ʹ��KDTree�����������뾶0.1����࿼��30���ھ�)
    pcd.EstimateNormals(geometry::KDTreeSearchParamHybrid(1, 50));
    //pcd.OrientNormalsToAlignWithDirection(Eigen::Vector3d(0.0, 0.0, 1.0));
    std::cout << "�������������" << std::endl;

    for (int i = 0;i < pcd.points_.size();i++)
      for(int k=0;k<3;k++)
        pcd.normals_[i][k] /= 10;


    // 3. ����KDTree������10�����
    geometry::KDTreeFlann kdtree(pcd);
    int query_index = 0;  // ��ѯ�������
    std::vector<int> indices(10);
    std::vector<double> distances(10);

    if (kdtree.SearchKNN(pcd.points_[query_index], 10, indices, distances) > 0) {
        std::cout << "�� " << query_index << " ��10�����������: ";
        for (size_t i = 0; i < indices.size(); ++i) {
            std::cout << indices[i] << " ";
        }
        std::cout << std::endl;
    }


    // 5. ������� (����������)
    if (io::WritePointCloud("C:\\Users\\31046\\Desktop\\city3D\\output.ply", pcd)) {
        std::cout << "���Ʊ���ɹ�" << std::endl;
    }
    else {
        std::cerr << "�������ʧ��" << std::endl;
    }

    return 0;
}
int plane(string path) {
    // 1. ��ȡ����
    geometry::PointCloud pcd;
    io::ReadPointCloud(path, pcd);

    // 2. ƽ��ָ����
    const double distance_threshold = 0.05;  // 2cm
    const int ransac_n = 4;
    const int num_iterations = 1000;         // ���ӵ�������
    const double probability = 0.8;        // 99.9% ���Ŷ�

    // 3. ִ��ƽ��ָ�
    auto [plane_model, inliers] = pcd.SegmentPlane(
        distance_threshold, ransac_n, num_iterations, probability);

    // 4. ��ȡƽ�����
    auto plane_cloud = pcd.SelectByIndex(inliers);
    auto remaining_cloud = pcd.SelectByIndex(inliers, true); // ��ѡ

    // 5. ������
    std::cout << "ƽ�淽��: " << plane_model.transpose() << std::endl;
    std::cout << "�ڵ�����: " << inliers.size() << std::endl;
    std::cout << "�������: " << remaining_cloud->points_.size() << std::endl;

    // 6. ���ӻ�
    plane_cloud->PaintUniformColor(Eigen::Vector3d(1, 0, 0));  // ��ɫƽ��
    remaining_cloud->PaintUniformColor(Eigen::Vector3d(0, 0, 1)); // ��ɫʣ���

    visualization::DrawGeometries({
        plane_cloud, remaining_cloud
        }, "ƽ��ָ���", 1600, 900);

    return 0;
}
int
main(int argc, char* argv[])
{
  param path = analyse_path(argv);  
  PCCPointSet3 pointCloud;
  double positionScale = 1000;                   //ת�ɺ���       
  ply::read(path.readPath, { "x", "y", "z" }, positionScale, pointCloud);

  plane(path.readPath);


  buildingSeg seg=buildingSeg(pointCloud);
  seg.compute_gird_picture();
  string base= "C:\\Users\\31046\\Desktop\\city3D\\";
  seg.save_image(base);
  extracted_contour(base+"��������.png", base + "extracted_contours.png",base+"extracted_contours_flip.png");

  return 0;
}

//ply::write(encoder_pointCloud, { "x", "y", "z" }, 1.00, { 0, 0, 0 }, path.savePath, false);