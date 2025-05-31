#include "TMC3.h"

#include <memory>
#include <filesystem>
#include "PCCTMC3Encoder.h"
#include "PCCTMC3Decoder.h"
#include "constants.h"
#include "ply.h"
#include "pointset_processing.h"
#include "program_options_lite.h"
#include "io_tlv.h"
#include "version.h"
#include "my_function.h"
using namespace std;
using namespace pcc;



vector<string>
Split(const string& s, const string& seperator)
{
  vector<string> ans;
  string token, str = s;
  size_t pos = 0;
  while ((pos = str.find(seperator)) != string::npos) {
    token = str.substr(0, pos);
    ans.push_back(token);
    str.erase(0, pos + seperator.length());
  }
  ans.push_back(str);
  return ans;
}
//1 逐行读取
vector<int>
read_category_num(string file)
{
  ifstream infile;
  infile.open(file.data());  //将文件流对象与文件连接起来
  assert(infile.is_open());  //若失败,则输出错误消息,并终止程序运行
  string s;
  vector<int> buffer;
  while (getline(infile, s)) {
    buffer.push_back(stoi(s));
  }
  infile.close();  //关闭文件输入流
  return buffer;
}

 PCCPointSet3
read_nuscence_bin(string path, int pointcount)
{
  PCCPointSet3 pointcloud;
  pointcloud.resize(pointcount);
  // 打开二进制文件以读取数据
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    // 处理文件打开失败的情况
    printf("文件打开失败");
  }

  // 假设我们知道文件中将存储多少个float数
  std::vector<float> data(5 * pointcount);

    // 读取数据到vector中
    file.read(
      reinterpret_cast<char*>(data.data()), 5 * pointcount * sizeof(float));
  for (int i = 0; i < pointcount; i++) {
    for (int k = 0; k <= 2; k++) {
      pointcloud[i][k] = std::round(1000 * data[5 * i + k]);  //转成毫米
    }
  }

  file.close();  // 关闭文件
  return pointcloud;
}




  PCCPointSet3
read_kitti_bin(string path, int pointcount)
{
  PCCPointSet3 pointcloud;
  pointcloud.resize(pointcount);
  // 打开二进制文件以读取数据
  std::ifstream file(path, std::ios::binary);

  if (!file) {
    // 处理文件打开失败的情况
    printf("文件打开失败");
  }

  // 假设我们知道文件中将存储多少个float数
  std::vector<float> data(4 * pointcount);

  // 读取数据到vector中
  file.read(
    reinterpret_cast<char*>(data.data()), 4 * pointcount * sizeof(float));
  for (int i = 0; i < pointcount; i++) {
    for (int k = 0; k <= 2; k++) {
      pointcloud[i][k] = std::round(1000 * data[4 * i + k]);  //转成毫米
    }
  }

  file.close();  // 关闭文件
  return pointcloud;
}
 void
write_original_pointcloud(PCCPointSet3 cloud1,string path)
 {
   
       ply::PropertyNameMap attrNames;
       Vec3<double> positionScale = {0, 0, 0};
       attrNames.position = {"x", "y", "z"};
       double plyScale = 1;
         ply::write(
           cloud1, attrNames, plyScale, positionScale, path, true);
       
 }

 vector<int>
 readlidar(string filepath)
 {
   vector<string> s = Split(filepath, "\\");
   string label_num_path = "";
   for (int i = 0; i < s.size();i++) {
       if (s[i] == "original_pointcloud") {

       string scnce_id = s[s.size()-2];
         label_num_path =
           label_num_path + "\\" + s[i] + "\\" + scnce_id + ".txt";
       break;
     }
       if (i!=0)
       label_num_path = label_num_path + "\\" + s[i];
       else
         label_num_path = label_num_path  + s[i];
   }

   vector<int> num_label = read_category_num(label_num_path);
      vector<string> id = Split(s[s.size() - 1], ".");
   int frame_id = stoi(id[0]);
   int pointcount = num_label[frame_id];


   string lidarpath;

      lidarpath = "";
   for (int i = 0; i < s.size(); i++) {
     if (s[i] == "kitti") {
       string scnce_id = s[s.size() - 2];
       lidarpath =
         lidarpath + s[i] + "\\label\\sequences\\"  + scnce_id + "\\labels\\"+id[0]+".label";
       break;
     }
     if (i!=0)
     lidarpath = lidarpath + "\\" + s[i];
     else
       lidarpath = s[i];
   }

   std::ifstream file(lidarpath, std::ios::binary);
   if (!file) {
     // 处理文件打开失败的情况
     printf("文件打开失败");
   }

   // 假设我们知道文件中将存储多少个float数
   std::vector<int> data(pointcount);

   // 读取数据到vector中
   file.read(reinterpret_cast<char*>(data.data()), pointcount * sizeof(float));

   
   file.close();  // 关闭文件
   return data;
 }


void
 write_lidar(vector<int> category,string path)
 {
  vector<string> s = Split(path, "\\");
   int n = s.size();
  vector<string> id = Split(s[n - 1], ".");

    string savepath;
   savepath = "H:\\luxiaoliang\\subsample_tmc13\\label\\"+s[n-2]+"\\"+id[0]+".txt";
   ofstream out(savepath, ios::trunc);
   if (!out.is_open()) {
     std::cout << "Failed to open file" << std::endl;
   }
   int num = category.size();
   for (int i = 0; i < num; i++) {
     out << category[i] << endl;
   }
   out.close();
 }


vector<string>
 read_token(string file)
 {
   ifstream infile;
   infile.open(file.data());  //将文件流对象与文件连接起来
   assert(infile.is_open());  //若失败,则输出错误消息,并终止程序运行
   string s;
   vector<string> buffer;
   while (getline(infile, s)) {
     buffer.push_back(s);
   }
   infile.close();  //关闭文件输入流
   return buffer;
 }




 param
analyse_path(char* argv[])
{
  param path;
  path.readPath = argv[1];
  path.savePath = argv[2];
  vector<string> readpath = Split(path.readPath,"=");
  path.readPath = readpath[1];
  vector<string> savepath = Split(path.savePath, "=");
  path.savePath = savepath[1];

  vector<string> s = Split(path.readPath, "\\");

  return path;

}

