
#include <opencv2/opencv.hpp>
#include <memory>
#include <filesystem>
#include "ply.h"
#include "my_function.h"

void extracted_contour(string read_path, string save_path,string flip) {
    Mat src = imread(read_path, IMREAD_COLOR);
    if (src.empty()) {
        cout << "无法加载点云图像" << endl;
    }
  
    // 2. 创建处理结果的图像副本
    Mat result = src.clone();

    // 3. 提取红色通道并进行阈值处理
    Mat redChannel;
    extractChannel(src, redChannel, 1); // 提取红色通道(索引2)
    threshold(redChannel, redChannel, 10, 255, THRESH_BINARY); // 二值化处理


    // 4. 形态学操作连接邻近点云
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    Mat morphed;
    morphologyEx(redChannel, morphed, MORPH_CLOSE, kernel, Point(-1, -1), 2);
   // cv::namedWindow("形态学处理结果", cv::WINDOW_NORMAL); // 允许调整窗口
    //imshow("形态学处理结果", morphed);
    //waitKey();
    // 5. 寻找轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(morphed, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

    // 6. 筛选主要的建筑物轮廓
    vector<vector<Point>> buildingContours;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        double perimeter = arcLength(contours[i], true);

        // 根据面积和周长筛选大型轮廓
        if (area > 500 && perimeter > 100) {
            buildingContours.push_back(contours[i]);

            //// 绘制轮廓边界框
            //RotatedRect minRect = minAreaRect(contours[i]);
            //Point2f rectPoints[4];
            //minRect.points(rectPoints);
            //for (int j = 0; j < 4; j++) {
            //    line(result, rectPoints[j], rectPoints[(j + 1) % 4], Scalar(255, 255, 0), 2);
            //}
        }
    }

    // 7. 绘制所有建筑物轮廓
    for (size_t i = 0; i < buildingContours.size(); i++) {
        drawContours(result, buildingContours, i, Scalar(255, 255, 0), 2);
    }




    // 4. 准备写入OBJ文件
    ofstream objFile("csa.obj");
    if (!objFile.is_open()) {
        cerr << "无法创建输出文件: " << endl;
        return;
    }

    // OBJ文件头
    objFile << "# 从轮廓生成的3D模型" << endl;
    objFile << "# 轮廓数量: " << contours.size() << endl;
    objFile << "# 顶点归一化到范围 [0,1] (x,y)" << endl << endl;

    int vertexIndex = 1;  // OBJ文件顶点索引从1开始
    vector<vector<int>> allVertexGroups;

    // 5. 处理每个轮廓
    for (size_t i = 0; i < contours.size(); i++) {
        vector<Point> contour = contours[i];

        // 轮廓归一化到[0,1]范围
        vector<Point2f> normalizedContour;
        for (const Point& p : contour) {
            normalizedContour.push_back(Point2f(
                static_cast<float>(p.x) / src.cols,
                1.0f - static_cast<float>(p.y) / src.rows  // 翻转y轴
            ));
        }

        // 为当前轮廓存储顶点索引
        vector<int> vertexGroup;

        // 添加底部和顶部顶点
        for (const Point2f& p : normalizedContour) {
            // 底部顶点 (z=0)
            objFile << "v " << p.x << " " << p.y << " 0.0" << endl;
            vertexGroup.push_back(vertexIndex++);

            // 顶部顶点 (z=height)
            objFile << "v " << p.x << " " << p.y << " " << 1 << endl;
            vertexGroup.push_back(vertexIndex++);
        }

        allVertexGroups.push_back(vertexGroup);
    }

    // 6. 创建面
    objFile << endl << "# 侧面 (四边形面)" << endl;
    for (const vector<int>& vertices : allVertexGroups) {
        int n = vertices.size() / 2;

        // 创建侧面四边形
        for (int i = 0; i < n; i++) {
            int next = (i + 1) % n;

            int bottomIdx1 = vertices[i * 2];
            int topIdx1 = vertices[i * 2 + 1];
            int bottomIdx2 = vertices[next * 2];
            int topIdx2 = vertices[next * 2 + 1];

            // 创建一个四边形面
            objFile << "f " << bottomIdx1 << " " << bottomIdx2 << " "
                << topIdx2 << " " << topIdx1 << endl;
        }
    }



    // 8. 显示处理过程中的图像
    //cv::namedWindow("红色通道提取", cv::WINDOW_NORMAL); // 允许调整窗口
    //cv::namedWindow("形态学处理结果", cv::WINDOW_NORMAL); // 允许调整窗口
    //cv::namedWindow("建筑物轮廓提取", cv::WINDOW_NORMAL); // 允许调整窗口

    //imshow("红色通道提取", redChannel);
    //imshow("形态学处理结果", morphed);
    //imshow("建筑物轮廓提取", result);

    // 9. 保存结果图像
    imwrite(save_path, result);
    Mat image_fliped;
    cv::flip(result, image_fliped, 0);
    imwrite(flip, image_fliped);

}

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

vector<plane> seg_plane::get_planes() {
     int i = 0;
     vector<plane> planes;

     for (;i < Cloud.getPointCount();i++) {
         if (Cloud.planeIdx[i] == -1) {

             cur_plane = plane();
             cur_plane.id = cur_planeId;
             cur_normal = normal[i];
             cur_center = Cloud[i];
             cur_plane.pointIdx.push_back(i);

             if (!Broad(i, 0))
                 continue;

             cur_plane.center = cur_center;
             cur_plane.normal = cur_normal;

             if (cur_plane.pointIdx.size() > th_pointCount) {
                 planes.push_back(cur_plane);
                 cur_planeId++;
             }
             else {
                 for (int i = 0;i < cur_plane.pointIdx.size();i++)
                 {
                     auto id = cur_plane.pointIdx[i];
                     Cloud.planeIdx[id] = -1;
                 }
             }
                

             
         }
     }

     return planes;
 }


bool seg_plane::Broad(int Idx,int depth) {
    vector<int> index = neigh[Idx];
    std::vector<int> selectedId;

    for (int idIndex = 1;idIndex < K;idIndex++) {
        auto id = index[idIndex];
        if (Cloud.planeIdx[id] <= 0) {
            Vec3<int> pVector = Cloud[id] - cur_center;
            auto distance_plane = abs(pVector * cur_normal);

            if (distance_plane <= th_thickness && cur_normal*normal[id]>=0.88) {
                selectedId.push_back(id);
                cur_plane.pointIdx.push_back(id);
                Cloud.planeIdx[id]=cur_planeId;
            }
               
        }
    }
    if (depth==0&&selectedId.size() < K - 1)
        return false;

    cur_normal = {0,0,0};
    cur_center = {0,0,0};
    for (int i = 0;i < cur_plane.pointIdx.size();i++)
    {
        auto id = cur_plane.pointIdx[i];
        cur_normal += normal[id];
        cur_center += Cloud[id];
    }
    cur_normal /= sqrt(cur_normal.getNorm2());
    cur_center /= cur_plane.pointIdx.size();

    for (int i = 0;i < selectedId.size();i++) {
        auto id = selectedId[i];
        Broad(id,depth+1);
    }

    return true;
}

void seg_plane::set_plane_color(vector<plane> &planes) {

    for (int i = 0;i < Cloud.getPointCount();i++) {
        Cloud.setColor(i, { 0,0,0 });
    }
        


    for (plane& p : planes) {
        Vec3<int> color = { 55+rand() % 200,55+rand() % 200,55+rand() % 200 };
        for (int i = 0;i < p.pointIdx.size();i++) {
            auto id = p.pointIdx[i];
            Cloud.setColor(id, color);
        }
    }
}