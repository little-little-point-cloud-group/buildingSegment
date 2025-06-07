
#include <opencv2/opencv.hpp>
#include <memory>
#include <filesystem>
#include "ply.h"
#include "my_function.h"

void extracted_contour(string read_path, string save_path,string flip) {
    Mat src = imread(read_path, IMREAD_COLOR);
    if (src.empty()) {
        cout << "�޷����ص���ͼ��" << endl;
    }
  
    // 2. ������������ͼ�񸱱�
    Mat result = src.clone();

    // 3. ��ȡ��ɫͨ����������ֵ����
    Mat redChannel;
    extractChannel(src, redChannel, 1); // ��ȡ��ɫͨ��(����2)
    threshold(redChannel, redChannel, 10, 255, THRESH_BINARY); // ��ֵ������


    // 4. ��̬ѧ���������ڽ�����
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    Mat morphed;
    morphologyEx(redChannel, morphed, MORPH_CLOSE, kernel, Point(-1, -1), 2);
   // cv::namedWindow("��̬ѧ������", cv::WINDOW_NORMAL); // �����������
    //imshow("��̬ѧ������", morphed);
    //waitKey();
    // 5. Ѱ������
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(morphed, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

    // 6. ɸѡ��Ҫ�Ľ���������
    vector<vector<Point>> buildingContours;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        double perimeter = arcLength(contours[i], true);

        // ����������ܳ�ɸѡ��������
        if (area > 500 && perimeter > 100) {
            buildingContours.push_back(contours[i]);

            //// ���������߽��
            //RotatedRect minRect = minAreaRect(contours[i]);
            //Point2f rectPoints[4];
            //minRect.points(rectPoints);
            //for (int j = 0; j < 4; j++) {
            //    line(result, rectPoints[j], rectPoints[(j + 1) % 4], Scalar(255, 255, 0), 2);
            //}
        }
    }

    // 7. �������н���������
    for (size_t i = 0; i < buildingContours.size(); i++) {
        drawContours(result, buildingContours, i, Scalar(255, 255, 0), 2);
    }




    // 4. ׼��д��OBJ�ļ�
    ofstream objFile("csa.obj");
    if (!objFile.is_open()) {
        cerr << "�޷���������ļ�: " << endl;
        return;
    }

    // OBJ�ļ�ͷ
    objFile << "# ���������ɵ�3Dģ��" << endl;
    objFile << "# ��������: " << contours.size() << endl;
    objFile << "# �����һ������Χ [0,1] (x,y)" << endl << endl;

    int vertexIndex = 1;  // OBJ�ļ�����������1��ʼ
    vector<vector<int>> allVertexGroups;

    // 5. ����ÿ������
    for (size_t i = 0; i < contours.size(); i++) {
        vector<Point> contour = contours[i];

        // ������һ����[0,1]��Χ
        vector<Point2f> normalizedContour;
        for (const Point& p : contour) {
            normalizedContour.push_back(Point2f(
                static_cast<float>(p.x) / src.cols,
                1.0f - static_cast<float>(p.y) / src.rows  // ��תy��
            ));
        }

        // Ϊ��ǰ�����洢��������
        vector<int> vertexGroup;

        // ��ӵײ��Ͷ�������
        for (const Point2f& p : normalizedContour) {
            // �ײ����� (z=0)
            objFile << "v " << p.x << " " << p.y << " 0.0" << endl;
            vertexGroup.push_back(vertexIndex++);

            // �������� (z=height)
            objFile << "v " << p.x << " " << p.y << " " << 1 << endl;
            vertexGroup.push_back(vertexIndex++);
        }

        allVertexGroups.push_back(vertexGroup);
    }

    // 6. ������
    objFile << endl << "# ���� (�ı�����)" << endl;
    for (const vector<int>& vertices : allVertexGroups) {
        int n = vertices.size() / 2;

        // ���������ı���
        for (int i = 0; i < n; i++) {
            int next = (i + 1) % n;

            int bottomIdx1 = vertices[i * 2];
            int topIdx1 = vertices[i * 2 + 1];
            int bottomIdx2 = vertices[next * 2];
            int topIdx2 = vertices[next * 2 + 1];

            // ����һ���ı�����
            objFile << "f " << bottomIdx1 << " " << bottomIdx2 << " "
                << topIdx2 << " " << topIdx1 << endl;
        }
    }



    // 8. ��ʾ��������е�ͼ��
    //cv::namedWindow("��ɫͨ����ȡ", cv::WINDOW_NORMAL); // �����������
    //cv::namedWindow("��̬ѧ������", cv::WINDOW_NORMAL); // �����������
    //cv::namedWindow("������������ȡ", cv::WINDOW_NORMAL); // �����������

    //imshow("��ɫͨ����ȡ", redChannel);
    //imshow("��̬ѧ������", morphed);
    //imshow("������������ȡ", result);

    // 9. ������ͼ��
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