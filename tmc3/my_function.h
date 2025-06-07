
#include<string>
#include<Vector>
#include <open3d/Open3D.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <filesystem>
#include "ply.h"


using namespace pcc;
using namespace cv;
using namespace std;
using namespace open3d;

class param {
public:
    string dataType;
    string readPath;
    string savePath;
    string num_sizePath;
    int frame;
};

struct plane {
    int id;       //id��Ҫ����0
    Vec3<double> normal;
    Vec3<int> center;
    std::vector<int>  pointIdx;
};


void extracted_contour(string read_path, string save_path, string flip);

param analyse_path(char* argv[]);


vector<string> Split(const string& s, const string& seperator);

// ƽ��ṹ�壺�洢ƽ����Ϣ
struct DetectedPlane {
    std::vector<size_t> indices;  // ���ڸ�ƽ��ĵ�����
    Eigen::Vector4d equation;     // ƽ�淽�� ax+by+cz+d=0
    Eigen::Vector3d normal;       // ƽ�淨����
    double d;                     // ƽ�泣����
};

template <int K>
void get_Normal_and_K_neighbor(PCCPointSet3& pointCloud, std::vector<Vec3<double>>& normal, std::vector<vector<int>>& neigh) {
    // 1. ��ȡ����
    geometry::PointCloud pcd;
    int pointCount = pointCloud.getPointCount();
    normal.resize(pointCount);
    neigh.resize(pointCount);

    pcd.points_.resize(pointCount);
    for (int i = 0;i < pointCount;i++) {
        for (int k = 0;k < 3;k++)
            pcd.points_[i][k] = pointCloud[i][k];
    }

    // 2. ���㷨���� (ʹ��KDTree�����������뾶0.1����࿼��30���ھ�)
    pcd.EstimateNormals(geometry::KDTreeSearchParamHybrid(100, 50));
    pcd.OrientNormalsToAlignWithDirection(Eigen::Vector3d(0.0, 0.0, 1.0));
    for (int i = 0;i < pointCount;i++) {
        for (int k = 0;k < 3;k++)
            normal[i][k] = pcd.normals_[i][k];
    }

    // 3. ����KDTree������10�����
    geometry::KDTreeFlann kdtree(pcd);
    std::vector<int> indices(K,-1);
    std::vector<double> distances(K,-1);

    for (int i = 0;i < pointCount;i++) {
        kdtree.SearchKNN(pcd.points_[i], K, indices, distances);
        neigh[i] = indices;
    }


    open3d::io::WritePointCloud("C:\\Users\\31046\\Desktop\\city3D\\output.ply", pcd);



}



class seg_plane {


public:



public:
    // ʹ�ó�ʼ���б�ֱ�Ӱ�����
    seg_plane(PCCPointSet3& pointCloud,std::vector<Vec3<double>>& normal,std::vector<std::vector<int>>& neigh,int num_neigh)
        : Cloud(pointCloud),    // ֱ�Ӱ󶨵��ⲿ����
        normal(normal),       // ������
        neigh(neigh),         // ������
        K(num_neigh) {
        Cloud.planeIdx.resize(Cloud.getPointCount(),-1);
    } 

    std::vector<plane> get_planes();

    bool Broad(int Idx, int depth);

    void set_plane_color(vector<plane> &planes);

private:
    PCCPointSet3& Cloud;  // ���ó�Ա��ֱ�Ӱ󶨵��ⲿ����
    std::vector<Vec3<double>>& normal;
    std::vector<std::vector<int>>& neigh;
    int K;
    int th_thickness = 300;   //�����ֵ������ָ�������Ϊ����ƽ����
    int th_pointCount = 400;
    int cur_planeId=1;
    Vec3<double> cur_normal;
    Vec3<int> cur_center;
    plane cur_plane;
};