
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
    int id;       //id需要大于0
    Vec3<double> normal;
    Vec3<int> center;
    std::vector<int>  pointIdx;
};


void extracted_contour(string read_path, string save_path, string flip);

param analyse_path(char* argv[]);


vector<string> Split(const string& s, const string& seperator);

// 平面结构体：存储平面信息
struct DetectedPlane {
    std::vector<size_t> indices;  // 属于该平面的点索引
    Eigen::Vector4d equation;     // 平面方程 ax+by+cz+d=0
    Eigen::Vector3d normal;       // 平面法向量
    double d;                     // 平面常数项
};

template <int K>
void get_Normal_and_K_neighbor(PCCPointSet3& pointCloud, std::vector<Vec3<double>>& normal, std::vector<vector<int>>& neigh) {
    // 1. 读取点云
    geometry::PointCloud pcd;
    int pointCount = pointCloud.getPointCount();
    normal.resize(pointCount);
    neigh.resize(pointCount);

    pcd.points_.resize(pointCount);
    for (int i = 0;i < pointCount;i++) {
        for (int k = 0;k < 3;k++)
            pcd.points_[i][k] = pointCloud[i][k];
    }

    // 2. 计算法向量 (使用KDTree搜索，搜索半径0.1，最多考虑30个邻居)
    pcd.EstimateNormals(geometry::KDTreeSearchParamHybrid(100, 50));
    pcd.OrientNormalsToAlignWithDirection(Eigen::Vector3d(0.0, 0.0, 1.0));
    for (int i = 0;i < pointCount;i++) {
        for (int k = 0;k < 3;k++)
            normal[i][k] = pcd.normals_[i][k];
    }

    // 3. 构建KDTree并查找10最近邻
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
    // 使用初始化列表直接绑定引用
    seg_plane(PCCPointSet3& pointCloud,std::vector<Vec3<double>>& normal,std::vector<std::vector<int>>& neigh,int num_neigh)
        : Cloud(pointCloud),    // 直接绑定到外部对象
        normal(normal),       // 绑定引用
        neigh(neigh),         // 绑定引用
        K(num_neigh) {
        Cloud.planeIdx.resize(Cloud.getPointCount(),-1);
    } 

    std::vector<plane> get_planes();

    bool Broad(int Idx, int depth);

    void set_plane_color(vector<plane> &planes);

private:
    PCCPointSet3& Cloud;  // 引用成员，直接绑定到外部对象
    std::vector<Vec3<double>>& normal;
    std::vector<std::vector<int>>& neigh;
    int K;
    int th_thickness = 300;   //厚度阈值，超过指定厚度认为不在平面上
    int th_pointCount = 400;
    int cur_planeId=1;
    Vec3<double> cur_normal;
    Vec3<int> cur_center;
    plane cur_plane;
};