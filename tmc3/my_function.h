
#include<string>
#include<Vector>
#include <open3d/Open3D.h>
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

