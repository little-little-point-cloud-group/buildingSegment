
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

// ƽ��ṹ�壺�洢ƽ����Ϣ
struct DetectedPlane {
    std::vector<size_t> indices;  // ���ڸ�ƽ��ĵ�����
    Eigen::Vector4d equation;     // ƽ�淽�� ax+by+cz+d=0
    Eigen::Vector3d normal;       // ƽ�淨����
    double d;                     // ƽ�泣����
};

