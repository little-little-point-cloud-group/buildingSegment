
#include<string>
#include<Vector>
#include <open3d/Open3D.h>
using namespace std;

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

