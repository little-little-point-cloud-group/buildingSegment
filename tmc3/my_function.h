
#include "TMC3.h"

#include <memory>

#include "PCCTMC3Encoder.h"
#include "PCCTMC3Decoder.h"
#include "constants.h"
#include "ply.h"
#include "pointset_processing.h"
#include "program_options_lite.h"
#include "io_tlv.h"
#include "version.h"
using namespace std;
using namespace pcc;

class param {
public:
  string dataType;
  string readPath;
  string savePath;
  string num_sizePath;
  int frame;
};

param analyse_path(char* argv[]);

vector<int> 
read_category_num(string file);

PCCPointSet3 read_nuscence_bin(string path, int pointcount);

PCCPointSet3 read_kitti_bin(string path, int pointcount);

void write_original_pointcloud(PCCPointSet3 cloud1, string path);

vector<string> Split(const string& s, const string& seperator);

vector<int> readlidar(string filepath);

void write_lidar(vector<int> category, string savepath);

vector<string> read_token(string file);

