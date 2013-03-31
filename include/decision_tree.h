#include <stdio.h>
#include <vector>


typedef  void*                     HANDLE;
typedef  std::vector<int   >       VecInt;
typedef  std::vector<double>       VecDouble;

enum TAIN_TYPE{
  MAX_INFO_GAIN       = 0,
  MAX_INFO_GAIN_RATIO = 1
};

// if train suc return 0; else reutrn -1
int train_model(const char*train_file, int type = MAX_INFO_GAIN);

// the format of test_model must be eq to train_file
int test_model(HANDLE model_handle, const char*test_file);

// load binary model
HANDLE load_model(const char*model_file);

// if suc:retur 0;else return -1;
int predict(HANDLE p_model, const VecInt &feature_ids,const VecDouble& feature_vals, double&pred);

// print model to stdout
void print_out_model(HANDLE model);
