#include <stdio.h>
#include <vector>
#include <list>
#include <map>
#include <cmath>
#include <string>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <deque>
#include <algorithm>
#include <decision_tree.h>

#ifndef DEBUG
#define DEBUG
#endif


typedef  std::vector<int   >       VecInt;
typedef  std::vector<double>       VecDouble;

struct feature_t {
  int     feature_id;
  double  feature_val;
};
struct instance_t {
  double    label;
  feature_t features[1024];
  int       feature_num;
};
struct DTNode{
  bool    is_leaf;
  int     child_num;
  int     feature_id;
  int     parent_feature_id;
  double  parent_feature_val;
  double  label;
  double  default_label;
  DTNode* p_child;
};

double g_info_gain_ratio_thr = 0.1;
double g_info_gain_thr       = 0.1;
int g_train_type             = MAX_INFO_GAIN;
FILE*  g_fp_model            = NULL;

int parse_input(const char*input_file, std::vector<instance_t>&instance_set, VecInt&feature_id_set);

bool cmp(const DTNode&p1, const DTNode&p2) {
  return p1.parent_feature_val < p2.parent_feature_val;
}

// 计算信息熵
int compute_entropy(std::vector<instance_t*>&instance_set, double&ret) {
  std::map<double, int  >entropy;
  double m_empirical_entropy = 0;
  int m_total_ins_num        = 0;
  int instance_num           = instance_set.size();
  instance_t*ptr             = NULL;
  for ( int index = 0; index < instance_num; ++index ) {
    ptr = instance_set[index];
    if ( entropy.end() == entropy.find(ptr->label)) {
      entropy[ptr->label] = 0;
    }
    entropy[ptr->label] ++;
    m_total_ins_num ++;
  }
  for ( std::map<double, int>::iterator it = entropy.begin(); it != entropy.end(); ++it ) {
    double m_sub_num = it->second;
    double p =  m_sub_num/m_total_ins_num;
    m_empirical_entropy += p*log(p)/log(2.0);
  }
  m_empirical_entropy *= -1;
  ret = m_empirical_entropy;
  return 0;
}

// 计算信息增益
int compute_information_gain(std::vector<instance_t*>&instance_set, int feature_id, double& info_gain) {
  if ( instance_set.empty() ) {
    return 0;
  }
  
  // 计算instance_set的经验熵
  double m_empirical_entropy = 0;
  compute_entropy(instance_set, m_empirical_entropy);
  
  // 计算条件熵
  std::map<double, std::vector<instance_t*> >sub_instance_set;
  feature_t* p_feature   = NULL;
  instance_t*p_instance  = NULL;
  double feature_val     = 0;
  int instance_num = instance_set.size();
  for ( int index = 0; index < instance_num; ++index ) {
    p_instance = instance_set[index];
    p_feature  = p_instance->features;
    feature_val= p_feature[feature_id].feature_val;
    if ( sub_instance_set.end() == sub_instance_set.find(feature_val) ) {
      sub_instance_set[feature_val] = std::vector<instance_t*>();
    }
    sub_instance_set[feature_val].push_back(p_instance);
  }
  double cur_entropy = 1.0;
  double condition_entropy = 0;
  double total_ins_num = instance_set.size();
  for ( std::map<double, std::vector<instance_t*> >:: iterator it = sub_instance_set.begin(); it != sub_instance_set.end(); ++it ) {
    compute_entropy(it->second, cur_entropy);
    condition_entropy += it->second.size()/total_ins_num*cur_entropy;
  }
 
  switch ( g_train_type ) {
    case MAX_INFO_GAIN:{
        // 熵增益
        info_gain = m_empirical_entropy-condition_entropy;
        break;
      }
    case MAX_INFO_GAIN_RATIO:{
        // 熵增益比
        info_gain = 1.0 - condition_entropy/m_empirical_entropy;                     
        break;
      }
    default:
      return -1;
  }
  
  return 0;
}

// 选择熵增益最大的feature_id
int select_max_info_gain(std::vector<instance_t*>&instance_set, std::vector<int>&feature_id_set, int &max_info_gain_fid, double&max_info_gain) {
  if ( feature_id_set.size() <= 0 || instance_set.size() <= 0 ) {
    return -1;
  } 
  max_info_gain_fid = -1;
  max_info_gain = 0;
  int feature_id_num = feature_id_set.size();
  double info_gain = 0;
  for ( int i = 0; i < feature_id_num; ++i ) {
    compute_information_gain(instance_set, feature_id_set[i], info_gain);
#ifdef DEBUG
    printf("++[%s:%d][%s]:%f\t%d\n", __FILE__, __LINE__, __FUNCTION__, info_gain, feature_id_set[i]);
#endif
    max_info_gain_fid  = max_info_gain > info_gain ? max_info_gain_fid : feature_id_set[i];
    max_info_gain = max_info_gain > info_gain ? max_info_gain : info_gain;
  }
  return 0;
}


// 是否是叶子
bool is_leaf(std::vector<instance_t*>&instance_set, std::vector<int>&feature_id_set) {
  if ( feature_id_set.size() <= 0 ) {
    return 0;
  }
  double label     = -1;
  int instance_num = instance_set.size();
  for ( int index = 0; index < instance_num; ++index ) {
    if ( -1 != label && instance_set[index]->label != label ) {
      return false;
    }
    label = instance_set[index]->label;
  }
  return true;
}

// 是否小于给定的熵增益阈值
bool  is_less_than_thr(std::vector<instance_t*>&instance_set, DTNode*p_root, double info_gain) {
  double label_t   = -1;
  int max_num      = 0;
  int instance_num = instance_set.size();
  std::map<double,  int>sub_instance_set;
  for ( int index = 0; index < instance_num; ++index ) {
    double label = instance_set[index]->label;
    sub_instance_set[label] += 1;
    label_t = max_num > sub_instance_set[label]?label_t:label;
    max_num = max_num > sub_instance_set[label]?max_num:sub_instance_set[label];
  }
  p_root->default_label = label_t;
  switch ( g_train_type ) {
    case MAX_INFO_GAIN:{
        if ( info_gain <= g_info_gain_thr ) {
          p_root->label = label_t;
          return true;
        }
        break;
      }
    case MAX_INFO_GAIN_RATIO:{
        // 熵增益比
        if ( info_gain <= g_info_gain_ratio_thr ) {
          p_root->label = label_t;
          return true;
        }
        break;
      }
    default:
      return -1;
  }
  return false;
}

// 实际的训练入口
int train_run(std::vector<instance_t*>&instance_set, std::vector<int>&feature_id_set, DTNode*p_root) {

  double max_info_gain  = 0;
  int max_info_gain_fid = 0;

  // 如果满足叶子的要求
  if ( is_leaf(instance_set, feature_id_set) ) {
    p_root->is_leaf       = true;
    p_root->feature_id    = -1;
    p_root->label         = instance_set[0]->label;
    p_root->default_label = instance_set[0]->label;
    return 0;
  }
  
  // 计算最大增益
  select_max_info_gain(instance_set, feature_id_set, max_info_gain_fid, max_info_gain);   
  #ifdef DEBUG
  printf("++[%s:%d][%s]fid:%d\tinfo_gain:%f\n", __FILE__, __LINE__, __FUNCTION__, max_info_gain_fid, max_info_gain);
  #endif
  // 如果max_info_gain小于给定的阈值，则当前节点设置为树的叶子节点
  if ( is_less_than_thr(instance_set, p_root, max_info_gain) ) {
    p_root->is_leaf = true;
    p_root->feature_id = -1;
    return 0;
  }

  // 通过max_info_gain_fid对应的feature_val，把instance_set进行划分
  std::map<double, std::vector<instance_t*> >split_instance_set;
  feature_t* p_feature  = NULL;
  instance_t*p_instance = NULL;
  double feature_val   = 0;
  int instance_num  = instance_set.size();
  int child_num = 0; 
  for ( int i = 0; i < instance_num; ++i ) {
    p_instance = instance_set[i];
    p_feature  = p_instance->features;
    feature_val = p_feature[max_info_gain_fid].feature_val;
    if ( split_instance_set.end() == split_instance_set.find(feature_val) ) {
      split_instance_set[feature_val] = std::vector<instance_t*>();
      child_num ++;
    }
    split_instance_set[feature_val].push_back(p_instance);
  }
  p_root->p_child = new DTNode[child_num];
  memset(p_root->p_child, 0, sizeof(DTNode)*child_num);
  p_root->child_num = child_num;
  p_root->feature_id = max_info_gain_fid;
  std::vector<int>sub_feature_id_set;
  for ( size_t i = 0; i < feature_id_set.size(); ++i ) {
    if ( feature_id_set[i] != max_info_gain_fid ) {
      sub_feature_id_set.push_back(feature_id_set[i]);
    }
  }

  // 对划分之后的每一个子树，进行递归训练，是一个dfs的训练
  int index = 0;
  for ( std::map<double, std::vector<instance_t*> >:: iterator it = split_instance_set.begin(); it != split_instance_set.end(); ++it ) {
    (p_root->p_child+index)->parent_feature_val = it->first;
    (p_root->p_child+index)->parent_feature_id  = p_root->feature_id;
    // std::cout<<(p_root->p_child+index)->parent_feature_id<<std::endl;
    train_run(it->second, sub_feature_id_set, p_root->p_child+index);
    index++;
  }
  
  // 排序是为了后面预测的时候可以用二分查找，来找到对应的val
  std::sort(p_root->p_child, p_root->p_child+index, cmp);
  
  return 0;
}

// 训练的对外入口
int train_decision_tree(std::vector<instance_t*>&instance_set, std::vector<int>&feature_id_set, DTNode*&p_root) {
  p_root = new DTNode;
  if ( NULL == p_root ) {
    return -1;
  }
  memset(p_root, 0, sizeof(DTNode));
  return train_run(instance_set, feature_id_set, p_root);
}

// 预测接口
int predict(DTNode*p_model, instance_t&instance, double &pred) {
  int feature_num = instance.feature_num;
  feature_t*features = instance.features;
  while ( true ) {
    if ( p_model->is_leaf ) {
      pred = p_model->label;
      return 0;
    }
    int feature_id = p_model->feature_id;
    if ( feature_id >= feature_num ) {
      pred = -1;
      return -1;
    }
    double feature_val = features[feature_id].feature_val;
    DTNode*p_child     = p_model->p_child;
    int m_start        = 0;
    int m_end          = p_model->child_num;
    int m_mid          = m_start+(m_end-m_start)/2;
    while ( true ) {
      if ( m_start > m_end ) {
        break;
      }
      if ( (p_child+m_mid)->parent_feature_val == feature_val ) {
        p_model = p_child+m_mid;
        break;
      } else if ( (p_child+m_mid)->parent_feature_val > feature_val ) {
        m_end = m_mid-1;
      } else {
        m_start = m_end+1;
      }
      m_mid = m_start+(m_end-m_start)/2;
    }
    if ( m_start > m_end ) {
      pred = -1;
      return -1;
    }
  }
  return 0;
}






// bfs 遍历
int bfs_traverse(DTNode*p_root, void(*action)(DTNode*)) {
  if ( NULL == p_root ) {
    return 0;
  }
  DTNode*p_model = NULL;
  std::deque<DTNode*>dt;
  dt.push_back(p_root);
  p_model = p_root;
  while ( !dt.empty() ) {
    p_model = dt.front();
    dt.pop_front();
    action(p_model);
    for ( int i = 0; i < p_model->child_num; ++i ) {
      dt.push_back(p_model->p_child+i);
    }
  }
  return 0;

}


void print_instance(DTNode*p_instance) {
    printf("++feature_id:%d\tis_leaf:%d\tchild_num:%d\tparent_feature_id:%d\tparent_feature_val:%f\tdefault_label:%f\tlabel:%f\n", p_instance->feature_id, p_instance->is_leaf, p_instance->child_num,p_instance->parent_feature_id, p_instance->parent_feature_val, p_instance->default_label, p_instance->label);
    return;
}

void write_model(DTNode*p_model) {
  print_instance(p_model);
  fwrite(p_model, sizeof(DTNode), 1, g_fp_model);
}

int save_model(DTNode*p_model) {
  return bfs_traverse(p_model, write_model);
}

// bfs遍历模型
int print_out_model(DTNode*p_root, void (*action)(DTNode*) = print_instance){
  return bfs_traverse(p_root, print_instance);
}

// bfs加载模型
int bfs_load_model(const char*model_file, DTNode*&p_model) {
  FILE*fp = fopen(model_file, "rb");
  if ( NULL == fp ) {
    fprintf(stderr, "%d open [%s] failed\n", __FILE__, model_file);
    return -1;
  }
#ifdef DEBUG
  DTNode*pt = new DTNode[10];
  fread(pt, sizeof(DTNode), 10, fp);
  std::cout<<"+++++++++++++++++++++"<<std::endl;
  for ( int i = 0; i < 5; ++i ) {
    print_instance(pt+i);
  }
  std::cout<<"+++++++++++++++++++++"<<std::endl;
  fseek(fp, 0, SEEK_SET);
#endif
  std::deque<DTNode*>dt;
  p_model = new DTNode;
  fread(p_model, sizeof(DTNode), 1, fp);
  dt.push_back(p_model);
  int m_total_num = 1;
  int m_child_num = 0;
  while ( !dt.empty() ) {
    DTNode*ptr = dt.front();
#ifdef DEBUG
    print_instance(ptr);
#endif
    dt.pop_front();
    m_child_num = ptr->child_num;
    fseek(fp, m_total_num*sizeof(DTNode), SEEK_SET);
    ptr->p_child = new DTNode[m_child_num];
    fread(ptr->p_child, sizeof(DTNode), m_child_num, fp);
    for ( int i = 0; i < m_child_num; ++i ) {
      dt.push_back(ptr->p_child+i);
    }
    m_total_num += m_child_num;
  }
  fclose(fp);
  return 0;
}

int load_model(const char*model_file, DTNode*&p_model) {
  return bfs_load_model(model_file, p_model);
}



int split_str(const char*input_str, std::vector<std::string>&split_ret, const char*seq) {
  if ( NULL == input_str ) {
    return 0;
  }
  split_ret.clear();
  int seq_len = strlen(seq);
  int input_str_len = strlen(input_str);
  char*cpstr = new char[input_str_len+1];
  char*s = cpstr;
  memcpy(cpstr, input_str, input_str_len);
  cpstr[input_str_len] = 0;
  char* pos = NULL;
  while ( NULL != (pos = strstr(cpstr, seq)) ) {
    *pos = 0;
    if ( cpstr != pos ) {
      split_ret.push_back(cpstr);
    }
    cpstr = pos+seq_len;
  }
  if ( 0 != *cpstr ) {
    split_ret.push_back(cpstr);
  }
  delete[] s;  
  return 0;
}

int parse_input(const char*input_file, std::vector<instance_t>&instance_set, VecInt&feature_id_set) {
  if ( NULL == input_file ) {
    return -1;
  }
  instance_set.clear();
  FILE* fp = fopen(input_file, "r");
  char line[1024] = {0};
  char*       seq = "\t";
  std::vector<std::string>ret;
  while ( NULL != fgets(line, 1024, fp) ) {
    split_str(line, ret, seq);
    if ( ret.size() <= 0 ) {
      fprintf(stderr, "split_error:%s\n", line);
      continue;
    }
    instance_t instance;
    std::vector<std::string>key_val;
    memset(&instance, 0, sizeof(instance));
    instance.label = atof(ret[0].c_str());
    int check = -1;
    for ( size_t i = 1; i < ret.size(); ++i ) {
      split_str(ret[i].c_str(), key_val, ":");
      if ( key_val.size() < 2 ) {
        fprintf(stderr, "split_error:%d:%s\n", __FILE__, ret[i].c_str());
        continue;
      }
      instance.features[instance.feature_num].feature_id  = atoi(key_val[0].c_str());
      instance.features[instance.feature_num].feature_val = atof(key_val[1].c_str());
      if ( instance.features[instance.feature_num].feature_id - check != 1 ) {
        fprintf(stderr, "[%s:%d]:input_error:%s\n", __FILE__, __LINE__, line);
        exit(1);
      }
      check = instance.features[instance.feature_num].feature_id; 
      instance.feature_num ++;
    }
    instance_set.push_back(instance);
  } 
 
  feature_t* features= instance_set[0].features;
  int feature_num    = instance_set[0].feature_num;
  for ( int index = 0; index < feature_num; ++index ) {
    feature_id_set.push_back(features[index].feature_id);
  } 
  fclose(fp);
  return 0;
}

int print_instance_out(std::vector<instance_t*>&instances) {
  printf("++instance_out:******************************\n");
  for ( size_t i = 0; i < instances.size(); ++i ) {
    printf("++%f\t", instances[i]->label);
    feature_t*features = instances[i]->features;
    for ( int j = 0; j < instances[i]->feature_num; ++j ) {
      printf("%d:%f\t", features[j].feature_id, features[j].feature_val);
    }
    printf("\n");
  }
  printf("++*************************************\n\n");
  return 0;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// 以下是对外提供的封装的接口
// print model
void print_out_model(HANDLE model) {
  DTNode*p_model = (DTNode*)model;
  print_out_model(p_model);
}

// predict
int predict(HANDLE p_model, const VecInt &feature_ids,const VecDouble& feature_vals, double&pred) {
  if ( feature_ids.size() != feature_vals.size() ) {
    fprintf(stderr, "[%s:%d]:input error\n", __FILE__, __LINE__);
    return -1;
  }
  DTNode* model = (DTNode*)p_model;
  instance_t instance;
  int feature_num = feature_ids.size();
  for ( int index = 0; index < feature_num; ++index ) {
    instance.features[instance.feature_num].feature_id  = feature_ids[index];
    instance.features[instance.feature_num].feature_val = feature_vals[index];
    instance.feature_num++;
  }
  
  return predict(model, instance, pred);
  
}

// train model
// int train_model(const char*train_file, enum TAIN_TYPE type){
int train_model(const char*train_file, int type){
  g_train_type = type;
  g_fp_model   = fopen("model.d", "w");
  if ( NULL == g_fp_model ) {
    fprintf(stderr, "++[%s:%d]open model.d failed\n", __FILE__, __LINE__);
    return -1;
  }
  DTNode*p_root = NULL;
  std::vector<instance_t  >instance_set;
  std::vector<int         >feature_id_set;
  std::vector<instance_t* >ptr;

  // 读入数据，并判断数据是否合法
  int m_ret = parse_input(train_file, instance_set, feature_id_set);
  if ( 0 != m_ret ) {
    fclose(g_fp_model);
    return -1;
  }

  // ptr存入instance的地址
  for ( size_t i = 0; i < instance_set.size(); ++i ) {
    ptr.push_back(&(instance_set[i]));
  }

  // 生成决策树
  m_ret = train_decision_tree(ptr, feature_id_set, p_root);
  if ( 0 != m_ret ) {
    fprintf(stderr, "[%s:%d]:train failed\n", __FILE__, __LINE__);
    fclose(g_fp_model);
    return -1;
  }

  // 保存model
  m_ret = save_model(p_root);
  if ( 0 != m_ret ) {
    fprintf(stderr, "[%s:%d]:train failed\n", __FILE__, __LINE__);
  }
  
  fclose(g_fp_model);

  return 0;
}

// load model
HANDLE load_model(const char*model_file) {
  DTNode*p_model = NULL;
  int m_ret =load_model(model_file, p_model);
  if ( 0 != m_ret ) {
    return NULL;
  }
  return (HANDLE)(p_model);
}

// test model
int test_model(HANDLE model_handle, const char*test_file) {
  if ( NULL == model_handle ) {
    return -1;
  }
  
  std::vector<instance_t>instance_set;
  VecInt feature_id_set;
  int m_ret = parse_input(test_file, instance_set, feature_id_set);
  if ( 0 != m_ret ) {
    fprintf(stderr, "[%s:%d]parser input failed\n", __FILE__, __LINE__);
    return -1;
  }

  double pred     = -1;
  DTNode* p_model = (DTNode*)model_handle;
  int instance_num= instance_set.size();
  int m_total_num = instance_num;
  int right_num   = 0;
  for ( int index = 0; index < instance_num; ++index ) {
    m_ret = predict(p_model, instance_set[index], pred);
    if ( 0 != m_ret ) {
      fprintf(stderr, "[%s:%d]predict failed\n", __FILE__, __LINE__);
      continue;
    }
    if ( pred == instance_set[index].label ) {
      right_num++;
    }
  }

  fprintf(stdout, "++Accuracy:%f\n", (double)right_num/m_total_num);

  return 0;
}
