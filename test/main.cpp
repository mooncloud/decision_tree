#include <decision_tree.h>

int main(int argc, char**args) {
  
  if ( argc < 2 ) {
    return -1;
  }
  
  int m_ret = train_model(args[1], 1);
  if ( 0 != m_ret ) {
    fprintf(stderr, "[%s:%d]:train failed\n", __FILE__, __LINE__);
    return -1;
  }

  HANDLE model = load_model(args[2]);
  if ( NULL == model ) {
    fprintf(stderr, "[%s:%d]load model failed\n", __FILE__, __LINE__);
    return -1;
  }
  
  test_model(model, args[1]);
  // print_out_model(model);
  return 0;
}
