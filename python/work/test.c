#include <stdlib.h>

#define DEFAULT_ITS 100000000

int main(int argc, char** argv){

  long long  its = (argc > 1) ? atoll(argv[1]) : DEFAULT_ITS;

  int x=rand();
  for(long long i=0; i<its; i++){
    x += x * 3 % x;
  }

  return x;
}