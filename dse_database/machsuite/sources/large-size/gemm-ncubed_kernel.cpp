#include "merlin_type_define.h"
#pragma ACCEL kernel

void gemm(double m1[1048576],double m2[1048576],double prod[1048576])
{
  int i;
  int j;
  int k;
  int k_col;
  int i_col;
  double mult;
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  outer:
  for (i = 0; i < 1024; i++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L1}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
    middle:
    for (j = 0; j < 1024; j++) {
      i_col = i * 1024;
      double sum = (double )0;
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
      inner:
      for (k = 0; k < 1024; k++) {
        k_col = k * 1024;
        mult = m1[i_col + k] * m2[k_col + j];
        sum += mult;
      }
      prod[i_col + j] = sum;
    }
  }
}
