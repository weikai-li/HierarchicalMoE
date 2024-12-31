#include "merlin_type_define.h"
#pragma ACCEL kernel

void bbgemm(double m1[1048576],double m2[1048576],double prod[1048576])
{
  int i;
  int k;
  int j;
  int jj;
  int kk;
  int i_row;
  int k_row;
  double temp_x;
  double mul;
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  loopjj:
// Standardize from: for(jj = 0;jj < 1024;jj += 64) {...}
  for (jj = 0; jj <= 15; jj++) {
    int _in_jj = 0 + 64L * jj;
    
#pragma ACCEL PIPELINE auto{__PIPE__L1}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
    loopkk:
// Standardize from: for(kk = 0;kk < 1024;kk += 64) {...}
    for (kk = 0; kk <= 15; kk++) {
      int _in_kk = 0 + 64L * kk;
      
#pragma ACCEL PIPELINE auto{__PIPE__L2}
      
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
      loopi:
      for (i = 0; i < 1024; ++i) {
        
#pragma ACCEL PIPELINE auto{__PIPE__L3}
        
#pragma ACCEL TILE FACTOR=auto{__TILE__L3}
        
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
        loopk:
        for (k = 0; k < 64; ++k) {
          i_row = i * 1024;
          k_row = (k + _in_kk) * 1024;
          temp_x = m1[i_row + k + _in_kk];
          
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L4}
          loopj:
          for (j = 0; j < 64; ++j) {
            mul = temp_x * m2[k_row + j + _in_jj];
            prod[i_row + j + _in_jj] += mul;
          }
        }
      }
    }
    kk = 960 + 64L;
  }
  jj = 960 + 64L;
}
