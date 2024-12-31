#pragma ACCEL KERNEL

void sub_atten(float data[512][768],float w1[768][64],float w2[768][64],float Q[512][64],float K[512][64],float atten[512][512])
{
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (int i = 0; i < 512; ++i) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L3}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L3}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
    for (int j = 0; j < 64; ++j) {
      
#pragma ACCEL PARALLEL reduction=Q FACTOR=auto{__PARA__L6} 
      for (int k = 0; k < 768; ++k) {
        Q[i][j] += data[i][k] * w1[k][j];
      }
    }
  }
  
#pragma ACCEL PIPELINE auto{__PIPE__L1}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
  for (int i = 0; i < 512; ++i) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L4}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L4}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L4}
    for (int j = 0; j < 64; ++j) {
      
#pragma ACCEL PARALLEL reduction=K FACTOR=auto{__PARA__L7} 
      for (int k = 0; k < 768; ++k) {
        K[i][j] += data[i][k] * w2[k][j];
      }
    }
  }
  
#pragma ACCEL PIPELINE auto{__PIPE__L2}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
  for (int i = 0; i < 512; ++i) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L5}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L5}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L5}
    for (int j = 0; j < 512; ++j) {
      
#pragma ACCEL PARALLEL reduction=atten FACTOR=auto{__PARA__L8} 
      for (int k = 0; k < 64; ++k) {
        atten[i][j] += Q[i][k] * K[j][k];
      }
    }
  }
}
