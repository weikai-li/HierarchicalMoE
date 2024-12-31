#pragma ACCEL KERNEL

void vmmv(float A[100][80],float a[100],float b[80],float B[80][80],float C[100][80],float x[80],float y[100])
{
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (int i = 0; i < 100; ++i) {
    
#pragma ACCEL PARALLEL reduction=A FACTOR=auto{__PARA__L3} 
    for (int j = 0; j < 80; ++j) {
      A[i][j] += a[i] * b[j];
    }
  }
  
#pragma ACCEL PIPELINE auto{__PIPE__L1}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
  for (int i = 0; i < 100; ++i) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L4}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L4}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L4}
    for (int j = 0; j < 80; ++j) {
      
#pragma ACCEL PARALLEL reduction=C FACTOR=auto{__PARA__L6} 
      for (int k = 0; k < 80; ++k) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  
#pragma ACCEL PIPELINE auto{__PIPE__L2}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
  for (int i = 0; i < 100; ++i) {
    
#pragma ACCEL PARALLEL reduction=y FACTOR=auto{__PARA__L5} 
    for (int j = 0; j < 80; ++j) {
      y[i] += C[i][j] * x[j];
    }
  }
}
