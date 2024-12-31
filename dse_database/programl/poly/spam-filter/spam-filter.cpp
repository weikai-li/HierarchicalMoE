#include "math.h"


static void read_data_1(float data[4608000],float training_instance[1024],int training_id)
{
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L5}
  READ_TRAINING_DATA:
  for (int i = 0; i < 1024; i++) {
    training_instance[i] = data[training_id * 1024 + i];
  }
}

static float dotProduct_1(float param[1024],float feature[1024])
{
  float result = (0);
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L6}
  DOT:
  for (int i = 0; i < 1024; i++) {
    float term = (param[i]*feature[i]);
    result+=term;
  }
  return result;
}

static float Sigmoid_1(float exponent)
{
  if (exponent>4) {
    return (1.0);
  }
   else {
    if (exponent<- 4) {
      return (0.0);
    }
     else {
      return 1.0 / (1.0 + expf(-exponent));
    }
  }
}

static void computeGradient_1(float grad[1024],float feature[1024],float scale)
{
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L7}
  GRAD:
  for (int i = 0; i < 1024; i++) {
    grad[i] = scale*feature[i];
  }
}

static void updateParameter_1(float param[1024],float grad[1024],float scale)
{
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L8}
  UPDATE:
  for (int i = 0; i < 1024; i++) {
    param[i]+=scale*grad[i];
  }
}


static void compute_1(float theta_local[1024],unsigned char label_local[4500],float training_instance[1024],int training_id)
{
// array for storing gradient
  static float gradient[1024];
// step size
  float step = (60000);
// do dot product with the parameter vector
  float dot = dotProduct_1(theta_local,training_instance);
// do sigmoid function
  float prob = Sigmoid_1(dot);
// get the label
  unsigned char training_label = label_local[training_id];
// compute gradient
  computeGradient_1(gradient,training_instance,prob-training_label);
// update the param vector
  updateParameter_1(theta_local,gradient, - step);
}

#pragma ACCEL KERNEL

void SgdLR(float data[4608000],unsigned char label[4500],float theta[1024])
{
// local version of the parameters
  static float theta_local[1024];
// local buffer of labels
  static unsigned char label_local[4500];
// array for storing one training instance
  static float training_instance[1024];
// initialize local parameters
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  PARAM_INIT:
  for (int i = 0; i < 1024; i++) {
    theta_local[i] = theta[i];
  }
// copy in labels
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
  LABEL_CP:
  for (int i = 0; i < 4500; i++) {
    label_local[i] = label[i];
  }
// main loop
// runs for multiple epochs
  
#pragma ACCEL PIPELINE auto{__PIPE__L2}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
  EPOCH:
  for (int epoch = 0; epoch < 5; epoch++) {
// in each epoch, go through each training instance in sequence
    
#pragma ACCEL PIPELINE auto{__PIPE__L4}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L4}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L4}
    TRAINING_INST:
    for (int training_id = 0; training_id < 4500; training_id++) {
// first reads in the training instance
      read_data_1(data,training_instance,training_id);
// compute part
      compute_1(theta_local,label_local,training_instance,training_id);
    }
  }
// stream out the result
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
  STREAM_OUT:
  for (int i = 0; i < 1024; i++) {
    theta[i] = theta_local[i];
  }
}
