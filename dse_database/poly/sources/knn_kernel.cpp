const unsigned long long m1 = (unsigned long long )0x5555555555555555;
const unsigned long long m2 = (unsigned long long )0x3333333333333333;
const unsigned long long m4 = (unsigned long long )0x0f0f0f0f0f0f0f0f;

static int popcount_1(unsigned long long x)
{
//put count of each 2 bits into those 2 bits
  x -= x >> 1 & m1;
//put count of each 4 bits into those 4 bits 
  x = (x & m2) + (x >> 2 & m2);
//put count of each 8 bits into those 8 bits 
  x = x + (x >> 4) & m4;
//put count of each 16 bits into their lowest 8 bits
  x += x >> 8;
//put count of each 32 bits into their lowest 8 bits
  x += x >> 16;
//put count of each 64 bits into their lowest 8 bits
  x += x >> 32;
  return (int )(x & ((unsigned long long )0x7f));
}

static void update_knn_1(unsigned long long test_inst[4],unsigned long long train_inst[4],int min_distances[3])
{
  int dist = 0;
  for (int i = 0; i < 4; i++) {
    unsigned long long diff = test_inst[i] ^ train_inst[i];
    dist += popcount_1(diff);
  }
  int max_dist = 0;
  int max_dist_id = 3 + 1;
  int k = 0;
// Find the max distance
  FIND_MAX_DIST:
  for (int k = 0; k < 3; ++k) {
    if (min_distances[k] > max_dist) {
      max_dist = min_distances[k];
      max_dist_id = k;
    }
  }
// Replace the entry with the max distance
  if (dist < max_dist) {
    min_distances[max_dist_id] = dist;
  }
  return ;
}

static unsigned char knn_vote_1(int knn_set[120])
{
// local buffers
// final K nearest neighbors
  int min_distance_list[3];
// labels for the K nearest neighbors
  int label_list[3];
// voting boxes
  int vote_list[10];
  int pos = 1000;
// initialize
  INIT_1:
  for (int i = 0; i < 3; i++) {
    min_distance_list[i] = 256;
    label_list[i] = 9;
  }
  INIT_2:
  for (int i = 0; i < 10; i++) {
    vote_list[i] = 0;
  }
// go through all the lanes
// do an insertion sort to keep a sorted neighbor list
  
#pragma ACCEL PIPELINE auto{__PIPE__L10}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L10}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L10}
  LANES:
  for (int i = 0; i < 40; i++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L14}
    INSERTION_SORT_OUTER:
    for (int j = 0; j < 3; j++) {
      pos = 1000;
      INSERTION_SORT_INNER:
      for (int r = 0; r < 3; r++) {
        pos = (knn_set[i * 3 + j] < min_distance_list[r] && pos > 3?r : pos);
      }
      INSERT:
// Standardize from: for(int r = 3;r > 0;r--) {...}
      for (int r = 0; r <= 2; r++) {
        int _in_r = 3 + -1 * r;
        if (_in_r - 1 > pos) {
          min_distance_list[_in_r - 1] = min_distance_list[_in_r - 2];
          label_list[_in_r - 1] = label_list[_in_r - 2];
        }
         else {
          if (_in_r - 1 == pos) {
            min_distance_list[_in_r - 1] = knn_set[i * 3 + j];
            label_list[_in_r - 1] = i / (40 / 10);
          }
        }
      }
    }
  }
// vote
  INCREMENT:
  for (int i = 0; i < 3; i++) {
    vote_list[label_list[i]] += 1;
  }
  unsigned char max_vote;
  max_vote = ((unsigned char )0);
// find the maximum value
  VOTE:
  for (int i = 0; i < 10; i++) {
    if (vote_list[i] >= vote_list[max_vote]) {
      max_vote = ((unsigned char )i);
    }
  }
  return max_vote;
}
#pragma ACCEL kernel

void DigitRec(unsigned long long global_training_set[72000],unsigned long long global_test_set[8000],unsigned char global_results[2000])
{
// This array stores K minimum distances per training set
  int knn_set[120];
  static unsigned long long training_set[72000];
  static unsigned long long test_set[8000];
  static unsigned char results[2000];
// for the second time
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (int i = 0; i < 18000 * 4; i++) {
    training_set[i] = global_training_set[i];
  }
// copy the test set
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
  for (int i = 0; i < 2000 * 4; i++) {
    test_set[i] = global_test_set[i];
  }
// loop through test set
  
#pragma ACCEL PIPELINE auto{__PIPE__L2}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
  TEST_LOOP:
  for (int t = 0; t < 2000; ++t) {
// fetch one instance
    unsigned long long test_instance[4];
    for (int i = 0; i < 4; ++i) {
      test_instance[i] = test_set[t * 4 + i];
    }
// Initialize the knn set
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L5}
    SET_KNN_SET:
    for (int i = 0; i < 3 * 40; ++i) {
// Note that the max distance is 256
      knn_set[i] = 256;
    }
    
#pragma ACCEL PIPELINE auto{__PIPE__L6}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L6}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L6}
    TRAINING_LOOP:
    for (int i = 0; i < 18000 / 40; ++i) {
      
#pragma ACCEL PIPELINE auto{__PIPE__L7}
      
#pragma ACCEL TILE FACTOR=auto{__TILE__L7}
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L7}
      LANES:
      for (int j = 0; j < 40; j++) {
// Read a new instance from the training set
        unsigned long long training_instance[4];
        for (int k = 0; k < 4; ++k) {
          training_instance[k] = training_set[(j * 18000 / 40 + i) * 4 + k];
        }
// Update the KNN set
        update_knn_1(test_instance,training_instance,&knn_set[j * 3]);
      }
    }
// Compute the final output
    unsigned char max_vote = knn_vote_1(knn_set);
    results[t] = max_vote;
  }
// copy the results out
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
  for (int i = 0; i < 2000; i++) {
    global_results[i] = results[i];
  }
}
