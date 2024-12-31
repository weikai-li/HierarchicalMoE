
typedef struct {unsigned char x0;unsigned char y0;unsigned char z0;unsigned char x1;unsigned char y1;unsigned char z1;unsigned char x2;unsigned char y2;unsigned char z2;}Triangle_3D;
typedef struct {unsigned char x0;unsigned char y0;unsigned char x1;unsigned char y1;unsigned char x2;unsigned char y2;unsigned char z;}Triangle_2D;
typedef struct {unsigned char x;unsigned char y;unsigned char z;unsigned char color;}CandidatePixel;
typedef struct {unsigned char x;unsigned char y;unsigned char color;}Pixel;


static void projection_1(Triangle_3D triangle_3d,Triangle_2D *triangle_2d,unsigned char angle)
{
// Setting camera to (0,0,-1), the canvas at z=0 plane
// The 3D model lies in z>0 space
// The coordinate on canvas is proportional to the corresponding coordinate 
// on space
  if (angle==0) {
    triangle_2d -> x0 = triangle_3d . x0;
    triangle_2d -> y0 = triangle_3d . y0;
    triangle_2d -> x1 = triangle_3d . x1;
    triangle_2d -> y1 = triangle_3d . y1;
    triangle_2d -> x2 = triangle_3d . x2;
    triangle_2d -> y2 = triangle_3d . y2;
    triangle_2d -> z = (triangle_3d . z0/3+triangle_3d . z1/3)+triangle_3d . z2/3;
  }
   else {
    if (angle==1) {
      triangle_2d -> x0 = triangle_3d . x0;
      triangle_2d -> y0 = triangle_3d . z0;
      triangle_2d -> x1 = triangle_3d . x1;
      triangle_2d -> y1 = triangle_3d . z1;
      triangle_2d -> x2 = triangle_3d . x2;
      triangle_2d -> y2 = triangle_3d . z2;
      triangle_2d -> z = (triangle_3d . y0/3+triangle_3d . y1/3)+triangle_3d . y2/3;
    }
     else {
      if (angle==2) {
        triangle_2d -> x0 = triangle_3d . z0;
        triangle_2d -> y0 = triangle_3d . y0;
        triangle_2d -> x1 = triangle_3d . z1;
        triangle_2d -> y1 = triangle_3d . y1;
        triangle_2d -> x2 = triangle_3d . z2;
        triangle_2d -> y2 = triangle_3d . y2;
        triangle_2d -> z = (triangle_3d . x0/3+triangle_3d . x1/3)+triangle_3d . x2/3;
      }
    }
  }
}

static int check_clockwise_1(Triangle_2D triangle_2d)
{
  int cw;
  cw = ((int )(((triangle_2d . x2-triangle_2d . x0)*(triangle_2d . y1-triangle_2d . y0)-(triangle_2d . y2-triangle_2d . y0)*(triangle_2d . x1-triangle_2d . x0))));
  return cw;
}

static void clockwise_vertices_1(Triangle_2D *triangle_2d)
{
  unsigned char tmp_x;
  unsigned char tmp_y;
  tmp_x = (triangle_2d -> x0);
  tmp_y = (triangle_2d -> y0);
  triangle_2d -> x0 = (triangle_2d -> x1);
  triangle_2d -> y0 = (triangle_2d -> y1);
  triangle_2d -> x1 = tmp_x;
  triangle_2d -> y1 = tmp_y;
}

static unsigned char find_min_1(unsigned char in0,unsigned char in1,unsigned char in2)
{
  if (in0<in1) {
    if (in0<in2) {
      return in0;
    }
     else {
      return in2;
    }
  }
   else {
    if (in1<in2) {
      return in1;
    }
     else {
      return in2;
    }
  }
}

static unsigned char find_max_1(unsigned char in0,unsigned char in1,unsigned char in2)
{
  if (in0>in1) {
    if (in0>in2) {
      return in0;
    }
     else {
      return in2;
    }
  }
   else {
    if (in1>in2) {
      return in1;
    }
     else {
      return in2;
    }
  }
}

static unsigned char rasterization1_1(Triangle_2D triangle_2d,unsigned char max_min[],Triangle_2D *triangle_2d_same,unsigned int max_index[])
{
// clockwise the vertices of input 2d triangle
  if (check_clockwise_1(triangle_2d) == 0) {
    return (1);
  }
  if (check_clockwise_1(triangle_2d) < 0) {
    clockwise_vertices_1(&triangle_2d);
  }
// copy the same 2D triangle
  triangle_2d_same -> x0 = triangle_2d . x0;
  triangle_2d_same -> y0 = triangle_2d . y0;
  triangle_2d_same -> x1 = triangle_2d . x1;
  triangle_2d_same -> y1 = triangle_2d . y1;
  triangle_2d_same -> x2 = triangle_2d . x2;
  triangle_2d_same -> y2 = triangle_2d . y2;
  triangle_2d_same -> z = triangle_2d . z;
// find the rectangle bounds of 2D triangles
  max_min[0] = find_min_1(triangle_2d . x0,triangle_2d . x1,triangle_2d . x2);
  max_min[1] = find_max_1(triangle_2d . x0,triangle_2d . x1,triangle_2d . x2);
  max_min[2] = find_min_1(triangle_2d . y0,triangle_2d . y1,triangle_2d . y2);
  max_min[3] = find_max_1(triangle_2d . y0,triangle_2d . y1,triangle_2d . y2);
  max_min[4] = max_min[1]-max_min[0];
// calculate index for searching pixels
  max_index[0] = (max_min[1]-max_min[0])*(max_min[3]-max_min[2]);
  return (0);
}

static unsigned char pixel_in_triangle_1(unsigned char x,unsigned char y,Triangle_2D triangle_2d)
{
  int pi0;
  int pi1;
  int pi2;
  pi0 = ((int )(((x-triangle_2d . x0)*(triangle_2d . y1-triangle_2d . y0)-(y-triangle_2d . y0)*(triangle_2d . x1-triangle_2d . x0))));
  pi1 = ((int )(((x-triangle_2d . x1)*(triangle_2d . y2-triangle_2d . y1)-(y-triangle_2d . y1)*(triangle_2d . x2-triangle_2d . x1))));
  pi2 = ((int )(((x-triangle_2d . x2)*(triangle_2d . y0-triangle_2d . y2)-(y-triangle_2d . y2)*(triangle_2d . x0-triangle_2d . x2))));
  return (pi0 >= 0 && pi1 >= 0 && pi2 >= 0);
}

static unsigned int rasterization2_1(unsigned char flag,unsigned char max_min[],unsigned int max_index[],Triangle_2D triangle_2d_same,CandidatePixel fragment2[])
{
// clockwise the vertices of input 2d triangle
  if ((bool )(flag)) {
    return (0);
  }
  unsigned char color = (100);
  unsigned int i = (0);
  
// #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
  RAST2:
  for (unsigned int k(0); k<max_index[0]; k ++ ) {
// assuming the bounding box is less than 32*32
    
#pragma HLS loop_tripcount max=1024
    unsigned char x = (max_min[0]+k%max_min[4]);
    unsigned char y = (max_min[2]+k/max_min[4]);
    if ((bool )(pixel_in_triangle_1(x,y,triangle_2d_same))) {
      fragment2[(long )(i)] . x = x;
      fragment2[(long )(i)] . y = y;
      fragment2[(long )(i)] . z = triangle_2d_same . z;
      fragment2[(long )(i)] . color = color;
      i ++ ;
    }
  }
  return i;
}

static unsigned int zculling_1(unsigned int counter,CandidatePixel fragments[],unsigned int size,Pixel pixels[])
{
// initilize the z-buffer in rendering first triangle for an image
  static unsigned char z_buffer[256][256];
  if (counter==0) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L3}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L3}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
    ZCULLING_INIT_ROW:
    for (unsigned int i(0); i<256; i ++ ) {
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L8}
      ZCULLING_INIT_COL:
      for (unsigned int j(0); j<256; j ++ ) {
        z_buffer[(long )(i)][(long )(j)] = 255;
      }
    }
  }
// pixel counter
  unsigned int pixel_cntr = (0);
// update z-buffer and pixels
  
// #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L4}
  ZCULLING:
  for (unsigned int n(0); n<size; n ++ ) {
// dzj: has to be less than 500, according to the size of fragment
    
#pragma HLS loop_tripcount max=500
    if (fragments[(long )(n)] . z<z_buffer[(long )(fragments[(long )(n)] . y)][(long )(fragments[(long )(n)] . x)]) {
      pixels[(long )(pixel_cntr)] . x = fragments[(long )(n)] . x;
      pixels[(long )(pixel_cntr)] . y = fragments[(long )(n)] . y;
      pixels[(long )(pixel_cntr)] . color = fragments[(long )(n)] . color;
      pixel_cntr ++ ;
      z_buffer[(long )(fragments[(long )(n)] . y)][(long )(fragments[(long )(n)] . x)] = fragments[(long )(n)] . z;
    }
  }
  return pixel_cntr;
}

static void coloringFB_1(unsigned int counter,unsigned int size_pixels,Pixel pixels[],unsigned char frame_buffer[256][256])
{
  if (counter==0) {
// initilize the framebuffer for a new image
    
#pragma ACCEL PIPELINE auto{__PIPE__L5}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L5}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L5}
    COLORING_FB_INIT_ROW:
    for (unsigned int i(0); i<256; i ++ ) {
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L9}
      COLORING_FB_INIT_COL:
      for (unsigned int j(0); j<256; j ++ ) {
        frame_buffer[(long )(i)][(long )(j)] = 0;
      }
    }
  }
// update the framebuffer
  
// #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L6}
  COLORING_FB:
  for (unsigned int i(0); i<size_pixels; i ++ ) {
// dzj: has to be less than 500, according to the size of fragment
    
#pragma HLS loop_tripcount max=500
    frame_buffer[(long )(pixels[(long )(i)] . x)][(long )(pixels[(long )(i)] . y)] = pixels[(long )(i)] . color;
  }
}

static void output_FB_1(unsigned char frame_buffer[256][256],unsigned int output[16384])
{
  unsigned int out_FB = (0);
  
#pragma ACCEL PIPELINE auto{__PIPE__L1}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
  OUTPUT_FB_ROW:
  for (unsigned int i(0); i<256; i ++ ) {
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L7}
    OUTPUT_FB_COL:
    for (unsigned int j(0); j<256; j = j+4) {
      out_FB |= (frame_buffer[(long )(i)][(long )((j+0))]);
      out_FB |= (frame_buffer[(long )(i)][(long )((j+1))] << 8);
      out_FB |= (frame_buffer[(long )(i)][(long )((j+2))] << 16);
      out_FB |= (frame_buffer[(long )(i)][(long )((j+3))] << 24);
      output[((i*256)/4+j/4)] = out_FB;
    }
  }
}

#pragma ACCEL kernel

void rendering(unsigned int input[9576],unsigned int output[16384])
{
// local variables
  Triangle_3D triangle_3ds;
  ;
  Triangle_2D triangle_2ds;
  ;
  Triangle_2D triangle_2ds_same;
  ;
  unsigned int size_fragment;
  ;
  CandidatePixel fragment[500];
  unsigned int size_pixels;
  ;
  Pixel pixels[500];
  unsigned char frame_buffer[256][256];
  unsigned char angle = (0);
  unsigned char max_min[5];
  unsigned int max_index[1];
  unsigned char flag;
// processing 3192 3D triangles
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  TRIANGLES:
  for (unsigned int i(0); i<3192; i ++ ) {
    unsigned int input_lo = input[(long )((3*i))];
    unsigned int input_mi = input[(long )((3*i+1))];
    unsigned int input_hi = input[(long )((3*i+2))];
    triangle_3ds . x0 = (input_lo&255);
    triangle_3ds . y0 = (input_lo >> 8) & 255;
    triangle_3ds . z0 = (input_lo >> 16) & 255;
    triangle_3ds . x1 = (input_lo >> 24) & 255;
    triangle_3ds . y1 = input_mi;
    triangle_3ds . z1 = (input_mi >> 8) & 255;
    triangle_3ds . x2 = (input_mi >> 16) & 255;
    triangle_3ds . y2 = (input_mi >> 24) & 255;
    triangle_3ds . z2 = input_hi&255;
// five stages for processing each 3D triangle
    projection_1(triangle_3ds,&triangle_2ds,angle);
    flag = rasterization1_1(triangle_2ds,max_min,&triangle_2ds_same,max_index);
    size_fragment = rasterization2_1(flag,max_min,max_index,triangle_2ds_same,fragment);
    size_pixels = zculling_1(i,fragment,size_fragment,pixels);
    coloringFB_1(i,size_pixels,pixels,frame_buffer);
  }
// output values: frame buffer
  output_FB_1(frame_buffer,output);
}
