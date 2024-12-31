; ModuleID = '3d-rendering.cpp'
source_filename = "3d-rendering.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.Triangle_3D = type { i8, i8, i8, i8, i8, i8, i8, i8, i8 }
%struct.Triangle_2D = type { i8, i8, i8, i8, i8, i8, i8 }
%struct.CandidatePixel = type { i8, i8, i8, i8 }
%struct.Pixel = type { i8, i8, i8 }

@_ZZL10zculling_1jP14CandidatePixeljP5PixelE8z_buffer = internal global [256 x [256 x i8]] zeroinitializer, align 16

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z9renderingPjS_(i32* %input, i32* %output) #0 {
entry:
  %input.addr = alloca i32*, align 8
  %output.addr = alloca i32*, align 8
  %triangle_3ds = alloca %struct.Triangle_3D, align 1
  %triangle_2ds = alloca %struct.Triangle_2D, align 1
  %triangle_2ds_same = alloca %struct.Triangle_2D, align 1
  %size_fragment = alloca i32, align 4
  %fragment = alloca [500 x %struct.CandidatePixel], align 16
  %size_pixels = alloca i32, align 4
  %pixels = alloca [500 x %struct.Pixel], align 16
  %frame_buffer = alloca [256 x [256 x i8]], align 16
  %angle = alloca i8, align 1
  %max_min = alloca [5 x i8], align 1
  %max_index = alloca [1 x i32], align 4
  %flag = alloca i8, align 1
  %i = alloca i32, align 4
  %input_lo = alloca i32, align 4
  %input_mi = alloca i32, align 4
  %input_hi = alloca i32, align 4
  %agg.tmp = alloca %struct.Triangle_3D, align 1
  %agg.tmp.coerce = alloca { i64, i8 }, align 1
  %agg.tmp29 = alloca %struct.Triangle_2D, align 1
  %tmp = alloca i56, align 8
  %agg.tmp33 = alloca %struct.Triangle_2D, align 1
  %tmp35 = alloca i56, align 8
  store i32* %input, i32** %input.addr, align 8
  store i32* %output, i32** %output.addr, align 8
  store i8 0, i8* %angle, align 1
  br label %TRIANGLES

TRIANGLES:                                        ; preds = %entry
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %TRIANGLES
  %0 = load i32, i32* %i, align 4
  %cmp = icmp ult i32 %0, 3192
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32*, i32** %input.addr, align 8
  %2 = load i32, i32* %i, align 4
  %mul = mul i32 3, %2
  %conv = zext i32 %mul to i64
  %arrayidx = getelementptr inbounds i32, i32* %1, i64 %conv
  %3 = load i32, i32* %arrayidx, align 4
  store i32 %3, i32* %input_lo, align 4
  %4 = load i32*, i32** %input.addr, align 8
  %5 = load i32, i32* %i, align 4
  %mul1 = mul i32 3, %5
  %add = add i32 %mul1, 1
  %conv2 = zext i32 %add to i64
  %arrayidx3 = getelementptr inbounds i32, i32* %4, i64 %conv2
  %6 = load i32, i32* %arrayidx3, align 4
  store i32 %6, i32* %input_mi, align 4
  %7 = load i32*, i32** %input.addr, align 8
  %8 = load i32, i32* %i, align 4
  %mul4 = mul i32 3, %8
  %add5 = add i32 %mul4, 2
  %conv6 = zext i32 %add5 to i64
  %arrayidx7 = getelementptr inbounds i32, i32* %7, i64 %conv6
  %9 = load i32, i32* %arrayidx7, align 4
  store i32 %9, i32* %input_hi, align 4
  %10 = load i32, i32* %input_lo, align 4
  %and = and i32 %10, 255
  %conv8 = trunc i32 %and to i8
  %x0 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3ds, i32 0, i32 0
  store i8 %conv8, i8* %x0, align 1
  %11 = load i32, i32* %input_lo, align 4
  %shr = lshr i32 %11, 8
  %and9 = and i32 %shr, 255
  %conv10 = trunc i32 %and9 to i8
  %y0 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3ds, i32 0, i32 1
  store i8 %conv10, i8* %y0, align 1
  %12 = load i32, i32* %input_lo, align 4
  %shr11 = lshr i32 %12, 16
  %and12 = and i32 %shr11, 255
  %conv13 = trunc i32 %and12 to i8
  %z0 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3ds, i32 0, i32 2
  store i8 %conv13, i8* %z0, align 1
  %13 = load i32, i32* %input_lo, align 4
  %shr14 = lshr i32 %13, 24
  %and15 = and i32 %shr14, 255
  %conv16 = trunc i32 %and15 to i8
  %x1 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3ds, i32 0, i32 3
  store i8 %conv16, i8* %x1, align 1
  %14 = load i32, i32* %input_mi, align 4
  %conv17 = trunc i32 %14 to i8
  %y1 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3ds, i32 0, i32 4
  store i8 %conv17, i8* %y1, align 1
  %15 = load i32, i32* %input_mi, align 4
  %shr18 = lshr i32 %15, 8
  %and19 = and i32 %shr18, 255
  %conv20 = trunc i32 %and19 to i8
  %z1 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3ds, i32 0, i32 5
  store i8 %conv20, i8* %z1, align 1
  %16 = load i32, i32* %input_mi, align 4
  %shr21 = lshr i32 %16, 16
  %and22 = and i32 %shr21, 255
  %conv23 = trunc i32 %and22 to i8
  %x2 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3ds, i32 0, i32 6
  store i8 %conv23, i8* %x2, align 1
  %17 = load i32, i32* %input_mi, align 4
  %shr24 = lshr i32 %17, 24
  %and25 = and i32 %shr24, 255
  %conv26 = trunc i32 %and25 to i8
  %y2 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3ds, i32 0, i32 7
  store i8 %conv26, i8* %y2, align 1
  %18 = load i32, i32* %input_hi, align 4
  %and27 = and i32 %18, 255
  %conv28 = trunc i32 %and27 to i8
  %z2 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3ds, i32 0, i32 8
  store i8 %conv28, i8* %z2, align 1
  %19 = bitcast %struct.Triangle_3D* %agg.tmp to i8*
  %20 = bitcast %struct.Triangle_3D* %triangle_3ds to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %19, i8* align 1 %20, i64 9, i1 false)
  %21 = load i8, i8* %angle, align 1
  %22 = bitcast { i64, i8 }* %agg.tmp.coerce to i8*
  %23 = bitcast %struct.Triangle_3D* %agg.tmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %22, i8* align 1 %23, i64 9, i1 false)
  %24 = getelementptr inbounds { i64, i8 }, { i64, i8 }* %agg.tmp.coerce, i32 0, i32 0
  %25 = load i64, i64* %24, align 1
  %26 = getelementptr inbounds { i64, i8 }, { i64, i8 }* %agg.tmp.coerce, i32 0, i32 1
  %27 = load i8, i8* %26, align 1
  call void @_ZL12projection_111Triangle_3DP11Triangle_2Dh(i64 %25, i8 %27, %struct.Triangle_2D* %triangle_2ds, i8 zeroext %21)
  %28 = bitcast %struct.Triangle_2D* %agg.tmp29 to i8*
  %29 = bitcast %struct.Triangle_2D* %triangle_2ds to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %28, i8* align 1 %29, i64 7, i1 false)
  %arraydecay = getelementptr inbounds [5 x i8], [5 x i8]* %max_min, i64 0, i64 0
  %arraydecay30 = getelementptr inbounds [1 x i32], [1 x i32]* %max_index, i64 0, i64 0
  %30 = bitcast i56* %tmp to i8*
  %31 = bitcast %struct.Triangle_2D* %agg.tmp29 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %30, i8* align 1 %31, i64 7, i1 false)
  %32 = load i56, i56* %tmp, align 8
  %call = call zeroext i8 @_ZL16rasterization1_111Triangle_2DPhPS_Pj(i56 %32, i8* %arraydecay, %struct.Triangle_2D* %triangle_2ds_same, i32* %arraydecay30)
  store i8 %call, i8* %flag, align 1
  %33 = load i8, i8* %flag, align 1
  %arraydecay31 = getelementptr inbounds [5 x i8], [5 x i8]* %max_min, i64 0, i64 0
  %arraydecay32 = getelementptr inbounds [1 x i32], [1 x i32]* %max_index, i64 0, i64 0
  %34 = bitcast %struct.Triangle_2D* %agg.tmp33 to i8*
  %35 = bitcast %struct.Triangle_2D* %triangle_2ds_same to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %34, i8* align 1 %35, i64 7, i1 false)
  %arraydecay34 = getelementptr inbounds [500 x %struct.CandidatePixel], [500 x %struct.CandidatePixel]* %fragment, i64 0, i64 0
  %36 = bitcast i56* %tmp35 to i8*
  %37 = bitcast %struct.Triangle_2D* %agg.tmp33 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %36, i8* align 1 %37, i64 7, i1 false)
  %38 = load i56, i56* %tmp35, align 8
  %call36 = call i32 @_ZL16rasterization2_1hPhPj11Triangle_2DP14CandidatePixel(i8 zeroext %33, i8* %arraydecay31, i32* %arraydecay32, i56 %38, %struct.CandidatePixel* %arraydecay34)
  store i32 %call36, i32* %size_fragment, align 4
  %39 = load i32, i32* %i, align 4
  %arraydecay37 = getelementptr inbounds [500 x %struct.CandidatePixel], [500 x %struct.CandidatePixel]* %fragment, i64 0, i64 0
  %40 = load i32, i32* %size_fragment, align 4
  %arraydecay38 = getelementptr inbounds [500 x %struct.Pixel], [500 x %struct.Pixel]* %pixels, i64 0, i64 0
  %call39 = call i32 @_ZL10zculling_1jP14CandidatePixeljP5Pixel(i32 %39, %struct.CandidatePixel* %arraydecay37, i32 %40, %struct.Pixel* %arraydecay38)
  store i32 %call39, i32* %size_pixels, align 4
  %41 = load i32, i32* %i, align 4
  %42 = load i32, i32* %size_pixels, align 4
  %arraydecay40 = getelementptr inbounds [500 x %struct.Pixel], [500 x %struct.Pixel]* %pixels, i64 0, i64 0
  %arraydecay41 = getelementptr inbounds [256 x [256 x i8]], [256 x [256 x i8]]* %frame_buffer, i64 0, i64 0
  call void @_ZL12coloringFB_1jjP5PixelPA256_h(i32 %41, i32 %42, %struct.Pixel* %arraydecay40, [256 x i8]* %arraydecay41)
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %43 = load i32, i32* %i, align 4
  %inc = add i32 %43, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %arraydecay42 = getelementptr inbounds [256 x [256 x i8]], [256 x [256 x i8]]* %frame_buffer, i64 0, i64 0
  %44 = load i32*, i32** %output.addr, align 8
  call void @_ZL11output_FB_1PA256_hPj([256 x i8]* %arraydecay42, i32* %44)
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @_ZL12projection_111Triangle_3DP11Triangle_2Dh(i64 %triangle_3d.coerce0, i8 %triangle_3d.coerce1, %struct.Triangle_2D* %triangle_2d, i8 zeroext %angle) #1 {
entry:
  %triangle_3d = alloca %struct.Triangle_3D, align 1
  %coerce = alloca { i64, i8 }, align 1
  %triangle_2d.addr = alloca %struct.Triangle_2D*, align 8
  %angle.addr = alloca i8, align 1
  %0 = getelementptr inbounds { i64, i8 }, { i64, i8 }* %coerce, i32 0, i32 0
  store i64 %triangle_3d.coerce0, i64* %0, align 1
  %1 = getelementptr inbounds { i64, i8 }, { i64, i8 }* %coerce, i32 0, i32 1
  store i8 %triangle_3d.coerce1, i8* %1, align 1
  %2 = bitcast %struct.Triangle_3D* %triangle_3d to i8*
  %3 = bitcast { i64, i8 }* %coerce to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %2, i8* align 1 %3, i64 9, i1 false)
  store %struct.Triangle_2D* %triangle_2d, %struct.Triangle_2D** %triangle_2d.addr, align 8
  store i8 %angle, i8* %angle.addr, align 1
  %4 = load i8, i8* %angle.addr, align 1
  %conv = zext i8 %4 to i32
  %cmp = icmp eq i32 %conv, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %x0 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3d, i32 0, i32 0
  %5 = load i8, i8* %x0, align 1
  %6 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %x01 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %6, i32 0, i32 0
  store i8 %5, i8* %x01, align 1
  %y0 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3d, i32 0, i32 1
  %7 = load i8, i8* %y0, align 1
  %8 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %y02 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %8, i32 0, i32 1
  store i8 %7, i8* %y02, align 1
  %x1 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3d, i32 0, i32 3
  %9 = load i8, i8* %x1, align 1
  %10 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %x13 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %10, i32 0, i32 2
  store i8 %9, i8* %x13, align 1
  %y1 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3d, i32 0, i32 4
  %11 = load i8, i8* %y1, align 1
  %12 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %y14 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %12, i32 0, i32 3
  store i8 %11, i8* %y14, align 1
  %x2 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3d, i32 0, i32 6
  %13 = load i8, i8* %x2, align 1
  %14 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %x25 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %14, i32 0, i32 4
  store i8 %13, i8* %x25, align 1
  %y2 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3d, i32 0, i32 7
  %15 = load i8, i8* %y2, align 1
  %16 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %y26 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %16, i32 0, i32 5
  store i8 %15, i8* %y26, align 1
  %z0 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3d, i32 0, i32 2
  %17 = load i8, i8* %z0, align 1
  %conv7 = zext i8 %17 to i32
  %div = sdiv i32 %conv7, 3
  %z1 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3d, i32 0, i32 5
  %18 = load i8, i8* %z1, align 1
  %conv8 = zext i8 %18 to i32
  %div9 = sdiv i32 %conv8, 3
  %add = add nsw i32 %div, %div9
  %z2 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3d, i32 0, i32 8
  %19 = load i8, i8* %z2, align 1
  %conv10 = zext i8 %19 to i32
  %div11 = sdiv i32 %conv10, 3
  %add12 = add nsw i32 %add, %div11
  %conv13 = trunc i32 %add12 to i8
  %20 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %z = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %20, i32 0, i32 6
  store i8 %conv13, i8* %z, align 1
  br label %if.end72

if.else:                                          ; preds = %entry
  %21 = load i8, i8* %angle.addr, align 1
  %conv14 = zext i8 %21 to i32
  %cmp15 = icmp eq i32 %conv14, 1
  br i1 %cmp15, label %if.then16, label %if.else42

if.then16:                                        ; preds = %if.else
  %x017 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3d, i32 0, i32 0
  %22 = load i8, i8* %x017, align 1
  %23 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %x018 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %23, i32 0, i32 0
  store i8 %22, i8* %x018, align 1
  %z019 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3d, i32 0, i32 2
  %24 = load i8, i8* %z019, align 1
  %25 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %y020 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %25, i32 0, i32 1
  store i8 %24, i8* %y020, align 1
  %x121 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3d, i32 0, i32 3
  %26 = load i8, i8* %x121, align 1
  %27 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %x122 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %27, i32 0, i32 2
  store i8 %26, i8* %x122, align 1
  %z123 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3d, i32 0, i32 5
  %28 = load i8, i8* %z123, align 1
  %29 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %y124 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %29, i32 0, i32 3
  store i8 %28, i8* %y124, align 1
  %x225 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3d, i32 0, i32 6
  %30 = load i8, i8* %x225, align 1
  %31 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %x226 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %31, i32 0, i32 4
  store i8 %30, i8* %x226, align 1
  %z227 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3d, i32 0, i32 8
  %32 = load i8, i8* %z227, align 1
  %33 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %y228 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %33, i32 0, i32 5
  store i8 %32, i8* %y228, align 1
  %y029 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3d, i32 0, i32 1
  %34 = load i8, i8* %y029, align 1
  %conv30 = zext i8 %34 to i32
  %div31 = sdiv i32 %conv30, 3
  %y132 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3d, i32 0, i32 4
  %35 = load i8, i8* %y132, align 1
  %conv33 = zext i8 %35 to i32
  %div34 = sdiv i32 %conv33, 3
  %add35 = add nsw i32 %div31, %div34
  %y236 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3d, i32 0, i32 7
  %36 = load i8, i8* %y236, align 1
  %conv37 = zext i8 %36 to i32
  %div38 = sdiv i32 %conv37, 3
  %add39 = add nsw i32 %add35, %div38
  %conv40 = trunc i32 %add39 to i8
  %37 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %z41 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %37, i32 0, i32 6
  store i8 %conv40, i8* %z41, align 1
  br label %if.end71

if.else42:                                        ; preds = %if.else
  %38 = load i8, i8* %angle.addr, align 1
  %conv43 = zext i8 %38 to i32
  %cmp44 = icmp eq i32 %conv43, 2
  br i1 %cmp44, label %if.then45, label %if.end

if.then45:                                        ; preds = %if.else42
  %z046 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3d, i32 0, i32 2
  %39 = load i8, i8* %z046, align 1
  %40 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %x047 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %40, i32 0, i32 0
  store i8 %39, i8* %x047, align 1
  %y048 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3d, i32 0, i32 1
  %41 = load i8, i8* %y048, align 1
  %42 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %y049 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %42, i32 0, i32 1
  store i8 %41, i8* %y049, align 1
  %z150 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3d, i32 0, i32 5
  %43 = load i8, i8* %z150, align 1
  %44 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %x151 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %44, i32 0, i32 2
  store i8 %43, i8* %x151, align 1
  %y152 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3d, i32 0, i32 4
  %45 = load i8, i8* %y152, align 1
  %46 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %y153 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %46, i32 0, i32 3
  store i8 %45, i8* %y153, align 1
  %z254 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3d, i32 0, i32 8
  %47 = load i8, i8* %z254, align 1
  %48 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %x255 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %48, i32 0, i32 4
  store i8 %47, i8* %x255, align 1
  %y256 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3d, i32 0, i32 7
  %49 = load i8, i8* %y256, align 1
  %50 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %y257 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %50, i32 0, i32 5
  store i8 %49, i8* %y257, align 1
  %x058 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3d, i32 0, i32 0
  %51 = load i8, i8* %x058, align 1
  %conv59 = zext i8 %51 to i32
  %div60 = sdiv i32 %conv59, 3
  %x161 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3d, i32 0, i32 3
  %52 = load i8, i8* %x161, align 1
  %conv62 = zext i8 %52 to i32
  %div63 = sdiv i32 %conv62, 3
  %add64 = add nsw i32 %div60, %div63
  %x265 = getelementptr inbounds %struct.Triangle_3D, %struct.Triangle_3D* %triangle_3d, i32 0, i32 6
  %53 = load i8, i8* %x265, align 1
  %conv66 = zext i8 %53 to i32
  %div67 = sdiv i32 %conv66, 3
  %add68 = add nsw i32 %add64, %div67
  %conv69 = trunc i32 %add68 to i8
  %54 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %z70 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %54, i32 0, i32 6
  store i8 %conv69, i8* %z70, align 1
  br label %if.end

if.end:                                           ; preds = %if.then45, %if.else42
  br label %if.end71

if.end71:                                         ; preds = %if.end, %if.then16
  br label %if.end72

if.end72:                                         ; preds = %if.end71, %if.then
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #2

; Function Attrs: noinline optnone uwtable
define internal zeroext i8 @_ZL16rasterization1_111Triangle_2DPhPS_Pj(i56 %triangle_2d.coerce, i8* %max_min, %struct.Triangle_2D* %triangle_2d_same, i32* %max_index) #0 {
entry:
  %retval = alloca i8, align 1
  %triangle_2d = alloca %struct.Triangle_2D, align 1
  %tmp = alloca i56, align 8
  %max_min.addr = alloca i8*, align 8
  %triangle_2d_same.addr = alloca %struct.Triangle_2D*, align 8
  %max_index.addr = alloca i32*, align 8
  %agg.tmp = alloca %struct.Triangle_2D, align 1
  %tmp1 = alloca i56, align 8
  %agg.tmp2 = alloca %struct.Triangle_2D, align 1
  %tmp3 = alloca i56, align 8
  store i56 %triangle_2d.coerce, i56* %tmp, align 8
  %0 = bitcast i56* %tmp to i8*
  %1 = bitcast %struct.Triangle_2D* %triangle_2d to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %1, i8* align 8 %0, i64 7, i1 false)
  store i8* %max_min, i8** %max_min.addr, align 8
  store %struct.Triangle_2D* %triangle_2d_same, %struct.Triangle_2D** %triangle_2d_same.addr, align 8
  store i32* %max_index, i32** %max_index.addr, align 8
  %2 = bitcast %struct.Triangle_2D* %agg.tmp to i8*
  %3 = bitcast %struct.Triangle_2D* %triangle_2d to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %2, i8* align 1 %3, i64 7, i1 false)
  %4 = bitcast i56* %tmp1 to i8*
  %5 = bitcast %struct.Triangle_2D* %agg.tmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %4, i8* align 1 %5, i64 7, i1 false)
  %6 = load i56, i56* %tmp1, align 8
  %call = call i32 @_ZL17check_clockwise_111Triangle_2D(i56 %6)
  %cmp = icmp eq i32 %call, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i8 1, i8* %retval, align 1
  br label %return

if.end:                                           ; preds = %entry
  %7 = bitcast %struct.Triangle_2D* %agg.tmp2 to i8*
  %8 = bitcast %struct.Triangle_2D* %triangle_2d to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %7, i8* align 1 %8, i64 7, i1 false)
  %9 = bitcast i56* %tmp3 to i8*
  %10 = bitcast %struct.Triangle_2D* %agg.tmp2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %9, i8* align 1 %10, i64 7, i1 false)
  %11 = load i56, i56* %tmp3, align 8
  %call4 = call i32 @_ZL17check_clockwise_111Triangle_2D(i56 %11)
  %cmp5 = icmp slt i32 %call4, 0
  br i1 %cmp5, label %if.then6, label %if.end7

if.then6:                                         ; preds = %if.end
  call void @_ZL20clockwise_vertices_1P11Triangle_2D(%struct.Triangle_2D* %triangle_2d)
  br label %if.end7

if.end7:                                          ; preds = %if.then6, %if.end
  %x0 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 0
  %12 = load i8, i8* %x0, align 1
  %13 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d_same.addr, align 8
  %x08 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %13, i32 0, i32 0
  store i8 %12, i8* %x08, align 1
  %y0 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 1
  %14 = load i8, i8* %y0, align 1
  %15 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d_same.addr, align 8
  %y09 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %15, i32 0, i32 1
  store i8 %14, i8* %y09, align 1
  %x1 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 2
  %16 = load i8, i8* %x1, align 1
  %17 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d_same.addr, align 8
  %x110 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %17, i32 0, i32 2
  store i8 %16, i8* %x110, align 1
  %y1 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 3
  %18 = load i8, i8* %y1, align 1
  %19 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d_same.addr, align 8
  %y111 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %19, i32 0, i32 3
  store i8 %18, i8* %y111, align 1
  %x2 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 4
  %20 = load i8, i8* %x2, align 1
  %21 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d_same.addr, align 8
  %x212 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %21, i32 0, i32 4
  store i8 %20, i8* %x212, align 1
  %y2 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 5
  %22 = load i8, i8* %y2, align 1
  %23 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d_same.addr, align 8
  %y213 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %23, i32 0, i32 5
  store i8 %22, i8* %y213, align 1
  %z = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 6
  %24 = load i8, i8* %z, align 1
  %25 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d_same.addr, align 8
  %z14 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %25, i32 0, i32 6
  store i8 %24, i8* %z14, align 1
  %x015 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 0
  %26 = load i8, i8* %x015, align 1
  %x116 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 2
  %27 = load i8, i8* %x116, align 1
  %x217 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 4
  %28 = load i8, i8* %x217, align 1
  %call18 = call zeroext i8 @_ZL10find_min_1hhh(i8 zeroext %26, i8 zeroext %27, i8 zeroext %28)
  %29 = load i8*, i8** %max_min.addr, align 8
  %arrayidx = getelementptr inbounds i8, i8* %29, i64 0
  store i8 %call18, i8* %arrayidx, align 1
  %x019 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 0
  %30 = load i8, i8* %x019, align 1
  %x120 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 2
  %31 = load i8, i8* %x120, align 1
  %x221 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 4
  %32 = load i8, i8* %x221, align 1
  %call22 = call zeroext i8 @_ZL10find_max_1hhh(i8 zeroext %30, i8 zeroext %31, i8 zeroext %32)
  %33 = load i8*, i8** %max_min.addr, align 8
  %arrayidx23 = getelementptr inbounds i8, i8* %33, i64 1
  store i8 %call22, i8* %arrayidx23, align 1
  %y024 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 1
  %34 = load i8, i8* %y024, align 1
  %y125 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 3
  %35 = load i8, i8* %y125, align 1
  %y226 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 5
  %36 = load i8, i8* %y226, align 1
  %call27 = call zeroext i8 @_ZL10find_min_1hhh(i8 zeroext %34, i8 zeroext %35, i8 zeroext %36)
  %37 = load i8*, i8** %max_min.addr, align 8
  %arrayidx28 = getelementptr inbounds i8, i8* %37, i64 2
  store i8 %call27, i8* %arrayidx28, align 1
  %y029 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 1
  %38 = load i8, i8* %y029, align 1
  %y130 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 3
  %39 = load i8, i8* %y130, align 1
  %y231 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 5
  %40 = load i8, i8* %y231, align 1
  %call32 = call zeroext i8 @_ZL10find_max_1hhh(i8 zeroext %38, i8 zeroext %39, i8 zeroext %40)
  %41 = load i8*, i8** %max_min.addr, align 8
  %arrayidx33 = getelementptr inbounds i8, i8* %41, i64 3
  store i8 %call32, i8* %arrayidx33, align 1
  %42 = load i8*, i8** %max_min.addr, align 8
  %arrayidx34 = getelementptr inbounds i8, i8* %42, i64 1
  %43 = load i8, i8* %arrayidx34, align 1
  %conv = zext i8 %43 to i32
  %44 = load i8*, i8** %max_min.addr, align 8
  %arrayidx35 = getelementptr inbounds i8, i8* %44, i64 0
  %45 = load i8, i8* %arrayidx35, align 1
  %conv36 = zext i8 %45 to i32
  %sub = sub nsw i32 %conv, %conv36
  %conv37 = trunc i32 %sub to i8
  %46 = load i8*, i8** %max_min.addr, align 8
  %arrayidx38 = getelementptr inbounds i8, i8* %46, i64 4
  store i8 %conv37, i8* %arrayidx38, align 1
  %47 = load i8*, i8** %max_min.addr, align 8
  %arrayidx39 = getelementptr inbounds i8, i8* %47, i64 1
  %48 = load i8, i8* %arrayidx39, align 1
  %conv40 = zext i8 %48 to i32
  %49 = load i8*, i8** %max_min.addr, align 8
  %arrayidx41 = getelementptr inbounds i8, i8* %49, i64 0
  %50 = load i8, i8* %arrayidx41, align 1
  %conv42 = zext i8 %50 to i32
  %sub43 = sub nsw i32 %conv40, %conv42
  %51 = load i8*, i8** %max_min.addr, align 8
  %arrayidx44 = getelementptr inbounds i8, i8* %51, i64 3
  %52 = load i8, i8* %arrayidx44, align 1
  %conv45 = zext i8 %52 to i32
  %53 = load i8*, i8** %max_min.addr, align 8
  %arrayidx46 = getelementptr inbounds i8, i8* %53, i64 2
  %54 = load i8, i8* %arrayidx46, align 1
  %conv47 = zext i8 %54 to i32
  %sub48 = sub nsw i32 %conv45, %conv47
  %mul = mul nsw i32 %sub43, %sub48
  %55 = load i32*, i32** %max_index.addr, align 8
  %arrayidx49 = getelementptr inbounds i32, i32* %55, i64 0
  store i32 %mul, i32* %arrayidx49, align 4
  store i8 0, i8* %retval, align 1
  br label %return

return:                                           ; preds = %if.end7, %if.then
  %56 = load i8, i8* %retval, align 1
  ret i8 %56
}

; Function Attrs: noinline optnone uwtable
define internal i32 @_ZL16rasterization2_1hPhPj11Triangle_2DP14CandidatePixel(i8 zeroext %flag, i8* %max_min, i32* %max_index, i56 %triangle_2d_same.coerce, %struct.CandidatePixel* %fragment2) #0 {
entry:
  %retval = alloca i32, align 4
  %triangle_2d_same = alloca %struct.Triangle_2D, align 1
  %tmp = alloca i56, align 8
  %flag.addr = alloca i8, align 1
  %max_min.addr = alloca i8*, align 8
  %max_index.addr = alloca i32*, align 8
  %fragment2.addr = alloca %struct.CandidatePixel*, align 8
  %color = alloca i8, align 1
  %i = alloca i32, align 4
  %k = alloca i32, align 4
  %x = alloca i8, align 1
  %y = alloca i8, align 1
  %agg.tmp = alloca %struct.Triangle_2D, align 1
  %tmp11 = alloca i56, align 8
  store i56 %triangle_2d_same.coerce, i56* %tmp, align 8
  %0 = bitcast i56* %tmp to i8*
  %1 = bitcast %struct.Triangle_2D* %triangle_2d_same to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %1, i8* align 8 %0, i64 7, i1 false)
  store i8 %flag, i8* %flag.addr, align 1
  store i8* %max_min, i8** %max_min.addr, align 8
  store i32* %max_index, i32** %max_index.addr, align 8
  store %struct.CandidatePixel* %fragment2, %struct.CandidatePixel** %fragment2.addr, align 8
  %2 = load i8, i8* %flag.addr, align 1
  %tobool = icmp ne i8 %2, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 0, i32* %retval, align 4
  br label %return

if.end:                                           ; preds = %entry
  store i8 100, i8* %color, align 1
  store i32 0, i32* %i, align 4
  br label %RAST2

RAST2:                                            ; preds = %if.end
  store i32 0, i32* %k, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %RAST2
  %3 = load i32, i32* %k, align 4
  %4 = load i32*, i32** %max_index.addr, align 8
  %arrayidx = getelementptr inbounds i32, i32* %4, i64 0
  %5 = load i32, i32* %arrayidx, align 4
  %cmp = icmp ult i32 %3, %5
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %6 = load i8*, i8** %max_min.addr, align 8
  %arrayidx1 = getelementptr inbounds i8, i8* %6, i64 0
  %7 = load i8, i8* %arrayidx1, align 1
  %conv = zext i8 %7 to i32
  %8 = load i32, i32* %k, align 4
  %9 = load i8*, i8** %max_min.addr, align 8
  %arrayidx2 = getelementptr inbounds i8, i8* %9, i64 4
  %10 = load i8, i8* %arrayidx2, align 1
  %conv3 = zext i8 %10 to i32
  %rem = urem i32 %8, %conv3
  %add = add i32 %conv, %rem
  %conv4 = trunc i32 %add to i8
  store i8 %conv4, i8* %x, align 1
  %11 = load i8*, i8** %max_min.addr, align 8
  %arrayidx5 = getelementptr inbounds i8, i8* %11, i64 2
  %12 = load i8, i8* %arrayidx5, align 1
  %conv6 = zext i8 %12 to i32
  %13 = load i32, i32* %k, align 4
  %14 = load i8*, i8** %max_min.addr, align 8
  %arrayidx7 = getelementptr inbounds i8, i8* %14, i64 4
  %15 = load i8, i8* %arrayidx7, align 1
  %conv8 = zext i8 %15 to i32
  %div = udiv i32 %13, %conv8
  %add9 = add i32 %conv6, %div
  %conv10 = trunc i32 %add9 to i8
  store i8 %conv10, i8* %y, align 1
  %16 = load i8, i8* %x, align 1
  %17 = load i8, i8* %y, align 1
  %18 = bitcast %struct.Triangle_2D* %agg.tmp to i8*
  %19 = bitcast %struct.Triangle_2D* %triangle_2d_same to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %18, i8* align 1 %19, i64 7, i1 false)
  %20 = bitcast i56* %tmp11 to i8*
  %21 = bitcast %struct.Triangle_2D* %agg.tmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %20, i8* align 1 %21, i64 7, i1 false)
  %22 = load i56, i56* %tmp11, align 8
  %call = call zeroext i8 @_ZL19pixel_in_triangle_1hh11Triangle_2D(i8 zeroext %16, i8 zeroext %17, i56 %22)
  %tobool12 = icmp ne i8 %call, 0
  br i1 %tobool12, label %if.then13, label %if.end26

if.then13:                                        ; preds = %for.body
  %23 = load i8, i8* %x, align 1
  %24 = load %struct.CandidatePixel*, %struct.CandidatePixel** %fragment2.addr, align 8
  %25 = load i32, i32* %i, align 4
  %conv14 = zext i32 %25 to i64
  %arrayidx15 = getelementptr inbounds %struct.CandidatePixel, %struct.CandidatePixel* %24, i64 %conv14
  %x16 = getelementptr inbounds %struct.CandidatePixel, %struct.CandidatePixel* %arrayidx15, i32 0, i32 0
  store i8 %23, i8* %x16, align 1
  %26 = load i8, i8* %y, align 1
  %27 = load %struct.CandidatePixel*, %struct.CandidatePixel** %fragment2.addr, align 8
  %28 = load i32, i32* %i, align 4
  %conv17 = zext i32 %28 to i64
  %arrayidx18 = getelementptr inbounds %struct.CandidatePixel, %struct.CandidatePixel* %27, i64 %conv17
  %y19 = getelementptr inbounds %struct.CandidatePixel, %struct.CandidatePixel* %arrayidx18, i32 0, i32 1
  store i8 %26, i8* %y19, align 1
  %z = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d_same, i32 0, i32 6
  %29 = load i8, i8* %z, align 1
  %30 = load %struct.CandidatePixel*, %struct.CandidatePixel** %fragment2.addr, align 8
  %31 = load i32, i32* %i, align 4
  %conv20 = zext i32 %31 to i64
  %arrayidx21 = getelementptr inbounds %struct.CandidatePixel, %struct.CandidatePixel* %30, i64 %conv20
  %z22 = getelementptr inbounds %struct.CandidatePixel, %struct.CandidatePixel* %arrayidx21, i32 0, i32 2
  store i8 %29, i8* %z22, align 1
  %32 = load i8, i8* %color, align 1
  %33 = load %struct.CandidatePixel*, %struct.CandidatePixel** %fragment2.addr, align 8
  %34 = load i32, i32* %i, align 4
  %conv23 = zext i32 %34 to i64
  %arrayidx24 = getelementptr inbounds %struct.CandidatePixel, %struct.CandidatePixel* %33, i64 %conv23
  %color25 = getelementptr inbounds %struct.CandidatePixel, %struct.CandidatePixel* %arrayidx24, i32 0, i32 3
  store i8 %32, i8* %color25, align 1
  %35 = load i32, i32* %i, align 4
  %inc = add i32 %35, 1
  store i32 %inc, i32* %i, align 4
  br label %if.end26

if.end26:                                         ; preds = %if.then13, %for.body
  br label %for.inc

for.inc:                                          ; preds = %if.end26
  %36 = load i32, i32* %k, align 4
  %inc27 = add i32 %36, 1
  store i32 %inc27, i32* %k, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %37 = load i32, i32* %i, align 4
  store i32 %37, i32* %retval, align 4
  br label %return

return:                                           ; preds = %for.end, %if.then
  %38 = load i32, i32* %retval, align 4
  ret i32 %38
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @_ZL10zculling_1jP14CandidatePixeljP5Pixel(i32 %counter, %struct.CandidatePixel* %fragments, i32 %size, %struct.Pixel* %pixels) #1 {
entry:
  %counter.addr = alloca i32, align 4
  %fragments.addr = alloca %struct.CandidatePixel*, align 8
  %size.addr = alloca i32, align 4
  %pixels.addr = alloca %struct.Pixel*, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %pixel_cntr = alloca i32, align 4
  %n = alloca i32, align 4
  store i32 %counter, i32* %counter.addr, align 4
  store %struct.CandidatePixel* %fragments, %struct.CandidatePixel** %fragments.addr, align 8
  store i32 %size, i32* %size.addr, align 4
  store %struct.Pixel* %pixels, %struct.Pixel** %pixels.addr, align 8
  %0 = load i32, i32* %counter.addr, align 4
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  br label %ZCULLING_INIT_ROW

ZCULLING_INIT_ROW:                                ; preds = %if.then
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc7, %ZCULLING_INIT_ROW
  %1 = load i32, i32* %i, align 4
  %cmp1 = icmp ult i32 %1, 256
  br i1 %cmp1, label %for.body, label %for.end9

for.body:                                         ; preds = %for.cond
  br label %ZCULLING_INIT_COL

ZCULLING_INIT_COL:                                ; preds = %for.body
  store i32 0, i32* %j, align 4
  br label %for.cond2

for.cond2:                                        ; preds = %for.inc, %ZCULLING_INIT_COL
  %2 = load i32, i32* %j, align 4
  %cmp3 = icmp ult i32 %2, 256
  br i1 %cmp3, label %for.body4, label %for.end

for.body4:                                        ; preds = %for.cond2
  %3 = load i32, i32* %i, align 4
  %conv = zext i32 %3 to i64
  %arrayidx = getelementptr inbounds [256 x [256 x i8]], [256 x [256 x i8]]* @_ZZL10zculling_1jP14CandidatePixeljP5PixelE8z_buffer, i64 0, i64 %conv
  %4 = load i32, i32* %j, align 4
  %conv5 = zext i32 %4 to i64
  %arrayidx6 = getelementptr inbounds [256 x i8], [256 x i8]* %arrayidx, i64 0, i64 %conv5
  store i8 -1, i8* %arrayidx6, align 1
  br label %for.inc

for.inc:                                          ; preds = %for.body4
  %5 = load i32, i32* %j, align 4
  %inc = add i32 %5, 1
  store i32 %inc, i32* %j, align 4
  br label %for.cond2

for.end:                                          ; preds = %for.cond2
  br label %for.inc7

for.inc7:                                         ; preds = %for.end
  %6 = load i32, i32* %i, align 4
  %inc8 = add i32 %6, 1
  store i32 %inc8, i32* %i, align 4
  br label %for.cond

for.end9:                                         ; preds = %for.cond
  br label %if.end

if.end:                                           ; preds = %for.end9, %entry
  store i32 0, i32* %pixel_cntr, align 4
  br label %ZCULLING

ZCULLING:                                         ; preds = %if.end
  store i32 0, i32* %n, align 4
  br label %for.cond10

for.cond10:                                       ; preds = %for.inc59, %ZCULLING
  %7 = load i32, i32* %n, align 4
  %8 = load i32, i32* %size.addr, align 4
  %cmp11 = icmp ult i32 %7, %8
  br i1 %cmp11, label %for.body12, label %for.end61

for.body12:                                       ; preds = %for.cond10
  %9 = load %struct.CandidatePixel*, %struct.CandidatePixel** %fragments.addr, align 8
  %10 = load i32, i32* %n, align 4
  %conv13 = zext i32 %10 to i64
  %arrayidx14 = getelementptr inbounds %struct.CandidatePixel, %struct.CandidatePixel* %9, i64 %conv13
  %z = getelementptr inbounds %struct.CandidatePixel, %struct.CandidatePixel* %arrayidx14, i32 0, i32 2
  %11 = load i8, i8* %z, align 1
  %conv15 = zext i8 %11 to i32
  %12 = load %struct.CandidatePixel*, %struct.CandidatePixel** %fragments.addr, align 8
  %13 = load i32, i32* %n, align 4
  %conv16 = zext i32 %13 to i64
  %arrayidx17 = getelementptr inbounds %struct.CandidatePixel, %struct.CandidatePixel* %12, i64 %conv16
  %y = getelementptr inbounds %struct.CandidatePixel, %struct.CandidatePixel* %arrayidx17, i32 0, i32 1
  %14 = load i8, i8* %y, align 1
  %conv18 = zext i8 %14 to i64
  %arrayidx19 = getelementptr inbounds [256 x [256 x i8]], [256 x [256 x i8]]* @_ZZL10zculling_1jP14CandidatePixeljP5PixelE8z_buffer, i64 0, i64 %conv18
  %15 = load %struct.CandidatePixel*, %struct.CandidatePixel** %fragments.addr, align 8
  %16 = load i32, i32* %n, align 4
  %conv20 = zext i32 %16 to i64
  %arrayidx21 = getelementptr inbounds %struct.CandidatePixel, %struct.CandidatePixel* %15, i64 %conv20
  %x = getelementptr inbounds %struct.CandidatePixel, %struct.CandidatePixel* %arrayidx21, i32 0, i32 0
  %17 = load i8, i8* %x, align 1
  %conv22 = zext i8 %17 to i64
  %arrayidx23 = getelementptr inbounds [256 x i8], [256 x i8]* %arrayidx19, i64 0, i64 %conv22
  %18 = load i8, i8* %arrayidx23, align 1
  %conv24 = zext i8 %18 to i32
  %cmp25 = icmp slt i32 %conv15, %conv24
  br i1 %cmp25, label %if.then26, label %if.end58

if.then26:                                        ; preds = %for.body12
  %19 = load %struct.CandidatePixel*, %struct.CandidatePixel** %fragments.addr, align 8
  %20 = load i32, i32* %n, align 4
  %conv27 = zext i32 %20 to i64
  %arrayidx28 = getelementptr inbounds %struct.CandidatePixel, %struct.CandidatePixel* %19, i64 %conv27
  %x29 = getelementptr inbounds %struct.CandidatePixel, %struct.CandidatePixel* %arrayidx28, i32 0, i32 0
  %21 = load i8, i8* %x29, align 1
  %22 = load %struct.Pixel*, %struct.Pixel** %pixels.addr, align 8
  %23 = load i32, i32* %pixel_cntr, align 4
  %conv30 = zext i32 %23 to i64
  %arrayidx31 = getelementptr inbounds %struct.Pixel, %struct.Pixel* %22, i64 %conv30
  %x32 = getelementptr inbounds %struct.Pixel, %struct.Pixel* %arrayidx31, i32 0, i32 0
  store i8 %21, i8* %x32, align 1
  %24 = load %struct.CandidatePixel*, %struct.CandidatePixel** %fragments.addr, align 8
  %25 = load i32, i32* %n, align 4
  %conv33 = zext i32 %25 to i64
  %arrayidx34 = getelementptr inbounds %struct.CandidatePixel, %struct.CandidatePixel* %24, i64 %conv33
  %y35 = getelementptr inbounds %struct.CandidatePixel, %struct.CandidatePixel* %arrayidx34, i32 0, i32 1
  %26 = load i8, i8* %y35, align 1
  %27 = load %struct.Pixel*, %struct.Pixel** %pixels.addr, align 8
  %28 = load i32, i32* %pixel_cntr, align 4
  %conv36 = zext i32 %28 to i64
  %arrayidx37 = getelementptr inbounds %struct.Pixel, %struct.Pixel* %27, i64 %conv36
  %y38 = getelementptr inbounds %struct.Pixel, %struct.Pixel* %arrayidx37, i32 0, i32 1
  store i8 %26, i8* %y38, align 1
  %29 = load %struct.CandidatePixel*, %struct.CandidatePixel** %fragments.addr, align 8
  %30 = load i32, i32* %n, align 4
  %conv39 = zext i32 %30 to i64
  %arrayidx40 = getelementptr inbounds %struct.CandidatePixel, %struct.CandidatePixel* %29, i64 %conv39
  %color = getelementptr inbounds %struct.CandidatePixel, %struct.CandidatePixel* %arrayidx40, i32 0, i32 3
  %31 = load i8, i8* %color, align 1
  %32 = load %struct.Pixel*, %struct.Pixel** %pixels.addr, align 8
  %33 = load i32, i32* %pixel_cntr, align 4
  %conv41 = zext i32 %33 to i64
  %arrayidx42 = getelementptr inbounds %struct.Pixel, %struct.Pixel* %32, i64 %conv41
  %color43 = getelementptr inbounds %struct.Pixel, %struct.Pixel* %arrayidx42, i32 0, i32 2
  store i8 %31, i8* %color43, align 1
  %34 = load i32, i32* %pixel_cntr, align 4
  %inc44 = add i32 %34, 1
  store i32 %inc44, i32* %pixel_cntr, align 4
  %35 = load %struct.CandidatePixel*, %struct.CandidatePixel** %fragments.addr, align 8
  %36 = load i32, i32* %n, align 4
  %conv45 = zext i32 %36 to i64
  %arrayidx46 = getelementptr inbounds %struct.CandidatePixel, %struct.CandidatePixel* %35, i64 %conv45
  %z47 = getelementptr inbounds %struct.CandidatePixel, %struct.CandidatePixel* %arrayidx46, i32 0, i32 2
  %37 = load i8, i8* %z47, align 1
  %38 = load %struct.CandidatePixel*, %struct.CandidatePixel** %fragments.addr, align 8
  %39 = load i32, i32* %n, align 4
  %conv48 = zext i32 %39 to i64
  %arrayidx49 = getelementptr inbounds %struct.CandidatePixel, %struct.CandidatePixel* %38, i64 %conv48
  %y50 = getelementptr inbounds %struct.CandidatePixel, %struct.CandidatePixel* %arrayidx49, i32 0, i32 1
  %40 = load i8, i8* %y50, align 1
  %conv51 = zext i8 %40 to i64
  %arrayidx52 = getelementptr inbounds [256 x [256 x i8]], [256 x [256 x i8]]* @_ZZL10zculling_1jP14CandidatePixeljP5PixelE8z_buffer, i64 0, i64 %conv51
  %41 = load %struct.CandidatePixel*, %struct.CandidatePixel** %fragments.addr, align 8
  %42 = load i32, i32* %n, align 4
  %conv53 = zext i32 %42 to i64
  %arrayidx54 = getelementptr inbounds %struct.CandidatePixel, %struct.CandidatePixel* %41, i64 %conv53
  %x55 = getelementptr inbounds %struct.CandidatePixel, %struct.CandidatePixel* %arrayidx54, i32 0, i32 0
  %43 = load i8, i8* %x55, align 1
  %conv56 = zext i8 %43 to i64
  %arrayidx57 = getelementptr inbounds [256 x i8], [256 x i8]* %arrayidx52, i64 0, i64 %conv56
  store i8 %37, i8* %arrayidx57, align 1
  br label %if.end58

if.end58:                                         ; preds = %if.then26, %for.body12
  br label %for.inc59

for.inc59:                                        ; preds = %if.end58
  %44 = load i32, i32* %n, align 4
  %inc60 = add i32 %44, 1
  store i32 %inc60, i32* %n, align 4
  br label %for.cond10

for.end61:                                        ; preds = %for.cond10
  %45 = load i32, i32* %pixel_cntr, align 4
  ret i32 %45
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @_ZL12coloringFB_1jjP5PixelPA256_h(i32 %counter, i32 %size_pixels, %struct.Pixel* %pixels, [256 x i8]* %frame_buffer) #1 {
entry:
  %counter.addr = alloca i32, align 4
  %size_pixels.addr = alloca i32, align 4
  %pixels.addr = alloca %struct.Pixel*, align 8
  %frame_buffer.addr = alloca [256 x i8]*, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %i10 = alloca i32, align 4
  store i32 %counter, i32* %counter.addr, align 4
  store i32 %size_pixels, i32* %size_pixels.addr, align 4
  store %struct.Pixel* %pixels, %struct.Pixel** %pixels.addr, align 8
  store [256 x i8]* %frame_buffer, [256 x i8]** %frame_buffer.addr, align 8
  %0 = load i32, i32* %counter.addr, align 4
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  br label %COLORING_FB_INIT_ROW

COLORING_FB_INIT_ROW:                             ; preds = %if.then
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc7, %COLORING_FB_INIT_ROW
  %1 = load i32, i32* %i, align 4
  %cmp1 = icmp ult i32 %1, 256
  br i1 %cmp1, label %for.body, label %for.end9

for.body:                                         ; preds = %for.cond
  br label %COLORING_FB_INIT_COL

COLORING_FB_INIT_COL:                             ; preds = %for.body
  store i32 0, i32* %j, align 4
  br label %for.cond2

for.cond2:                                        ; preds = %for.inc, %COLORING_FB_INIT_COL
  %2 = load i32, i32* %j, align 4
  %cmp3 = icmp ult i32 %2, 256
  br i1 %cmp3, label %for.body4, label %for.end

for.body4:                                        ; preds = %for.cond2
  %3 = load [256 x i8]*, [256 x i8]** %frame_buffer.addr, align 8
  %4 = load i32, i32* %i, align 4
  %conv = zext i32 %4 to i64
  %arrayidx = getelementptr inbounds [256 x i8], [256 x i8]* %3, i64 %conv
  %5 = load i32, i32* %j, align 4
  %conv5 = zext i32 %5 to i64
  %arrayidx6 = getelementptr inbounds [256 x i8], [256 x i8]* %arrayidx, i64 0, i64 %conv5
  store i8 0, i8* %arrayidx6, align 1
  br label %for.inc

for.inc:                                          ; preds = %for.body4
  %6 = load i32, i32* %j, align 4
  %inc = add i32 %6, 1
  store i32 %inc, i32* %j, align 4
  br label %for.cond2

for.end:                                          ; preds = %for.cond2
  br label %for.inc7

for.inc7:                                         ; preds = %for.end
  %7 = load i32, i32* %i, align 4
  %inc8 = add i32 %7, 1
  store i32 %inc8, i32* %i, align 4
  br label %for.cond

for.end9:                                         ; preds = %for.cond
  br label %if.end

if.end:                                           ; preds = %for.end9, %entry
  br label %COLORING_FB

COLORING_FB:                                      ; preds = %if.end
  store i32 0, i32* %i10, align 4
  br label %for.cond11

for.cond11:                                       ; preds = %for.inc24, %COLORING_FB
  %8 = load i32, i32* %i10, align 4
  %9 = load i32, i32* %size_pixels.addr, align 4
  %cmp12 = icmp ult i32 %8, %9
  br i1 %cmp12, label %for.body13, label %for.end26

for.body13:                                       ; preds = %for.cond11
  %10 = load %struct.Pixel*, %struct.Pixel** %pixels.addr, align 8
  %11 = load i32, i32* %i10, align 4
  %conv14 = zext i32 %11 to i64
  %arrayidx15 = getelementptr inbounds %struct.Pixel, %struct.Pixel* %10, i64 %conv14
  %color = getelementptr inbounds %struct.Pixel, %struct.Pixel* %arrayidx15, i32 0, i32 2
  %12 = load i8, i8* %color, align 1
  %13 = load [256 x i8]*, [256 x i8]** %frame_buffer.addr, align 8
  %14 = load %struct.Pixel*, %struct.Pixel** %pixels.addr, align 8
  %15 = load i32, i32* %i10, align 4
  %conv16 = zext i32 %15 to i64
  %arrayidx17 = getelementptr inbounds %struct.Pixel, %struct.Pixel* %14, i64 %conv16
  %x = getelementptr inbounds %struct.Pixel, %struct.Pixel* %arrayidx17, i32 0, i32 0
  %16 = load i8, i8* %x, align 1
  %conv18 = zext i8 %16 to i64
  %arrayidx19 = getelementptr inbounds [256 x i8], [256 x i8]* %13, i64 %conv18
  %17 = load %struct.Pixel*, %struct.Pixel** %pixels.addr, align 8
  %18 = load i32, i32* %i10, align 4
  %conv20 = zext i32 %18 to i64
  %arrayidx21 = getelementptr inbounds %struct.Pixel, %struct.Pixel* %17, i64 %conv20
  %y = getelementptr inbounds %struct.Pixel, %struct.Pixel* %arrayidx21, i32 0, i32 1
  %19 = load i8, i8* %y, align 1
  %conv22 = zext i8 %19 to i64
  %arrayidx23 = getelementptr inbounds [256 x i8], [256 x i8]* %arrayidx19, i64 0, i64 %conv22
  store i8 %12, i8* %arrayidx23, align 1
  br label %for.inc24

for.inc24:                                        ; preds = %for.body13
  %20 = load i32, i32* %i10, align 4
  %inc25 = add i32 %20, 1
  store i32 %inc25, i32* %i10, align 4
  br label %for.cond11

for.end26:                                        ; preds = %for.cond11
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @_ZL11output_FB_1PA256_hPj([256 x i8]* %frame_buffer, i32* %output) #1 {
entry:
  %frame_buffer.addr = alloca [256 x i8]*, align 8
  %output.addr = alloca i32*, align 8
  %out_FB = alloca i32, align 4
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  store [256 x i8]* %frame_buffer, [256 x i8]** %frame_buffer.addr, align 8
  store i32* %output, i32** %output.addr, align 8
  store i32 0, i32* %out_FB, align 4
  br label %OUTPUT_FB_ROW

OUTPUT_FB_ROW:                                    ; preds = %entry
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc34, %OUTPUT_FB_ROW
  %0 = load i32, i32* %i, align 4
  %cmp = icmp ult i32 %0, 256
  br i1 %cmp, label %for.body, label %for.end35

for.body:                                         ; preds = %for.cond
  br label %OUTPUT_FB_COL

OUTPUT_FB_COL:                                    ; preds = %for.body
  store i32 0, i32* %j, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %OUTPUT_FB_COL
  %1 = load i32, i32* %j, align 4
  %cmp2 = icmp ult i32 %1, 256
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %2 = load [256 x i8]*, [256 x i8]** %frame_buffer.addr, align 8
  %3 = load i32, i32* %i, align 4
  %conv = zext i32 %3 to i64
  %arrayidx = getelementptr inbounds [256 x i8], [256 x i8]* %2, i64 %conv
  %4 = load i32, i32* %j, align 4
  %add = add i32 %4, 0
  %conv4 = zext i32 %add to i64
  %arrayidx5 = getelementptr inbounds [256 x i8], [256 x i8]* %arrayidx, i64 0, i64 %conv4
  %5 = load i8, i8* %arrayidx5, align 1
  %conv6 = zext i8 %5 to i32
  %6 = load i32, i32* %out_FB, align 4
  %or = or i32 %6, %conv6
  store i32 %or, i32* %out_FB, align 4
  %7 = load [256 x i8]*, [256 x i8]** %frame_buffer.addr, align 8
  %8 = load i32, i32* %i, align 4
  %conv7 = zext i32 %8 to i64
  %arrayidx8 = getelementptr inbounds [256 x i8], [256 x i8]* %7, i64 %conv7
  %9 = load i32, i32* %j, align 4
  %add9 = add i32 %9, 1
  %conv10 = zext i32 %add9 to i64
  %arrayidx11 = getelementptr inbounds [256 x i8], [256 x i8]* %arrayidx8, i64 0, i64 %conv10
  %10 = load i8, i8* %arrayidx11, align 1
  %conv12 = zext i8 %10 to i32
  %shl = shl i32 %conv12, 8
  %11 = load i32, i32* %out_FB, align 4
  %or13 = or i32 %11, %shl
  store i32 %or13, i32* %out_FB, align 4
  %12 = load [256 x i8]*, [256 x i8]** %frame_buffer.addr, align 8
  %13 = load i32, i32* %i, align 4
  %conv14 = zext i32 %13 to i64
  %arrayidx15 = getelementptr inbounds [256 x i8], [256 x i8]* %12, i64 %conv14
  %14 = load i32, i32* %j, align 4
  %add16 = add i32 %14, 2
  %conv17 = zext i32 %add16 to i64
  %arrayidx18 = getelementptr inbounds [256 x i8], [256 x i8]* %arrayidx15, i64 0, i64 %conv17
  %15 = load i8, i8* %arrayidx18, align 1
  %conv19 = zext i8 %15 to i32
  %shl20 = shl i32 %conv19, 16
  %16 = load i32, i32* %out_FB, align 4
  %or21 = or i32 %16, %shl20
  store i32 %or21, i32* %out_FB, align 4
  %17 = load [256 x i8]*, [256 x i8]** %frame_buffer.addr, align 8
  %18 = load i32, i32* %i, align 4
  %conv22 = zext i32 %18 to i64
  %arrayidx23 = getelementptr inbounds [256 x i8], [256 x i8]* %17, i64 %conv22
  %19 = load i32, i32* %j, align 4
  %add24 = add i32 %19, 3
  %conv25 = zext i32 %add24 to i64
  %arrayidx26 = getelementptr inbounds [256 x i8], [256 x i8]* %arrayidx23, i64 0, i64 %conv25
  %20 = load i8, i8* %arrayidx26, align 1
  %conv27 = zext i8 %20 to i32
  %shl28 = shl i32 %conv27, 24
  %21 = load i32, i32* %out_FB, align 4
  %or29 = or i32 %21, %shl28
  store i32 %or29, i32* %out_FB, align 4
  %22 = load i32, i32* %out_FB, align 4
  %23 = load i32*, i32** %output.addr, align 8
  %24 = load i32, i32* %i, align 4
  %mul = mul i32 %24, 256
  %div = udiv i32 %mul, 4
  %25 = load i32, i32* %j, align 4
  %div30 = udiv i32 %25, 4
  %add31 = add i32 %div, %div30
  %idxprom = zext i32 %add31 to i64
  %arrayidx32 = getelementptr inbounds i32, i32* %23, i64 %idxprom
  store i32 %22, i32* %arrayidx32, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %26 = load i32, i32* %j, align 4
  %add33 = add i32 %26, 4
  store i32 %add33, i32* %j, align 4
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.inc34

for.inc34:                                        ; preds = %for.end
  %27 = load i32, i32* %i, align 4
  %inc = add i32 %27, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end35:                                        ; preds = %for.cond
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @_ZL17check_clockwise_111Triangle_2D(i56 %triangle_2d.coerce) #1 {
entry:
  %triangle_2d = alloca %struct.Triangle_2D, align 1
  %tmp = alloca i56, align 8
  %cw = alloca i32, align 4
  store i56 %triangle_2d.coerce, i56* %tmp, align 8
  %0 = bitcast i56* %tmp to i8*
  %1 = bitcast %struct.Triangle_2D* %triangle_2d to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %1, i8* align 8 %0, i64 7, i1 false)
  %x2 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 4
  %2 = load i8, i8* %x2, align 1
  %conv = zext i8 %2 to i32
  %x0 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 0
  %3 = load i8, i8* %x0, align 1
  %conv1 = zext i8 %3 to i32
  %sub = sub nsw i32 %conv, %conv1
  %y1 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 3
  %4 = load i8, i8* %y1, align 1
  %conv2 = zext i8 %4 to i32
  %y0 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 1
  %5 = load i8, i8* %y0, align 1
  %conv3 = zext i8 %5 to i32
  %sub4 = sub nsw i32 %conv2, %conv3
  %mul = mul nsw i32 %sub, %sub4
  %y2 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 5
  %6 = load i8, i8* %y2, align 1
  %conv5 = zext i8 %6 to i32
  %y06 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 1
  %7 = load i8, i8* %y06, align 1
  %conv7 = zext i8 %7 to i32
  %sub8 = sub nsw i32 %conv5, %conv7
  %x1 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 2
  %8 = load i8, i8* %x1, align 1
  %conv9 = zext i8 %8 to i32
  %x010 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 0
  %9 = load i8, i8* %x010, align 1
  %conv11 = zext i8 %9 to i32
  %sub12 = sub nsw i32 %conv9, %conv11
  %mul13 = mul nsw i32 %sub8, %sub12
  %sub14 = sub nsw i32 %mul, %mul13
  store i32 %sub14, i32* %cw, align 4
  %10 = load i32, i32* %cw, align 4
  ret i32 %10
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @_ZL20clockwise_vertices_1P11Triangle_2D(%struct.Triangle_2D* %triangle_2d) #1 {
entry:
  %triangle_2d.addr = alloca %struct.Triangle_2D*, align 8
  %tmp_x = alloca i8, align 1
  %tmp_y = alloca i8, align 1
  store %struct.Triangle_2D* %triangle_2d, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %0 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %x0 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %0, i32 0, i32 0
  %1 = load i8, i8* %x0, align 1
  store i8 %1, i8* %tmp_x, align 1
  %2 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %y0 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %2, i32 0, i32 1
  %3 = load i8, i8* %y0, align 1
  store i8 %3, i8* %tmp_y, align 1
  %4 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %x1 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %4, i32 0, i32 2
  %5 = load i8, i8* %x1, align 1
  %6 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %x01 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %6, i32 0, i32 0
  store i8 %5, i8* %x01, align 1
  %7 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %y1 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %7, i32 0, i32 3
  %8 = load i8, i8* %y1, align 1
  %9 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %y02 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %9, i32 0, i32 1
  store i8 %8, i8* %y02, align 1
  %10 = load i8, i8* %tmp_x, align 1
  %11 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %x13 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %11, i32 0, i32 2
  store i8 %10, i8* %x13, align 1
  %12 = load i8, i8* %tmp_y, align 1
  %13 = load %struct.Triangle_2D*, %struct.Triangle_2D** %triangle_2d.addr, align 8
  %y14 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %13, i32 0, i32 3
  store i8 %12, i8* %y14, align 1
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal zeroext i8 @_ZL10find_min_1hhh(i8 zeroext %in0, i8 zeroext %in1, i8 zeroext %in2) #1 {
entry:
  %retval = alloca i8, align 1
  %in0.addr = alloca i8, align 1
  %in1.addr = alloca i8, align 1
  %in2.addr = alloca i8, align 1
  store i8 %in0, i8* %in0.addr, align 1
  store i8 %in1, i8* %in1.addr, align 1
  store i8 %in2, i8* %in2.addr, align 1
  %0 = load i8, i8* %in0.addr, align 1
  %conv = zext i8 %0 to i32
  %1 = load i8, i8* %in1.addr, align 1
  %conv1 = zext i8 %1 to i32
  %cmp = icmp slt i32 %conv, %conv1
  br i1 %cmp, label %if.then, label %if.else6

if.then:                                          ; preds = %entry
  %2 = load i8, i8* %in0.addr, align 1
  %conv2 = zext i8 %2 to i32
  %3 = load i8, i8* %in2.addr, align 1
  %conv3 = zext i8 %3 to i32
  %cmp4 = icmp slt i32 %conv2, %conv3
  br i1 %cmp4, label %if.then5, label %if.else

if.then5:                                         ; preds = %if.then
  %4 = load i8, i8* %in0.addr, align 1
  store i8 %4, i8* %retval, align 1
  br label %return

if.else:                                          ; preds = %if.then
  %5 = load i8, i8* %in2.addr, align 1
  store i8 %5, i8* %retval, align 1
  br label %return

if.else6:                                         ; preds = %entry
  %6 = load i8, i8* %in1.addr, align 1
  %conv7 = zext i8 %6 to i32
  %7 = load i8, i8* %in2.addr, align 1
  %conv8 = zext i8 %7 to i32
  %cmp9 = icmp slt i32 %conv7, %conv8
  br i1 %cmp9, label %if.then10, label %if.else11

if.then10:                                        ; preds = %if.else6
  %8 = load i8, i8* %in1.addr, align 1
  store i8 %8, i8* %retval, align 1
  br label %return

if.else11:                                        ; preds = %if.else6
  %9 = load i8, i8* %in2.addr, align 1
  store i8 %9, i8* %retval, align 1
  br label %return

return:                                           ; preds = %if.else11, %if.then10, %if.else, %if.then5
  %10 = load i8, i8* %retval, align 1
  ret i8 %10
}

; Function Attrs: noinline nounwind optnone uwtable
define internal zeroext i8 @_ZL10find_max_1hhh(i8 zeroext %in0, i8 zeroext %in1, i8 zeroext %in2) #1 {
entry:
  %retval = alloca i8, align 1
  %in0.addr = alloca i8, align 1
  %in1.addr = alloca i8, align 1
  %in2.addr = alloca i8, align 1
  store i8 %in0, i8* %in0.addr, align 1
  store i8 %in1, i8* %in1.addr, align 1
  store i8 %in2, i8* %in2.addr, align 1
  %0 = load i8, i8* %in0.addr, align 1
  %conv = zext i8 %0 to i32
  %1 = load i8, i8* %in1.addr, align 1
  %conv1 = zext i8 %1 to i32
  %cmp = icmp sgt i32 %conv, %conv1
  br i1 %cmp, label %if.then, label %if.else6

if.then:                                          ; preds = %entry
  %2 = load i8, i8* %in0.addr, align 1
  %conv2 = zext i8 %2 to i32
  %3 = load i8, i8* %in2.addr, align 1
  %conv3 = zext i8 %3 to i32
  %cmp4 = icmp sgt i32 %conv2, %conv3
  br i1 %cmp4, label %if.then5, label %if.else

if.then5:                                         ; preds = %if.then
  %4 = load i8, i8* %in0.addr, align 1
  store i8 %4, i8* %retval, align 1
  br label %return

if.else:                                          ; preds = %if.then
  %5 = load i8, i8* %in2.addr, align 1
  store i8 %5, i8* %retval, align 1
  br label %return

if.else6:                                         ; preds = %entry
  %6 = load i8, i8* %in1.addr, align 1
  %conv7 = zext i8 %6 to i32
  %7 = load i8, i8* %in2.addr, align 1
  %conv8 = zext i8 %7 to i32
  %cmp9 = icmp sgt i32 %conv7, %conv8
  br i1 %cmp9, label %if.then10, label %if.else11

if.then10:                                        ; preds = %if.else6
  %8 = load i8, i8* %in1.addr, align 1
  store i8 %8, i8* %retval, align 1
  br label %return

if.else11:                                        ; preds = %if.else6
  %9 = load i8, i8* %in2.addr, align 1
  store i8 %9, i8* %retval, align 1
  br label %return

return:                                           ; preds = %if.else11, %if.then10, %if.else, %if.then5
  %10 = load i8, i8* %retval, align 1
  ret i8 %10
}

; Function Attrs: noinline nounwind optnone uwtable
define internal zeroext i8 @_ZL19pixel_in_triangle_1hh11Triangle_2D(i8 zeroext %x, i8 zeroext %y, i56 %triangle_2d.coerce) #1 {
entry:
  %triangle_2d = alloca %struct.Triangle_2D, align 1
  %tmp = alloca i56, align 8
  %x.addr = alloca i8, align 1
  %y.addr = alloca i8, align 1
  %pi0 = alloca i32, align 4
  %pi1 = alloca i32, align 4
  %pi2 = alloca i32, align 4
  store i56 %triangle_2d.coerce, i56* %tmp, align 8
  %0 = bitcast i56* %tmp to i8*
  %1 = bitcast %struct.Triangle_2D* %triangle_2d to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %1, i8* align 8 %0, i64 7, i1 false)
  store i8 %x, i8* %x.addr, align 1
  store i8 %y, i8* %y.addr, align 1
  %2 = load i8, i8* %x.addr, align 1
  %conv = zext i8 %2 to i32
  %x0 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 0
  %3 = load i8, i8* %x0, align 1
  %conv1 = zext i8 %3 to i32
  %sub = sub nsw i32 %conv, %conv1
  %y1 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 3
  %4 = load i8, i8* %y1, align 1
  %conv2 = zext i8 %4 to i32
  %y0 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 1
  %5 = load i8, i8* %y0, align 1
  %conv3 = zext i8 %5 to i32
  %sub4 = sub nsw i32 %conv2, %conv3
  %mul = mul nsw i32 %sub, %sub4
  %6 = load i8, i8* %y.addr, align 1
  %conv5 = zext i8 %6 to i32
  %y06 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 1
  %7 = load i8, i8* %y06, align 1
  %conv7 = zext i8 %7 to i32
  %sub8 = sub nsw i32 %conv5, %conv7
  %x1 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 2
  %8 = load i8, i8* %x1, align 1
  %conv9 = zext i8 %8 to i32
  %x010 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 0
  %9 = load i8, i8* %x010, align 1
  %conv11 = zext i8 %9 to i32
  %sub12 = sub nsw i32 %conv9, %conv11
  %mul13 = mul nsw i32 %sub8, %sub12
  %sub14 = sub nsw i32 %mul, %mul13
  store i32 %sub14, i32* %pi0, align 4
  %10 = load i8, i8* %x.addr, align 1
  %conv15 = zext i8 %10 to i32
  %x116 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 2
  %11 = load i8, i8* %x116, align 1
  %conv17 = zext i8 %11 to i32
  %sub18 = sub nsw i32 %conv15, %conv17
  %y2 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 5
  %12 = load i8, i8* %y2, align 1
  %conv19 = zext i8 %12 to i32
  %y120 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 3
  %13 = load i8, i8* %y120, align 1
  %conv21 = zext i8 %13 to i32
  %sub22 = sub nsw i32 %conv19, %conv21
  %mul23 = mul nsw i32 %sub18, %sub22
  %14 = load i8, i8* %y.addr, align 1
  %conv24 = zext i8 %14 to i32
  %y125 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 3
  %15 = load i8, i8* %y125, align 1
  %conv26 = zext i8 %15 to i32
  %sub27 = sub nsw i32 %conv24, %conv26
  %x2 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 4
  %16 = load i8, i8* %x2, align 1
  %conv28 = zext i8 %16 to i32
  %x129 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 2
  %17 = load i8, i8* %x129, align 1
  %conv30 = zext i8 %17 to i32
  %sub31 = sub nsw i32 %conv28, %conv30
  %mul32 = mul nsw i32 %sub27, %sub31
  %sub33 = sub nsw i32 %mul23, %mul32
  store i32 %sub33, i32* %pi1, align 4
  %18 = load i8, i8* %x.addr, align 1
  %conv34 = zext i8 %18 to i32
  %x235 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 4
  %19 = load i8, i8* %x235, align 1
  %conv36 = zext i8 %19 to i32
  %sub37 = sub nsw i32 %conv34, %conv36
  %y038 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 1
  %20 = load i8, i8* %y038, align 1
  %conv39 = zext i8 %20 to i32
  %y240 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 5
  %21 = load i8, i8* %y240, align 1
  %conv41 = zext i8 %21 to i32
  %sub42 = sub nsw i32 %conv39, %conv41
  %mul43 = mul nsw i32 %sub37, %sub42
  %22 = load i8, i8* %y.addr, align 1
  %conv44 = zext i8 %22 to i32
  %y245 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 5
  %23 = load i8, i8* %y245, align 1
  %conv46 = zext i8 %23 to i32
  %sub47 = sub nsw i32 %conv44, %conv46
  %x048 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 0
  %24 = load i8, i8* %x048, align 1
  %conv49 = zext i8 %24 to i32
  %x250 = getelementptr inbounds %struct.Triangle_2D, %struct.Triangle_2D* %triangle_2d, i32 0, i32 4
  %25 = load i8, i8* %x250, align 1
  %conv51 = zext i8 %25 to i32
  %sub52 = sub nsw i32 %conv49, %conv51
  %mul53 = mul nsw i32 %sub47, %sub52
  %sub54 = sub nsw i32 %mul43, %mul53
  store i32 %sub54, i32* %pi2, align 4
  %26 = load i32, i32* %pi0, align 4
  %cmp = icmp sge i32 %26, 0
  br i1 %cmp, label %land.lhs.true, label %land.end

land.lhs.true:                                    ; preds = %entry
  %27 = load i32, i32* %pi1, align 4
  %cmp55 = icmp sge i32 %27, 0
  br i1 %cmp55, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %land.lhs.true
  %28 = load i32, i32* %pi2, align 4
  %cmp56 = icmp sge i32 %28, 0
  br label %land.end

land.end:                                         ; preds = %land.rhs, %land.lhs.true, %entry
  %29 = phi i1 [ false, %land.lhs.true ], [ false, %entry ], [ %cmp56, %land.rhs ]
  %conv57 = zext i1 %29 to i8
  ret i8 %conv57
}

attributes #0 = { noinline optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { argmemonly nounwind willreturn }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"Ubuntu clang version 10.0.1-++20211003085942+ef32c611aa21-1~exp1~20211003090334.2"}
