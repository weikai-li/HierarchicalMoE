; ModuleID = 'knn.cpp'
source_filename = "knn.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@_ZZ8DigitRecPyS_PhE12training_set = internal global [72000 x i64] zeroinitializer, align 16
@_ZZ8DigitRecPyS_PhE8test_set = internal global [8000 x i64] zeroinitializer, align 16
@_ZZ8DigitRecPyS_PhE7results = internal global [2000 x i8] zeroinitializer, align 16

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z8DigitRecPyS_Ph(i64* %global_training_set, i64* %global_test_set, i8* %global_results) #0 {
entry:
  %global_training_set.addr = alloca i64*, align 8
  %global_test_set.addr = alloca i64*, align 8
  %global_results.addr = alloca i8*, align 8
  %knn_set = alloca [120 x i32], align 16
  %i = alloca i32, align 4
  %i3 = alloca i32, align 4
  %t = alloca i32, align 4
  %test_instance = alloca [4 x i64], align 16
  %i17 = alloca i32, align 4
  %i28 = alloca i32, align 4
  %i37 = alloca i32, align 4
  %j = alloca i32, align 4
  %training_instance = alloca [4 x i64], align 16
  %k = alloca i32, align 4
  %max_vote = alloca i8, align 1
  %i74 = alloca i32, align 4
  store i64* %global_training_set, i64** %global_training_set.addr, align 8
  store i64* %global_test_set, i64** %global_test_set.addr, align 8
  store i8* %global_results, i8** %global_results.addr, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 72000
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i64*, i64** %global_training_set.addr, align 8
  %2 = load i32, i32* %i, align 4
  %idxprom = sext i32 %2 to i64
  %arrayidx = getelementptr inbounds i64, i64* %1, i64 %idxprom
  %3 = load i64, i64* %arrayidx, align 8
  %4 = load i32, i32* %i, align 4
  %idxprom1 = sext i32 %4 to i64
  %arrayidx2 = getelementptr inbounds [72000 x i64], [72000 x i64]* @_ZZ8DigitRecPyS_PhE12training_set, i64 0, i64 %idxprom1
  store i64 %3, i64* %arrayidx2, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32, i32* %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  store i32 0, i32* %i3, align 4
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc11, %for.end
  %6 = load i32, i32* %i3, align 4
  %cmp5 = icmp slt i32 %6, 8000
  br i1 %cmp5, label %for.body6, label %for.end13

for.body6:                                        ; preds = %for.cond4
  %7 = load i64*, i64** %global_test_set.addr, align 8
  %8 = load i32, i32* %i3, align 4
  %idxprom7 = sext i32 %8 to i64
  %arrayidx8 = getelementptr inbounds i64, i64* %7, i64 %idxprom7
  %9 = load i64, i64* %arrayidx8, align 8
  %10 = load i32, i32* %i3, align 4
  %idxprom9 = sext i32 %10 to i64
  %arrayidx10 = getelementptr inbounds [8000 x i64], [8000 x i64]* @_ZZ8DigitRecPyS_PhE8test_set, i64 0, i64 %idxprom9
  store i64 %9, i64* %arrayidx10, align 8
  br label %for.inc11

for.inc11:                                        ; preds = %for.body6
  %11 = load i32, i32* %i3, align 4
  %inc12 = add nsw i32 %11, 1
  store i32 %inc12, i32* %i3, align 4
  br label %for.cond4

for.end13:                                        ; preds = %for.cond4
  br label %TEST_LOOP

TEST_LOOP:                                        ; preds = %for.end13
  store i32 0, i32* %t, align 4
  br label %for.cond14

for.cond14:                                       ; preds = %for.inc71, %TEST_LOOP
  %12 = load i32, i32* %t, align 4
  %cmp15 = icmp slt i32 %12, 2000
  br i1 %cmp15, label %for.body16, label %for.end73

for.body16:                                       ; preds = %for.cond14
  store i32 0, i32* %i17, align 4
  br label %for.cond18

for.cond18:                                       ; preds = %for.inc25, %for.body16
  %13 = load i32, i32* %i17, align 4
  %cmp19 = icmp slt i32 %13, 4
  br i1 %cmp19, label %for.body20, label %for.end27

for.body20:                                       ; preds = %for.cond18
  %14 = load i32, i32* %t, align 4
  %mul = mul nsw i32 %14, 4
  %15 = load i32, i32* %i17, align 4
  %add = add nsw i32 %mul, %15
  %idxprom21 = sext i32 %add to i64
  %arrayidx22 = getelementptr inbounds [8000 x i64], [8000 x i64]* @_ZZ8DigitRecPyS_PhE8test_set, i64 0, i64 %idxprom21
  %16 = load i64, i64* %arrayidx22, align 8
  %17 = load i32, i32* %i17, align 4
  %idxprom23 = sext i32 %17 to i64
  %arrayidx24 = getelementptr inbounds [4 x i64], [4 x i64]* %test_instance, i64 0, i64 %idxprom23
  store i64 %16, i64* %arrayidx24, align 8
  br label %for.inc25

for.inc25:                                        ; preds = %for.body20
  %18 = load i32, i32* %i17, align 4
  %inc26 = add nsw i32 %18, 1
  store i32 %inc26, i32* %i17, align 4
  br label %for.cond18

for.end27:                                        ; preds = %for.cond18
  br label %SET_KNN_SET

SET_KNN_SET:                                      ; preds = %for.end27
  store i32 0, i32* %i28, align 4
  br label %for.cond29

for.cond29:                                       ; preds = %for.inc34, %SET_KNN_SET
  %19 = load i32, i32* %i28, align 4
  %cmp30 = icmp slt i32 %19, 120
  br i1 %cmp30, label %for.body31, label %for.end36

for.body31:                                       ; preds = %for.cond29
  %20 = load i32, i32* %i28, align 4
  %idxprom32 = sext i32 %20 to i64
  %arrayidx33 = getelementptr inbounds [120 x i32], [120 x i32]* %knn_set, i64 0, i64 %idxprom32
  store i32 256, i32* %arrayidx33, align 4
  br label %for.inc34

for.inc34:                                        ; preds = %for.body31
  %21 = load i32, i32* %i28, align 4
  %inc35 = add nsw i32 %21, 1
  store i32 %inc35, i32* %i28, align 4
  br label %for.cond29

for.end36:                                        ; preds = %for.cond29
  br label %TRAINING_LOOP

TRAINING_LOOP:                                    ; preds = %for.end36
  store i32 0, i32* %i37, align 4
  br label %for.cond38

for.cond38:                                       ; preds = %for.inc65, %TRAINING_LOOP
  %22 = load i32, i32* %i37, align 4
  %cmp39 = icmp slt i32 %22, 450
  br i1 %cmp39, label %for.body40, label %for.end67

for.body40:                                       ; preds = %for.cond38
  br label %LANES

LANES:                                            ; preds = %for.body40
  store i32 0, i32* %j, align 4
  br label %for.cond41

for.cond41:                                       ; preds = %for.inc62, %LANES
  %23 = load i32, i32* %j, align 4
  %cmp42 = icmp slt i32 %23, 40
  br i1 %cmp42, label %for.body43, label %for.end64

for.body43:                                       ; preds = %for.cond41
  store i32 0, i32* %k, align 4
  br label %for.cond44

for.cond44:                                       ; preds = %for.inc55, %for.body43
  %24 = load i32, i32* %k, align 4
  %cmp45 = icmp slt i32 %24, 4
  br i1 %cmp45, label %for.body46, label %for.end57

for.body46:                                       ; preds = %for.cond44
  %25 = load i32, i32* %j, align 4
  %mul47 = mul nsw i32 %25, 18000
  %div = sdiv i32 %mul47, 40
  %26 = load i32, i32* %i37, align 4
  %add48 = add nsw i32 %div, %26
  %mul49 = mul nsw i32 %add48, 4
  %27 = load i32, i32* %k, align 4
  %add50 = add nsw i32 %mul49, %27
  %idxprom51 = sext i32 %add50 to i64
  %arrayidx52 = getelementptr inbounds [72000 x i64], [72000 x i64]* @_ZZ8DigitRecPyS_PhE12training_set, i64 0, i64 %idxprom51
  %28 = load i64, i64* %arrayidx52, align 8
  %29 = load i32, i32* %k, align 4
  %idxprom53 = sext i32 %29 to i64
  %arrayidx54 = getelementptr inbounds [4 x i64], [4 x i64]* %training_instance, i64 0, i64 %idxprom53
  store i64 %28, i64* %arrayidx54, align 8
  br label %for.inc55

for.inc55:                                        ; preds = %for.body46
  %30 = load i32, i32* %k, align 4
  %inc56 = add nsw i32 %30, 1
  store i32 %inc56, i32* %k, align 4
  br label %for.cond44

for.end57:                                        ; preds = %for.cond44
  %arraydecay = getelementptr inbounds [4 x i64], [4 x i64]* %test_instance, i64 0, i64 0
  %arraydecay58 = getelementptr inbounds [4 x i64], [4 x i64]* %training_instance, i64 0, i64 0
  %31 = load i32, i32* %j, align 4
  %mul59 = mul nsw i32 %31, 3
  %idxprom60 = sext i32 %mul59 to i64
  %arrayidx61 = getelementptr inbounds [120 x i32], [120 x i32]* %knn_set, i64 0, i64 %idxprom60
  call void @_ZL12update_knn_1PyS_Pi(i64* %arraydecay, i64* %arraydecay58, i32* %arrayidx61)
  br label %for.inc62

for.inc62:                                        ; preds = %for.end57
  %32 = load i32, i32* %j, align 4
  %inc63 = add nsw i32 %32, 1
  store i32 %inc63, i32* %j, align 4
  br label %for.cond41

for.end64:                                        ; preds = %for.cond41
  br label %for.inc65

for.inc65:                                        ; preds = %for.end64
  %33 = load i32, i32* %i37, align 4
  %inc66 = add nsw i32 %33, 1
  store i32 %inc66, i32* %i37, align 4
  br label %for.cond38

for.end67:                                        ; preds = %for.cond38
  %arraydecay68 = getelementptr inbounds [120 x i32], [120 x i32]* %knn_set, i64 0, i64 0
  %call = call zeroext i8 @_ZL10knn_vote_1Pi(i32* %arraydecay68)
  store i8 %call, i8* %max_vote, align 1
  %34 = load i8, i8* %max_vote, align 1
  %35 = load i32, i32* %t, align 4
  %idxprom69 = sext i32 %35 to i64
  %arrayidx70 = getelementptr inbounds [2000 x i8], [2000 x i8]* @_ZZ8DigitRecPyS_PhE7results, i64 0, i64 %idxprom69
  store i8 %34, i8* %arrayidx70, align 1
  br label %for.inc71

for.inc71:                                        ; preds = %for.end67
  %36 = load i32, i32* %t, align 4
  %inc72 = add nsw i32 %36, 1
  store i32 %inc72, i32* %t, align 4
  br label %for.cond14

for.end73:                                        ; preds = %for.cond14
  store i32 0, i32* %i74, align 4
  br label %for.cond75

for.cond75:                                       ; preds = %for.inc82, %for.end73
  %37 = load i32, i32* %i74, align 4
  %cmp76 = icmp slt i32 %37, 2000
  br i1 %cmp76, label %for.body77, label %for.end84

for.body77:                                       ; preds = %for.cond75
  %38 = load i32, i32* %i74, align 4
  %idxprom78 = sext i32 %38 to i64
  %arrayidx79 = getelementptr inbounds [2000 x i8], [2000 x i8]* @_ZZ8DigitRecPyS_PhE7results, i64 0, i64 %idxprom78
  %39 = load i8, i8* %arrayidx79, align 1
  %40 = load i8*, i8** %global_results.addr, align 8
  %41 = load i32, i32* %i74, align 4
  %idxprom80 = sext i32 %41 to i64
  %arrayidx81 = getelementptr inbounds i8, i8* %40, i64 %idxprom80
  store i8 %39, i8* %arrayidx81, align 1
  br label %for.inc82

for.inc82:                                        ; preds = %for.body77
  %42 = load i32, i32* %i74, align 4
  %inc83 = add nsw i32 %42, 1
  store i32 %inc83, i32* %i74, align 4
  br label %for.cond75

for.end84:                                        ; preds = %for.cond75
  ret void
}

; Function Attrs: noinline optnone uwtable
define internal void @_ZL12update_knn_1PyS_Pi(i64* %test_inst, i64* %train_inst, i32* %min_distances) #0 {
entry:
  %test_inst.addr = alloca i64*, align 8
  %train_inst.addr = alloca i64*, align 8
  %min_distances.addr = alloca i32*, align 8
  %dist = alloca i32, align 4
  %i = alloca i32, align 4
  %diff = alloca i64, align 8
  %max_dist = alloca i32, align 4
  %max_dist_id = alloca i32, align 4
  %k = alloca i32, align 4
  %k3 = alloca i32, align 4
  store i64* %test_inst, i64** %test_inst.addr, align 8
  store i64* %train_inst, i64** %train_inst.addr, align 8
  store i32* %min_distances, i32** %min_distances.addr, align 8
  store i32 0, i32* %dist, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 4
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i64*, i64** %test_inst.addr, align 8
  %2 = load i32, i32* %i, align 4
  %idxprom = sext i32 %2 to i64
  %arrayidx = getelementptr inbounds i64, i64* %1, i64 %idxprom
  %3 = load i64, i64* %arrayidx, align 8
  %4 = load i64*, i64** %train_inst.addr, align 8
  %5 = load i32, i32* %i, align 4
  %idxprom1 = sext i32 %5 to i64
  %arrayidx2 = getelementptr inbounds i64, i64* %4, i64 %idxprom1
  %6 = load i64, i64* %arrayidx2, align 8
  %xor = xor i64 %3, %6
  store i64 %xor, i64* %diff, align 8
  %7 = load i64, i64* %diff, align 8
  %call = call i32 @_ZL10popcount_1y(i64 %7)
  %8 = load i32, i32* %dist, align 4
  %add = add nsw i32 %8, %call
  store i32 %add, i32* %dist, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %9 = load i32, i32* %i, align 4
  %inc = add nsw i32 %9, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  store i32 0, i32* %max_dist, align 4
  store i32 4, i32* %max_dist_id, align 4
  store i32 0, i32* %k, align 4
  br label %FIND_MAX_DIST

FIND_MAX_DIST:                                    ; preds = %for.end
  store i32 0, i32* %k3, align 4
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc12, %FIND_MAX_DIST
  %10 = load i32, i32* %k3, align 4
  %cmp5 = icmp slt i32 %10, 3
  br i1 %cmp5, label %for.body6, label %for.end14

for.body6:                                        ; preds = %for.cond4
  %11 = load i32*, i32** %min_distances.addr, align 8
  %12 = load i32, i32* %k3, align 4
  %idxprom7 = sext i32 %12 to i64
  %arrayidx8 = getelementptr inbounds i32, i32* %11, i64 %idxprom7
  %13 = load i32, i32* %arrayidx8, align 4
  %14 = load i32, i32* %max_dist, align 4
  %cmp9 = icmp sgt i32 %13, %14
  br i1 %cmp9, label %if.then, label %if.end

if.then:                                          ; preds = %for.body6
  %15 = load i32*, i32** %min_distances.addr, align 8
  %16 = load i32, i32* %k3, align 4
  %idxprom10 = sext i32 %16 to i64
  %arrayidx11 = getelementptr inbounds i32, i32* %15, i64 %idxprom10
  %17 = load i32, i32* %arrayidx11, align 4
  store i32 %17, i32* %max_dist, align 4
  %18 = load i32, i32* %k3, align 4
  store i32 %18, i32* %max_dist_id, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body6
  br label %for.inc12

for.inc12:                                        ; preds = %if.end
  %19 = load i32, i32* %k3, align 4
  %inc13 = add nsw i32 %19, 1
  store i32 %inc13, i32* %k3, align 4
  br label %for.cond4

for.end14:                                        ; preds = %for.cond4
  %20 = load i32, i32* %dist, align 4
  %21 = load i32, i32* %max_dist, align 4
  %cmp15 = icmp slt i32 %20, %21
  br i1 %cmp15, label %if.then16, label %if.end19

if.then16:                                        ; preds = %for.end14
  %22 = load i32, i32* %dist, align 4
  %23 = load i32*, i32** %min_distances.addr, align 8
  %24 = load i32, i32* %max_dist_id, align 4
  %idxprom17 = sext i32 %24 to i64
  %arrayidx18 = getelementptr inbounds i32, i32* %23, i64 %idxprom17
  store i32 %22, i32* %arrayidx18, align 4
  br label %if.end19

if.end19:                                         ; preds = %if.then16, %for.end14
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal zeroext i8 @_ZL10knn_vote_1Pi(i32* %knn_set) #1 {
entry:
  %knn_set.addr = alloca i32*, align 8
  %min_distance_list = alloca [3 x i32], align 4
  %label_list = alloca [3 x i32], align 4
  %vote_list = alloca [10 x i32], align 16
  %pos = alloca i32, align 4
  %i = alloca i32, align 4
  %i3 = alloca i32, align 4
  %i12 = alloca i32, align 4
  %j = alloca i32, align 4
  %r = alloca i32, align 4
  %r31 = alloca i32, align 4
  %_in_r = alloca i32, align 4
  %i73 = alloca i32, align 4
  %max_vote = alloca i8, align 1
  %i85 = alloca i32, align 4
  store i32* %knn_set, i32** %knn_set.addr, align 8
  store i32 1000, i32* %pos, align 4
  br label %INIT_1

INIT_1:                                           ; preds = %entry
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %INIT_1
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 3
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4
  %idxprom = sext i32 %1 to i64
  %arrayidx = getelementptr inbounds [3 x i32], [3 x i32]* %min_distance_list, i64 0, i64 %idxprom
  store i32 256, i32* %arrayidx, align 4
  %2 = load i32, i32* %i, align 4
  %idxprom1 = sext i32 %2 to i64
  %arrayidx2 = getelementptr inbounds [3 x i32], [3 x i32]* %label_list, i64 0, i64 %idxprom1
  store i32 9, i32* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %3 = load i32, i32* %i, align 4
  %inc = add nsw i32 %3, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  br label %INIT_2

INIT_2:                                           ; preds = %for.end
  store i32 0, i32* %i3, align 4
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc9, %INIT_2
  %4 = load i32, i32* %i3, align 4
  %cmp5 = icmp slt i32 %4, 10
  br i1 %cmp5, label %for.body6, label %for.end11

for.body6:                                        ; preds = %for.cond4
  %5 = load i32, i32* %i3, align 4
  %idxprom7 = sext i32 %5 to i64
  %arrayidx8 = getelementptr inbounds [10 x i32], [10 x i32]* %vote_list, i64 0, i64 %idxprom7
  store i32 0, i32* %arrayidx8, align 4
  br label %for.inc9

for.inc9:                                         ; preds = %for.body6
  %6 = load i32, i32* %i3, align 4
  %inc10 = add nsw i32 %6, 1
  store i32 %inc10, i32* %i3, align 4
  br label %for.cond4

for.end11:                                        ; preds = %for.cond4
  br label %LANES

LANES:                                            ; preds = %for.end11
  store i32 0, i32* %i12, align 4
  br label %for.cond13

for.cond13:                                       ; preds = %for.inc70, %LANES
  %7 = load i32, i32* %i12, align 4
  %cmp14 = icmp slt i32 %7, 40
  br i1 %cmp14, label %for.body15, label %for.end72

for.body15:                                       ; preds = %for.cond13
  br label %INSERTION_SORT_OUTER

INSERTION_SORT_OUTER:                             ; preds = %for.body15
  store i32 0, i32* %j, align 4
  br label %for.cond16

for.cond16:                                       ; preds = %for.inc67, %INSERTION_SORT_OUTER
  %8 = load i32, i32* %j, align 4
  %cmp17 = icmp slt i32 %8, 3
  br i1 %cmp17, label %for.body18, label %for.end69

for.body18:                                       ; preds = %for.cond16
  store i32 1000, i32* %pos, align 4
  br label %INSERTION_SORT_INNER

INSERTION_SORT_INNER:                             ; preds = %for.body18
  store i32 0, i32* %r, align 4
  br label %for.cond19

for.cond19:                                       ; preds = %for.inc28, %INSERTION_SORT_INNER
  %9 = load i32, i32* %r, align 4
  %cmp20 = icmp slt i32 %9, 3
  br i1 %cmp20, label %for.body21, label %for.end30

for.body21:                                       ; preds = %for.cond19
  %10 = load i32*, i32** %knn_set.addr, align 8
  %11 = load i32, i32* %i12, align 4
  %mul = mul nsw i32 %11, 3
  %12 = load i32, i32* %j, align 4
  %add = add nsw i32 %mul, %12
  %idxprom22 = sext i32 %add to i64
  %arrayidx23 = getelementptr inbounds i32, i32* %10, i64 %idxprom22
  %13 = load i32, i32* %arrayidx23, align 4
  %14 = load i32, i32* %r, align 4
  %idxprom24 = sext i32 %14 to i64
  %arrayidx25 = getelementptr inbounds [3 x i32], [3 x i32]* %min_distance_list, i64 0, i64 %idxprom24
  %15 = load i32, i32* %arrayidx25, align 4
  %cmp26 = icmp slt i32 %13, %15
  br i1 %cmp26, label %land.lhs.true, label %cond.false

land.lhs.true:                                    ; preds = %for.body21
  %16 = load i32, i32* %pos, align 4
  %cmp27 = icmp sgt i32 %16, 3
  br i1 %cmp27, label %cond.true, label %cond.false

cond.true:                                        ; preds = %land.lhs.true
  %17 = load i32, i32* %r, align 4
  br label %cond.end

cond.false:                                       ; preds = %land.lhs.true, %for.body21
  %18 = load i32, i32* %pos, align 4
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %17, %cond.true ], [ %18, %cond.false ]
  store i32 %cond, i32* %pos, align 4
  br label %for.inc28

for.inc28:                                        ; preds = %cond.end
  %19 = load i32, i32* %r, align 4
  %inc29 = add nsw i32 %19, 1
  store i32 %inc29, i32* %r, align 4
  br label %for.cond19

for.end30:                                        ; preds = %for.cond19
  br label %INSERT

INSERT:                                           ; preds = %for.end30
  store i32 0, i32* %r31, align 4
  br label %for.cond32

for.cond32:                                       ; preds = %for.inc64, %INSERT
  %20 = load i32, i32* %r31, align 4
  %cmp33 = icmp sle i32 %20, 2
  br i1 %cmp33, label %for.body34, label %for.end66

for.body34:                                       ; preds = %for.cond32
  %21 = load i32, i32* %r31, align 4
  %mul35 = mul nsw i32 -1, %21
  %add36 = add nsw i32 3, %mul35
  store i32 %add36, i32* %_in_r, align 4
  %22 = load i32, i32* %_in_r, align 4
  %sub = sub nsw i32 %22, 1
  %23 = load i32, i32* %pos, align 4
  %cmp37 = icmp sgt i32 %sub, %23
  br i1 %cmp37, label %if.then, label %if.else

if.then:                                          ; preds = %for.body34
  %24 = load i32, i32* %_in_r, align 4
  %sub38 = sub nsw i32 %24, 2
  %idxprom39 = sext i32 %sub38 to i64
  %arrayidx40 = getelementptr inbounds [3 x i32], [3 x i32]* %min_distance_list, i64 0, i64 %idxprom39
  %25 = load i32, i32* %arrayidx40, align 4
  %26 = load i32, i32* %_in_r, align 4
  %sub41 = sub nsw i32 %26, 1
  %idxprom42 = sext i32 %sub41 to i64
  %arrayidx43 = getelementptr inbounds [3 x i32], [3 x i32]* %min_distance_list, i64 0, i64 %idxprom42
  store i32 %25, i32* %arrayidx43, align 4
  %27 = load i32, i32* %_in_r, align 4
  %sub44 = sub nsw i32 %27, 2
  %idxprom45 = sext i32 %sub44 to i64
  %arrayidx46 = getelementptr inbounds [3 x i32], [3 x i32]* %label_list, i64 0, i64 %idxprom45
  %28 = load i32, i32* %arrayidx46, align 4
  %29 = load i32, i32* %_in_r, align 4
  %sub47 = sub nsw i32 %29, 1
  %idxprom48 = sext i32 %sub47 to i64
  %arrayidx49 = getelementptr inbounds [3 x i32], [3 x i32]* %label_list, i64 0, i64 %idxprom48
  store i32 %28, i32* %arrayidx49, align 4
  br label %if.end63

if.else:                                          ; preds = %for.body34
  %30 = load i32, i32* %_in_r, align 4
  %sub50 = sub nsw i32 %30, 1
  %31 = load i32, i32* %pos, align 4
  %cmp51 = icmp eq i32 %sub50, %31
  br i1 %cmp51, label %if.then52, label %if.end

if.then52:                                        ; preds = %if.else
  %32 = load i32*, i32** %knn_set.addr, align 8
  %33 = load i32, i32* %i12, align 4
  %mul53 = mul nsw i32 %33, 3
  %34 = load i32, i32* %j, align 4
  %add54 = add nsw i32 %mul53, %34
  %idxprom55 = sext i32 %add54 to i64
  %arrayidx56 = getelementptr inbounds i32, i32* %32, i64 %idxprom55
  %35 = load i32, i32* %arrayidx56, align 4
  %36 = load i32, i32* %_in_r, align 4
  %sub57 = sub nsw i32 %36, 1
  %idxprom58 = sext i32 %sub57 to i64
  %arrayidx59 = getelementptr inbounds [3 x i32], [3 x i32]* %min_distance_list, i64 0, i64 %idxprom58
  store i32 %35, i32* %arrayidx59, align 4
  %37 = load i32, i32* %i12, align 4
  %div = sdiv i32 %37, 4
  %38 = load i32, i32* %_in_r, align 4
  %sub60 = sub nsw i32 %38, 1
  %idxprom61 = sext i32 %sub60 to i64
  %arrayidx62 = getelementptr inbounds [3 x i32], [3 x i32]* %label_list, i64 0, i64 %idxprom61
  store i32 %div, i32* %arrayidx62, align 4
  br label %if.end

if.end:                                           ; preds = %if.then52, %if.else
  br label %if.end63

if.end63:                                         ; preds = %if.end, %if.then
  br label %for.inc64

for.inc64:                                        ; preds = %if.end63
  %39 = load i32, i32* %r31, align 4
  %inc65 = add nsw i32 %39, 1
  store i32 %inc65, i32* %r31, align 4
  br label %for.cond32

for.end66:                                        ; preds = %for.cond32
  br label %for.inc67

for.inc67:                                        ; preds = %for.end66
  %40 = load i32, i32* %j, align 4
  %inc68 = add nsw i32 %40, 1
  store i32 %inc68, i32* %j, align 4
  br label %for.cond16

for.end69:                                        ; preds = %for.cond16
  br label %for.inc70

for.inc70:                                        ; preds = %for.end69
  %41 = load i32, i32* %i12, align 4
  %inc71 = add nsw i32 %41, 1
  store i32 %inc71, i32* %i12, align 4
  br label %for.cond13

for.end72:                                        ; preds = %for.cond13
  br label %INCREMENT

INCREMENT:                                        ; preds = %for.end72
  store i32 0, i32* %i73, align 4
  br label %for.cond74

for.cond74:                                       ; preds = %for.inc82, %INCREMENT
  %42 = load i32, i32* %i73, align 4
  %cmp75 = icmp slt i32 %42, 3
  br i1 %cmp75, label %for.body76, label %for.end84

for.body76:                                       ; preds = %for.cond74
  %43 = load i32, i32* %i73, align 4
  %idxprom77 = sext i32 %43 to i64
  %arrayidx78 = getelementptr inbounds [3 x i32], [3 x i32]* %label_list, i64 0, i64 %idxprom77
  %44 = load i32, i32* %arrayidx78, align 4
  %idxprom79 = sext i32 %44 to i64
  %arrayidx80 = getelementptr inbounds [10 x i32], [10 x i32]* %vote_list, i64 0, i64 %idxprom79
  %45 = load i32, i32* %arrayidx80, align 4
  %add81 = add nsw i32 %45, 1
  store i32 %add81, i32* %arrayidx80, align 4
  br label %for.inc82

for.inc82:                                        ; preds = %for.body76
  %46 = load i32, i32* %i73, align 4
  %inc83 = add nsw i32 %46, 1
  store i32 %inc83, i32* %i73, align 4
  br label %for.cond74

for.end84:                                        ; preds = %for.cond74
  store i8 0, i8* %max_vote, align 1
  br label %VOTE

VOTE:                                             ; preds = %for.end84
  store i32 0, i32* %i85, align 4
  br label %for.cond86

for.cond86:                                       ; preds = %for.inc96, %VOTE
  %47 = load i32, i32* %i85, align 4
  %cmp87 = icmp slt i32 %47, 10
  br i1 %cmp87, label %for.body88, label %for.end98

for.body88:                                       ; preds = %for.cond86
  %48 = load i32, i32* %i85, align 4
  %idxprom89 = sext i32 %48 to i64
  %arrayidx90 = getelementptr inbounds [10 x i32], [10 x i32]* %vote_list, i64 0, i64 %idxprom89
  %49 = load i32, i32* %arrayidx90, align 4
  %50 = load i8, i8* %max_vote, align 1
  %idxprom91 = zext i8 %50 to i64
  %arrayidx92 = getelementptr inbounds [10 x i32], [10 x i32]* %vote_list, i64 0, i64 %idxprom91
  %51 = load i32, i32* %arrayidx92, align 4
  %cmp93 = icmp sge i32 %49, %51
  br i1 %cmp93, label %if.then94, label %if.end95

if.then94:                                        ; preds = %for.body88
  %52 = load i32, i32* %i85, align 4
  %conv = trunc i32 %52 to i8
  store i8 %conv, i8* %max_vote, align 1
  br label %if.end95

if.end95:                                         ; preds = %if.then94, %for.body88
  br label %for.inc96

for.inc96:                                        ; preds = %if.end95
  %53 = load i32, i32* %i85, align 4
  %inc97 = add nsw i32 %53, 1
  store i32 %inc97, i32* %i85, align 4
  br label %for.cond86

for.end98:                                        ; preds = %for.cond86
  %54 = load i8, i8* %max_vote, align 1
  ret i8 %54
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @_ZL10popcount_1y(i64 %x) #1 {
entry:
  %x.addr = alloca i64, align 8
  store i64 %x, i64* %x.addr, align 8
  %0 = load i64, i64* %x.addr, align 8
  %shr = lshr i64 %0, 1
  %and = and i64 %shr, 6148914691236517205
  %1 = load i64, i64* %x.addr, align 8
  %sub = sub i64 %1, %and
  store i64 %sub, i64* %x.addr, align 8
  %2 = load i64, i64* %x.addr, align 8
  %and1 = and i64 %2, 3689348814741910323
  %3 = load i64, i64* %x.addr, align 8
  %shr2 = lshr i64 %3, 2
  %and3 = and i64 %shr2, 3689348814741910323
  %add = add i64 %and1, %and3
  store i64 %add, i64* %x.addr, align 8
  %4 = load i64, i64* %x.addr, align 8
  %5 = load i64, i64* %x.addr, align 8
  %shr4 = lshr i64 %5, 4
  %add5 = add i64 %4, %shr4
  %and6 = and i64 %add5, 1085102592571150095
  store i64 %and6, i64* %x.addr, align 8
  %6 = load i64, i64* %x.addr, align 8
  %shr7 = lshr i64 %6, 8
  %7 = load i64, i64* %x.addr, align 8
  %add8 = add i64 %7, %shr7
  store i64 %add8, i64* %x.addr, align 8
  %8 = load i64, i64* %x.addr, align 8
  %shr9 = lshr i64 %8, 16
  %9 = load i64, i64* %x.addr, align 8
  %add10 = add i64 %9, %shr9
  store i64 %add10, i64* %x.addr, align 8
  %10 = load i64, i64* %x.addr, align 8
  %shr11 = lshr i64 %10, 32
  %11 = load i64, i64* %x.addr, align 8
  %add12 = add i64 %11, %shr11
  store i64 %add12, i64* %x.addr, align 8
  %12 = load i64, i64* %x.addr, align 8
  %and13 = and i64 %12, 127
  %conv = trunc i64 %and13 to i32
  ret i32 %conv
}

attributes #0 = { noinline optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"Ubuntu clang version 10.0.1-++20211003085942+ef32c611aa21-1~exp1~20211003090334.2"}
