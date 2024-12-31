; ModuleID = 'att-3mm-fuse.cpp'
source_filename = "att-3mm-fuse.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_Z9sub_attenPA768_fPA64_fS2_S2_S2_PA512_f([768 x float]* %data, [64 x float]* %w1, [64 x float]* %w2, [64 x float]* %Q, [64 x float]* %K, [512 x float]* %atten) #0 {
entry:
  %data.addr = alloca [768 x float]*, align 8
  %w1.addr = alloca [64 x float]*, align 8
  %w2.addr = alloca [64 x float]*, align 8
  %Q.addr = alloca [64 x float]*, align 8
  %K.addr = alloca [64 x float]*, align 8
  %atten.addr = alloca [512 x float]*, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  %j20 = alloca i32, align 4
  %k24 = alloca i32, align 4
  %j48 = alloca i32, align 4
  %k52 = alloca i32, align 4
  store [768 x float]* %data, [768 x float]** %data.addr, align 8
  store [64 x float]* %w1, [64 x float]** %w1.addr, align 8
  store [64 x float]* %w2, [64 x float]** %w2.addr, align 8
  store [64 x float]* %Q, [64 x float]** %Q.addr, align 8
  store [64 x float]* %K, [64 x float]** %K.addr, align 8
  store [512 x float]* %atten, [512 x float]** %atten.addr, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc76, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 512
  br i1 %cmp, label %for.body, label %for.end78

for.body:                                         ; preds = %for.cond
  store i32 0, i32* %j, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc17, %for.body
  %1 = load i32, i32* %j, align 4
  %cmp2 = icmp slt i32 %1, 64
  br i1 %cmp2, label %for.body3, label %for.end19

for.body3:                                        ; preds = %for.cond1
  store i32 0, i32* %k, align 4
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc, %for.body3
  %2 = load i32, i32* %k, align 4
  %cmp5 = icmp slt i32 %2, 768
  br i1 %cmp5, label %for.body6, label %for.end

for.body6:                                        ; preds = %for.cond4
  %3 = load [768 x float]*, [768 x float]** %data.addr, align 8
  %4 = load i32, i32* %i, align 4
  %idxprom = sext i32 %4 to i64
  %arrayidx = getelementptr inbounds [768 x float], [768 x float]* %3, i64 %idxprom
  %5 = load i32, i32* %k, align 4
  %idxprom7 = sext i32 %5 to i64
  %arrayidx8 = getelementptr inbounds [768 x float], [768 x float]* %arrayidx, i64 0, i64 %idxprom7
  %6 = load float, float* %arrayidx8, align 4
  %7 = load [64 x float]*, [64 x float]** %w1.addr, align 8
  %8 = load i32, i32* %k, align 4
  %idxprom9 = sext i32 %8 to i64
  %arrayidx10 = getelementptr inbounds [64 x float], [64 x float]* %7, i64 %idxprom9
  %9 = load i32, i32* %j, align 4
  %idxprom11 = sext i32 %9 to i64
  %arrayidx12 = getelementptr inbounds [64 x float], [64 x float]* %arrayidx10, i64 0, i64 %idxprom11
  %10 = load float, float* %arrayidx12, align 4
  %mul = fmul float %6, %10
  %11 = load [64 x float]*, [64 x float]** %Q.addr, align 8
  %12 = load i32, i32* %i, align 4
  %idxprom13 = sext i32 %12 to i64
  %arrayidx14 = getelementptr inbounds [64 x float], [64 x float]* %11, i64 %idxprom13
  %13 = load i32, i32* %j, align 4
  %idxprom15 = sext i32 %13 to i64
  %arrayidx16 = getelementptr inbounds [64 x float], [64 x float]* %arrayidx14, i64 0, i64 %idxprom15
  %14 = load float, float* %arrayidx16, align 4
  %add = fadd float %14, %mul
  store float %add, float* %arrayidx16, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body6
  %15 = load i32, i32* %k, align 4
  %inc = add nsw i32 %15, 1
  store i32 %inc, i32* %k, align 4
  br label %for.cond4

for.end:                                          ; preds = %for.cond4
  br label %for.inc17

for.inc17:                                        ; preds = %for.end
  %16 = load i32, i32* %j, align 4
  %inc18 = add nsw i32 %16, 1
  store i32 %inc18, i32* %j, align 4
  br label %for.cond1

for.end19:                                        ; preds = %for.cond1
  store i32 0, i32* %j20, align 4
  br label %for.cond21

for.cond21:                                       ; preds = %for.inc45, %for.end19
  %17 = load i32, i32* %j20, align 4
  %cmp22 = icmp slt i32 %17, 64
  br i1 %cmp22, label %for.body23, label %for.end47

for.body23:                                       ; preds = %for.cond21
  store i32 0, i32* %k24, align 4
  br label %for.cond25

for.cond25:                                       ; preds = %for.inc42, %for.body23
  %18 = load i32, i32* %k24, align 4
  %cmp26 = icmp slt i32 %18, 768
  br i1 %cmp26, label %for.body27, label %for.end44

for.body27:                                       ; preds = %for.cond25
  %19 = load [768 x float]*, [768 x float]** %data.addr, align 8
  %20 = load i32, i32* %i, align 4
  %idxprom28 = sext i32 %20 to i64
  %arrayidx29 = getelementptr inbounds [768 x float], [768 x float]* %19, i64 %idxprom28
  %21 = load i32, i32* %k24, align 4
  %idxprom30 = sext i32 %21 to i64
  %arrayidx31 = getelementptr inbounds [768 x float], [768 x float]* %arrayidx29, i64 0, i64 %idxprom30
  %22 = load float, float* %arrayidx31, align 4
  %23 = load [64 x float]*, [64 x float]** %w2.addr, align 8
  %24 = load i32, i32* %k24, align 4
  %idxprom32 = sext i32 %24 to i64
  %arrayidx33 = getelementptr inbounds [64 x float], [64 x float]* %23, i64 %idxprom32
  %25 = load i32, i32* %j20, align 4
  %idxprom34 = sext i32 %25 to i64
  %arrayidx35 = getelementptr inbounds [64 x float], [64 x float]* %arrayidx33, i64 0, i64 %idxprom34
  %26 = load float, float* %arrayidx35, align 4
  %mul36 = fmul float %22, %26
  %27 = load [64 x float]*, [64 x float]** %K.addr, align 8
  %28 = load i32, i32* %i, align 4
  %idxprom37 = sext i32 %28 to i64
  %arrayidx38 = getelementptr inbounds [64 x float], [64 x float]* %27, i64 %idxprom37
  %29 = load i32, i32* %j20, align 4
  %idxprom39 = sext i32 %29 to i64
  %arrayidx40 = getelementptr inbounds [64 x float], [64 x float]* %arrayidx38, i64 0, i64 %idxprom39
  %30 = load float, float* %arrayidx40, align 4
  %add41 = fadd float %30, %mul36
  store float %add41, float* %arrayidx40, align 4
  br label %for.inc42

for.inc42:                                        ; preds = %for.body27
  %31 = load i32, i32* %k24, align 4
  %inc43 = add nsw i32 %31, 1
  store i32 %inc43, i32* %k24, align 4
  br label %for.cond25

for.end44:                                        ; preds = %for.cond25
  br label %for.inc45

for.inc45:                                        ; preds = %for.end44
  %32 = load i32, i32* %j20, align 4
  %inc46 = add nsw i32 %32, 1
  store i32 %inc46, i32* %j20, align 4
  br label %for.cond21

for.end47:                                        ; preds = %for.cond21
  store i32 0, i32* %j48, align 4
  br label %for.cond49

for.cond49:                                       ; preds = %for.inc73, %for.end47
  %33 = load i32, i32* %j48, align 4
  %cmp50 = icmp slt i32 %33, 512
  br i1 %cmp50, label %for.body51, label %for.end75

for.body51:                                       ; preds = %for.cond49
  store i32 0, i32* %k52, align 4
  br label %for.cond53

for.cond53:                                       ; preds = %for.inc70, %for.body51
  %34 = load i32, i32* %k52, align 4
  %cmp54 = icmp slt i32 %34, 64
  br i1 %cmp54, label %for.body55, label %for.end72

for.body55:                                       ; preds = %for.cond53
  %35 = load [64 x float]*, [64 x float]** %Q.addr, align 8
  %36 = load i32, i32* %i, align 4
  %idxprom56 = sext i32 %36 to i64
  %arrayidx57 = getelementptr inbounds [64 x float], [64 x float]* %35, i64 %idxprom56
  %37 = load i32, i32* %k52, align 4
  %idxprom58 = sext i32 %37 to i64
  %arrayidx59 = getelementptr inbounds [64 x float], [64 x float]* %arrayidx57, i64 0, i64 %idxprom58
  %38 = load float, float* %arrayidx59, align 4
  %39 = load [64 x float]*, [64 x float]** %K.addr, align 8
  %40 = load i32, i32* %j48, align 4
  %idxprom60 = sext i32 %40 to i64
  %arrayidx61 = getelementptr inbounds [64 x float], [64 x float]* %39, i64 %idxprom60
  %41 = load i32, i32* %k52, align 4
  %idxprom62 = sext i32 %41 to i64
  %arrayidx63 = getelementptr inbounds [64 x float], [64 x float]* %arrayidx61, i64 0, i64 %idxprom62
  %42 = load float, float* %arrayidx63, align 4
  %mul64 = fmul float %38, %42
  %43 = load [512 x float]*, [512 x float]** %atten.addr, align 8
  %44 = load i32, i32* %i, align 4
  %idxprom65 = sext i32 %44 to i64
  %arrayidx66 = getelementptr inbounds [512 x float], [512 x float]* %43, i64 %idxprom65
  %45 = load i32, i32* %j48, align 4
  %idxprom67 = sext i32 %45 to i64
  %arrayidx68 = getelementptr inbounds [512 x float], [512 x float]* %arrayidx66, i64 0, i64 %idxprom67
  %46 = load float, float* %arrayidx68, align 4
  %add69 = fadd float %46, %mul64
  store float %add69, float* %arrayidx68, align 4
  br label %for.inc70

for.inc70:                                        ; preds = %for.body55
  %47 = load i32, i32* %k52, align 4
  %inc71 = add nsw i32 %47, 1
  store i32 %inc71, i32* %k52, align 4
  br label %for.cond53

for.end72:                                        ; preds = %for.cond53
  br label %for.inc73

for.inc73:                                        ; preds = %for.end72
  %48 = load i32, i32* %j48, align 4
  %inc74 = add nsw i32 %48, 1
  store i32 %inc74, i32* %j48, align 4
  br label %for.cond49

for.end75:                                        ; preds = %for.cond49
  br label %for.inc76

for.inc76:                                        ; preds = %for.end75
  %49 = load i32, i32* %i, align 4
  %inc77 = add nsw i32 %49, 1
  store i32 %inc77, i32* %i, align 4
  br label %for.cond

for.end78:                                        ; preds = %for.cond
  ret void
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"Ubuntu clang version 10.0.1-++20211003085942+ef32c611aa21-1~exp1~20211003090334.2"}
