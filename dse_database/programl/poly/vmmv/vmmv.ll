; ModuleID = 'vmmv.cpp'
source_filename = "vmmv.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_Z4vmmvPA80_fPfS1_S0_S0_S1_S1_([80 x float]* %A, float* %a, float* %b, [80 x float]* %B, [80 x float]* %C, float* %x, float* %y) #0 {
entry:
  %A.addr = alloca [80 x float]*, align 8
  %a.addr = alloca float*, align 8
  %b.addr = alloca float*, align 8
  %B.addr = alloca [80 x float]*, align 8
  %C.addr = alloca [80 x float]*, align 8
  %x.addr = alloca float*, align 8
  %y.addr = alloca float*, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %i13 = alloca i32, align 4
  %j17 = alloca i32, align 4
  %k = alloca i32, align 4
  %i47 = alloca i32, align 4
  %j51 = alloca i32, align 4
  store [80 x float]* %A, [80 x float]** %A.addr, align 8
  store float* %a, float** %a.addr, align 8
  store float* %b, float** %b.addr, align 8
  store [80 x float]* %B, [80 x float]** %B.addr, align 8
  store [80 x float]* %C, [80 x float]** %C.addr, align 8
  store float* %x, float** %x.addr, align 8
  store float* %y, float** %y.addr, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc10, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end12

for.body:                                         ; preds = %for.cond
  store i32 0, i32* %j, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %1 = load i32, i32* %j, align 4
  %cmp2 = icmp slt i32 %1, 80
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %2 = load float*, float** %a.addr, align 8
  %3 = load i32, i32* %i, align 4
  %idxprom = sext i32 %3 to i64
  %arrayidx = getelementptr inbounds float, float* %2, i64 %idxprom
  %4 = load float, float* %arrayidx, align 4
  %5 = load float*, float** %b.addr, align 8
  %6 = load i32, i32* %j, align 4
  %idxprom4 = sext i32 %6 to i64
  %arrayidx5 = getelementptr inbounds float, float* %5, i64 %idxprom4
  %7 = load float, float* %arrayidx5, align 4
  %mul = fmul float %4, %7
  %8 = load [80 x float]*, [80 x float]** %A.addr, align 8
  %9 = load i32, i32* %i, align 4
  %idxprom6 = sext i32 %9 to i64
  %arrayidx7 = getelementptr inbounds [80 x float], [80 x float]* %8, i64 %idxprom6
  %10 = load i32, i32* %j, align 4
  %idxprom8 = sext i32 %10 to i64
  %arrayidx9 = getelementptr inbounds [80 x float], [80 x float]* %arrayidx7, i64 0, i64 %idxprom8
  %11 = load float, float* %arrayidx9, align 4
  %add = fadd float %11, %mul
  store float %add, float* %arrayidx9, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %12 = load i32, i32* %j, align 4
  %inc = add nsw i32 %12, 1
  store i32 %inc, i32* %j, align 4
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.inc10

for.inc10:                                        ; preds = %for.end
  %13 = load i32, i32* %i, align 4
  %inc11 = add nsw i32 %13, 1
  store i32 %inc11, i32* %i, align 4
  br label %for.cond

for.end12:                                        ; preds = %for.cond
  store i32 0, i32* %i13, align 4
  br label %for.cond14

for.cond14:                                       ; preds = %for.inc44, %for.end12
  %14 = load i32, i32* %i13, align 4
  %cmp15 = icmp slt i32 %14, 100
  br i1 %cmp15, label %for.body16, label %for.end46

for.body16:                                       ; preds = %for.cond14
  store i32 0, i32* %j17, align 4
  br label %for.cond18

for.cond18:                                       ; preds = %for.inc41, %for.body16
  %15 = load i32, i32* %j17, align 4
  %cmp19 = icmp slt i32 %15, 80
  br i1 %cmp19, label %for.body20, label %for.end43

for.body20:                                       ; preds = %for.cond18
  store i32 0, i32* %k, align 4
  br label %for.cond21

for.cond21:                                       ; preds = %for.inc38, %for.body20
  %16 = load i32, i32* %k, align 4
  %cmp22 = icmp slt i32 %16, 80
  br i1 %cmp22, label %for.body23, label %for.end40

for.body23:                                       ; preds = %for.cond21
  %17 = load [80 x float]*, [80 x float]** %A.addr, align 8
  %18 = load i32, i32* %i13, align 4
  %idxprom24 = sext i32 %18 to i64
  %arrayidx25 = getelementptr inbounds [80 x float], [80 x float]* %17, i64 %idxprom24
  %19 = load i32, i32* %k, align 4
  %idxprom26 = sext i32 %19 to i64
  %arrayidx27 = getelementptr inbounds [80 x float], [80 x float]* %arrayidx25, i64 0, i64 %idxprom26
  %20 = load float, float* %arrayidx27, align 4
  %21 = load [80 x float]*, [80 x float]** %B.addr, align 8
  %22 = load i32, i32* %k, align 4
  %idxprom28 = sext i32 %22 to i64
  %arrayidx29 = getelementptr inbounds [80 x float], [80 x float]* %21, i64 %idxprom28
  %23 = load i32, i32* %j17, align 4
  %idxprom30 = sext i32 %23 to i64
  %arrayidx31 = getelementptr inbounds [80 x float], [80 x float]* %arrayidx29, i64 0, i64 %idxprom30
  %24 = load float, float* %arrayidx31, align 4
  %mul32 = fmul float %20, %24
  %25 = load [80 x float]*, [80 x float]** %C.addr, align 8
  %26 = load i32, i32* %i13, align 4
  %idxprom33 = sext i32 %26 to i64
  %arrayidx34 = getelementptr inbounds [80 x float], [80 x float]* %25, i64 %idxprom33
  %27 = load i32, i32* %j17, align 4
  %idxprom35 = sext i32 %27 to i64
  %arrayidx36 = getelementptr inbounds [80 x float], [80 x float]* %arrayidx34, i64 0, i64 %idxprom35
  %28 = load float, float* %arrayidx36, align 4
  %add37 = fadd float %28, %mul32
  store float %add37, float* %arrayidx36, align 4
  br label %for.inc38

for.inc38:                                        ; preds = %for.body23
  %29 = load i32, i32* %k, align 4
  %inc39 = add nsw i32 %29, 1
  store i32 %inc39, i32* %k, align 4
  br label %for.cond21

for.end40:                                        ; preds = %for.cond21
  br label %for.inc41

for.inc41:                                        ; preds = %for.end40
  %30 = load i32, i32* %j17, align 4
  %inc42 = add nsw i32 %30, 1
  store i32 %inc42, i32* %j17, align 4
  br label %for.cond18

for.end43:                                        ; preds = %for.cond18
  br label %for.inc44

for.inc44:                                        ; preds = %for.end43
  %31 = load i32, i32* %i13, align 4
  %inc45 = add nsw i32 %31, 1
  store i32 %inc45, i32* %i13, align 4
  br label %for.cond14

for.end46:                                        ; preds = %for.cond14
  store i32 0, i32* %i47, align 4
  br label %for.cond48

for.cond48:                                       ; preds = %for.inc68, %for.end46
  %32 = load i32, i32* %i47, align 4
  %cmp49 = icmp slt i32 %32, 100
  br i1 %cmp49, label %for.body50, label %for.end70

for.body50:                                       ; preds = %for.cond48
  store i32 0, i32* %j51, align 4
  br label %for.cond52

for.cond52:                                       ; preds = %for.inc65, %for.body50
  %33 = load i32, i32* %j51, align 4
  %cmp53 = icmp slt i32 %33, 80
  br i1 %cmp53, label %for.body54, label %for.end67

for.body54:                                       ; preds = %for.cond52
  %34 = load [80 x float]*, [80 x float]** %C.addr, align 8
  %35 = load i32, i32* %i47, align 4
  %idxprom55 = sext i32 %35 to i64
  %arrayidx56 = getelementptr inbounds [80 x float], [80 x float]* %34, i64 %idxprom55
  %36 = load i32, i32* %j51, align 4
  %idxprom57 = sext i32 %36 to i64
  %arrayidx58 = getelementptr inbounds [80 x float], [80 x float]* %arrayidx56, i64 0, i64 %idxprom57
  %37 = load float, float* %arrayidx58, align 4
  %38 = load float*, float** %x.addr, align 8
  %39 = load i32, i32* %j51, align 4
  %idxprom59 = sext i32 %39 to i64
  %arrayidx60 = getelementptr inbounds float, float* %38, i64 %idxprom59
  %40 = load float, float* %arrayidx60, align 4
  %mul61 = fmul float %37, %40
  %41 = load float*, float** %y.addr, align 8
  %42 = load i32, i32* %i47, align 4
  %idxprom62 = sext i32 %42 to i64
  %arrayidx63 = getelementptr inbounds float, float* %41, i64 %idxprom62
  %43 = load float, float* %arrayidx63, align 4
  %add64 = fadd float %43, %mul61
  store float %add64, float* %arrayidx63, align 4
  br label %for.inc65

for.inc65:                                        ; preds = %for.body54
  %44 = load i32, i32* %j51, align 4
  %inc66 = add nsw i32 %44, 1
  store i32 %inc66, i32* %j51, align 4
  br label %for.cond52

for.end67:                                        ; preds = %for.cond52
  br label %for.inc68

for.inc68:                                        ; preds = %for.end67
  %45 = load i32, i32* %i47, align 4
  %inc69 = add nsw i32 %45, 1
  store i32 %inc69, i32* %i47, align 4
  br label %for.cond48

for.end70:                                        ; preds = %for.cond48
  ret void
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"Ubuntu clang version 10.0.1-++20211003085942+ef32c611aa21-1~exp1~20211003090334.2"}
