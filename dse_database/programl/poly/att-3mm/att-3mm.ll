; ModuleID = 'att-3mm.cpp'
source_filename = "att-3mm.cpp"
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
  %i23 = alloca i32, align 4
  %j27 = alloca i32, align 4
  %k31 = alloca i32, align 4
  %i58 = alloca i32, align 4
  %j62 = alloca i32, align 4
  %k66 = alloca i32, align 4
  store [768 x float]* %data, [768 x float]** %data.addr, align 8
  store [64 x float]* %w1, [64 x float]** %w1.addr, align 8
  store [64 x float]* %w2, [64 x float]** %w2.addr, align 8
  store [64 x float]* %Q, [64 x float]** %Q.addr, align 8
  store [64 x float]* %K, [64 x float]** %K.addr, align 8
  store [512 x float]* %atten, [512 x float]** %atten.addr, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc20, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 512
  br i1 %cmp, label %for.body, label %for.end22

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
  br label %for.inc20

for.inc20:                                        ; preds = %for.end19
  %17 = load i32, i32* %i, align 4
  %inc21 = add nsw i32 %17, 1
  store i32 %inc21, i32* %i, align 4
  br label %for.cond

for.end22:                                        ; preds = %for.cond
  store i32 0, i32* %i23, align 4
  br label %for.cond24

for.cond24:                                       ; preds = %for.inc55, %for.end22
  %18 = load i32, i32* %i23, align 4
  %cmp25 = icmp slt i32 %18, 512
  br i1 %cmp25, label %for.body26, label %for.end57

for.body26:                                       ; preds = %for.cond24
  store i32 0, i32* %j27, align 4
  br label %for.cond28

for.cond28:                                       ; preds = %for.inc52, %for.body26
  %19 = load i32, i32* %j27, align 4
  %cmp29 = icmp slt i32 %19, 64
  br i1 %cmp29, label %for.body30, label %for.end54

for.body30:                                       ; preds = %for.cond28
  store i32 0, i32* %k31, align 4
  br label %for.cond32

for.cond32:                                       ; preds = %for.inc49, %for.body30
  %20 = load i32, i32* %k31, align 4
  %cmp33 = icmp slt i32 %20, 768
  br i1 %cmp33, label %for.body34, label %for.end51

for.body34:                                       ; preds = %for.cond32
  %21 = load [768 x float]*, [768 x float]** %data.addr, align 8
  %22 = load i32, i32* %i23, align 4
  %idxprom35 = sext i32 %22 to i64
  %arrayidx36 = getelementptr inbounds [768 x float], [768 x float]* %21, i64 %idxprom35
  %23 = load i32, i32* %k31, align 4
  %idxprom37 = sext i32 %23 to i64
  %arrayidx38 = getelementptr inbounds [768 x float], [768 x float]* %arrayidx36, i64 0, i64 %idxprom37
  %24 = load float, float* %arrayidx38, align 4
  %25 = load [64 x float]*, [64 x float]** %w2.addr, align 8
  %26 = load i32, i32* %k31, align 4
  %idxprom39 = sext i32 %26 to i64
  %arrayidx40 = getelementptr inbounds [64 x float], [64 x float]* %25, i64 %idxprom39
  %27 = load i32, i32* %j27, align 4
  %idxprom41 = sext i32 %27 to i64
  %arrayidx42 = getelementptr inbounds [64 x float], [64 x float]* %arrayidx40, i64 0, i64 %idxprom41
  %28 = load float, float* %arrayidx42, align 4
  %mul43 = fmul float %24, %28
  %29 = load [64 x float]*, [64 x float]** %K.addr, align 8
  %30 = load i32, i32* %i23, align 4
  %idxprom44 = sext i32 %30 to i64
  %arrayidx45 = getelementptr inbounds [64 x float], [64 x float]* %29, i64 %idxprom44
  %31 = load i32, i32* %j27, align 4
  %idxprom46 = sext i32 %31 to i64
  %arrayidx47 = getelementptr inbounds [64 x float], [64 x float]* %arrayidx45, i64 0, i64 %idxprom46
  %32 = load float, float* %arrayidx47, align 4
  %add48 = fadd float %32, %mul43
  store float %add48, float* %arrayidx47, align 4
  br label %for.inc49

for.inc49:                                        ; preds = %for.body34
  %33 = load i32, i32* %k31, align 4
  %inc50 = add nsw i32 %33, 1
  store i32 %inc50, i32* %k31, align 4
  br label %for.cond32

for.end51:                                        ; preds = %for.cond32
  br label %for.inc52

for.inc52:                                        ; preds = %for.end51
  %34 = load i32, i32* %j27, align 4
  %inc53 = add nsw i32 %34, 1
  store i32 %inc53, i32* %j27, align 4
  br label %for.cond28

for.end54:                                        ; preds = %for.cond28
  br label %for.inc55

for.inc55:                                        ; preds = %for.end54
  %35 = load i32, i32* %i23, align 4
  %inc56 = add nsw i32 %35, 1
  store i32 %inc56, i32* %i23, align 4
  br label %for.cond24

for.end57:                                        ; preds = %for.cond24
  store i32 0, i32* %i58, align 4
  br label %for.cond59

for.cond59:                                       ; preds = %for.inc90, %for.end57
  %36 = load i32, i32* %i58, align 4
  %cmp60 = icmp slt i32 %36, 512
  br i1 %cmp60, label %for.body61, label %for.end92

for.body61:                                       ; preds = %for.cond59
  store i32 0, i32* %j62, align 4
  br label %for.cond63

for.cond63:                                       ; preds = %for.inc87, %for.body61
  %37 = load i32, i32* %j62, align 4
  %cmp64 = icmp slt i32 %37, 512
  br i1 %cmp64, label %for.body65, label %for.end89

for.body65:                                       ; preds = %for.cond63
  store i32 0, i32* %k66, align 4
  br label %for.cond67

for.cond67:                                       ; preds = %for.inc84, %for.body65
  %38 = load i32, i32* %k66, align 4
  %cmp68 = icmp slt i32 %38, 64
  br i1 %cmp68, label %for.body69, label %for.end86

for.body69:                                       ; preds = %for.cond67
  %39 = load [64 x float]*, [64 x float]** %Q.addr, align 8
  %40 = load i32, i32* %i58, align 4
  %idxprom70 = sext i32 %40 to i64
  %arrayidx71 = getelementptr inbounds [64 x float], [64 x float]* %39, i64 %idxprom70
  %41 = load i32, i32* %k66, align 4
  %idxprom72 = sext i32 %41 to i64
  %arrayidx73 = getelementptr inbounds [64 x float], [64 x float]* %arrayidx71, i64 0, i64 %idxprom72
  %42 = load float, float* %arrayidx73, align 4
  %43 = load [64 x float]*, [64 x float]** %K.addr, align 8
  %44 = load i32, i32* %j62, align 4
  %idxprom74 = sext i32 %44 to i64
  %arrayidx75 = getelementptr inbounds [64 x float], [64 x float]* %43, i64 %idxprom74
  %45 = load i32, i32* %k66, align 4
  %idxprom76 = sext i32 %45 to i64
  %arrayidx77 = getelementptr inbounds [64 x float], [64 x float]* %arrayidx75, i64 0, i64 %idxprom76
  %46 = load float, float* %arrayidx77, align 4
  %mul78 = fmul float %42, %46
  %47 = load [512 x float]*, [512 x float]** %atten.addr, align 8
  %48 = load i32, i32* %i58, align 4
  %idxprom79 = sext i32 %48 to i64
  %arrayidx80 = getelementptr inbounds [512 x float], [512 x float]* %47, i64 %idxprom79
  %49 = load i32, i32* %j62, align 4
  %idxprom81 = sext i32 %49 to i64
  %arrayidx82 = getelementptr inbounds [512 x float], [512 x float]* %arrayidx80, i64 0, i64 %idxprom81
  %50 = load float, float* %arrayidx82, align 4
  %add83 = fadd float %50, %mul78
  store float %add83, float* %arrayidx82, align 4
  br label %for.inc84

for.inc84:                                        ; preds = %for.body69
  %51 = load i32, i32* %k66, align 4
  %inc85 = add nsw i32 %51, 1
  store i32 %inc85, i32* %k66, align 4
  br label %for.cond67

for.end86:                                        ; preds = %for.cond67
  br label %for.inc87

for.inc87:                                        ; preds = %for.end86
  %52 = load i32, i32* %j62, align 4
  %inc88 = add nsw i32 %52, 1
  store i32 %inc88, i32* %j62, align 4
  br label %for.cond63

for.end89:                                        ; preds = %for.cond63
  br label %for.inc90

for.inc90:                                        ; preds = %for.end89
  %53 = load i32, i32* %i58, align 4
  %inc91 = add nsw i32 %53, 1
  store i32 %inc91, i32* %i58, align 4
  br label %for.cond59

for.end92:                                        ; preds = %for.cond59
  ret void
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"Ubuntu clang version 10.0.1-++20211003085942+ef32c611aa21-1~exp1~20211003090334.2"}
