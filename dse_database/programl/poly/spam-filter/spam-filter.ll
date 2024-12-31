; ModuleID = 'spam-filter.cpp'
source_filename = "spam-filter.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@_ZZ5SgdLRPfPhS_E11theta_local = internal global [1024 x float] zeroinitializer, align 16
@_ZZ5SgdLRPfPhS_E11label_local = internal global [4500 x i8] zeroinitializer, align 16
@_ZZ5SgdLRPfPhS_E17training_instance = internal global [1024 x float] zeroinitializer, align 16
@_ZZL9compute_1PfPhS_iE8gradient = internal global [1024 x float] zeroinitializer, align 16

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z5SgdLRPfPhS_(float* %data, i8* %label, float* %theta) #0 {
entry:
  %data.addr = alloca float*, align 8
  %label.addr = alloca i8*, align 8
  %theta.addr = alloca float*, align 8
  %i = alloca i32, align 4
  %i3 = alloca i32, align 4
  %epoch = alloca i32, align 4
  %training_id = alloca i32, align 4
  %i26 = alloca i32, align 4
  store float* %data, float** %data.addr, align 8
  store i8* %label, i8** %label.addr, align 8
  store float* %theta, float** %theta.addr, align 8
  br label %PARAM_INIT

PARAM_INIT:                                       ; preds = %entry
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %PARAM_INIT
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 1024
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load float*, float** %theta.addr, align 8
  %2 = load i32, i32* %i, align 4
  %idxprom = sext i32 %2 to i64
  %arrayidx = getelementptr inbounds float, float* %1, i64 %idxprom
  %3 = load float, float* %arrayidx, align 4
  %4 = load i32, i32* %i, align 4
  %idxprom1 = sext i32 %4 to i64
  %arrayidx2 = getelementptr inbounds [1024 x float], [1024 x float]* @_ZZ5SgdLRPfPhS_E11theta_local, i64 0, i64 %idxprom1
  store float %3, float* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32, i32* %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  br label %LABEL_CP

LABEL_CP:                                         ; preds = %for.end
  store i32 0, i32* %i3, align 4
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc11, %LABEL_CP
  %6 = load i32, i32* %i3, align 4
  %cmp5 = icmp slt i32 %6, 4500
  br i1 %cmp5, label %for.body6, label %for.end13

for.body6:                                        ; preds = %for.cond4
  %7 = load i8*, i8** %label.addr, align 8
  %8 = load i32, i32* %i3, align 4
  %idxprom7 = sext i32 %8 to i64
  %arrayidx8 = getelementptr inbounds i8, i8* %7, i64 %idxprom7
  %9 = load i8, i8* %arrayidx8, align 1
  %10 = load i32, i32* %i3, align 4
  %idxprom9 = sext i32 %10 to i64
  %arrayidx10 = getelementptr inbounds [4500 x i8], [4500 x i8]* @_ZZ5SgdLRPfPhS_E11label_local, i64 0, i64 %idxprom9
  store i8 %9, i8* %arrayidx10, align 1
  br label %for.inc11

for.inc11:                                        ; preds = %for.body6
  %11 = load i32, i32* %i3, align 4
  %inc12 = add nsw i32 %11, 1
  store i32 %inc12, i32* %i3, align 4
  br label %for.cond4

for.end13:                                        ; preds = %for.cond4
  br label %EPOCH

EPOCH:                                            ; preds = %for.end13
  store i32 0, i32* %epoch, align 4
  br label %for.cond14

for.cond14:                                       ; preds = %for.inc23, %EPOCH
  %12 = load i32, i32* %epoch, align 4
  %cmp15 = icmp slt i32 %12, 5
  br i1 %cmp15, label %for.body16, label %for.end25

for.body16:                                       ; preds = %for.cond14
  br label %TRAINING_INST

TRAINING_INST:                                    ; preds = %for.body16
  store i32 0, i32* %training_id, align 4
  br label %for.cond17

for.cond17:                                       ; preds = %for.inc20, %TRAINING_INST
  %13 = load i32, i32* %training_id, align 4
  %cmp18 = icmp slt i32 %13, 4500
  br i1 %cmp18, label %for.body19, label %for.end22

for.body19:                                       ; preds = %for.cond17
  %14 = load float*, float** %data.addr, align 8
  %15 = load i32, i32* %training_id, align 4
  call void @_ZL11read_data_1PfS_i(float* %14, float* getelementptr inbounds ([1024 x float], [1024 x float]* @_ZZ5SgdLRPfPhS_E17training_instance, i64 0, i64 0), i32 %15)
  %16 = load i32, i32* %training_id, align 4
  call void @_ZL9compute_1PfPhS_i(float* getelementptr inbounds ([1024 x float], [1024 x float]* @_ZZ5SgdLRPfPhS_E11theta_local, i64 0, i64 0), i8* getelementptr inbounds ([4500 x i8], [4500 x i8]* @_ZZ5SgdLRPfPhS_E11label_local, i64 0, i64 0), float* getelementptr inbounds ([1024 x float], [1024 x float]* @_ZZ5SgdLRPfPhS_E17training_instance, i64 0, i64 0), i32 %16)
  br label %for.inc20

for.inc20:                                        ; preds = %for.body19
  %17 = load i32, i32* %training_id, align 4
  %inc21 = add nsw i32 %17, 1
  store i32 %inc21, i32* %training_id, align 4
  br label %for.cond17

for.end22:                                        ; preds = %for.cond17
  br label %for.inc23

for.inc23:                                        ; preds = %for.end22
  %18 = load i32, i32* %epoch, align 4
  %inc24 = add nsw i32 %18, 1
  store i32 %inc24, i32* %epoch, align 4
  br label %for.cond14

for.end25:                                        ; preds = %for.cond14
  br label %STREAM_OUT

STREAM_OUT:                                       ; preds = %for.end25
  store i32 0, i32* %i26, align 4
  br label %for.cond27

for.cond27:                                       ; preds = %for.inc34, %STREAM_OUT
  %19 = load i32, i32* %i26, align 4
  %cmp28 = icmp slt i32 %19, 1024
  br i1 %cmp28, label %for.body29, label %for.end36

for.body29:                                       ; preds = %for.cond27
  %20 = load i32, i32* %i26, align 4
  %idxprom30 = sext i32 %20 to i64
  %arrayidx31 = getelementptr inbounds [1024 x float], [1024 x float]* @_ZZ5SgdLRPfPhS_E11theta_local, i64 0, i64 %idxprom30
  %21 = load float, float* %arrayidx31, align 4
  %22 = load float*, float** %theta.addr, align 8
  %23 = load i32, i32* %i26, align 4
  %idxprom32 = sext i32 %23 to i64
  %arrayidx33 = getelementptr inbounds float, float* %22, i64 %idxprom32
  store float %21, float* %arrayidx33, align 4
  br label %for.inc34

for.inc34:                                        ; preds = %for.body29
  %24 = load i32, i32* %i26, align 4
  %inc35 = add nsw i32 %24, 1
  store i32 %inc35, i32* %i26, align 4
  br label %for.cond27

for.end36:                                        ; preds = %for.cond27
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @_ZL11read_data_1PfS_i(float* %data, float* %training_instance, i32 %training_id) #1 {
entry:
  %data.addr = alloca float*, align 8
  %training_instance.addr = alloca float*, align 8
  %training_id.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store float* %data, float** %data.addr, align 8
  store float* %training_instance, float** %training_instance.addr, align 8
  store i32 %training_id, i32* %training_id.addr, align 4
  br label %READ_TRAINING_DATA

READ_TRAINING_DATA:                               ; preds = %entry
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %READ_TRAINING_DATA
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 1024
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load float*, float** %data.addr, align 8
  %2 = load i32, i32* %training_id.addr, align 4
  %mul = mul nsw i32 %2, 1024
  %3 = load i32, i32* %i, align 4
  %add = add nsw i32 %mul, %3
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds float, float* %1, i64 %idxprom
  %4 = load float, float* %arrayidx, align 4
  %5 = load float*, float** %training_instance.addr, align 8
  %6 = load i32, i32* %i, align 4
  %idxprom1 = sext i32 %6 to i64
  %arrayidx2 = getelementptr inbounds float, float* %5, i64 %idxprom1
  store float %4, float* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %7 = load i32, i32* %i, align 4
  %inc = add nsw i32 %7, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; Function Attrs: noinline optnone uwtable
define internal void @_ZL9compute_1PfPhS_i(float* %theta_local, i8* %label_local, float* %training_instance, i32 %training_id) #0 {
entry:
  %theta_local.addr = alloca float*, align 8
  %label_local.addr = alloca i8*, align 8
  %training_instance.addr = alloca float*, align 8
  %training_id.addr = alloca i32, align 4
  %step = alloca float, align 4
  %dot = alloca float, align 4
  %prob = alloca float, align 4
  %training_label = alloca i8, align 1
  store float* %theta_local, float** %theta_local.addr, align 8
  store i8* %label_local, i8** %label_local.addr, align 8
  store float* %training_instance, float** %training_instance.addr, align 8
  store i32 %training_id, i32* %training_id.addr, align 4
  store float 6.000000e+04, float* %step, align 4
  %0 = load float*, float** %theta_local.addr, align 8
  %1 = load float*, float** %training_instance.addr, align 8
  %call = call float @_ZL12dotProduct_1PfS_(float* %0, float* %1)
  store float %call, float* %dot, align 4
  %2 = load float, float* %dot, align 4
  %call1 = call float @_ZL9Sigmoid_1f(float %2)
  store float %call1, float* %prob, align 4
  %3 = load i8*, i8** %label_local.addr, align 8
  %4 = load i32, i32* %training_id.addr, align 4
  %idxprom = sext i32 %4 to i64
  %arrayidx = getelementptr inbounds i8, i8* %3, i64 %idxprom
  %5 = load i8, i8* %arrayidx, align 1
  store i8 %5, i8* %training_label, align 1
  %6 = load float*, float** %training_instance.addr, align 8
  %7 = load float, float* %prob, align 4
  %8 = load i8, i8* %training_label, align 1
  %conv = zext i8 %8 to i32
  %conv2 = sitofp i32 %conv to float
  %sub = fsub float %7, %conv2
  call void @_ZL17computeGradient_1PfS_f(float* getelementptr inbounds ([1024 x float], [1024 x float]* @_ZZL9compute_1PfPhS_iE8gradient, i64 0, i64 0), float* %6, float %sub)
  %9 = load float*, float** %theta_local.addr, align 8
  %10 = load float, float* %step, align 4
  %fneg = fneg float %10
  call void @_ZL17updateParameter_1PfS_f(float* %9, float* getelementptr inbounds ([1024 x float], [1024 x float]* @_ZZL9compute_1PfPhS_iE8gradient, i64 0, i64 0), float %fneg)
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal float @_ZL12dotProduct_1PfS_(float* %param, float* %feature) #1 {
entry:
  %param.addr = alloca float*, align 8
  %feature.addr = alloca float*, align 8
  %result = alloca float, align 4
  %i = alloca i32, align 4
  %term = alloca float, align 4
  store float* %param, float** %param.addr, align 8
  store float* %feature, float** %feature.addr, align 8
  store float 0.000000e+00, float* %result, align 4
  br label %DOT

DOT:                                              ; preds = %entry
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %DOT
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 1024
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load float*, float** %param.addr, align 8
  %2 = load i32, i32* %i, align 4
  %idxprom = sext i32 %2 to i64
  %arrayidx = getelementptr inbounds float, float* %1, i64 %idxprom
  %3 = load float, float* %arrayidx, align 4
  %4 = load float*, float** %feature.addr, align 8
  %5 = load i32, i32* %i, align 4
  %idxprom1 = sext i32 %5 to i64
  %arrayidx2 = getelementptr inbounds float, float* %4, i64 %idxprom1
  %6 = load float, float* %arrayidx2, align 4
  %mul = fmul float %3, %6
  store float %mul, float* %term, align 4
  %7 = load float, float* %term, align 4
  %8 = load float, float* %result, align 4
  %add = fadd float %8, %7
  store float %add, float* %result, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %9 = load i32, i32* %i, align 4
  %inc = add nsw i32 %9, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %10 = load float, float* %result, align 4
  ret float %10
}

; Function Attrs: noinline nounwind optnone uwtable
define internal float @_ZL9Sigmoid_1f(float %exponent) #1 {
entry:
  %retval = alloca float, align 4
  %exponent.addr = alloca float, align 4
  store float %exponent, float* %exponent.addr, align 4
  %0 = load float, float* %exponent.addr, align 4
  %cmp = fcmp ogt float %0, 4.000000e+00
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store float 1.000000e+00, float* %retval, align 4
  br label %return

if.else:                                          ; preds = %entry
  %1 = load float, float* %exponent.addr, align 4
  %cmp1 = fcmp olt float %1, -4.000000e+00
  br i1 %cmp1, label %if.then2, label %if.else3

if.then2:                                         ; preds = %if.else
  store float 0.000000e+00, float* %retval, align 4
  br label %return

if.else3:                                         ; preds = %if.else
  %2 = load float, float* %exponent.addr, align 4
  %fneg = fneg float %2
  %call = call float @expf(float %fneg) #3
  %conv = fpext float %call to double
  %add = fadd double 1.000000e+00, %conv
  %div = fdiv double 1.000000e+00, %add
  %conv4 = fptrunc double %div to float
  store float %conv4, float* %retval, align 4
  br label %return

return:                                           ; preds = %if.else3, %if.then2, %if.then
  %3 = load float, float* %retval, align 4
  ret float %3
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @_ZL17computeGradient_1PfS_f(float* %grad, float* %feature, float %scale) #1 {
entry:
  %grad.addr = alloca float*, align 8
  %feature.addr = alloca float*, align 8
  %scale.addr = alloca float, align 4
  %i = alloca i32, align 4
  store float* %grad, float** %grad.addr, align 8
  store float* %feature, float** %feature.addr, align 8
  store float %scale, float* %scale.addr, align 4
  br label %GRAD

GRAD:                                             ; preds = %entry
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %GRAD
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 1024
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load float, float* %scale.addr, align 4
  %2 = load float*, float** %feature.addr, align 8
  %3 = load i32, i32* %i, align 4
  %idxprom = sext i32 %3 to i64
  %arrayidx = getelementptr inbounds float, float* %2, i64 %idxprom
  %4 = load float, float* %arrayidx, align 4
  %mul = fmul float %1, %4
  %5 = load float*, float** %grad.addr, align 8
  %6 = load i32, i32* %i, align 4
  %idxprom1 = sext i32 %6 to i64
  %arrayidx2 = getelementptr inbounds float, float* %5, i64 %idxprom1
  store float %mul, float* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %7 = load i32, i32* %i, align 4
  %inc = add nsw i32 %7, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @_ZL17updateParameter_1PfS_f(float* %param, float* %grad, float %scale) #1 {
entry:
  %param.addr = alloca float*, align 8
  %grad.addr = alloca float*, align 8
  %scale.addr = alloca float, align 4
  %i = alloca i32, align 4
  store float* %param, float** %param.addr, align 8
  store float* %grad, float** %grad.addr, align 8
  store float %scale, float* %scale.addr, align 4
  br label %UPDATE

UPDATE:                                           ; preds = %entry
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %UPDATE
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 1024
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load float, float* %scale.addr, align 4
  %2 = load float*, float** %grad.addr, align 8
  %3 = load i32, i32* %i, align 4
  %idxprom = sext i32 %3 to i64
  %arrayidx = getelementptr inbounds float, float* %2, i64 %idxprom
  %4 = load float, float* %arrayidx, align 4
  %mul = fmul float %1, %4
  %5 = load float*, float** %param.addr, align 8
  %6 = load i32, i32* %i, align 4
  %idxprom1 = sext i32 %6 to i64
  %arrayidx2 = getelementptr inbounds float, float* %5, i64 %idxprom1
  %7 = load float, float* %arrayidx2, align 4
  %add = fadd float %7, %mul
  store float %add, float* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %8 = load i32, i32* %i, align 4
  %inc = add nsw i32 %8, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; Function Attrs: nounwind
declare dso_local float @expf(float) #2

attributes #0 = { noinline optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"Ubuntu clang version 10.0.1-++20211003085942+ef32c611aa21-1~exp1~20211003090334.2"}
