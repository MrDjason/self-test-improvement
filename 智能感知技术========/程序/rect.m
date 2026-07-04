%窗函数
%Tao：窗长度；t：时间轴；fs：采样频率
%注：本函数没有考虑参数异常的情况。同学们可以自行完善。
function W = rect(Tao,t,fs)
W = ones(1,length(t));
b = find(t > Tao-1/fs);
a = find(t < 0);
W(b) = 0;
W(a) = 0;
