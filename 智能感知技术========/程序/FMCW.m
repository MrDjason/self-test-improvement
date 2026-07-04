%FMCW信号处理仿真，实现目标距离和速度测量
close all;clear all;clc;
%===================================系统参数
%光速
c = 3e8; 
%K波段
f0 = 24e9;     
%信号带宽150MHz
B = 150e6;                                                
%信号时宽20us
T = 20e-6;   
%调频斜率
K = B/T;    
%重复周期数，也就是完成一次目标探测所需要的发射信号个数
Num = 50;                
%仿真采样频率，需满足对发射信号的采样，满足带通采样定理，2fH/(m+1)<fs<fL/m,  0<=m<=N-1,  fs>=2B，N为不大于fs/B的最大正整数
fs = 500e6; 
%如果探测目标最大距离是150米，对应混频后最大的中频频率是多少？？？

%时间轴
t = 0:1/fs:T*Num-1/fs; 

%===================================目标参数
%目标个数
TarNum = 1;
%目标多普勒频率和径向速度
Fd = 4.5e3;               V = Fd/f0*c/2;
%目标起始距离
Dis0 = 32;               
%目标距离和时间的关系，匀速直线运动
Dis = Dis0-V*t;
%目标距离导致的延迟
DelayTime = (Dis0-V*t)/c*2;        

%===================================示例，生成一个周期的发射信号和回波信号。
%-------------------一个周期的发射信号
%窗函数，在0<=t<=T之间为1，其余为0
W = rect(T,t,fs);  
%发射信号：线性调频信号与窗函数的乘积
St = exp(j*2*pi*(f0*(t)+0.5*K*t.^2)).*W; 
%接收信号：发射信号在时间维的延迟
%窗函数延迟
Wr = rect(T,t-DelayTime,fs); 
%发射信号延迟形成接收信号
Sr = exp(j*2*pi*(f0*(t-DelayTime)+0.5*K*(t-DelayTime).^2)).*Wr; 
figure
plot(real(Sr))

%===================================目标建模
%发射信号
St = zeros(1,length(t));       
%接收信号
Sr = zeros(1,length(t)); 
%用循环依次生成50个发射周期的发射信号和目标回波信号
for Loop = 1:Num
    %发射窗函数，每个周期对应相应的时间延迟(Loop-1)*T
    Wt = rect(T,(t-(Loop-1)*T),fs); 
    %发射信号，每个周期对应相应的时间延迟(Loop-1)*T
    StTemp = exp(j*2*pi*(f0*(t-(Loop-1)*T)+0.5*K*(t-(Loop-1)*T).^2)).*Wt;  
    %接收窗函数，每个周期对应相应的时间延迟(Loop-1)*T和目标回波延迟DelayTime
    Wr = rect(T,t-(Loop-1)*T-DelayTime,fs); 
    %接收信号，每个周期对应相应的时间延迟(Loop-1)*T和目标回波延迟DelayTime
    SrTemp = exp(j*2*pi*(f0*(t-(Loop-1)*T-DelayTime)+0.5*K*(t-(Loop-1)*T-DelayTime).^2)).*Wr;   
    %每个周期的发射信号相加，对应所有周期的发射信号
    St = St+StTemp; 
    %每个周期的接收信号相加，对应所有周期的接收信号
    Sr = Sr+SrTemp;
    figure(1)
    plot(Wt);
    figure(2)
    plot(real(StTemp))
%     pause(0.3)
end
figure
plot(real(St))

%混频
Smix = Sr.*(St').';

%===================================信号处理
%数据重排，排列为50×T*fs。
Sd = (reshape(Smix.',10000,Num)).'; 
%剔除无效点
Sd(:,1:fix(Dis0/c*2*fs)) = [];

%------------------画出混频后的中频信号
figure
plot(real(Sd(1,:)))

%-------------------------------------距离维FFT
%沿着距离维进行FFT
SDis = fftshift(fft(Sd,T*fs,2));

%----------------------画出二维图
%x轴坐标
x = -fs/2:fs/10000:fs/2-fs/10000;       d = x/K*c/2;
%y轴坐标
y = 1:50;
figure
mesh(d,y,abs(SDis))
xlabel('距离');   ylabel('发射周期')

%----------------------画出第一个发射周期的频率维处理结果
%横坐标
x = -fs/2:fs/10000:fs/2-fs/10000;
figure
plot(x,abs(SDis(1,:)))
xlabel('频率');   ylabel('幅度');

%----------------------画出第一个发射周期的距离维处理结果
%横坐标
d = x/K*c/2;
figure
plot(d,abs(SDis(1,:)))
xlabel('距离');   ylabel('幅度');

%-------------------------------------距离速度二维FFT
%先沿着距离维进行FFT，再沿着发射周期维进行FFT
Sfft = fftshift(fft2(Sd,100,10000));

%----------------------画出二维图
%x轴坐标
x = -fs/2:fs/10000:fs/2-fs/10000;       d = x/K*c/2;
%y轴坐标
y = -1/T/2:1/T/100:1/T/2-1/T/100;
figure
mesh(d,y,abs(Sfft))
xlabel('距离');       ylabel('速度')

%----------------------画出目标所在位置的速度维结果
%x轴坐标
y = -1/T/2:1/T/100:1/T/2-1/T/100; 
figure
plot(y,abs(Sfft(:,4969)))
xlabel('速度');       ylabel('幅度')





