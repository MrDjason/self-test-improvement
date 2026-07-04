%DBF仿真
clear all; close all; clc;
%光速
C = 3e8;
%阵元个数
ArrayNum = 8;
%工作频率
FreqReal = 16.8e9;
%阵元间距
d = 0.009;
%目标角度
Theta = 0;

%加权值，dB
Weight = [ 1 1 1 1 1 1 1 1 ];
%幅度加权
AltLog = 20*log10(ones(1,ArrayNum)*9)-Weight; 
%各阵元目标回波相位
Phase = d*sind(Theta).*(0:ArrayNum-1)/C*FreqReal*360;
%导向适量
Steer = exp(j*2*pi*Phase/360);

%波束形成
Index = 0;
for Loop = -90:0.1:90
     Index = Index+1;
     %生成移相
     PhaseLoop = [d*sind(Loop).*(0:ArrayNum/2-1)/C*FreqReal*360,d*sind(Loop).*(ArrayNum/2:ArrayNum-1)/C*FreqReal*360];
     SteerLoop = 10.^(AltLog/20).*exp(-j*2*pi*PhaseLoop/360);
     Beam(Index) = Steer*SteerLoop.';
end

[a,X] = max(Beam)


figure
plot(20*log10(abs(Beam)/max(abs(Beam))),'r');grid on;hold on;
