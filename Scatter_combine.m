
set(0,'defaultTextInterpreter','latex');
set(0,'defaultLegendInterpreter','latex');
file_dir = dir('C:\Users\s194086\Desktop\Simulation_plots\Data_files');
Files = extractfield(file_dir,'name');
Files(1:2) = [];

Glass = load(fullfile('C:\Users\s194086\Desktop\Simulation_plots\Data_files',Files{3}));
Gold = load(fullfile('C:\Users\s194086\Desktop\Simulation_plots\Data_files',Files{2}));
Gold_45 = load(fullfile('C:\Users\s194086\Desktop\Simulation_plots\Data_files',Files{6}));
Gold_unnormalized = load(fullfile('C:\Users\s194086\Desktop\Simulation_plots\Data_files',Files{4}));
Green = load(fullfile('C:\Users\s194086\Desktop\Simulation_plots\Data_files',Files{5}));
Brown = load(fullfile('C:\Users\s194086\Desktop\Simulation_plots\Data_files',Files{8}));
LightGreen = load(fullfile('C:\Users\s194086\Desktop\Simulation_plots\Data_files',Files{7}));
Red = load(fullfile('C:\Users\s194086\Desktop\Simulation_plots\Data_files',Files{10}));
Silicon = load(fullfile('C:\Users\s194086\Desktop\Simulation_plots\Data_files',Files{13}));
Gold_10000 = load(fullfile('C:\Users\s194086\Desktop\Simulation_plots\Data_files',Files{15}));
Gold_25000 = load(fullfile('C:\Users\s194086\Desktop\Simulation_plots\Data_files',Files{16}));
Combined = load(fullfile('C:\Users\s194086\Desktop\Simulation_plots\Data_files',Files{1}));
Gold_30 = load(fullfile('C:\Users\s194086\Desktop\Simulation_plots\Data_files',Files{19}));
Gold_60 = load(fullfile('C:\Users\s194086\Desktop\Simulation_plots\Data_files',Files{18}));
Gold_0 = load(fullfile('C:\Users\s194086\Desktop\Simulation_plots\Data_files',Files{17}));

np = numel(Gold.Power(1,:));

azimuth = linspace(0,358,180);
polar = linspace(0,90,np).';

pA = reshape(repmat(azimuth,[np,1]),[],1);
pP = repmat(polar,numel(azimuth),1);


%%
figure(1)
plot(Gold.Polar_filter,Gold.avg_pol,'Color','blue','Marker','o','LineStyle','--','LineWidth',2)
hold on
plot(Gold_45.Polar_filter,Gold_45.avg_pol,'Color','red','Marker','diamond','LineStyle','--','LineWidth',2)
plot(Gold_unnormalized.Polar_filter,Gold_unnormalized.avg_pol,'Color','black','Marker','square','LineStyle','--','LineWidth',2)
grid('on')
set(gca,'YScale','log')
xlabel('Scatter Angle $\theta_s$ [$^\circ$]',FontSize = 26)
ylabel('B[1/sr]',FontSize = 26)
grid('on')
legend({'$\mathrm{B_{Glass + Enc}}$','$\mathrm{B_{Silicon}}$','$\mathrm{B_{Glass + Enc} + B_{Silicon}}$'},'Interpreter','latex','FontSize',24)

%% Other power. Detector extraction. 

M_Solid_Ang = repmat((Glass.Solid_ang),[1,numel(Glass.M_power(1,:))]);
%Irradiance_in = P./ A_n; %./ source_size^2*pi;

%Irradiance_out =  M_power ./ (A_n .* cosd(Polar_filter).');%(arc_pol*az_arc); Sensor Area/ cos(theta) / cos(theta_i)M_power
%Irradiance_avg = mean(Irradiance_out,2);

%brdf_M = (Irradiance_out ./ (M_Solid_Ang)) / Irradiance_in';


A_n = 1.8295;

Irradiance_in = 1./ A_n; %./ source_size^2*pi;

Irradiance_out_glass =  Glass.M_power ./ (A_n .* cosd(Glass.Polar_filter).');%(arc_pol*az_arc); Sensor Area/ cos(theta) / cos(theta_i)M_power
Irradiance_avg_glass = mean(Irradiance_out_glass,2);

Irradiance_out_Silicon =  Silicon.M_power ./ (A_n .* cosd(Silicon.Polar_filter).'); %Silicon.M_Power' ./ (A_n .* cosd(Silicon.Polar_filter).');%(arc_pol*az_arc); Sensor Area/ cos(theta) / cos(theta_i)M_power
Irradiance_avg_Silicon = mean(Irradiance_out_Silicon,2);


brdf_Glass = (Irradiance_out_glass./(M_Solid_Ang))./ (Irradiance_in.'); 
brdf_Silicon = (Irradiance_out_Silicon./(Silicon.Solid_ang))./ (Irradiance_in.'); 



figure(2)
plot(Glass.Polar_filter,mean(brdf_Glass,2),'Color','blue','Marker','o','LineStyle','--','LineWidth',2)
hold on
plot(Silicon.Polar_filter,mean(brdf_Silicon,2),'Color','red','Marker','diamond','LineStyle','--','LineWidth',2)
plot(Silicon.Polar_filter,mean(brdf_Silicon,2)+ mean(brdf_Glass,2),'Color','black','Marker','square','LineStyle','--','LineWidth',2)
grid('on')
set(gca,'YScale','log')
xlabel('Scatter Angle $\theta_s$ [$^\circ$]',FontSize = 26)
ylabel('B[1/sr]',FontSize = 26)
grid('on')
legend({'$\mathrm{B_{Glass + Enc}}$','$\mathrm{B_{Silicon}}$','$\mathrm{B_{Glass + Enc} + B_{Silicon}}$'},'Interpreter','latex','FontSize',24)





%% Radiant flux calc Vs tracked.

%PP_avg = mean(Power,1).';




%% All samples simulated 
mycolors = [0,1,0;0,0,1;0.8706,0.8706,0.1294;1,0,0;0.4000,0.0745,0.1373;0,0.60,0]; %{[0,0,1],[0,1,0],[0.8706,0.8706,0.1294],[1,0,0],[0.4000,0.0745,0.1373],[0,0.60,0]};
%colorization = dictionary;
%colors = "_" + ["Lightgreen","Ref","Gold","Red","Brown","Green"]

pol = cat(2,-flip(Gold.Polar_filter(~isnan(Gold.brdf_M(:,91)))),Gold.Polar_filter(~isnan(Gold.brdf_M(:,1))));
pol2 = cat(2,-flip(Gold.Polar_filter(~isnan(Gold_0.brdf_M(:,91)))),Gold.Polar_filter(~isnan(Gold_0.brdf_M(:,1))));

n = 5;
%Gold.brdf_M(~isnan(Gold.brdf_M(:,1)),1)
%Gold.brdf_M(~isnan(Gold.brdf_M(:,91)),91)
cat(1,flip(Gold.brdf_M(~isnan(Gold.brdf_M(:,91)),91)),Gold.brdf_M(~isnan(Gold.brdf_M(:,1)),1));
pol = mean_function(pol,n);
pol2 = mean_function(pol2,n);
gold = mean_function(cat(1,(Gold.brdf_M(~isnan(Gold.brdf_M(:,91)),91)),Gold.brdf_M(~isnan(Gold.brdf_M(:,1)),1)),n);
gold_45 = mean_function(cat(1,(Gold_45.brdf_M(~isnan(Gold_45.brdf_M(:,91)),91)),Gold_45.brdf_M(~isnan(Gold_45.brdf_M(:,1)),1)),n);
gold_un =  mean_function(cat(1,(Gold_unnormalized.brdf_M(~isnan(Gold_unnormalized.brdf_M(:,91)),91)),Gold_unnormalized.brdf_M(~isnan(Gold_unnormalized.brdf_M(:,1)),1)),n);
red = mean_function(cat(1,(Red.brdf_M(~isnan(Red.brdf_M(:,91)),91)),Red.brdf_M(~isnan(Red.brdf_M(:,1)),1)),n);
lightgreen = mean_function(cat(1,(LightGreen.brdf_M(~isnan(LightGreen.brdf_M(:,91)),91)),LightGreen.brdf_M(~isnan(LightGreen.brdf_M(:,1)),1)),n);
green = mean_function(cat(1,(Green.brdf_M(~isnan(Green.brdf_M(:,91)),91)),Green.brdf_M(~isnan(Green.brdf_M(:,1)),1)),n);
brown = mean_function(cat(1,(Brown.brdf_M(~isnan(Brown.brdf_M(:,91)),91)),Brown.brdf_M(~isnan(Brown.brdf_M(:,1)),1)),n);
silicon = mean_function(cat(1,(brdf_Silicon(~isnan(brdf_Silicon(:,91)),91)),brdf_Silicon(~isnan(brdf_Silicon(:,1)),1)),n);
combi = mean_function(cat(1,(Combined.brdf_M(~isnan(Combined.brdf_M(:,91)),91)),Combined.brdf_M(~isnan(Combined.brdf_M(:,1)),1)),n);
gold_30 = mean_function(cat(1,(Gold_30.brdf_M(~isnan(Gold_30.brdf_M(:,91)),91)),Gold_30.brdf_M(~isnan(Gold_30.brdf_M(:,1)),1)),n);
gold_60 = mean_function(cat(1,(Gold_60.brdf_M(~isnan(Gold_60.brdf_M(:,91)),91)),Gold_60.brdf_M(~isnan(Gold_60.brdf_M(:,1)),1)),n);
gold_0 = mean_function(cat(1,(Gold_0.brdf_M(~isnan(Gold_0.brdf_M(:,91)),91)),Gold_0.brdf_M(~isnan(Gold_0.brdf_M(:,1)),1)),n);


% cat(1,flip(Gold.brdf_M(:,91)),Gold.brdf_M(:,1))
%% Moving mean values 

figure(10)
%plot(Polar_filter(2:end),avg_pol(2:end),'Color','black','Marker','square','LineStyle','--','LineWidth',2)
plot(pol,gold,'Color',mycolors(3,:),'Marker','square','LineStyle','-','LineWidth',2)
hold on
plot(pol,red,'Color',mycolors(4,:),'Marker','square','LineStyle','-','LineWidth',2)
plot(pol,lightgreen,'Color',mycolors(1,:),'Marker','square','LineStyle','-','LineWidth',2)
plot(pol,green,'Color',mycolors(6,:),'Marker','square','LineStyle','-','LineWidth',2)
plot(pol,brown,'Color',mycolors(5,:),'Marker','square','LineStyle','-','LineWidth',2)
%plot(Polar_filter(1:end),brdf_M(:,1),'Color','blue','Marker','square','LineStyle','none','LineWidth',2)%
%plot(-Polar_filter(1:end),flip(brdf_M(:,91)),'Color','red','Marker','square','LineStyle','none','LineWidth',2)
grid('on')
set(gca,'FontSize',18)
lgd = legend({'Gold','Red','Lightgreen','Green','Brown'},'Interpreter','latex','FontSize',18);
title(lgd,'Input samples')
xlabel('Polar Angle $\theta_p [^\circ]$','FontSize',20)
ylabel('$B_S (\theta_i = 15^\circ)$ [1/sr]','FontSize',20)
set(gca,'YScale','log','YTick',[10^(-6),10^(-5),10^(-4),10^(-3),10^(-2),10^(-1),10^0,10^1,10^2])
xlim([-90,90])

%%
exp_data = [NaN,0.0241,0.0213,0.0470,0.0240,0.0140,0.0120,0.0119,0.0126,0.0143,0.0157,0.0162,0.0194,0.0251,0.0235,NaN,0.0201,0.0197,0.0212,0.0247,0.0321,0.0467,0.0675,0.0490,0.0190,0.0134,0.0118,0.0113,0.0111,0.0111,0.0114,0.0119,0.0131,0.0153,0.0201,0.0347,NaN];
figure(100)
plot(pol,silicon,'Color','black','Marker','square','LineStyle','-','LineWidth',2)
hold on
plot(cat(2,flip(-Gold.Polar_filter),Gold.Polar_filter),cat(1,(brdf_Silicon(:,91)),brdf_Silicon(:,1)),'Color','red','Marker','square','LineStyle','none','LineWidth',2)
plot(linspace(-90,90,37),exp_data,'Color','blue','Marker','square','LineStyle','-','LineWidth',2)
set(gca,'YScale','log')
set(gca,'FontSize',18)
grid('on')
xlabel('Polar Angle $\theta_p [^\circ]$','FontSize',20)
ylabel('$B_S (\theta_i = 15^\circ)$ [1/sr]','FontSize',20)
set(gca,'YScale','log')
xlim([-90,90])
legend({'$B_{Silicon,mean}$','$B_{Silicon,raw}$','$B_{exp}$'},'Interpreter','latex','FontSize',18)

%% Combined

figure(8)
plot(pol,combi,'Color','black','Marker','square','LineStyle','-','LineWidth',2)
hold on
plot(cat(2,flip(-Gold.Polar_filter),Gold.Polar_filter),cat(1,(Combined.brdf_M(:,91)),Combined.brdf_M(:,1)),'Color','red','Marker','square','LineStyle','none','LineWidth',2)
set(gca,'YScale','log')
set(gca,'FontSize',18)
grid('on')
xlabel('Polar Angle $\theta_p [^\circ]$','FontSize',20)
ylabel('$B_S (\theta_i = 15^\circ)$ [1/sr]','FontSize',20)
set(gca,'YScale','log')
xlim([-90,90])
legend({'$B_{Combined,mean}$','$B_{Combined,raw}$'},'Interpreter','latex','FontSize',18)

%% Other comparisons 
figure(8)
plot(pol,silicon,'Color','black','Marker','square','LineStyle','-','LineWidth',2)
hold on
plot(cat(2,flip(-Gold.Polar_filter),Gold.Polar_filter),cat(1,(brdf_Silicon(:,91)),brdf_Silicon(:,1)),'Color','red','Marker','square','LineStyle','none','LineWidth',2)
set(gca,'YScale','log')
set(gca,'FontSize',18)
grid('on')
xlabel('Polar Angle $\theta_p [^\circ]$','FontSize',20)
ylabel('$B_S (\theta_i = 15^\circ)$ [1/sr]','FontSize',20)
set(gca,'YScale','log')
xlim([-90,90])
legend({'$B_{Silicon,mean}$','$B_{Silicon,raw}$'},'Interpreter','latex','FontSize',18)

figure(12)
%plot(pol2,gold_0,'Color','black','Marker','square','LineStyle','-','LineWidth',2)
plot(pol,gold,'Color','red','Marker','square','LineStyle','-','LineWidth',2)
hold on 
%plot(pol,gold,'Color','red','Marker','square','LineStyle','-','LineWidth',2) % mycolors(3,:)
%plot(pol,gold_30,'Color','#FFA500','Marker','square','LineStyle','-','LineWidth',2)
%plot(pol,gold_45,'Color','#A020F0','Marker','square','LineStyle','-','LineWidth',2)
%plot(pol,gold_60,'Color',mycolors(6,:),'Marker','square','LineStyle','-','LineWidth',2)
plot(pol,gold_un,'Color','black','Marker','square','LineStyle','-','LineWidth',2)
grid('on')
set(gca,'FontSize',18)
%lgd = legend({'$B_{gold}(\theta_i = 15^\circ)$','$B_{gold}(\theta_i = 45^\circ)$','$B_{gold,unnorm}(\theta_i = 15^\circ)$'},'Interpreter','latex','FontSize',18,'Location','northwest');
%lgd = legend({'0','15','30','45','60'},'Interpreter','latex','FontSize',18,'Location','northwest');
%title(lgd,'$\theta_i [^\circ]$')
%title(lgd,'Input samples')
lgd = legend({'Normalized','Unnormalized'},'Interpreter','latex','FontSize',18,'Location','northwest');
xlabel('Polar Angle $\theta_p [^\circ]$','FontSize',20)
ylabel('$B_S$ [1/sr]','FontSize',20)
set(gca,'YScale','log','YTick',[10^(-6),10^(-5),10^(-4),10^(-3),10^(-2),10^(-1),10^0,10^1,10^2])
xlim([-90,90])

%%


%% Final contribution

figure(11)
%plot(Polar_filter(2:end),avg_pol(2:end),'Color','black','Marker','square','LineStyle','--','LineWidth',2)
plot(pol,gold+silicon,'Color',mycolors(3,:),'Marker','square','LineStyle','-','LineWidth',2)
hold on
plot(pol,red+silicon,'Color',mycolors(4,:),'Marker','square','LineStyle','-','LineWidth',2)
plot(pol,lightgreen+silicon,'Color',mycolors(1,:),'Marker','square','LineStyle','-','LineWidth',2)
plot(pol,green+silicon,'Color',mycolors(6,:),'Marker','square','LineStyle','-','LineWidth',2)
plot(pol,brown+silicon,'Color',mycolors(5,:),'Marker','square','LineStyle','-','LineWidth',2)
%plot(Polar_filter(1:end),brdf_M(:,1),'Color','blue','Marker','square','LineStyle','none','LineWidth',2)%
%plot(-Polar_filter(1:end),flip(brdf_M(:,91)),'Color','red','Marker','square','LineStyle','none','LineWidth',2)
grid('on')
set(gca,'FontSize',18)
lgd = legend({'Gold','Red','Lightgreen','Green','Brown'},'Interpreter','latex','FontSize',18);
title(lgd,'Input samples')
xlabel('Polar Angle $\theta_p [^\circ]$','FontSize',20)
ylabel('$B_S (\theta_i = 15^\circ)$ [1/sr]','FontSize',20)
set(gca,'YScale','log','YTick',[10^(-6),10^(-5),10^(-4),10^(-3),10^(-2),10^(-1),10^0,10^1,10^2])
xlim([-90,90])

%%
pol = cat(2,-flip(Gold.Polar_filter),Gold.Polar_filter);

figure(3)
%plot(Polar_filter(2:end),avg_pol(2:end),'Color','black','Marker','square','LineStyle','--','LineWidth',2)
plot(pol,cat(1,(Gold.brdf_M(:,91)),Gold.brdf_M(:,1)),'Color',mycolors(3,:),'Marker','square','LineStyle','none','LineWidth',2)
hold on
plot(pol,cat(1,(brdf_Silicon(:,91)),brdf_Silicon(:,1)),'Color','blue','Marker','square','LineStyle','none','LineWidth',2)
plot(pol,cat(1,(brdf_Silicon(:,91)),brdf_Silicon(:,1))+cat(1,(Gold.brdf_M(:,91)),Gold.brdf_M(:,1)) ,'Color','black','Marker','square','LineStyle','none','LineWidth',2)

%plot(pol,cat(1,flip(Silicon.brdf_M(:,91)),Gold.brdf_M(:,1)),'Color',mycolors(3,:),'Marker','square','LineStyle','none','LineWidth',2)
%plot(pol,cat(1,flip(Red.brdf_M(:,91)),Red.brdf_M(:,1)),'Color',mycolors(4,:),'Marker','square','LineStyle','none','LineWidth',2)
%plot(pol,cat(1,flip(LightGreen.brdf_M(:,91)),LightGreen.brdf_M(:,1)),'Color',mycolors(1,:),'Marker','square','LineStyle','none','LineWidth',2)
%plot(pol,cat(1,flip(Green.brdf_M(:,91)),Green.brdf_M(:,1)),'Color',mycolors(6,:),'Marker','square','LineStyle','none','LineWidth',2)
%plot(pol,cat(1,flip(Brown.brdf_M(:,91)),Brown.brdf_M(:,1)),'Color',mycolors(5,:),'Marker','square','LineStyle','none','LineWidth',2)
%plot(Polar_filter(1:end),brdf_M(:,1),'Color','blue','Marker','square','LineStyle','none','LineWidth',2)%
%plot(-Polar_filter(1:end),flip(brdf_M(:,91)),'Color','red','Marker','square','LineStyle','none','LineWidth',2)
grid('on')
set(gca,'FontSize',18)
%legend({'Gold','Red','Lightgreen','Green','Brown'},'Interpreter','latex','FontSize',16)
grid('on')
set(gca,'YScale','log')
legend({'$\mathrm{B_{Glass + Enc}}$','$\mathrm{B_{Silicon}}$','$\mathrm{B_{Glass + Enc} + B_{Silicon}}$'},'Interpreter','latex','FontSize',24,'Location','northwest')
xlabel('Polar Angle $\theta_p [^\circ]$','FontSize',20)
ylabel('$B_S (\theta_i = 15^\circ)$ [1/sr]','FontSize',20)
set(gca,'YScale','log')
xlim([-90,90])
%%
figure(3)
%plot(Polar_filter(2:end),avg_pol(2:end),'Color','black','Marker','square','LineStyle','--','LineWidth',2)
plot(pol,cat(1,(Gold.brdf_M(:,91)),Gold.brdf_M(:,1)),'Color',mycolors(3,:),'Marker','square','LineStyle','none','LineWidth',2)
hold on
plot(pol,cat(1,(Gold_10000.brdf_M(:,91)),Gold_10000.brdf_M(:,1)),'Color','red','Marker','square','LineStyle','none','LineWidth',2)
plot(pol,cat(1,(Gold_25000.brdf_M(:,91)),Gold_25000.brdf_M(:,1)),'Color','black','Marker','square','LineStyle','none','LineWidth',2)

set(gca,'FontSize',18)
%legend({'Gold','Red','Lightgreen','Green','Brown'},'Interpreter','latex','FontSize',16)
grid('on')
set(gca,'YScale','log')
lgd = legend({'5000','10000','25000'},'Interpreter','latex','FontSize',24,'Location','northwest');
title(lgd,'$N_{rays}[\#]$ for $B_{gold}$','FontSize',18,'Interpreter','latex')
xlabel('Polar Angle $\theta_p [^\circ]$','FontSize',20)
ylabel('$B_S (\theta_i = 15^\circ)$ [1/sr]','FontSize',20)
set(gca,'YScale','log')
xlim([-90,90])

%% Pure silicon
figure(5)
plot(pol,cat(1,(brdf_Silicon(:,91)),brdf_Silicon(:,1)),'Color','blue','Marker','square','LineStyle','none','LineWidth',2)
set(gca,'YScale','log')
grid('on')
xlabel('Polar Angle $\theta_p [^\circ]$','FontSize',20)
ylabel('$B_S (\theta_s = 15^\circ)$ [1/sr]','FontSize',20)
set(gca,'YScale','log')
xlim([-90,90])

%% Polar plots

figure;
polarscatter(deg2rad(pA), pP,20,reshape(Silicon.M_power.*10^6,[],1),'filled')
% 
colormap(jet(255))
cb = colorbar();
title(cb,'Radiant Flux $\Phi_s [\mathrm{\mu W}]$','Fontsize',18,'Interpreter','latex')
rticks(0:15:90)
%rticklabels({'\theta_p = 0\circ','\theta_p = 15\circ','\theta_p = 30\circ','\theta_p = 45\circ','\theta_p = 60\circ','\theta_p = 75\circ','\theta_p = 90\circ'}')
%pax = gca;
set(gca,"CLim",[0,0.005*10^(3)]) 
set(gca,'Layer','Top')
set(gca,'ThetaColor','blue','Rcolor','#FAFF00','FontSize',18,'FontWeight','bold','LineWidth',3)

%% RMSE

file_dir = dir('C:\Users\s194086\Documents\Zemax\ZOS-API Projects\MATLABZOSConnection');
Files = extractfield(file_dir,'name');

Lenz = load(fullfile('C:\Users\s194086\Documents\Zemax\ZOS-API Projects\MATLABZOSConnection',Files{3}));
%load(Files{1})
zero_point = round(numel(Lenz.white_rel_sensor_ang(:,1))/2);
theta = Lenz.f_ellipse_angles;
alpha = Lenz.white_rel_sensor_ang(:,1);
brdf = Lenz.BRDF_brdf;
%Lenz.BRDF_brdf(~isnan(Lenz.BRDF_brdf(:,2,:)),:,1);
%cat(1,flip(Gold.brdf_M(~isnan(Gold.brdf_M(:,91)),91)),Gold.brdf_M(~isnan(Gold.brdf_M(:,1)),1));
%brdf(~isnan(brdf(:,1,8)),1,8)
PV_names = ['Green','Gold','Ref','Brown','Red','LightGreen'];
sim_list = [green,gold,gold_45,brown,red,lightgreen];
RMSE = [];
STD = [];
for n=1:6
    if n == 3
        continue
    else
    exp_meas = brdf(~isnan(brdf(:,2,n)),2,n); 
    RMSE = [RMSE,rmse(sim_list(n),exp_meas)]
    
    end
end


%% FUnctions

function result=mean_function(data,n)
result = zeros(1, floor(length(data) / n));

% Calculate the mean every n points
for i = 1:floor(length(data) / n)
    result(i) = mean(data((i-1)*n + 1:i*n));
end
end



