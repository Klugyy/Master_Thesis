close all
clear all

file_dir = dir(fullfile('..\','Processed_data/'));
Files = extractfield(file_dir,'name');
Files(1:2) = [];

Enc = load(fullfile('..\','Processed_data/',Files{4}));
Sim = load(fullfile('..\','Processed_data/',Files{5}));
Sim_fin = load(fullfile('..\','Processed_data/',Files{6}));

brdf_encaps = Enc.BRDF_brdf; %Green,Red, Gold. 
brdf_sim = Sim.BRDF_brdf; % Gold,Ref,red,lightgreen,brown,green,glass, silicon 
theta = Sim.f_ellipse_angles;
alpha = Sim.white_rel_sensor_ang(:,1);



%% First attempt of isolating encapsulant brdf for Green,Red and Gold. Simple subtraction.
set(0,'defaultTextInterpreter','latex')
set(0,'defaultLegendInterpreter','latex')
mycolors = [0,0,0;1,0,0;1,0.75,0;0,1,0;0,0.5,0;0,0,1;0,0,0.5;1,1,0;0.5,0.5,0;1,0,1;0.5,0,0.5];
PV_modules = ["Gold","Blue","Red","LightGreen","Brown","Green","Glass","Silicon"];
symbol = ["o","^","Pentagram","diamond","square",">"];
co = get(gca,'ColorOrder'); % Initial
set(gca, 'ColorOrder', mycolors(1:5,:), 'NextPlot', 'replacechildren');
brdf_calc = abs(brdf_sim(:,:,2) - brdf_sim);
brdf_calc_2 = abs(brdf_sim ./ brdf_sim(:,:,2)).*brdf_sim(:,:,2); %abs(brdf_sim(:,:,2)-brdf_sim)./(brdf_sim); 
brdf_calc_3 = zeros(size(brdf_calc_2));
brdf_calc_3(19-6:19+6,:,:) = abs(brdf_sim(19-6:19+6,:,2) - brdf_sim(19-6:19+6,:,:));
brdf_calc_3(1:19-6,:,:) = abs(brdf_sim(1:19-6,:,2) ./ brdf_sim(1:19-6,:,:)).*brdf_sim(1:19-6,:,:);
brdf_calc_3(19+6:end,:,:) = abs(brdf_sim(19+6:end,:,2) ./ brdf_sim(19+6:end,:,:)).*brdf_sim(19+6:end,:,:);
angle_color = [0,0,0;1,0,0;1,0.6,0;0.6,0.1,1;0,0.6,0];





figure(1)
for i=1:numel(theta)
p1(i) = plot(alpha,brdf_sim(:,i,2),'LineStyle','--','Marker',symbol(1),'LineWidth',2,'Color',angle_color(i,:));
hold on
p1_2(i) = plot(alpha,brdf_sim(:,i,2),'LineStyle','--','Marker',symbol(1),'LineWidth',2,'Color',angle_color(i,:));
p2(i) = plot(alpha,brdf_sim(:,i,1),'LineStyle','--','Marker',symbol(2),'LineWidth',2,'Color',angle_color(i,:)); % Gold
%p3 = plot(alpha,brdf_sim(:,:,3),'LineStyle','--','Marker',symbol(3),'LineWidth',2); % Red
%p4 = plot(alpha,brdf_sim(:,:,6),'LineStyle','none','Marker',symbol(4),'LineWidth',2); % Green
end
xlim([-90,90])
set(gca,'FontSize',16)
xlabel('Relative Sensor angle $\alpha [^\circ]$ ','FontSize',22)
ylabel('B[1/sr] ','FontSize',22)
hold off
set(gca,'YScale','log')
grid('on')
lgd = legend(p1,{'0','15','30','45','60'},'Interpreter','latex','Location','northeast','FontSize',16,'Orientation','vertical');% {sprintf('%s = %d','$\theta$',theta(1)),sprintf('%s = %d','$\theta_e$',theta(2)),sprintf('%s = %d','$\theta_e$',theta(3)),sprintf('%s = %d','$\theta_e$',theta(4)),sprintf('%s = %d','$\theta_e$',theta(5))}
title(lgd,'$\theta_i [^\circ]$')
ah1 = axes('position',get(gca,'position'),'visible','off');
lgd2 = legend(ah1,[p1_2(1),p2(1)],{'Reference','Gold'},'Interpreter','latex','Location','southwest','FontSize',18,'Orientation','horizontal');
%%
figure(2)
for i=1:numel(theta)
p1(i) = plot(alpha,brdf_sim(:,i,2),'LineStyle','--','Marker',symbol(1),'LineWidth',2,'Color',angle_color(i,:));
hold on
p1_2(i) = plot(alpha,brdf_sim(:,i,2),'LineStyle','--','Marker',symbol(1),'LineWidth',2,'Color',angle_color(i,:));
p2(i) = plot(alpha,brdf_sim(:,i,3),'LineStyle','--','Marker',symbol(2),'LineWidth',2,'Color',angle_color(i,:)); % Gold
%p3 = plot(alpha,brdf_sim(:,:,3),'LineStyle','--','Marker',symbol(3),'LineWidth',2); % Red
%p4 = plot(alpha,brdf_sim(:,:,6),'LineStyle','none','Marker',symbol(4),'LineWidth',2); % Green
end
set(gca,'FontSize',16)
xlim([-90,90])
xlabel('Relative Sensor angle $\alpha [^\circ]$ ','FontSize',22)
ylabel('B [1/sr] ','FontSize',22)
hold off
set(gca,'YScale','log')
grid('on')
lgd = legend(p1,{'0','15','30','45','60'},'Interpreter','latex','Location','northeast','FontSize',16,'Orientation','vertical');% {sprintf('%s = %d','$\theta$',theta(1)),sprintf('%s = %d','$\theta_e$',theta(2)),sprintf('%s = %d','$\theta_e$',theta(3)),sprintf('%s = %d','$\theta_e$',theta(4)),sprintf('%s = %d','$\theta_e$',theta(5))}
%lgd = legend(p1,{sprintf('%s = %d','$\theta$',theta(1)),sprintf('%s = %d','$\theta$',theta(2)),sprintf('%s = %d','$\theta$',theta(3)),sprintf('%s = %d','$\theta$',theta(4)),sprintf('%s = %d','$\theta$',theta(5))},'Interpreter','latex','Location','northeast','FontSize',16,'Orientation','vertical');
title(lgd,'$\theta_i [^\circ]$')
ah1 = axes('position',get(gca,'position'),'visible','off');
lgd2 = legend(ah1,[p1_2(1),p2(1)],{'Reference','Red'},'Interpreter','latex','Location','southwest','FontSize',16,'Orientation','horizontal');



%% The calculated encapsulant brdf.
% This seems to be done for the green encapsulant. 
figure(3)
hold on
for i=1:numel(theta)
if i == 0
p1(i) = plot(alpha+5,brdf_encaps(:,i,1),'LineStyle','-','Marker',symbol(1),'LineWidth',2,'Color',angle_color(i,:));
p11(i) = plot(alpha+5,brdf_encaps(:,i,1),'LineStyle','-','Marker',symbol(1),'LineWidth',2,'Color',angle_color(i,:));
else
    p1(i) = plot(alpha-5,brdf_encaps(:,i,1),'LineStyle','-','Marker',symbol(1),'LineWidth',2,'Color',angle_color(i,:));
    p11(i) = plot(alpha-5,brdf_encaps(:,i,1),'LineStyle','-','Marker',symbol(1),'LineWidth',2,'Color',angle_color(i,:));

end

p2(i) = plot(alpha,brdf_calc(:,i,6),'LineStyle','-','Marker',symbol(2),'LineWidth',2,'Color',angle_color(i,:));
RR = sqrt(sum((brdf_encaps(~isnan(brdf_encaps(:,i,1)),i,1) - brdf_calc(~isnan(brdf_calc(:,i,6)),i,6)).^2) / (numel(brdf_calc(~isnan(brdf_calc(:,i,6)),i,6))-1));
R = rmse(brdf_encaps(~isnan(brdf_encaps(:,i,1)),i,1),brdf_calc(~isnan(brdf_calc(:,i,6)),i,6));
annotation('textbox',[0.25,0.6-0.05*i,0.3,0.3],'String',"RMSE:" + num2str(round(RR,2)),'BackgroundColor',angle_color(i,:),'FontSize',16,'FitBoxToText','on','Color','white')
end
hold off
set(gca,'YScale','log')
set(gca,'FontSize',18)
grid('on')
xlim([-90,90])
xlabel('Relative Sensor Angle $\alpha [^\circ]$ ','FontSize',22)
ylabel('B [1/sr] ','FontSize',22)
%lgd = legend(p1,{sprintf('%s = %d','$\theta$',theta(1)),sprintf('%s = %d','$\theta$',theta(2)),sprintf('%s = %d','$\theta$',theta(3)),sprintf('%s = %d','$\theta$',theta(4)),sprintf('%s = %d','$\theta$',theta(5))},'Interpreter','latex','Location','northeast','FontSize',16,'Orientation','vertical');
ah1 = axes('position',get(gca,'position'),'visible','off');
lgd = legend(p1,{'0','15','30','45','60'},'Interpreter','latex','Location','northeast','FontSize',16,'Orientation','vertical');% {sprintf('%s = %d','$\theta$',theta(1)),sprintf('%s = %d','$\theta_e$',theta(2)),sprintf('%s = %d','$\theta_e$',theta(3)),sprintf('%s = %d','$\theta_e$',theta(4)),sprintf('%s = %d','$\theta_e$',theta(5))}
%lgd = legend(p1,{sprintf('%s = %d','$\theta$',theta(1)),sprintf('%s = %d','$\theta$',theta(2)),sprintf('%s = %d','$\theta$',theta(3)),sprintf('%s = %d','$\theta$',theta(4)),sprintf('%s = %d','$\theta$',theta(5))},'Interpreter','latex','Location','northeast','FontSize',16,'Orientation','vertical');
title(lgd,'$\theta_i [^\circ]$')
lgd2 = legend(ah1,[p11(1),p2(1)],{'Encapsulant:Measured','Encapsulant calculated $|B_{color}-B_{ref}|$'},'Interpreter','latex','Location','northwest','FontSize',20,'Orientation','horizontal');

%%

figure(4)
hold on
for i=1:numel(theta)
if i == 0
p1(i) = plot(alpha+5,brdf_encaps(:,i,1),'LineStyle','-','Marker',symbol(1),'LineWidth',2,'Color',angle_color(i,:));
p11(i) = plot(alpha+5,brdf_encaps(:,i,1),'LineStyle','-','Marker',symbol(1),'LineWidth',2,'Color',angle_color(i,:));
else
    p1(i) = plot(alpha-5,brdf_encaps(:,i,1),'LineStyle','-','Marker',symbol(1),'LineWidth',2,'Color',angle_color(i,:));
    p11(i) = plot(alpha-5,brdf_encaps(:,i,1),'LineStyle','-','Marker',symbol(1),'LineWidth',2,'Color',angle_color(i,:));

end
brdf_encaps(~isnan(brdf_encaps(:,i,1)),i,1) - brdf_calc_2(~isnan(brdf_calc_2(:,i,6)),i,6).^2 / (numel(brdf_calc_2(~isnan(brdf_calc_2(:,i,6)),i,6))-1)
p2(i) = plot(alpha,brdf_calc_2(:,i,6),'LineStyle','-','Marker',symbol(2),'LineWidth',2,'Color',angle_color(i,:));
RR = sqrt(sum((brdf_encaps(~isnan(brdf_encaps(:,i,1)),i,1) - brdf_calc_2(~isnan(brdf_calc_2(:,i,6)),i,6)).^2) / (numel(brdf_calc_2(~isnan(brdf_calc_2(:,i,6)),i,6))-1));
annotation('textbox',[0.25,0.6-0.05*i,0.3,0.3],'String',"RMSE:" + num2str(round(RR,2)),'BackgroundColor',angle_color(i,:),'FontSize',16,'FitBoxToText','on','Color','white')
end
hold off
set(gca,'YScale','log')
grid('on')
set(gca,'FontSize',18)
xlim([-90,90])
xlabel('Relative Sensor Angle $\alpha [^\circ]$ ','FontSize',22)
ylabel('Brdf [1/sr] ','FontSize',22)
lgd = legend(p1,{'0','15','30','45','60'},'Interpreter','latex','Location','northeast','FontSize',16,'Orientation','vertical');% {sprintf('%s = %d','$\theta$',theta(1)),sprintf('%s = %d','$\theta_e$',theta(2)),sprintf('%s = %d','$\theta_e$',theta(3)),sprintf('%s = %d','$\theta_e$',theta(4)),sprintf('%s = %d','$\theta_e$',theta(5))}
%lgd = legend(p1,{sprintf('%s = %d','$\theta$',theta(1)),sprintf('%s = %d','$\theta$',theta(2)),sprintf('%s = %d','$\theta$',theta(3)),sprintf('%s = %d','$\theta$',theta(4)),sprintf('%s = %d','$\theta$',theta(5))},'Interpreter','latex','Location','northeast','FontSize',16,'Orientation','vertical');
title(lgd,'$\theta_i [^\circ]$')
%lgd = legend(p1,{sprintf('%s = %d','$\theta$',theta(1)),sprintf('%s = %d','$\theta$',theta(2)),sprintf('%s = %d','$\theta$',theta(3)),sprintf('%s = %d','$\theta$',theta(4)),sprintf('%s = %d','$\theta$',theta(5))},'Interpreter','latex','Location','northeast','FontSize',16,'Orientation','vertical');
ah1 = axes('position',get(gca,'position'),'visible','off');
lgd2 = legend(ah1,[p11(1),p2(1)],{'Encapsulant:Measured','Encapsulant calculated $(B_{color}/B_{ref}) \cdot B_{ref}$'},'Interpreter','latex','Location','northwest','FontSize',16,'Orientation','horizontal');

%%

figure(5)
hold on
for i=1:numel(theta)
if i == 0
p1(i) = plot(alpha+5,brdf_encaps(:,i,1),'LineStyle','-','Marker',symbol(1),'LineWidth',2,'Color',mycolors(i,:));
p11(i) = plot(alpha+5,brdf_encaps(:,i,1),'LineStyle','-','Marker',symbol(1),'LineWidth',2,'Color',mycolors(i,:));
else
    p1(i) = plot(alpha-5,brdf_encaps(:,i,1),'LineStyle','-','Marker',symbol(1),'LineWidth',2,'Color',mycolors(i,:));
    p11(i) = plot(alpha-5,brdf_encaps(:,i,1),'LineStyle','-','Marker',symbol(1),'LineWidth',2,'Color',mycolors(i,:));

end
%brdf_encaps(~isnan(brdf_encaps(:,i,1)),i,1) - brdf_calc_3(~isnan(brdf_calc_3(:,i,6)),i,6).^2 / (numel(brdf_calc_2(~isnan(brdf_calc_3(:,i,6)),i,6))-1)
p2(i) = plot(alpha,brdf_calc_3(:,i,6),'LineStyle','-','Marker',symbol(2),'LineWidth',2,'Color',mycolors(i,:));
RR = sqrt(sum((brdf_encaps(~isnan(brdf_encaps(:,i,1)),i,1) - brdf_calc_3(~isnan(brdf_calc_3(:,i,6)),i,6)).^2) / (numel(brdf_calc_3(~isnan(brdf_calc_3(:,i,6)),i,6))-1));
annotation('textbox',[0.25,0.6-0.05*i,0.3,0.3],'String',"RMSE:" + num2str(round(RR,2)),'BackgroundColor',mycolors(i,:),'FontSize',16,'FitBoxToText','on','Color','white')
end
hold off
set(gca,'YScale','log')
grid('on')
xlabel('Sensor angle $\alpha [^\circ]$ ','FontSize',16)
ylabel('Brdf [1/sr] ','FontSize',16)
lgd = legend(p1,{sprintf('%s = %d','$\theta$',theta(1)),sprintf('%s = %d','$\theta$',theta(2)),sprintf('%s = %d','$\theta$',theta(3)),sprintf('%s = %d','$\theta$',theta(4)),sprintf('%s = %d','$\theta$',theta(5))},'Interpreter','latex','Location','northeast','FontSize',16,'Orientation','vertical');
ah1 = axes('position',get(gca,'position'),'visible','off');
lgd2 = legend(ah1,[p11(1),p2(1)],{'Encapsulant:Measured','Encapsulant calculated : Mixed '},'Interpreter','latex','Location','southwest','FontSize',16,'Orientation','horizontal');


%% Plotting the scattering from mie scattering
%ax2 = axes('Position',[0.2,0.5,0.25,0.25]);
%ax1 = axes('Position',[0.2 0.1 1 1]);
%ax2.XTickLabel = []
%ax2.YTickLabel = [];
for n_c = 0:10
mycolors = [0,0,0;1,0,0;1,0.75,0;0,1,0;0,0.5,0;0,0,1;0,0,0.5;0.5,0.5,0;1,0,1;0.5,0,0.5;0.7,0.3,0.15];
n_encaps = 1.4; %+ 6.52*10^(-7)*1j;
n_TiO2 = 2.62; % 1.4+0.1*n_c ; % 1.330
wav = (550)*10^(-9); % in m
rho =0.1*10^(-6) + 100*n_c*10^(-8); %50*10^(-6); %25^(-6);%100*n_c*10^(-8); %0.263*10^(-6);%; % in m Radius of sphere.
x = (2*pi*rho*n_encaps) / wav; %3;% %(2*pi*rho*n_encaps) / wav;
m = n_TiO2/n_encaps;%1.330 - 1*10^(-8)*j;%1.022+ 1.119*10^(-2)*1j; %1.330 - 1*10^(-8)*j; %n_TiO2/n_encaps ;%1.33 - 1e-8*j;%; % 1.330+i*1*10^(-8); %1.022+ 1.119*10^(-2)*1j;
theta = linspace(0,360,361);
mu = (cosd(theta));
I_perp = zeros(1,numel(mu));
I_para = zeros(1,numel(mu));
Q = Mie(m,x);% [qext qsca qabs qb asy qratio]
a = Q(2)/Q(1);
for i=1:numel(mu)
S12 = Mie_S12(x,m,mu(i));
S(i,:) = S12;
I_perp(i) = abs(S12(1))^2;
I_para(i) = abs(S12(2))^2;
end

beta = (I_perp+I_para)/2;
norm = 4*pi*x^2*Q(2);
tot = 2*pi*trapz(beta.*sind(theta),mu);%2*pi*sum(beta.*abs(mu(1)-mu(2)));
P = (I_perp- I_para) ./ (I_perp+I_para);
str = strcat(num2str(round(rho*10^6,5)),'=',num2str(round(x)));
figure(6)
%plot(theta,I_perp,'Color','blue','LineWidth',2)
%hold on
%plot(theta,I_para,'Color','green','LineWidth',2)
%plot(theta,beta,'Color','black','LineWidth',2)
plot(linspace(-90,90,181),beta(1,91:181+90)./tot,'Color',mycolors(n_c+1,:),'DisplayName',str,'LineWidth',2) % num2str(round((m),2))
hold on
set(gca,'YScale','log')
%ylim([10^(-2),10^2])
%xlim([91,181+90])
xlim([-90,90])
lgx = legend('FontSize',20,'Location','eastoutside');
title(lgx,'$\rho [\mathrm{\mu m}] = x_s$','FontSize',16) % \mathrm{Re}(m) = \mathrm{Re} \left[\frac{n_{TiO_2}}{n_{enc}} \right]\rho [\mathrm{\mu m}] = x_s 
%legend({'$I_{\perp}$','$I_{\parallel}$','$\beta$'},'Interpreter','latex','FontSize',22)
grid('on')
set(gca,'FontSize',18)
xlabel('Polar Angle $\theta_p [^\circ]$','FontSize',22)
ylabel('Normalized scattering intensity $\tilde{\beta}$ [1/sr]','FontSize',22)

beta_norm = beta./tot;
beta_plot = beta_norm(1,91:5:181+90)';
fighandler = figure(7);
%p1= plot(alpha,brdf_sim(:,2,2),'LineStyle','-.','Marker',symbol(1),'LineWidth',2,'Color','red');
%p1_2(i) =
%p3 = plot(alpha, beta_norm(1,91:5:181+90)' ,'LineStyle','-.','Marker',symbol(3),'LineWidth',2,'Color','black'); %beta(1,91:5:181+90))'
plot(alpha, beta_plot + brdf_sim(:,2,2),'LineStyle','-.','Marker',symbol(4)','LineWidth',2,'Color',mycolors(n_c+1,:));
hold on
%box on 
%indexOfinterest = (alpha <= 25 & alpha > 0);
%plot(ax2,alpha(indexOfinterest),beta_plot(indexOfinterest) + brdf_sim(indexOfinterest,2,2),'LineStyle','-.','Marker',symbol(4)','LineWidth',2,'Color',mycolors(n_c+1,:))

%p4 = plot(Sim_fin.white_rel_sensor_ang(:,2),Sim_fin.BRDF_brdf(:,2,1),'LineStyle','-.','Marker','none','LineWidth',2,'Color','magenta');
%p3 = plot(alpha,brdf_sim(:,:,3),'LineStyle','--','Marker',symbol(3),'LineWidth',2); % Red
%p4 = plot(alpha,brdf_sim(:,:,6),'LineStyle','none','Marker',symbol(4),'LineWidth',2); % Green

%legend({'$B_{glass}$','$B_{Gold}$','$B_{Model}$','$B_{Gold} + B_{Model}$'},'Interpreter','latex','FontSize',18)
%lgd = legend(p1,{sprintf('%s = %d','$\theta$',theta(1)),sprintf('%s = %d','$\theta$',theta(2)),sprintf('%s = %d','$\theta$',theta(3)),sprintf('%s = %d','$\theta$',theta(4)),sprintf('%s = %d','$\theta$',theta(5))},'Interpreter','latex','Location','northeast','FontSize',16,'Orientation','vertical');
%ah1 = axes('position',get(gca,'position'),'visible','off');
%lgd2 = legend(ah1,[p1_2(1),p2(1)],{'Reference','Red'},'Interpreter','latex','Location','southwest','FontSize',16,'Orientation','horizontal');
end

plot(alpha,brdf_sim(:,2,2),'LineStyle','--','Marker',symbol(1),'LineWidth',2,'Color','cyan');%
hold on
p2 = plot(alpha,brdf_sim(:,2,3),'LineStyle','-.','Marker',symbol(2),'LineWidth',2,'Color','blue'); % Gold
set(gca,'FontSize',18)
xlabel('Relative Sensor Angle $\alpha [^\circ]$ ','FontSize',24)
ylabel(' Brdf B[1/sr] ','FontSize',24)
legend({'$B_{Gold} + B_{Model}$ ','','','','','','','','','','','$B_{glass}$','$B_{Gold}$'},'Interpreter','latex','FontSize',20)
%hold off
set(gca,'YScale','log')
xlim([-90,90])
grid('on')

global toolId
toolId = 1234;
magnifyOnFigure(...
        fighandler,...
        'units', 'pixels',...
        'magnifierShape', 'square',...
        'initialPositionSecondaryAxes', [326.933 259.189 125 125],...
        'initialPositionMagnifier',     [1000 750 150 300],...    
        'mode', 'interactive',...    
        'displayLinkStyle', 'straight',...        
        'edgeWidth', 2,...
        'edgeColor', 'black',...
        'secondaryAxesXLim',[0,25],...
        'secondaryAxesXLim', [0,2],...
        'secondaryAxesFaceColor', [0.91 0.91 0.91]... 
            ); 
disp('Press a key...')
%'secondaryAxesXLim': Initial XLim value of the secondary axes
%'secondaryAxesYLim': Initial YLim value of the secondary axes
pause;
%}
%% Parameter change

x = 1.71;
m = 1;

I_perp = zeros(1,numel(mu));
I_para = zeros(1,numel(mu));
Q = Mie(m,x);% [qext qsca qabs qb asy qratio]
a = Q(2)/Q(1);
for i=1:numel(mu)
S12 = Mie_S12(x,m,mu(i));
S(i,:) = S12;
I_perp(i) = abs(S12(1))^2;
I_para(i) = abs(S12(2))^2;
end

beta = (I_perp+I_para)/2;
norm = 4*pi*x^2*Q(2);
tot = 2*pi*trapz(beta.*sind(theta),mu);%2*pi*sum(beta.*abs(mu(1)-mu(2)));


%% Mie checkup
pin =zeros(numel(theta),round(x + 4*x^(1/3) + 2));
tin = zeros(numel(theta),round(x + 4*x^(1/3) + 2));
for zug=1:numel(theta)
pt=Mie_pt(mu(zug),round(x + 4*x^(1/3) + 2));
pin(zug,:) =pt(1,:);
tin(zug,:) =pt(2,:);
end
ratio = [0,5/3,11/6,2,35/15,50/21,70/28,100/35];
ax = gca;
%co = get(gca,'ColorOrder'); % Initial
%set(ax, 'ColorOrder', mycolors(1:11,:), 'NextPlot', 'replacechildren');
figure(11)
for i=1:8
%ax.ColorOrder = mycolors;
plot(theta,pin(:,i),'-.','LineWidth',2,'Color',mycolors(i,:))
hold on
end
hold off
grid('on')
set(gca,'FontSize',20)
legend({'n=1','n=2','n=3','n=4','n=5','n=6','n=7','n=8'},'FontSize',20,'Location','best')
xlabel('Polar Angle $\theta_p [^\circ]$','FontSize',26)
ylabel('$\pi_n$','FontSize',44)
xlim([0,180])
%%
%co = get(gca,'ColorOrder'); % Initial
%set(gca, 'ColorOrder', mycolors(2:11,:), 'NextPlot', 'replacechildren');
figure(12)
for i=1:8
plot(theta,tin(:,i),'-.','LineWidth',2,'Color',mycolors(i,:))
hold on
end
hold off
grid('on')
set(gca,'FontSize',20)
legend({'n=1','n=2','n=3','n=4','n=5','n=6','n=7','n=8'},'FontSize',20,'Location','best')
xlabel('Polar Angle $\theta_p [^\circ]$','FontSize',26)
ylabel('$\tau_n$','FontSize',44)
xlim([0,180])

%figure(13)
%plot(ratio)



%% Overlap prediction theta constant
%Sim_fin.white_rel_sensor_ang(:,2)

beta_norm = beta./tot;
beta_plot = cat(2,cat(2,beta_norm(1,91:5:181),beta(1,181+5:5:181+20)),beta_norm(181+25:5:181+90));


figure(10)
p1 = plot(alpha,brdf_sim(:,2,2),'LineStyle','-.','Marker',symbol(1),'LineWidth',2,'Color','red');
hold on
%p1_2(i) = plot(alpha,brdf_sim(:,2,2),'LineStyle','--','Marker',symbol(1),'LineWidth',2,'Color','red');
p2 = plot(alpha,brdf_sim(:,2,3),'LineStyle','-.','Marker',symbol(2),'LineWidth',2,'Color','blue'); % Gold
p3 = plot(alpha, beta_norm(1,91:5:181+90)' ,'LineStyle','-.','Marker',symbol(3),'LineWidth',2,'Color','black'); %beta(1,91:5:181+90))'
p4 = plot(alpha,beta_norm(1,91:5:181+90)'+ brdf_sim(:,2,2),'LineStyle','-.','Marker',symbol(4)','LineWidth',2,'Color','green');
%p4 = plot(Sim_fin.white_rel_sensor_ang(:,2),Sim_fin.BRDF_brdf(:,2,1),'LineStyle','-.','Marker','none','LineWidth',2,'Color','magenta');
%p3 = plot(alpha,brdf_sim(:,:,3),'LineStyle','--','Marker',symbol(3),'LineWidth',2); % Red
%p4 = plot(alpha,brdf_sim(:,:,6),'LineStyle','none','Marker',symbol(4),'LineWidth',2); % Green
xlabel('Relative Sensor Angle $\alpha [^\circ]$ ','FontSize',16)
ylabel('Brdf [1/sr] ','FontSize',16)
hold off
set(gca,'YScale','log')
grid('on')
legend({'$B_{glass}$','$B_{Gold}$','$B_{Model}$','$B_{Gold} + B_{Model}$'},'Interpreter','latex','FontSize',18)
%lgd = legend(p1,{sprintf('%s = %d','$\theta$',theta(1)),sprintf('%s = %d','$\theta$',theta(2)),sprintf('%s = %d','$\theta$',theta(3)),sprintf('%s = %d','$\theta$',theta(4)),sprintf('%s = %d','$\theta$',theta(5))},'Interpreter','latex','Location','northeast','FontSize',16,'Orientation','vertical');
%ah1 = axes('position',get(gca,'position'),'visible','off');
%lgd2 = legend(ah1,[p1_2(1),p2(1)],{'Reference','Red'},'Interpreter','latex','Location','southwest','FontSize',16,'Orientation','horizontal');


%% Get encaps data 

filename = "Encaps_TIS_v2.txt";

% Getting correct formatting of data.
Source = 'Measured';
Symmetry = 'PlaneSymmetrical';
Spectralcontent = 'Monochrome';
ScatterType = 'BRDF';
SampleRotation = 90;
AngleOfIncidence = theta;
ScatterAzimuth = [0,180];
ScatterRadial = linspace(0,150,150/5 +1 );%alpha(19:end);
SpectralContentincicator = 'Monochrome';
spacing = 15; % Angle spacing. 

%% Writing to doc.

DIR=fopen(filename,'wt+'); % Clear and open file. 
% Making doc header from variables. 
writelines("Source  "+ Source,filename,WriteMode='append')
writelines("Symmetry  " + Symmetry,filename,WriteMode = 'append')
writelines("SpectralContent  " + Spectralcontent,filename,WriteMode = 'append')
writelines("ScatterType  " + ScatterType,filename,WriteMode = 'append')
writelines("SampleRotation  " + SampleRotation,filename,WriteMode = 'append')
writelines(num2str(0),filename,WriteMode = 'append')
writelines("AngleOfIncidence  " + num2str(numel(theta)),filename,WriteMode = 'append')
writematrix(theta,filename,'Delimiter','tab',WriteMode = 'append')
writelines("ScatterAzimuth " + num2str(numel(ScatterAzimuth)),filename,WriteMode = 'append')
writematrix(ScatterAzimuth,filename,'Delimiter','tab',WriteMode = 'append')
writelines("ScatterRadial " + num2str(numel(ScatterRadial)),filename,WriteMode = 'append')
writematrix(ScatterRadial,filename,'Delimiter','tab',WriteMode = 'append')
writelines('',filename,WriteMode = 'append')
writelines(Spectralcontent,filename,'WriteMode','append')
writelines("DataBegin",filename,'WriteMode','append')

idx  = 0;
%for n=1:numel(ScatterAzimuth)
for m=1:numel(theta)
%B_mean = mean(brdf_silicon(~isnan(brdf_silicon(:,m)),m));
%TIS = TIS*2*pi*sum(B_mean*cos())
brdf_calc(isnan(brdf_calc(:,m,1)),m,1) = 0;
TIS =  tis(brdf_calc(19:end,m,1),deg2rad(5)) %brdf_silicon(~isnan(brdf_silicon(:,m))
data = zeros(1,numel(ScatterRadial));
ID = find( ScatterRadial >= 0+idx & ScatterRadial <= 90+idx);
writelines("TIS " + num2str(TIS),filename,'WriteMode','append')
data(1,ID) = round(brdf_calc(19:end,m,1),3).';
data(1,isnan(data(1,:))) = 0;%data(1,i)
writematrix(data,filename,'Delimiter','tab',WriteMode = 'append')
data(1,ID) = flip(round(brdf_calc(1:19,m,1),3).');
data(1,isnan(data(1,:))) = 0;%data(1,i)
writematrix(data,filename,'Delimiter','tab',WriteMode = 'append')
idx = idx + spacing;
%end
end

writelines("DataEnd",filename,'WriteMode','append')


function TIS = tis(B,dth)
th = 0:dth:pi/2;
for i=1:length(th)
        cossin(i) = cos(th(i))*sin(th(i));
end
%B
 TIS = (sum(B.*cossin'))*dth;%*dphi;
 
end


%% Functions to Calculate the mie scattering 

%Mie_abcd(m,x)
%Mie_ab(m,x)
%[an,bn] = Mie_ab(m,x,N_max);
%pt=Mie_pt(mu(1),round(x + 4*x^(1/3) + 2));

%Mie_S12(m,x,mu)
function [an,bn] = Mie_ab(m,x)
nmax =round(x + 4*x^(1/3) + 2);
z=m.*x; % Input to bessel functions.
nmx=round(max(nmax,abs(z))+16); % Stated by source
n=(1:nmax); nu = (n+0.5); %
sx=sqrt(0.5*pi*x); % Prefix between Bessel and spherical Bessel.
px=sx.*besselj(nu,x); % Spherical Bessel of first kind
p1x=[sin(x), px(1:nmax-1)]; % Recurrence formula to obtain higher order.
chx=-sx.*bessely(nu,x); % Spherical Bessel of second kind
ch1x=[cos(x), chx(1:nmax-1)]; % Recurrence formula to obtain higher order.
gsx=px-1i*chx; gs1x=p1x-1i*ch1x; % Ricatti Bessel function xi
dnx(nmx)=0+0i;
for j=nmx:-1:2      % Computation of Dn(z) according to (4.89) of B+H (1983). Logartimic derivative of Riccati-Bessel function. 
    dnx(j-1)=j./z-1/(dnx(j)+j./z); % Downward recurrence formula. 
end
dn=dnx(n);          % Dn(z), n=1 to nmax
da=dn./m+n./x; % using expansion to simplify expression.
db=m.*dn+n./x; % using expansion to simplify expression
an=(da.*px-p1x)./(da.*gsx-gs1x); % an mie coefficient
bn=(db.*px-p1x)./(db.*gsx-gs1x); % bn mie coefficient
end

function result = Mie_S12(m, x, u)
% Computation of Mie Scattering functions S1 and S2

% for complex refractive index m=m'+im", 

% size parameter x=k0*a, and u=cos(scattering angle),

% where k0=vacuum wave number, a=sphere radius;

% s. p. 111-114, Bohren and Huffman (1983) BEWI:TDD122

% C. Mätzler, May 2002
nmax=round(2+x+4*x^(1/3));
[an,bn]=Mie_ab(m,x);
pt=Mie_pt(u,nmax);
pin =pt(1,:);
tin =pt(2,:);
n=(1:nmax);
n2=(2*n+1)./(n.*(n+1));
pin=n2.*pin;
tin=n2.*tin;
tin = tin;
S1=sum((an.*pin+bn.*tin)); % (an*pin'+bn*tin')

S2=sum((an.*tin+bn.*pin));%(an*tin'+bn*pin');%sum((an.*tin+bn.*pin));
result=[S1;S2];
end

function result=Mie_pt(u,nmax)
% pi_n and tau_n, -1 <= u <= 1, n1 integer from 1 to nmax 
% angular functions used in Mie Theory
% Bohren and Huffman (1983), p. 94 - 95
p(1)=1; % 
t(1)=u;
p(2)=3*u; 
t(2)=3*cosd(2*acosd(u));%6*(cosd(u))^2-3;%3*cosd(2*acosd(u)); % %
for n1=3:nmax
    p1= (((2*n1)-1)./(n1-1).*u).*p(n1-1);%((2*n1-1)./(n1-1)).*p(n1-1).*u;
    p2=(n1./(n1-1)).*p(n1-2);
    p(n1)=p1-p2;
    t1=(n1*u).*p(n1);
    t2=(n1+1).*p(n1-1);
    t(n1)=t1-t2;
end
result=[p;t];
end

function result = Mie(m, x)

% Computation of Mie Efficiencies for given 

% complex refractive-index ratio m=m'+im" 

% and size parameter x=k0*a, where k0= wave number in ambient 

% medium, a=sphere radius, using complex Mie Coefficients

% an and bn for n=1 to nmax,

% s. Bohren and Huffman (1983) BEWI:TDD122, p. 103,119-122,477.

% Result: m', m", x, efficiencies for extinction (qext), 

% scattering (qsca), absorption (qabs), backscattering (qb), 

% asymmetry parameter (asy=<costeta>) and (qratio=qb/qsca).

% Uses the function "Mie_ab" for an and bn, for n=1 to nmax.

% C. Mätzler, May 2002, revised July 2002.



if x==0                 % To avoid a singularity at x=0

    result=[0 0 0 0 0 1.5];

elseif x>0              % This is the normal situation

    nmax=round(2+x+4*x.^(1/3));

    n1=nmax-1;

    n=(1:nmax);cn=2*n+1; c1n=n.*(n+2)./(n+1); c2n=cn./n./(n+1);

    x2=x.*x;

    [an,bn]=Mie_ab(m,x);

    anp=(real(an(1,:))); anpp=(imag(an(1,:)));

    bnp=(real(bn(1,:))); bnpp=(imag(bn(1,:)));

    g1(1:4,nmax)=[0; 0; 0; 0]; % displaced numbers used for

    g1(1,1:n1)=anp(2:nmax);    % asymmetry parameter, p. 120

    g1(2,1:n1)=anpp(2:nmax);

    g1(3,1:n1)=bnp(2:nmax);

    g1(4,1:n1)=bnpp(2:nmax);   

    dn=cn.*(anp+bnp);

    q=sum(dn);

    qext=2*q/x2;

    en=cn.*(anp.*anp+anpp.*anpp+bnp.*bnp+bnpp.*bnpp);

    q=sum(en);

    qsca=2*q/x2;

    qabs=qext-qsca;

    fn=(an(1,:)-bn(1,:)).*cn;

    gn=(-1).^n;

    f(3,:)=fn.*gn;

    q=sum(f(3,:));

    qb=q*q'/x2;

    asy1=c1n.*(anp.*g1(1,:)+anpp.*g1(2,:)+bnp.*g1(3,:)+bnpp.*g1(4,:));

    asy2=c2n.*(anp.*bnp+anpp.*bnpp);

    asy=4/x2*sum(asy1+asy2)/qsca;

    qratio=qb/qsca;

    result=[qext qsca qabs qb asy qratio];

end
end
