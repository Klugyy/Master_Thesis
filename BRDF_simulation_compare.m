clear all

file_dir = dir(fullfile('..\','Processed_data/'));
Files = extractfield(file_dir,'name');
Files(1:2) = [];

Sim = load(fullfile('..\','Processed_data/',Files{5}));
Sim_fine = load(fullfile('..\','Processed_data/',Files{6}));

wavelengths = Sim.wavelengths;
zero_point = round(numel(Sim.white_rel_sensor_ang(:,1))/2);
theta = Sim.f_ellipse_angles; % Incidence angle/sample angle.
alpha = Sim.white_rel_sensor_ang(:,1); % Called emergince angle. Angle between specular reflection and surface normal. 
brdf = Sim.BRDF_brdf;
white_brdf = Sim.white_brdf;

set(0,'defaultTextInterpreter','latex')
set(0,'defaultLegendInterpreter','latex')

%% Lambertian reflector BRDF 
ref_angles = [];
for i=4:3:numel(Sim.white_rel_sensor_ang(:,1))/2
    ref_angles =[ref_angles,Sim.white_rel_sensor_ang(i,1),-Sim.white_rel_sensor_ang(i,1)]; 
end
ref_angles = [ref_angles,0];
ref_angles = ref_angles(end:-1:1);
output = [];
for n=1:numel(ref_angles)
output = [output,find(Sim.white_rel_sensor_ang(:,1)==ref_angles(n))];
end
mycolors = [0,0,0;1,0,0;1,0.75,0;0,1,0;0,0.5,0;0,0,1;0,0,0.5;1,1,0;0.5,0.5,0;1,0,1;0.5,0,0.5];
white_brdf((Sim.white_brdf(:,1) < 0.01),1)
%Sim.white_brdf(find(Sim.white_brdf(:,i) < 0.001),i) = NaN;
%white_brdf = Sim.white_brdf;



%%
white_ang = flip(linspace(-60,60,9));
figure;
for i=1:numel(white_brdf(1,:))
%white_brdf((Sim.white_brdf(:,i) < 0.01),i)
%white_brdf((Sim.white_brdf(:,i) < 0.001),i) = NaN;
[val,I] = min(white_brdf(:,i));
white_brdf(I,i) = NaN;
plot(alpha,white_brdf(:,i),'-o','DisplayName',sprintf('Angle of incidence = %d',white_ang(i)),'Color',mycolors(i,:))
hold on 
end
xlabel('Sensor angle $\alpha [^\circ]$',FontSize = 16)
ylabel('Spectralon brdf [1/sr]',FontSize=16)
legend(FontSize = 16)
grid('on')
%xlim([0,90])

%% BRDF encaps, testing glass 

PV_modules = ["Gold","Reference","Red","LightGreen","Brown","Green","Glass","Silicon"];
mycolors = [0.8706,0.8706,0.1294;0,0,1;1,0,0;0,1,0;0.4000,0.0745,0.1373;0,0.60,0;0,0.5,0.5;0,0,0];
figure(2)
for i=1:numel(brdf(1,1,:))
    plot(alpha,brdf(:,2,i),'Marker','o' ,'DisplayName',PV_modules(i),'LineWidth',2,'Color',mycolors(i,:))
    hold on
end
hold off
ylim([0,6])
xlim([-90,90])
xlabel('Relative Sensor Angle $\alpha [^\circ]$ ','FontSize',20)
ylabel('B($\theta_i = 15^\circ$)[1/sr]' ,'FontSize',20)
set(gca,'LineWidth',2,'FontSize',20,'YScale','log');
grid()
ylim([10^(-3),5])
xlim([-90,90])
lgd = legend('Location','northeast');
title(lgd,'Samples')

% Wish to interpolate or fit the peaks to a gaussian. 
%%
Gauss = {};%zeros(numel(Sim_fine.white_sample_ang(:,1)),numel(theta));

figure(30)
for i=2:numel(theta)
    [A,sigma,gauss] = Gaussian(Sim_fine.white_rel_sensor_ang(:,i),Sim_fine.BRDF_brdf(:,i,1));
    Gauss{i} = gauss;
 h1 =    plot(Sim_fine.white_rel_sensor_ang(:,i),Sim_fine.BRDF_brdf(:,i,1),'Marker','o' ,'Color',mycolors(i,:),'LineStyle','none',linewidth = 2);
    hold on
    plot(Sim_fine.white_rel_sensor_ang(:,i),gauss(Sim_fine.white_rel_sensor_ang(:,i)),'LineStyle','--','Color',mycolors(i,:),linewidth = 2)


end
hold off
%ylim([0,6])
xlim([-90,90])
xlabel('Sensor angle $\alpha [^\circ]$ ','FontSize',20)
ylabel('B[1/sr]' ,'FontSize',20)
set(gca,'LineWidth',2,'FontSize',20);
grid()
lgd = legend({sprintf('%s = %d','$\theta$',theta(2)),'',sprintf('%s = %d','$\theta$',theta(3)),'',sprintf('%s = %d','$\theta$',theta(4)),'',sprintf('%s = %d','$\theta$',theta(i)),''},'Interpreter','latex','Location','northeastoutside');
%title(lgd,'$\theta[^\circ]$')
annotation('textbox',[0.3,0.5,0.3,0.3],'String','O Experimental data','FitBoxToText','on',FontSize=16)
annotation('textbox',[0.3,0.45,0.3,0.3],'String',sprintf('- - - Gaussian fit'),'FitBoxToText','on',FontSize = 16)

% Move so the files are in the correct order. See PV-modules.
brdf_i = Sim_fine.BRDF_brdf(:,:,4); % Saving brown data.
Sim_fine.BRDF_brdf(:,:,4) =  Sim_fine.BRDF_brdf(:,:,6); % Switching in lightgreen for brown.
brdf_ii = Sim_fine.BRDF_brdf(:,:,5); % Saving DRKgreen data.
Sim_fine.BRDF_brdf(:,:,5) = brdf_i; % Saving Brown data in DRKgreen data.
Sim_fine.BRDF_brdf(:,:,5) = brdf_i; % Saving DRKgreen data in Lightgreen data.



% Set the two data sets together. 

%kat_alpha = zeros([],numel(theta));
%kat_brdf = brdf;


for i=1:numel(PV_modules)-1
    for n=2:numel(theta)
        q = Gauss{n};
       idx_find = find(alpha == Sim_fine.white_rel_sensor_ang(1,n)):find(alpha==Sim_fine.white_rel_sensor_ang(end,n));
       kat_alpha(:,n) = cat(1,alpha(1:idx_find(1)-1,1), Sim_fine.white_rel_sensor_ang(:,n), alpha(idx_find(end)+1:end,1));
       kat_brdf(:,n,i) = cat(1,brdf(1:idx_find(1)-1,n,i), q(Sim_fine.white_rel_sensor_ang(:,n)) , brdf(idx_find(end)+1:end,n,i)); %Sim_fine.BRDF_brdf(:,n,i)
      % kat =  cat(1,brdf(find(alpha == Sim_fine.white_rel_sensor_ang(1,n)):find(alpha==Sim_fine.white_rel_sensor_ang(end,n)),n,i),q(Sim_fine.white_rel_sensor_ang(:,n)));
    end
end


figure(31)
for i=2:numel(theta)
subplot(2,1,1)
plot(kat_alpha(:,i),kat_brdf(:,i,2),'LineWidth',2,'LineStyle','--','Color',mycolors(i,:))
hold on
plot(alpha,brdf(:,i,2),'Marker','o' ,'Color',mycolors(i,:),'LineStyle','none',linewidth = 2)
xlim([-90,90])
xlabel('Sensor angle $\alpha [^\circ]$ ','FontSize',20)
ylabel('B[1/sr]' ,'FontSize',20)
set(gca,'LineWidth',2,'FontSize',20);
grid('on')
annotation('textbox',[0.5,0.5,0.3,0.3],'FitBoxToText','on','String','--- Gaussian fit data','FontSize',16)
annotation('textbox',[0.5,0.45,0.3,0.3],'FitBoxToText','on','String','O Rough data','FontSize',16)
subplot(2,1,2)
plot(kat_alpha(:,i),kat_brdf(:,i,6),'LineWidth',2,'LineStyle','--','Color',mycolors(i,:))
hold on
plot(alpha,brdf(:,i,6),'Marker','o' ,'Color',mycolors(i,:),'LineStyle','none',linewidth = 2)
annotation('textbox',[0.5,0.05,0.3,0.3],'FitBoxToText','on','String','--- Gaussian fit data','FontSize',16)
annotation('textbox',[0.5,0,0.3,0.3],'FitBoxToText','on','String','O Rough data','FontSize',16)
end
hold off
%ylim([0,6])
xlim([-90,90])
xlabel('Sensor angle $\alpha [^\circ]$ ','FontSize',20)
ylabel('B[1/sr]' ,'FontSize',20)
set(gca,'LineWidth',2,'FontSize',20);
grid()
lgd = legend({sprintf('%s = %d','$\theta$',theta(2)),'',sprintf('%s = %d','$\theta$',theta(3)),'',sprintf('%s = %d','$\theta$',theta(4)),'',sprintf('%s = %d','$\theta$',theta(i)),''},'Interpreter','latex','Location','northeastoutside');
lgd.Position(1) = 0.92;
lgd.Position(2) = 0.45;

%% BRDF glass and silicon

figure(3)
for i=2:numel(theta)
ax1 = subplot(2,1,1);
plot(alpha,brdf(:,i,7),'Marker','o' ,'color',mycolors(i,:),'DisplayName',sprintf('%.2f',theta(i)),linewidth = 2)
hold on
end
hold off
ylim([0,10])
xlim([-90,90])
xlabel('Sensor angle $\alpha [^\circ]$ ','FontSize',20)
ylabel('B[1/sr]' ,'FontSize',20)
set(gca,'LineWidth',2,'FontSize',20);
grid()
lgd = legend('Location','northeastoutside');
title(lgd,'$\theta[^\circ]$')
set(gca,'YScale','log')

for i=2:numel(theta)
ax2 = subplot(2,1,2);hold on
plot(alpha,brdf(:,i,8),'Marker','o' ,'color',mycolors(i,:),'DisplayName',sprintf('%.2f',theta(i)),linewidth = 2)
end
hold off
ylim([0,0.1])
xlim([-90,90])
xlabel('Sensor angle $\alpha [^\circ]$ ','FontSize',20)
ylabel('B[1/sr]' ,'FontSize',20)
set(gca,'LineWidth',2,'FontSize',20);
grid()
lgd = legend('Location','northeastoutside');
title(lgd,'$\theta[^\circ]$')
linkaxes([ax1,ax2],'x')
set(gca,'YScale','log')

% Assuming completely diffuse silicon. Finding average.

silicon_diffuse = 4*pi.*[mean(brdf(~isnan(brdf(:,1,8)),1,8)),mean(brdf(~isnan(brdf(:,2,8)),2,8)),mean(brdf(~isnan(brdf(:,3,8)),3,8)),mean(brdf(~isnan(brdf(:,4,8)),4,8)),mean(brdf(~isnan(brdf(:,5,8)),5,8))];

%% Test data

emer = linspace(-70,70,140/10 + 1);
brdf_20 = [0.007,0.005,0.005,0.006,0.012,NaN,0.052,0.154,0.605,5.091,0.436,0.151,0.058,0.027,0.013]; 
brdf_20_az30 = [NaN,0.006,0.009,0.007,0.012,0.023,0.057,0.157,0.483,0.69,0.238,0.096,0.041,0.019,0.013];
brdf_0 = [0.007,0.008,0.012,0.022,0.053,0.152,0.558,NaN,0.384,0.121,0.042,0.019,0.009,0.006,0.007];

x_t = x_disp(theta,alpha,0);

[x_tt,brdf_t] = x_sort(x_t(1,:),brdf(:,5,4));
[A_non,B_non,g_non,fit_non] = ABgfit_fit(x_tt,brdf_t);

[A_bon,B_bon,g_bon,fit_bon] = direct_fit(x_tt,brdf_t);


TIS = tis(A_non,B_non,g_non,0,1,1)

%{
x_disp(0,emer,0)
x_t = x_disp(0,emer,0);

[x_t,brdf_t] = x_sort(x_t,brdf_0);
[A_non,B_non,g_non,fit_non] = ABgfit(x_t,brdf_t);


TIS = tis(A_non,B_non,g_non,0,1,1)

[A,B,g,fitting] = ABgfit(x_t,brdf_t./TIS);

tis(A,B,g,0,1,1)

%}
figure;
loglog(x_tt(1,:),brdf_t.','LineWidth',2,'Color','blue','DisplayName','Raw data','LineStyle','none','Marker','o')
hold on
grid("on")
loglog(x_tt,fit_non,'LineWidth',2,'Color','red','DisplayName',sprintf('Non corrected fit: A= %f B= %f g= %f',[A_non,B_non,g_non]))
loglog(x_tt,fit_bon,'LineWidth',2,'Color','green','DisplayName',sprintf('Corrected fit: A= %f B= %f g= %f',[A_bon,B_bon,g_bon]))
xlabel('Displacement vector $\bar{x}$',fontsize = 20)
ylabel('BRDF value [1/sr]',fontsize = 20)
legend()



%% Model parameter fitting 
%test = x_disp(30,30,7);

%[x_test,b_test] = x_sort(test,brdf(25,3,7))


az = linspace(-180,180,360/5 +1);
mycolors =[1,1,0;0,0,1;1,0,0;0,1,0;0.5,0,0;0,0.5,0;0,0,0;0,0,0.5]; %[0,0,0;1,0,0;1,0.75,0;0,1,0;0,0.5,0;0,0,1;0,0,0.5;1,1,0;0.5,0.5,0;1,0,1;0.5,0,0.5];

alpha_pos = alpha(90/5 + 1:end,1);

% Testting the calculation of the displacement vector.
%{
figure(4)
for i=1:numel(brdf(1,1,:))-7
x_pos = x_disp(30,alpha_pos,0);
x_ng = x_disp(theta,alpha(1:90/5 + 1,1),0);
[x_new,brdf_glass] =  x_sort(x_pos(1,:),brdf(90/5 + 1:end,3,i));
[x_new_ng,brdf_glass_ng] =  x_sort(x_ng(3,:),brdf(1:90/5 + 1,3,i));

plot(x_pos(1,:),brdf(90/5 + 1:end,3,i),'-o','LineWidth',2,'DisplayName',PV_modules(i),'color',mycolors(i,:)) 
hold on
plot(x_new,brdf_glass,'-o','LineWidth',2,'LineStyle','--','DisplayName',PV_modules(i)+' Sorted','color','black')
end
grid('on')
xlabel('Displacement vector $\bar{x}$',fontsize = 20)
ylabel('BRDF value [1/sr]',fontsize = 20)
set(gca,'XScale','log','YScale','log')
legend()

%alpha = alpha(3:end-3)
%}

%% Forward/backward scattering 
idx_scat = zeros(numel(theta),1);
for i=1:numel(theta)
idx_scat(i) = find(theta(i) == alpha);
end

x = x_disp(theta,alpha,0);

figure(4)

for i=1
for n=1:numel(theta)
x_forward = x(n,idx_scat(n):end);
x_back =  x(n,1:idx_scat(n));
[x_forward_new,brdf_forward] =  x_sort(x_forward,brdf(idx_scat(n):end,n,i));
[x_back_new,brdf_backward] =  x_sort(x_back,brdf(1:idx_scat(n),n,i));
ax1 = subplot(2,1,1);
plot(x_forward_new,brdf_forward,'-o','LineWidth',2,'DisplayName',sprintf('forward scatter incidence = %d',theta(n)),'color',mycolors(n,:)) 
hold on

grid('on')
xlabel('Displacement vector $\bar{x}$',fontsize = 20)
ylabel('BRDF value [1/sr]',fontsize = 20)
set(gca,'XScale','log','YScale','log')
linkaxes([ax1,ax2],'xy')
legend(FontSize = 16)
ylim([0,max(brdf(:,n,i))])
xlim([0,max(x(n,:))])
ax2 = subplot(2,1,2);
plot(x_back_new,brdf_backward,'-o','LineWidth',2,'LineStyle','--','DisplayName',sprintf('Backward scatter incidence = %d',theta(n)),'color',mycolors(n,:))
hold on
end
end
grid('on')
xlabel('Displacement vector $\bar{x}$',fontsize = 16)
ylabel('BRDF value [1/sr]',fontsize = 20)
set(gca,'XScale','log','YScale','log')
linkaxes([ax1,ax2],'xy')
legend(FontSize = 16)
%ylim([0,max(brdf(:,n,i))])

%%
str_leg = '%s = %d [%s]';

angle_color = [0,0,0;1,0,0;1,0.6,0;0.6,0.1,1;0,0.6,0];

figure(7)
for i=1:numel(theta)
x_forward = x_disp(theta(i),alpha(idx_scat(i):end,1),0);
x_back = x_disp(theta(i),alpha(1:idx_scat(i),1),0);
plot(alpha(idx_scat(i):end,1),x_forward,'-o','LineWidth',2,'color',angle_color(i,:)) 
hold on
plot(alpha(1:idx_scat(i),1),x_back,'LineWidth',2,'LineStyle','--','color',angle_color(i,:))
end
grid('on')
ylabel('Displacement vector $\bar{x}$',fontsize = 24)
xlabel('Relative Sensor Angle $\alpha [^\circ]$',fontsize = 24)
set(gca,'Xtick',alpha(1:3:end,1),'FontSize',18)
annotation('textbox',[0.5,0.5,0.2,0.2],'String','-- Backward Scattered','FitBoxToText','on','FontSize',16,'Color','black')
annotation('textbox',[0.5,0.45,0.2,0.2],'String','-o Forward Scattered','FitBoxToText','on','FontSize',16,'Color','black')
%annotation('textbox',dim,'String',sprintf(leg_str,'$\frac{A}{B + |\bar{x}|^g}$'),'FitBoxToText','on','Interpreter','latex',FontSize=16,Color='black')
%set(gca,'XScale','log','YScale','log')
lgd = legend({num2str(0),'',num2str(15),'',num2str(30),'',num2str(45),'',num2str(60),''},'FontSize',20);
%lgd = legend({sprintf(str_leg,'$\theta_e$',theta(1),'$^\circ$'),'',sprintf(str_leg,'$\theta_e$',theta(2),'$^\circ$'),'',sprintf(str_leg,'$\theta$',theta(3),'$^\circ$'),'',sprintf(str_leg,'$\theta$',theta(4),'$^\circ$'),'',sprintf(str_leg,'$\theta$',theta(5),'$^\circ$'),''},'interpreter','latex',FontSize=  20);
title(lgd,'$\theta_i [^\circ]$','FontSize',18)


%%
%{
fitting = dictionary;
for i=1:numel(brdf(1,1,:))
    for n=1:numel(theta) % Choose a PV module.
         x_forward = x(n,idx_scat(n):end);
         x_back =  x(n,1:idx_scat(n));
        [x_forward_new,brdf_new] =  x_sort(x(n,:),brdf(90/5 + 1:end,n,i));
       % x_new
        fitting(PV_modules(1,i) + "x" + num2str(theta(n))) = {x_forward_new};
        fitting(PV_modules(1,i) + "BRDF" + num2str(theta(n))) = {brdf_new};
        %M_x(n,:) = {x_new};
        %M_brdf(n,:) = {brdf_new};
    end
    
end
%}
PV_names = {'Gold','Reference','Red','LightGreen','Brown','Green','Glass','Silicon'};
PV_angles = {'Zero','Fifteen','Thirty','Fourtyfive','Sixty'};
%RMSE = array(numel(PV_names),numel(theta));
%RSquared = array(numel(PV_names),numel(theta));
TIS_struct = struct('TIS_val',[],'coeffs',[],'Fit',[],'rmse',[],'Rsquared',[]);
Fitting = struct('x',[],'brdf',[],'coeffs',[],'Fit',[],'rmse',[],'Rsquared',[],'TIS',TIS_struct); % ,'A',[],'B',[],'g',[],'rmse',[],'Rsquared',[]
Forward_backward = struct('Forward',Fitting,'Backward',Fitting);
Fit_Data = struct('Raw_Data',Forward_backward,'Log_fit',Forward_backward,'Direct_Fit',Forward_backward);
%Fit_method = struct('Raw_data',[],'Log_fit',[],'Direct_Fit',[]);
     
Polar_angles_fit_data = struct('Zero',Fit_Data,'Fifteen',Fit_Data,'Thirty',Fit_Data,'Fourtyfive',Fit_Data,'Sixty',Fit_Data);
 % Polar_angles_fit_data = struct('Zero',Fit_method,'Fifteen',Fit_method,'Thirty',Fit_method,'Fourtyfive',Fit_method,'Sixty',Fit_method);
    
for i=1:numel(PV_names)
    
    d.(PV_names{i}).sample = PV_names{i};
    d.(PV_names{i}).Fit_Data = Polar_angles_fit_data;
    fn_sample = fieldnames(d);
    fn = fieldnames(d.Gold.Fit_Data);
    for n=1:numel(theta) % Choose a PV module.
           
        % Initialzing struct.
        %full_angle_data = d.(fn_sample{i}).Fit_Data.(fn{n});
      %  d.(PV_names{i}).Data.Zero(PV_angles{n}).Fit_Data = Fit_Data;
      %  d.(PV_names{i}).Data(PV_angles{n}).Fit_Method = Fit_method;
        %d.(PV_names{i}).Data.PV_angles{n}.Fit_Method = Fit_Data;
        
        % Calculating arrays. 
         x_forward = x(n,idx_scat(n):end);
         x_back =  x(n,1:idx_scat(n));
        [x_forward_new,brdf_forward_new] =  x_sort(x_forward,brdf(idx_scat(n):end,n,i));
        [x_backward_new,brdf_backward_new] = x_sort(x_back,brdf(1:idx_scat(n),n,i));
        

        % Calculating fits with different methods.
        [A_f,B_f,g_f,fit_fit,gop_lin] = ABgfit_fit(x_forward_new,brdf_forward_new);
        [A_d,B_d,g_d,fit_d,gop_dir] = direct_fit(x_forward_new,brdf_forward_new);
        % Forward data.
        d.(fn_sample{i}).Fit_Data.(fn{n}).Raw_Data.Forward.x = x_forward_new;
        d.(fn_sample{i}).Fit_Data.(fn{n}).Raw_Data.Forward.brdf = brdf_forward_new;

        SSR = sum( (fit_fit-brdf_forward_new ).^2);
        TSS = sum( (brdf_forward_new -mean(brdf_forward_new)).^2);
        r2 =  1 - (SSR/TSS);
        rms_own = sqrt(  mean((brdf_forward_new - fit_fit).^2) );
        

        d.(fn_sample{i}).Fit_Data.(fn{n}).Log_fit.Forward.coeffs = [A_f,B_f,g_f];
        d.(fn_sample{i}).Fit_Data.(fn{n}).Log_fit.Forward.Fit = fit_fit;
        d.(fn_sample{i}).Fit_Data.(fn{n}).Log_fit.Forward.rmse = rms_own;%gop_lin.rmse;
        d.(fn_sample{i}).Fit_Data.(fn{n}).Log_fit.Forward.Rsquared = r2;%gop_lin.rsquare;

        SSR = sum( (fit_d-brdf_forward_new ).^2);
        r2 =  1 - (SSR/TSS);
        rms_own = sqrt(  mean((brdf_forward_new - fit_d).^2) );

        d.(fn_sample{i}).Fit_Data.(fn{n}).Direct_Fit.Forward.coeffs = [A_d,B_d,g_d];
        d.(fn_sample{i}).Fit_Data.(fn{n}).Direct_Fit.Forward.Fit = fit_d;
        d.(fn_sample{i}).Fit_Data.(fn{n}).Direct_Fit.Forward.rmse = rms_own;%gop_dir.rmse;
        d.(fn_sample{i}).Fit_Data.(fn{n}).Direct_Fit.Forward.Rsquared =r2; %gop_dir.rsquare;

        TIS = tis(A_d,B_d,g_d,theta(n),deg2rad(1),deg2rad(1));
        %[A_d,B_d,g_d,fit_d,gop_dir] = direct_fit(x_forward_new,brdf_forward_new./TIS);
         
        % TIS = tis(A_d,B_d,g_d,theta(n),deg2rad(1),deg2rad(1));
         [A_d,B_d,g_d,fit_d,gop_dir] = direct_fit(x_forward_new,brdf_forward_new./TIS);
         d.(fn_sample{i}).Fit_Data.(fn{n}).Direct_Fit.Forward.TIS.TIS_val =  tis(A_d,B_d,g_d,theta(n),deg2rad(1),deg2rad(1));
         d.(fn_sample{i}).Fit_Data.(fn{n}).Direct_Fit.Forward.TIS.coeffs = [A_d,B_d,g_d];
          d.(fn_sample{i}).Fit_Data.(fn{n}).Direct_Fit.Forward.TIS.coeffs
         d.(fn_sample{i}).Fit_Data.(fn{n}).Direct_Fit.Forward.TIS.Fit = fit_d;
         d.(fn_sample{i}).Fit_Data.(fn{n}).Direct_Fit.Forward.TIS.rmse = gop_dir.rmse;
         d.(fn_sample{i}).Fit_Data.(fn{n}).Direct_Fit.Forward.TIS.Rsquared = gop_dir.rsquare;
  
        
        % Calculating fits for backward data.
        [A_f,B_f,g_f,fit_fit,gop_lin] = ABgfit_fit(x_backward_new,brdf_backward_new);
        [A_d,B_d,g_d,fit_d,gop_dir] = direct_fit(x_backward_new,brdf_backward_new);
        
        % Inserting backward data. 
        d.(fn_sample{i}).Fit_Data.(fn{n}).Raw_Data.Backward.x = x_backward_new;
        d.(fn_sample{i}).Fit_Data.(fn{n}).Raw_Data.Backward.brdf = brdf_backward_new;

        d.(fn_sample{i}).Fit_Data.(fn{n}).Log_fit.Backward.coeffs = [A_f,B_f,g_f];
        d.(fn_sample{i}).Fit_Data.(fn{n}).Log_fit.Backward.Fit = fit_fit;

        SSR = sum( (fit_fit-brdf_backward_new ).^2);
        TSS = sum( (brdf_backward_new -mean(brdf_backward_new)).^2);
        r2 =  1 - (SSR/TSS);
        rms_own = sqrt(mean((brdf_backward_new - fit_fit).^2) );
        
      %  rmse(fit_fit,brdf_backward_new)
        d.(fn_sample{i}).Fit_Data.(fn{n}).Log_fit.Backward.rmse = rms_own;  %gop_lin.rmse;
        d.(fn_sample{i}).Fit_Data.(fn{n}).Log_fit.Backward.Rsquared = r2;%gop_lin.rsquare;

        SSR = sum( (fit_d-brdf_backward_new ).^2);
        r2 =  1 - (SSR/TSS);
        rms_own = sqrt(  mean((brdf_backward_new - fit_d).^2) );
        TIS = tis(A_d,B_d,g_d,theta(n),deg2rad(1),deg2rad(1));
       % [A_d,B_d,g_d,fit_d,gop_dir] = direct_fit(x_backward_new,brdf_backward_new./TIS);
         
        % TIS = tis(A_d,B_d,g_d,theta(n),deg2rad(1),deg2rad(1));
         %[A_d,B_d,g_d,fit_d,gop_dir] = direct_fit(x_forward_new,brdf_forward_new./TIS);
        d.(fn_sample{i}).Fit_Data.(fn{n}).Direct_Fit.Backward.TIS.TIS_val = TIS; %tis(A_d,B_d,g_d,theta(n),deg2rad(1),deg2rad(1));
        d.(fn_sample{i}).Fit_Data.(fn{n}).Direct_Fit.Backward.coeffs = [A_d,B_d,g_d];
        d.(fn_sample{i}).Fit_Data.(fn{n}).Direct_Fit.Backward.Fit = fit_d;
        d.(fn_sample{i}).Fit_Data.(fn{n}).Direct_Fit.Backward.rmse = rms_own; %gop_dir.rmse;
        d.(fn_sample{i}).Fit_Data.(fn{n}).Direct_Fit.Backward.Rsquared =  r2; %gop_dir.rsquare;

        
        

        % Inserting into structs.
        
       % fitting(PV_modules(1,i) + "forward" + num2str(theta(n))) = {x_forward_new,brdf_forward_new};
       % fitting(PV_modules(1,i) + "BRDF" + num2str(theta(n))) = {brdf_forward_new};
        
        % Different fitting methods and getting their data.
       % fitting(PV_modules(1,i) + "Fit" + num2str(theta(n))) = {brdf_forward_new};
        
        %M_x(n,:) = {x_new};
        %M_brdf(n,:) = {brdf_new};
    end
    
end

%% Fit plotting
Leg_2 = '%s for %s = %d';
leg_str = '- - - Fit: %s';
leg_str_2 = ' ... Fit: %s';
% Subplot for each fittype for all angles. (Forward and backward
% scattering)
sample = 1;
symbol = ["o","^","Pentagram","diamond","square"];
dim = [0.35,0.45,0.3,0.3];
 % Choosing sample and Angle.
fig = figure(8);
for i=1:numel(theta)
ax1 = subplot(2,2,1);
%scatter3(ones(1,numel(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.x)).*theta(i),d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.brdf)
hold on
%plot3(ones(1,numel(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.x)).*theta(i),d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Direct_Fit.Forward.Fit)
scatter(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.brdf,26,mycolors(i+1,:),'filled','Marker',symbol(i))
hold on
plot(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Direct_Fit.Forward.Fit,'LineWidth',1,'LineStyle','--','color',mycolors(i+1,:))
annotation('textbox',dim,'String',sprintf(leg_str,'$\frac{A}{B + |\bar{x}|^g}$'),'FitBoxToText','on','Interpreter','latex',FontSize=16,Color='black')
%plot(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Log_fit.Forward.Fit,'LineWidth',2,'LineStyle',':','color',mycolors(i,:))
set(gca,'YScale','log')
grid('on')
legend({sprintf(Leg_2,'Forward','$\theta$',theta(1)),'',sprintf(Leg_2,'Forward','$\theta$',theta(2)),'',sprintf(Leg_2,'Forward','$\theta$',theta(3)),'',sprintf(Leg_2,'Forward','$\theta$',theta(4)),'',sprintf(Leg_2,'Forward','$\theta$',theta(5)),''},FontSize = 14)
hold off
ax2 = subplot(2,2,2);
scatter(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Backward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Backward.brdf,26,mycolors(i+1,:),'filled','Marker',symbol(i))
hold on
plot(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Backward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Direct_Fit.Backward.Fit,'LineWidth',1,'LineStyle','--','color',mycolors(i+1,:));
annotation('textbox',[0.75,0.45,0.3,0.3],'String',sprintf(leg_str,'$\frac{A}{B + |\bar{x}|^g}$'),'FitBoxToText','on','Interpreter','latex',FontSize=16,Color='black')
legend({sprintf(Leg_2,'Backward','$\theta$',theta(1)),'',sprintf(Leg_2,'Backward','$\theta$',theta(2)),'',sprintf(Leg_2,'Backward','$\theta$',theta(3)),'',sprintf(Leg_2,'Backward','$\theta$',theta(4)),'',sprintf(Leg_2,'Backward','$\theta$',theta(5)),''},FontSize = 14)
grid('on')
set(gca,'YScale','log')

%hold off

ax3 = subplot(2,2,3);
scatter(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.brdf,26,mycolors(i+1,:),'filled','Marker',symbol(i))
hold on
plot(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Log_fit.Forward.Fit,'LineWidth',2,'LineStyle',':','color',mycolors(i+1,:))
set(gca,'YScale','log')
grid('on')
annotation('textbox',[0.25,0,0.3,0.3],'String',sprintf(leg_str_2,'$log(A) -g \cdot log(|\bar{x}|)$'),'FitBoxToText','on','Interpreter','latex',FontSize=16,Color='black')
legend('',{sprintf(Leg_2,'Forward','$\theta$',theta(1)),'',sprintf(Leg_2,'Forward','$\theta$',theta(2)),'',sprintf(Leg_2,'Forward','$\theta$',theta(3)),'',sprintf(Leg_2,'Forward','$\theta$',theta(4)),'',sprintf(Leg_2,'Forward','$\theta$',theta(5))},FontSize = 14)
ax4 = subplot(2,2,4);
scatter(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Backward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Backward.brdf,26,mycolors(i+1,:),'filled','Marker',symbol(i))
hold on
plot(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Backward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Log_fit.Backward.Fit,'LineWidth',2,'LineStyle',':','color',mycolors(i+1,:))
grid('on')
annotation('textbox',[0.7,0,0.3,0.3],'String',sprintf(leg_str_2,'$log(A) -g \cdot log(|\bar{x}|)$'),'FitBoxToText','on','Interpreter','latex',FontSize=16,Color='black')
legend({sprintf(Leg_2,'Backward','$\theta$',theta(1)),'',sprintf(Leg_2,'Backward','$\theta$',theta(2)),'',sprintf(Leg_2,'Backward','$\theta$',theta(3)),'',sprintf(Leg_2,'Backward','$\theta$',theta(4)),'',sprintf(Leg_2,'Backward','$\theta$',theta(5)),''},FontSize = 14)


%plot(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Backward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Log_fit.Backward.Fit,'LineWidth',2,'LineStyle',':','color',mycolors(i,:))

end
set(gca,'YScale','log')
%h = get(gca,'Children');
%set(gca,'Children',cat(1,h(1:2:end),h(2:2:end)))
%get(gca,'Children')

linkaxes([ax1,ax2],'xy')
han=axes(fig,'visible','off'); 
han.XLabel.Visible='on';
han.YLabel.Visible='on';
han.Title.Visible='on';
xlabel(han,'Displacement vector $\bar{x}$',Fontsize = 20);
ylabel(han,'BRDF [1/sr]',Fontsize = 20);
%title(han,sprintf('Angular BRDF comparison for sample %s',lower(fn_sample{sample})),FontSize= 20)
hold off
%% 
% Comparison of fit / backward scattering across sample types.


%% RMSE and Rsquared heatmaps.
% Rmse / Rsquared for each fit at each angle. Table maybe?
% Sample out of one axis and rmse is the color and angle is the other
% direction.
RMSE_log_for = zeros(numel(PV_names),numel(theta));
RMSE_log_back = zeros(numel(PV_names),numel(theta));
RMSE_dir_for = zeros(numel(PV_names),numel(theta));
RMSE_dir_back = zeros(numel(PV_names),numel(theta));
RSquared_log_for = zeros(numel(PV_names),numel(theta));
RSquared_log_back = zeros(numel(PV_names),numel(theta));
RSquared_dir_for = zeros(numel(PV_names),numel(theta));
RSquared_dir_back = zeros(numel(PV_names),numel(theta));
tis_array = zeros(numel(PV_names),numel(theta));
for i=1:numel(fn_sample)
    for n=1:numel(theta)
         RMSE_log_for(i,n) = d.(fn_sample{i}).Fit_Data.(fn{n}).Log_fit.Forward.rmse;
         RMSE_log_back(i,n) = d.(fn_sample{i}).Fit_Data.(fn{n}).Log_fit.Backward.rmse;
         RMSE_dir_for(i,n) = d.(fn_sample{i}).Fit_Data.(fn{n}).Direct_Fit.Forward.rmse;
         RMSE_dir_back(i,n) = d.(fn_sample{i}).Fit_Data.(fn{n}).Direct_Fit.Backward.rmse;

         RSquared_log_for(i,n) = d.(fn_sample{i}).Fit_Data.(fn{n}).Log_fit.Forward.Rsquared;
         RSquared_log_back(i,n) = d.(fn_sample{i}).Fit_Data.(fn{n}).Log_fit.Backward.Rsquared;
         RSquared_dir_for(i,n) = d.(fn_sample{i}).Fit_Data.(fn{n}).Direct_Fit.Forward.Rsquared;
         RSquared_dir_back(i,n) = d.(fn_sample{i}).Fit_Data.(fn{n}).Direct_Fit.Backward.Rsquared;

         tis_array(i,n)  =  d.(fn_sample{i}).Fit_Data.(fn{n}).Direct_Fit.Forward.TIS.TIS_val;
         tis_array_back(i,n) = d.(fn_sample{i}).Fit_Data.(fn{n}).Direct_Fit.Backward.TIS.TIS_val;
         
         
    end

end

text_color = "black";
figure(9)
imagesc(abs(round(RSquared_dir_for,4)))
for i = 1:size(theta, 2)
    for j = 1:numel(PV_names)
        % Draw a rectangle for each cell with the specified color
     %   color_condition = squeeze(abs(rbg_new_Lenz(idx_angles(i), j, :)));
        rectangle('Position', [i-0.5, j-0.5, 1, 1]);
        % Check the condition and set the text color
   % if all(color_condition == 1)
   %     text_color = 'black';
   % else
   %     text_color = 'white';
   % end
        text(i, j, num2str(abs(round(RSquared_dir_for(j, i),4)), '%.4f'), 'HorizontalAlignment', 'center', 'Color', text_color,'EdgeColor', 'none');    
    end
end
hold off;
colorbar()
xlabel('Incidence Angle $\theta_i [^\circ]$','FontSize',22);
ylabel('Samples','FontSize',22);
%xlim([0.5, size(PV_names, 2) + 0.5]);
%ylim([0.5, size(theta, 2) + 0.5]);
yticks(1:size(PV_names, 2));
xticks(1:size(theta, 2));
yticklabels(PV_names);
xticklabels({'0','15','30','45','60'});
colormap(gca, summer(50))
annotation('textbox',[0.87,0.88,0.1,0.1],'String','R^2','FitBoxToText','on','Color','black','FontSize',16,'EdgeColor',[1,1,1],'BackgroundColor','none')


%%

figure(10)
imagesc(abs(round(RSquared_dir_back,4)))
for i = 1:size(theta, 2)
    for j = 1:numel(PV_names)
        % Draw a rectangle for each cell with the specified color
        rectangle('Position', [i-0.5, j-0.5, 1, 1]);
        text(i, j, num2str(abs(round(RSquared_dir_back(j, i),4)), '%.4f'), 'HorizontalAlignment', 'center', 'Color', text_color,'EdgeColor', 'none');    
    end
end
hold off;
colorbar()
xlabel('Incidence Angle $\theta_i [^\circ]$','FontSize',22);
ylabel('Samples','FontSize',22);
yticks(1:size(PV_names, 2));
xticks(1:size(theta, 2));
yticklabels(PV_names);
xticklabels({'0','15','30','45','60'});
colormap(gca, summer(50))
annotation('textbox',[0.87,0.88,0.1,0.1],'String','R^2','FitBoxToText','on','Color','black','FontSize',16,'EdgeColor',[1,1,1],'BackgroundColor','none')





%%
%{
figure(9)
h = heatmap(theta,PV_names(1,:),abs(round(RSquared_dir_for,4)),'Colormap',hot,'ColorLimits',[0,1]);
annotation('textbox',[0.87,0.88,0.1,0.1],'String','R^2','FitBoxToText','on','Color','black','FontSize',16,'EdgeColor',[1,1,1],'BackgroundColor','none','FaceAlpha',1)
set(gca,'XLabel','Incidence angle \theta_i [\circ]')
set(gca,'YLabel','Samples')
%set(gca,'Title','Rsquared :  Rsquared_{dir,f}')
h.XLabel = strcat('\fontsize{20}',h.XLabel);
h.YLabel = strcat('\fontsize{20}',h.YLabel);
%h.Title = strcat('\fontsize{24}',h.Title);
%To set the font size of tick labels only, use the same approach. 
h.XDisplayLabels = strcat('\fontsize{12}',h.XDisplayLabels);
h.YDisplayLabels = strcat('\fontsize{12}',h.YDisplayLabels);
%set(gca,'FontSize',18)
%}

%%
figure(10)
h = heatmap(theta,PV_names(1,:),round(RSquared_dir_back,4),'Colormap',hot,'ColorLimits',[0,1]);
annotation('textbox',[0.87,0.88,0.1,0.1],'String','R^2','FitBoxToText','on','Color','black','FontSize',16,'EdgeColor',[1,1,1],'BackgroundColor','none')
set(gca,'XLabel','Incidence angle \theta_i [\circ]')
set(gca,'YLabel','Samples')
%set(gca,'Title','Rsquared_{dir,b}')
h.XLabel = strcat('\fontsize{20}',h.XLabel);
h.YLabel = strcat('\fontsize{20}',h.YLabel);
%h.Title = strcat('\fontsize{24}',h.Title);
%To set the font size of tick labels only, use the same approach. 
h.XDisplayLabels = strcat('\fontsize{14}',h.XDisplayLabels);
h.YDisplayLabels = strcat('\fontsize{14}',h.YDisplayLabels);

%%
figure(11)
h = heatmap(theta,PV_names(1,:),RSquared_log_for,'Colormap',hot,'ColorLimits',[0,1]);
annotation('textbox',[0.87,0.88,0.1,0.1],'String','R^2','FitBoxToText','on','Color','white','FontSize',16,'EdgeColor',[1,1,1],'BackgroundColor','black')
set(gca,'XLabel','Incidence angle \theta [\circ]')
set(gca,'YLabel','Samples')
set(gca,'Title','Rsquared_{log,f}')
h.XLabel = strcat('\fontsize{20}',h.XLabel);
h.YLabel = strcat('\fontsize{20}',h.YLabel);
h.Title = strcat('\fontsize{24}',h.Title);
%To set the font size of tick labels only, use the same approach. 
h.XDisplayLabels = strcat('\fontsize{12}',h.XDisplayLabels);
h.YDisplayLabels = strcat('\fontsize{12}',h.YDisplayLabels);


figure(12)
h = heatmap(theta,PV_names(1,:),RSquared_log_back,'Colormap',hot,'ColorLimits',[0,1]);
annotation('textbox',[0.87,0.88,0.1,0.1],'String','R^2','FitBoxToText','on','Color','white','FontSize',16,'EdgeColor',[1,1,1],'BackgroundColor','black')
set(gca,'XLabel','Incidence angle \theta [\circ]')
set(gca,'YLabel','Samples')
set(gca,'Title','Rsquared_{log,b}')
h.XLabel = strcat('\fontsize{20}',h.XLabel);
h.YLabel = strcat('\fontsize{20}',h.YLabel);
h.Title = strcat('\fontsize{24}',h.Title);
%To set the font size of tick labels only, use the same approach. 
h.XDisplayLabels = strcat('\fontsize{12}',h.XDisplayLabels);
h.YDisplayLabels = strcat('\fontsize{12}',h.YDisplayLabels);




%%

%{
figure(9)
h = heatmap(theta,PV_names(1,:),RMSE_log_for,'Colormap',hot,'ColorLimits',[0,0.5]);
annotation('textbox',[0.87,0.88,0.1,0.1],'String','RMSE [1/Sr]','FitBoxToText','on','Color','white','FontSize',16,'EdgeColor',[1,1,1],'BackgroundColor','black')
set(gca,'XLabel','\theta [\circ]')
set(gca,'YLabel','Samples')
h.XLabel = strcat('\fontsize{20}',h.XLabel);
h.YLabel = strcat('\fontsize{20}',h.YLabel);
%To set the font size of tick labels only, use the same approach. 
h.XDisplayLabels = strcat('\fontsize{12}',h.XDisplayLabels);
h.YDisplayLabels = strcat('\fontsize{12}',h.YDisplayLabels);


figure(10)
h = heatmap(theta,PV_names(1,:),RMSE_log_back,'Colormap',hot,'ColorLimits',[0,0.5]);
annotation('textbox',[0.87,0.88,0.1,0.1],'String','RMSE [1/Sr]','FitBoxToText','on','Color','white','FontSize',16,'EdgeColor',[1,1,1],'BackgroundColor','black')
set(gca,'XLabel','\theta [\circ]')
set(gca,'YLabel','Samples')
h.XLabel = strcat('\fontsize{20}',h.XLabel);
h.YLabel = strcat('\fontsize{20}',h.YLabel);
%To set the font size of tick labels only, use the same approach. 
h.XDisplayLabels = strcat('\fontsize{12}',h.XDisplayLabels);
h.YDisplayLabels = strcat('\fontsize{12}',h.YDisplayLabels);
%}

%{
figure(11)
h = heatmap(theta,PV_names(1,:),RMSE_log_for - RMSE_log_back,'Colormap',[1,0,0;0,0,0],'ColorLimits',[min(min(RMSE_log_for - RMSE_log_back)),-min(min(RMSE_log_for - RMSE_log_back))]);
annotation('textbox',[0.87,0.88,0.1,0.1],'String','RMSE [1/Sr]','FitBoxToText','on','Color','white','FontSize',16,'EdgeColor',[1,1,1],'BackgroundColor','black')
set(gca,'XLabel','Incidence angle \theta [\circ]')
set(gca,'YLabel','Samples')
set(gca,'Title','RMSE comparison:  RMSE_{log,f} - RMSE_{log,b}')
h.XLabel = strcat('\fontsize{20}',h.XLabel);
h.YLabel = strcat('\fontsize{20}',h.YLabel);
h.Title = strcat('\fontsize{24}',h.Title);
%To set the font size of tick labels only, use the same approach. 
h.XDisplayLabels = strcat('\fontsize{12}',h.XDisplayLabels);
h.YDisplayLabels = strcat('\fontsize{12}',h.YDisplayLabels);


figure(12)
h = heatmap(theta,PV_names(1,:),RMSE_dir_for - RMSE_dir_back,'Colormap',[1,0,0;0,0,0],'ColorLimits',[min(min(RMSE_log_for - RMSE_log_back)),-min(min(RMSE_log_for - RMSE_log_back))]);
annotation('textbox',[0.87,0.88,0.1,0.1],'String','RMSE [1/Sr]','FitBoxToText','on','Color','white','FontSize',16,'EdgeColor',[1,1,1],'BackgroundColor','black')
set(gca,'XLabel','Incidence angle \theta [\circ]')
set(gca,'YLabel','Samples')
set(gca,'Title','RMSE comparison:  RMSE_{dir,f} - RMSE_{dir,b}')
h.XLabel = strcat('\fontsize{20}',h.XLabel);
h.YLabel = strcat('\fontsize{20}',h.YLabel);
%To set the font size of tick labels only, use the same approach. 
h.XDisplayLabels = strcat('\fontsize{12}',h.XDisplayLabels);
h.YDisplayLabels = strcat('\fontsize{12}',h.YDisplayLabels);
h.Title = strcat('\fontsize{24}',h.Title);


figure(13)
h = heatmap(theta,PV_names(1,:),RMSE_dir_for - RMSE_log_for,'Colormap',[1,0,0;0,0,0],'ColorLimits',[min(min(RMSE_log_for - RMSE_log_back)),-min(min(RMSE_log_for - RMSE_log_back))]);
annotation('textbox',[0.87,0.88,0.1,0.1],'String','RMSE [1/Sr]','FitBoxToText','on','Color','white','FontSize',16,'EdgeColor',[1,1,1],'BackgroundColor','black')
set(gca,'XLabel','\theta [\circ]')
set(gca,'YLabel','Samples')
set(gca,'Title','RMSE comparison:  RMSE_{dir,f} - RMSE_{log,f}')
h.XLabel = strcat('\fontsize{20}',h.XLabel);
h.YLabel = strcat('\fontsize{20}',h.YLabel);
%To set the font size of tick labels only, use the same approach. 
h.XDisplayLabels = strcat('\fontsize{12}',h.XDisplayLabels);
h.YDisplayLabels = strcat('\fontsize{12}',h.YDisplayLabels);
h.Title = strcat('\fontsize{24}',h.Title);

%}
%% Polar Error average 
%colororder({'r','r','y'})
c2 = [0,0.5,0];
fig = figure(14);
yyaxis('left')
plot(theta,mean(RSquared_log_for),'-s','LineWidth',2,'LineStyle','--','MarkerSize',8,'MarkerFaceColor','r','MarkerEdgeColor','R','Color','r')
hold on
plot(theta,mean(RSquared_log_back),'-d','LineWidth',2,'LineStyle','--','MarkerSize',8,'MarkerFaceColor','r','MarkerEdgeColor','R','Color','r')
plot(theta,mean(RSquared_dir_for),'-v','LineWidth',2,'LineStyle','--','MarkerSize',8,'MarkerFaceColor','r','MarkerEdgeColor','R','Color','r')
plot(theta,mean(RSquared_dir_back),'-o','LineWidth',2,'LineStyle','--','MarkerSize',8,'MarkerFaceColor','r','MarkerEdgeColor','R','Color','r')
ylabel('Mean Sample Fit $R^2$',Fontsize = 20)
set(gca,'YColor',[1,0,0]);
yyaxis('right')
plot(theta,mean(RMSE_log_for),'-s','LineWidth',2,'LineStyle',':','MarkerSize',8,'MarkerFaceColor',c2,'MarkerEdgeColor',c2,'Color',c2)
plot(theta,mean(RMSE_log_back),'-d','LineWidth',2,'LineStyle',':','MarkerSize',8,'MarkerFaceColor',c2,'MarkerEdgeColor',c2,'Color',c2)
plot(theta,mean(RMSE_dir_for),'-v','LineWidth',2,'LineStyle',':','MarkerSize',8,'MarkerFaceColor',c2,'MarkerEdgeColor',c2,'Color',c2)
plot(theta,mean(RMSE_dir_back),'-o','LineWidth',2,'LineStyle',':','MarkerSize',8,'MarkerFaceColor',c2,'MarkerEdgeColor',c2,'Color',c2)
grid('on')
ylabel('Fit parameter RMSE [$^\circ$]','Rotation',270,Fontsize = 20)
xlabel('Incidence angle $\theta_i [^\circ]$',FontSize =20)
set(gca,'YColor',[0,0.5,0]);
lgd = legend({'$R^2$: $\mathrm{FS_{log}}$','$R^2$: $\mathrm{BS_{log}}$','$R^2$ $\mathrm{FS_{dir}}$','$R^2$: $\mathrm{BS_{dir}}$','RMSE: $\mathrm{FS_{log}}$','RMSE: $\mathrm{BS_{log}}$','RMSE $\mathrm{FS_{dir}}$','RMSE: $\mathrm{BS_{dir}}$'},'Interpreter','latex','Location','best',FontSize = 16);
title(lgd,'Fit precision')
lgd.Position(1) = 0.5; %- lgd.Position(3)/2; 
lgd.Position(2) = 0.35; %- lgd.Position(4)/2;
%ax = axes;
set(gca,'FontSize',18)
%ax.YAxis(1).Color = [1,0,0];
%ax.ColorOrder(1,:) = [1,0,0];
%ax.ColorOrder(2,:) = [0.63,0.07,0.18];
%set(fig,'defaultAxesColorOrder',[1,0,0;0,1,1])

%{
figure(14)
h = heatmap(theta,PV_names(1,:),RMSE_dir_back - RMSE_log_back,'Colormap',hot,'ColorLimits',[min(min(RMSE_log_for - RMSE_log_back)),-min(min(RMSE_log_for - RMSE_log_back))]);
annotation('textbox',[0.87,0.88,0.1,0.1],'String','RMSE [1/Sr]','FitBoxToText','on','Color','white','FontSize',16,'EdgeColor',[1,1,1],'BackgroundColor','black')
set(gca,'XLabel','\theta [\circ]')
set(gca,'YLabel','Samples')
h.XLabel = strcat('\fontsize{20}',h.XLabel);
h.YLabel = strcat('\fontsize{20}',h.YLabel);
%To set the font size of tick labels only, use the same approach. 
h.XDisplayLabels = strcat('\fontsize{12}',h.XDisplayLabels);
h.YDisplayLabels = strcat('\fontsize{12}',h.YDisplayLabels);

%}

%% Sample comparison 

Leg_2 = '%s';
leg_str = '- - - Fit: %s';
% Subplot for each fittype for all angles. (Forward and backward
% scattering)
sample = 1;
symbol = ["o","^","Pentagram","diamond","square","> ","hexagram","v"];
dim = [0.35,0.40,0.3,0.3];
 % Choosing sample and Angle.
i = 2;
fig = figure(15);
for sample=1:numel(PV_names)
ax1 = subplot(2,2,1);
hold on
%scatter3(ones(1,numel(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.x)).*theta(i),d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.brdf)
%plot3(ones(1,numel(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.x)).*theta(i),d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Direct_Fit.Forward.Fit)
scatter(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.brdf,26,mycolors(sample,:),'filled','Marker',symbol(sample))
hold on
plot(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Direct_Fit.Forward.Fit,'LineWidth',1,'LineStyle','--','color',mycolors(sample,:))
annotation('textbox',dim,'String',sprintf(leg_str,'$\frac{A}{B + |\bar{x}|^g}$'),'FitBoxToText','on','Interpreter','latex',FontSize=16,Color='black')
%plot(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Log_fit.Forward.Fit,'LineWidth',2,'LineStyle',':','color',mycolors(i,:))
set(gca,'YScale','log')
grid('on')
lgd = legend({sprintf(Leg_2,fn_sample{1}),'',sprintf(Leg_2,fn_sample{2}),'',sprintf(Leg_2,fn_sample{3}),'',sprintf(Leg_2,fn_sample{4}),'',sprintf(Leg_2,fn_sample{5}),'',sprintf(Leg_2,fn_sample{6}),'',sprintf(Leg_2,fn_sample{7}),'',sprintf(Leg_2,fn_sample{8}),''},'Location','northeast',FontSize = 10);
title(lgd,'FS Samples','FontWeight','bold')
%fig_size = get(ax1,"Position");
%lgd_size = get(lgd,'position');
%fig_size(3) = fig_size(3) + lgd_size(3);
%set(ax1,'position',fig_size)
xlim([0,1])
hold off
ax2 = subplot(2,2,2);
scatter(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Backward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Backward.brdf,26,mycolors(sample,:),'filled','Marker',symbol(sample))
hold on
plot(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Backward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Direct_Fit.Backward.Fit,'LineWidth',1,'LineStyle','--','color',mycolors(sample,:));
annotation('textbox',[0.8,0.40,0.3,0.3],'String',sprintf(leg_str,'$\frac{A}{B + |\bar{x}|^g}$'),'FitBoxToText','on','Interpreter','latex',FontSize=16,Color='black')
lgd = legend({sprintf(Leg_2,fn_sample{1}),'',sprintf(Leg_2,fn_sample{2}),'',sprintf(Leg_2,fn_sample{3}),'',sprintf(Leg_2,fn_sample{4}),'',sprintf(Leg_2,fn_sample{5}),'',sprintf(Leg_2,fn_sample{6}),'',sprintf(Leg_2,fn_sample{7}),'',sprintf(Leg_2,fn_sample{8}),''},'Location','northeast',FontSize = 10);
title(lgd,'BS Samples','FontWeight','bold')
grid('on')
set(gca,'YScale','log')
xlim([0,2])
ax3 = subplot(2,2,3);
scatter(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.brdf,26,mycolors(sample,:),'filled','Marker',symbol(sample))
hold on
plot(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Log_fit.Forward.Fit,'LineWidth',2,'LineStyle',':','color',mycolors(sample,:))
set(gca,'YScale','log')
grid('on')
annotation('textbox',[0.28,0.07,0.3,0.3],'String',sprintf(leg_str_2,'$log(A) -g \cdot log(|\bar{x}|)$'),'FitBoxToText','on','Interpreter','latex',FontSize=16,Color='black')
lgd = legend({sprintf(Leg_2,fn_sample{1}),'',sprintf(Leg_2,fn_sample{2}),'',sprintf(Leg_2,fn_sample{3}),'',sprintf(Leg_2,fn_sample{4}),'',sprintf(Leg_2,fn_sample{5}),'',sprintf(Leg_2,fn_sample{6}),'',sprintf(Leg_2,fn_sample{7}),'',sprintf(Leg_2,fn_sample{8}),''},'Location','southeast',FontSize = 10);
title(lgd,'FS Samples','FontWeight','bold')
xlim([0,1])
ax4 = subplot(2,2,4);
scatter(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Backward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Backward.brdf,26,mycolors(sample,:),'filled','Marker',symbol(sample))
hold on
plot(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Backward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Log_fit.Backward.Fit,'LineWidth',2,'LineStyle',':','color',mycolors(sample,:))
grid('on')
annotation('textbox',[0.7,0.07,0.3,0.3],'String',sprintf(leg_str_2,'$log(A) -g \cdot log(|\bar{x}|)$'),'FitBoxToText','on','Interpreter','latex',FontSize=16,Color='black')
lgd = legend({sprintf(Leg_2,fn_sample{1}),'',sprintf(Leg_2,fn_sample{2}),'',sprintf(Leg_2,fn_sample{3}),'',sprintf(Leg_2,fn_sample{4}),'',sprintf(Leg_2,fn_sample{5}),'',sprintf(Leg_2,fn_sample{6}),'',sprintf(Leg_2,fn_sample{7}),'',sprintf(Leg_2,fn_sample{8}),''},'Location','southeast',FontSize = 10);
title(lgd,'BS Samples','FontWeight','bold')
xlim([0,2])

%plot(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Backward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Log_fit.Backward.Fit,'LineWidth',2,'LineStyle',':','color',mycolors(i,:))

end

set(gca,'YScale','log')
%linkaxes([ax1,ax2],'xy')
han=axes(fig,'visible','off'); 
han.XLabel.Visible='on';
han.YLabel.Visible='on';
han.Title.Visible='on';
xlabel(han,'Displacement vector $\bar{x}$',Fontsize = 20);
ylabel(han,'BRDF [1/sr]',Fontsize = 20);
%title(han,sprintf('BRDF sample comparsion for Incidence = %d [Deg]',(theta(i))),FontSize= 20)
hold off

%%
mycolors =[1,1,0;0,0,1;1,0,0;0,1,0;0.5,0,0;0,0.5,0;0,0,0;0,0,0.5]; %[0,0,0;1,0,0;1,0.75,0;0,1,0;0,0.5,0;0,0,1;0,0,0.5;1,1,0;0.5,0.5,0;1,0,1;0.5,0,0.5];
symbol = ["o","^","Pentagram","diamond","square","> ","hexagram","v"];
Leg_2 = '%s';
leg_str = '- - - Fit: %s';
i = 2;
figure;
for sample=1:numel(PV_names)
%ax1 = subplot(2,2,1);
%hold on
%scatter3(ones(1,numel(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.x)).*theta(i),d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.brdf)
%plot3(ones(1,numel(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.x)).*theta(i),d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Direct_Fit.Forward.Fit)
scatter(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.brdf,85,mycolors(sample,:),'filled','Marker',symbol(sample) )
hold on
plot(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Direct_Fit.Forward.Fit,'LineWidth',1,'LineStyle','--','color',mycolors(sample,:),'LineWidth',3)
annotation('textbox',[0.5,0.7,0.2,0.2],'String',sprintf(leg_str,'$\frac{A}{B + |\vec{x}|^g}$'),'FitBoxToText','on','Interpreter','latex',FontSize=22,Color='black')
%plot(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Log_fit.Forward.Fit,'LineWidth',2,'LineStyle',':','color',mycolors(i,:))
set(gca,'YScale','log')
grid('on')
lgd = legend({sprintf(Leg_2,fn_sample{1}),'',sprintf(Leg_2,fn_sample{2}),'',sprintf(Leg_2,fn_sample{3}),'',sprintf(Leg_2,fn_sample{4}),'',sprintf(Leg_2,fn_sample{5}),'',sprintf(Leg_2,fn_sample{6}),'',sprintf(Leg_2,fn_sample{7}),'',sprintf(Leg_2,fn_sample{8}),''},'Location','northeast',FontSize = 16);
title(lgd,'FS Samples','FontWeight','bold')
end
set(gca,'YScale','log','FontSize',16)
xlim([0,0.8])
ylim([10^(-4),1])
xlabel('Displacement vector $\vec{x}$',Fontsize = 24);
ylabel('B($\theta_i = 15^\circ$) [1/sr]',Fontsize = 24);
hold off
%%
figure;
for sample=1:numel(PV_names)
scatter(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Backward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Backward.brdf,85,mycolors(sample,:),'filled','Marker',symbol(sample))
hold on
plot(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Backward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Direct_Fit.Backward.Fit,'LineWidth',1,'LineStyle','--','color',mycolors(sample,:),'LineWidth',3);
annotation('textbox',[0.5,0.7,0.2,0.2],'String',sprintf(leg_str,'$\frac{A}{B + |\vec{x}|^g}$'),'FitBoxToText','on','Interpreter','latex',FontSize=20,Color='black')
lgd = legend({sprintf(Leg_2,fn_sample{1}),'',sprintf(Leg_2,fn_sample{2}),'',sprintf(Leg_2,fn_sample{3}),'',sprintf(Leg_2,fn_sample{4}),'',sprintf(Leg_2,fn_sample{5}),'',sprintf(Leg_2,fn_sample{6}),'',sprintf(Leg_2,fn_sample{7}),'',sprintf(Leg_2,fn_sample{8}),''},'Location','southwest',FontSize = 18);
title(lgd,'BS Samples','FontWeight','bold')
grid('on')
end
set(gca,'YScale','log','FontSize',18)
xlim([0,1.3])
ylim([10^(-5),2])
xlabel('Displacement vector $\vec{x}$',Fontsize = 24);
ylabel('B($\theta_i = 15^\circ$) [1/sr]',Fontsize = 24);

%%
figure;
for sample=1:numel(PV_names)
scatter(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.brdf,85,mycolors(sample,:),'filled','Marker',symbol(sample))
hold on
plot(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Forward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Log_fit.Forward.Fit,'LineWidth',2,'LineStyle',':','color',mycolors(sample,:),'LineWidth',3)
set(gca,'YScale','log')
grid('on')
annotation('textbox',[0.35,0.7,0.2,0.2],'String',sprintf(leg_str_2,'$log(A) -g \cdot log(|\vec{x}|)$'),'FitBoxToText','on','Interpreter','latex',FontSize=20,Color='black')
lgd = legend({sprintf(Leg_2,fn_sample{1}),'',sprintf(Leg_2,fn_sample{2}),'',sprintf(Leg_2,fn_sample{3}),'',sprintf(Leg_2,fn_sample{4}),'',sprintf(Leg_2,fn_sample{5}),'',sprintf(Leg_2,fn_sample{6}),'',sprintf(Leg_2,fn_sample{7}),'',sprintf(Leg_2,fn_sample{8}),''},'Location','northeast',FontSize = 18);
title(lgd,'FS Samples','FontWeight','bold')
end
xlim([0,0.8])
set(gca,'YScale','log','FontSize',18)
xlabel('Displacement vector $\vec{x}$',Fontsize = 24);
ylabel('B($\theta_i = 15^\circ$) [1/sr]',Fontsize = 24);
ylim([10^(-4),1])


%%

figure;
for sample=1:numel(PV_names)

scatter(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Backward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Backward.brdf,85,mycolors(sample,:),'filled','Marker',symbol(sample))
hold on
plot(d.(fn_sample{sample}).Fit_Data.(fn{i}).Raw_Data.Backward.x,d.(fn_sample{sample}).Fit_Data.(fn{i}).Log_fit.Backward.Fit,'LineWidth',2,'LineStyle',':','color',mycolors(sample,:),'LineWidth',3)
grid('on')
annotation('textbox',[0.35,0.7,0.2,0.2],'String',sprintf(leg_str_2,'$log(A) -g \cdot log(|\vec{x}|)$'),'FitBoxToText','on','Interpreter','latex',FontSize=20,Color='black')
lgd = legend({sprintf(Leg_2,fn_sample{1}),'',sprintf(Leg_2,fn_sample{2}),'',sprintf(Leg_2,fn_sample{3}),'',sprintf(Leg_2,fn_sample{4}),'',sprintf(Leg_2,fn_sample{5}),'',sprintf(Leg_2,fn_sample{6}),'',sprintf(Leg_2,fn_sample{7}),'',sprintf(Leg_2,fn_sample{8}),''},'Location','southwest',FontSize = 18);
title(lgd,'BS Samples','FontWeight','bold')
end
set(gca,'YScale','log','FontSize',18)
xlabel('Displacement vector $\vec{x}$',Fontsize = 24);
ylabel('B($\theta_i = 15^\circ$) [1/sr]',Fontsize = 24);
xlim([0,1.3])
ylim([10^(-5),2])

%% TIS plots 

figure(17)
imagesc(abs(round(tis_array(1:6,:),2)))
for i = 1:size(theta, 2)
    for j = 1:numel(PV_names)-2
        % Draw a rectangle for each cell with the specified color
        rectangle('Position', [i-0.5, j-0.5, 1, 1]);
        text(i, j, num2str(abs(round(tis_array(j, i),2)), '%.2f'), 'HorizontalAlignment', 'center', 'Color', text_color,'EdgeColor', 'none','FontSize',16);    
    end
end
hold off;
colorbar()
clim([0,1.4])
set(gca,'FontSize',16)
xlabel('Incidence Angle $\theta_i [^\circ]$','FontSize',28);
ylabel('Samples','FontSize',28);
yticks(1:size(PV_names, 2));
xticks(1:size(theta, 2));
yticklabels(PV_names);
xticklabels({'0','15','30','45','60'});
colormap(gca, summer(50))

annotation('textbox',[0.85,0.88,0.1,0.1],'String','TIS','FitBoxToText','on','Color','black','FontSize',16,'EdgeColor',[1,1,1],'BackgroundColor','none')


%figure(17)
%h = heatmap(theta,PV_names(1:6),tis_array(1:6,:),'Colormap',hot);
%annotation('textbox',[0.88,0.89,0.1,0.1],'String','TIS','FitBoxToText','on','Color','white','FontSize',16,'EdgeColor',[1,1,1],'BackgroundColor','black')
%set(gca,'XLabel','Sample Angle \theta [\circ]')
%set(gca,'YLabel','Samples')
%clim([min(min(tis_array_back)),0.001])
%h.XLabel = strcat('\fontsize{20}',h.XLabel);
%h.YLabel = strcat('\fontsize{20}',h.YLabel);
%To set the font size of tick labels only, use the same approach. 
%h.XDisplayLabels = strcat('\fontsize{12}',h.XDisplayLabels);
%h.YDisplayLabels = strcat('\fontsize{12}',h.YDisplayLabels);
%%
figure(18)
imagesc(abs(round(tis_array_back(1:6,:),2)))
for i = 1:size(theta, 2)
    for j = 1:numel(PV_names)-2
        % Draw a rectangle for each cell with the specified color
        rectangle('Position', [i-0.5, j-0.5, 1, 1]);
        text(i, j, num2str(abs(round(tis_array_back(j, i),6)), '%.6f'), 'HorizontalAlignment', 'center', 'Color', text_color,'EdgeColor', 'none','FontSize',16);    
    end
end
hold off;
colorbar()
clim([0,1.4])
set(gca,'FontSize',16)
xlabel('Incidence Angle $\theta_i [^\circ]$','FontSize',28);
ylabel('Samples','FontSize',28);
yticks(1:size(PV_names, 2));
xticks(1:size(theta, 2));
yticklabels(PV_names);
xticklabels({'0','15','30','45','60'});
colormap(gca, summer(50))

annotation('textbox',[0.85,0.88,0.1,0.1],'String','TIS','FitBoxToText','on','Color','black','FontSize',16,'EdgeColor',[1,1,1],'BackgroundColor','none')

%figure(18)
%heatmap(theta,PV_names,tis_array_back,'Colormap',hot);
%annotation('textbox',[0.87,0.88,0.1,0.1],'String','TIS','FitBoxToText','on','Color','white','FontSize',16,'EdgeColor',[1,1,1],'BackgroundColor','black')
%set(gca,'XLabel','\theta [\circ]')
%set(gca,'YLabel','Samples')
%hh.XLabel = strcat('\fontsize{20}',h.XLabel);
%hh.YLabel = strcat('\fontsize{20}',h.YLabel);
%To set the font size of tick labels only, use the same approach. 
%h.XDisplayLabels = strcat('\fontsize{12}',h.XDisplayLabels);
%h.YDisplayLabels = strcat('\fontsize{12}',h.YDisplayLabels);





%{
figure(18)
for n=1:numel(theta)
    d.(fn_sample{1}).Fit_Data.(fn{n}).Direct_Fit.Forward.TIS.Fit
    
plot(d.(fn_sample{7}).Fit_Data.(fn{n}).Raw_Data.Forward.x,d.(fn_sample{1}).Fit_Data.(fn{n}).Direct_Fit.Forward.TIS.Fit,'-o','LineWidth',2,'LineStyle',':','Color',mycolors(n+1,:))
hold on
plot(d.(fn_sample{7}).Fit_Data.(fn{n}).Raw_Data.Forward.x,d.(fn_sample{1}).Fit_Data.(fn{n}).Raw_Data.Forward.brdf,'-o','LineWidth',2,'LineStyle',':','Color','black')
end
set(gca,'YScale','log')
grid('on')
%}

%% Saving coefficients to doc. 

filename = "ABg_coeffs_forward.dat";

% open file identifier
DIR=fopen(filename,'wt+'); % Clear and open file. 

for k=1:length(fn_sample)
name_str = sprintf('NM "%s" ',fn_sample{k});
Wav_str = sprintf('WV %.4f',0);
writelines(name_str,filename,WriteMode='append')
writelines(Wav_str,filename,WriteMode = 'append')
for m=1:numel(theta)
Data_str = sprintf('DT %d %.2f %f %.12f %f',[m,theta(m),d.(fn_sample{k}).Fit_Data.(fn{m}).Direct_Fit.Forward.coeffs]);
writelines(Data_str,filename,WriteMode = 'append')
end
end


%%

filename = "ABg_coeffs_backward.dat";

% open file identifier
DIR=fopen(filename,'wt+'); % Clear and open file. 

for k=1:length(fn_sample)
name_str = sprintf('NM "%s" ',fn_sample{k});
Wav_str = sprintf('WV %.4f',0);
writelines(name_str,filename,WriteMode='append')
writelines(Wav_str,filename,WriteMode = 'append')
for m=1:numel(theta)
Data_str = sprintf('DT %d %.2f %f %.12f %f',[m,theta(m),d.(fn_sample{k}).Fit_Data.(fn{m}).Direct_Fit.Backward.coeffs]);
writelines(Data_str,filename,WriteMode = 'append')
end
end

%% Combined
%format shortEng
%format compact
%nf = java.text.DecimalFormat;
%nf.setMinimumFractionDigits(4);
%nf.DecimalFormatSymbols(".")
%nf.format(10)
filename = "ABg_coeffs_Combined_Normalized_true.dat";

% open file identifier
method = ["FL","FD","BL","BD"];
DIR=fopen(filename,'wt+'); % Clear and open file. 
for u=1:numel(method)
for k=1:length(fn_sample)
name_str = sprintf('NM "%s%s"',upper(method(u)),upper(fn_sample{k}));
Wav_str = sprintf('WV %.4E',0);
writelines(name_str,filename,WriteMode='append')
writelines(Wav_str,filename,WriteMode = 'append')
for m=1:numel(theta)
    if u==1
        Data_str = sprintf('DT %d %.2E %E %.12E %E',[m-1,theta(m),d.(fn_sample{k}).Fit_Data.(fn{m}).Log_fit.Forward.coeffs]);
        writelines(Data_str,filename,WriteMode = 'append')
    elseif u==2
        Data_str = sprintf('DT %d %.2E %E %.12E %E',[m-1,theta(m),d.(fn_sample{k}).Fit_Data.(fn{m}).Direct_Fit.Forward.TIS.coeffs]);
        writelines(Data_str,filename,WriteMode = 'append')
    elseif u==3
        Data_str = sprintf('DT %d %.2E %E %.12E %E',[m-1,theta(m),d.(fn_sample{k}).Fit_Data.(fn{m}).Log_fit.Backward.coeffs]);
        writelines(Data_str,filename,WriteMode = 'append')
    elseif u==4
       Data_str = sprintf('DT %d %.2E %E %.12E %E',[m-1,theta(m),d.(fn_sample{k}).Fit_Data.(fn{m}).Direct_Fit.Backward.TIS.coeffs]);
        writelines(Data_str,filename,WriteMode = 'append')
    end
end
end
end

%%
%pvx = fitting{"Goldx30"};
%pvb = fitting{"GoldBRDF30"};
%[A,B,g,fit_poly] = ABgfit(pvx,pvb);
%TIS = abs(tis(A,B,g,theta(2),1,1));
%[A_f,B_f,g_f,fit_fit,gop_lin]= ABgfit_fit(pvx,pvb);
%[A_c,B_c,g_c,fit_c] = ABgfit(pvx,pvb./TIS);
%[A_d,B_d,g_d,fit_d,gop_dir] = direct_fit(pvx,pvb,A,B,g);
%TIS = (tis(A_f,B_f,g_f,theta(3),1,1));
%[A_f,B_f,g_f,fit_f,gop_dir] = direct_fit(pvx,pvb./TIS,A_f,B_f,g_f);
% Litterature value of ABg model. For incidence angles
%A_lit = [0.0018,0.0016,0.0015,0.002,0.004,0.0016];
%B_lit = [0.0095,0.001,0.0005,0.0005,0.0002,2*10^(-5)];
%g_lit = [2.1,2.1,2.3,2.05,1.85,2.3];
%fit_lit = A_lit(2)./(B_lit(2)+abs(pvx).^g_lit(2));
%[NA,sigma,gauss] = Gaussian(pvx,pvb);
%{
leg_str = 'Fit: %s with A = %.3f, B = %.3f, g = %.3f';
leg_str_2 =' Glass sample at  %s = %.2f';
%tis(A_c,B_c,g_c,theta(2),1,1)
figure(5)
plot((pvx),(pvb),'-o','Linestyle','--','LineWidth',2,'Color','blue')
%loglog(pvx,fit,'LineWidth',2,'Color','red')
hold on
%loglog(pvx,fit_poly,'DisplayName',sprintf('Non-Corrected polyfit: A= %f B= %f g= %f',[A,B,g]),'LineWidth',2,'Color','red')
%plot(pvx,fit_poly,'-o','LineWidth',2,'Color','red')
%loglog(pvx,fit_c,'DisplayName',sprintf('TIS corrected: A= %f B= %f g= %f',[A_c,B_c,g_c]),'LineWidth',2,'Color','yellow')
plot(pvx,fit_fit,'DisplayName',sprintf('Non-corrected Linear log fit: A= %f B= %f g= %f',[A_f,B_f,g_f]),'LineStyle','--','LineWidth',2,'Color','green')
%loglog(pvx,fit_d,'DisplayName',sprintf('Direct Fit: A= %f B= %f g= %f',[A_d,B_d,g_d]),'LineWidth',2,'Color','magenta')
%plot(pvx,log10(A)-g*log10(abs(pvx)),'LineWidth',2,'Linestyle','--','Color','black')
plot(pvx,fit_d,'LineWidth',2,'Linestyle','--','Color','magenta')
%plot(pvx,gauss,'-o','LineWidth',2,'Color','cyan')
%loglog(pvx,fit_lit,'DisplayName',sprintf('Litterature Fit: A= %f B= %f g= %f',[A_lit(2),B_lit(2),g_lit(2)]),'Linestyle','--','LineWidth',2,'Color','black')
ylim([10^(-3),5])
xlabel('Displacement vector $\bar{x}$',fontsize = 20)
ylabel('BRDF value [1/sr]',fontsize = 20)
legend({sprintf(leg_str_2,'$\theta_i$',30),sprintf(leg_str,'$log(A)-g \cdot log(|\bar{x}|)$',A,B,g),sprintf(leg_str,'$\frac{A}{B + |\bar{x}|^g}$',A_d,B_d,g_d)},'interpreter','latex',fontsize=16)
grid('on')
set(gca,'XScale','log','YScale','log')
%{
figure(6)
plot((pvx),(pvb),'DisplayName','Experimental data','LineWidth',2,'Color','blue')
%loglog(pvx,fit,'LineWidth',2,'Color','red')
hold on
plot(pvx,fit_poly,'DisplayName',sprintf('Non-Corrected polyfit: A= %f B= %f g= %f',[A,B,g]),'LineWidth',2,'Color','red')
%loglog(pvx,fit_c,'DisplayName',sprintf('Corrected fit: A= %f B= %f g= %f',[A_c,B_c,g_c]),'LineWidth',2,'Color','green')
plot(pvx,fit_fit,'DisplayName',sprintf('Non-corrected Linear log fit: A= %f B= %f g= %f',[A_f,B_f,g_f]),'LineStyle','--','LineWidth',2,'Color','green')
plot(pvx,fit_d,'DisplayName',sprintf('Direct Fit: A= %f B= %f g= %f',[A_d,B_d,g_d]),'LineWidth',2,'Color','magenta')
plot(pvx,fit_lit,'DisplayName',sprintf('Litterature Fit: A= %f B= %f g= %f',[A_lit(2),B_lit(2),g_lit(2)]),'Linestyle','--','LineWidth',2,'Color','black')
ylim([10^(-3),5])
xlabel('Displacement vector $\bar{x}$',fontsize = 20)
ylabel('BRDF value [1/sr]',fontsize = 20)
legend(fontsize=16)
grid('on')
%}


%}
%% Functions

tis(0.02,0.0031,2.8,0,deg2rad(5),deg2rad(5))

function x=x_disp(incidence,emergence,azimuth) % Incidence = theta, emergence = alpha.
x = zeros(numel(incidence),numel(emergence));
for i=1:numel(incidence)
    for n=1:numel(emergence)
        x(i,n) = sqrt((sind(emergence(n))*cosd(azimuth)-sind(incidence(i)))^2 + (sind(emergence(n))*sind(azimuth))^2);
    end
end
end

function [x_end,brdf] = x_sort(x,brdf_old) % Wishing to sort the x matrix. This is done for every independent incidence angle.
x_new = zeros(size(x));
[x_new,k] = sort(x,2,'ascend');
for i=1:numel(k)
brdf_new(i) = brdf_old(k(i));
end

j=1;
brdf = [];
x_end = [];
%%numel(brdf_new)
%numel(k)
%numel(brdf_old)
for i=1:numel(brdf_new)
 % ~isnan(brdf_new(i))
 if ~isnan(brdf_new(i)) && x_new(i) ~= 0 %|| brdf_new(i) ~= max(brdf_new)
    brdf= [brdf,brdf_new(i)];
    x_end = [x_end,x_new(i)];
  %  x_end
    j=j+1;
 %else
 %    disp('ERROR')
 %    x_new(i)
 %    brdf_new(i)
 end
 
% brdf_new = reshape(brdf_new,size(x_new));
end


end

function [A,sigma,gauss] = Gaussian(x,brdf)
%sigma_start = std(brdf);
%fo = fitoptions('Method','NonLinearLeastSquares','StartPoint',[1/(sqrt(2*pi*sigma_start)),sigma_start],'Lower',[0,0]);
%ft = fittype('A*exp((-x^2)/(2*B^2))','options',fo);
[curve,gof] = fit(x,brdf,'gauss1') %fit(x,brdf,ft);

c = coeffvalues(curve);
A = c(1);
sigma = c(2);
gauss = curve;
%gauss = A.*exp(-abs(x).^2 ./ sigma.^2);
end

function [A,B,g,fit_dir,gof] = direct_fit(x,brdf)
fo = fitoptions('Method','NonLinearLeastSquares','StartPoint',[0.01,0.01,2],'Lower',[0,10^(-12),0]);
ft = fittype('A / (B+x^g)','options',fo);
[curve,gof] = fit(abs(x.'),brdf.',ft);

c = coeffvalues(curve);
A = c(1);
B = c(2);
g = c(3);
fit_dir = A./(B+abs(x).^g);


end

function[A,B,g,fit_fit,gof] = ABgfit_fit(x,brdf)
[f1,gof] = fit(log10(x.'),log10(brdf.'),'poly1');
c = coeffvalues(f1);
g = -c(1);
A = 10^c(2);
B = A/brdf(1);
fit_fit = A./(B+abs(x).^g);
end

function [A,B,g,fit_poly] = ABgfit(x,brdf)
 p = polyfit(log10(x),log10(brdf),1); % Linear fit in log domain. [p,S,mu]
% Calculation of coefficient. when x>>B then log(A)- g*log(|x|)
g = -p(1);
A = 10^(p(2));
B = A/brdf(1); % Specular part described by A/B.
fit_poly = A./(B+abs(x).^g); % log(A) -g*log(|x|)
end


function TIS = tis(A,B,g,incidence,dth,dphi)
th = 0:dth:pi/2;
phi = 0:dphi:2*pi;
% Calc of eachx and product of integration of TIS.
for i=1:length(th)
    for n=1:length(phi)
        x_tis(i,n) = sqrt((sin(th(i))*cos(phi(n))-sin(incidence))^2 + (sin(th(i))*sin(phi(n)))^2 );
        cossin(i,n) = cos(th(i))*sin(th(i));
    end
end
%sum(A./(B+abs(x_tis).^g))
%B+abs(x_tis).^g
% Integration of the resulting arrays.
 TIS = (sum(A./(B+abs(x_tis(:,1)).^g).*cossin(:,1)))*dth*dphi;
 %TIS = sum(A./(B+x_tis.^g).*cossin)*dth*phi;

end







