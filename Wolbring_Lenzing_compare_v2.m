clear all

file_dir = dir(fullfile('..\','Processed_data/'));
Files = extractfield(file_dir,'name');
Files(1:2) = [];

Lenz = load(fullfile('..\','Processed_data/',Files{1}));
Wol = load(fullfile('..\','Processed_data/',Files{3}));
Encaps = load(fullfile('..\','Processed_data/',Files{4}));

wavelengths = Wol.wavelengths;
zero_point = round(numel(Wol.white_rel_sensor_ang(:,1))/2);
theta = Wol.f_ellipse_angles;
alpha = Wol.white_rel_sensor_ang(:,1);

set(0,'defaultTextInterpreter','latex')
set(0,'defaultLegendInterpreter','latex')
%% White measurement check 

ref_angles = [];

for i=4:3:numel(Wol.white_rel_sensor_ang(:,1))/2
    ref_angles =[ref_angles,Wol.white_rel_sensor_ang(i,1),-Wol.white_rel_sensor_ang(i,1)]; 
end
ref_angles = [ref_angles,0];
ref_angles = ref_angles(end:-1:1);
output = [];
for n=1:numel(ref_angles)
output = [output,find(Wol.white_rel_sensor_ang(:,1)==ref_angles(n))];
end

mycolors = [0,0,0;1,0,0;1,0.75,0;0,1,0;0,0.5,0;0,0,1;0,0,0.5;1,1,0;0.5,0.5,0;1,0,1;0.5,0,0.5];
figure;
for i=1:numel(ref_angles)%4:3:numel(white_rel_sensor_ang(:,6))-3
output(i)
plot(Wol.wavelengths,Wol.white_spectral_brdf{output(i),5},'color',mycolors(i,:),'DisplayName',sprintf('%.2f',ref_angles(i)),linewidth = 2);
hold on
end
hold off
title('Spectral BRDF white Lambertian reflector',fontsize = 20)
ylim([0,0.35])
xlim([310,960])
xlabel('Wavelength $\lambda$[nm] ','FontSize',20)
ylabel('$B_\lambda(\theta = 0^\circ)$[1/sr]' ,'FontSize',20)
set(gca,'LineWidth',2,'FontSize',20);
grid()
lgd = legend('Location','eastoutside');
title(lgd,'$\alpha[^\circ]$')

figure;
for i=1:numel(ref_angles)%4:3:numel(white_rel_sensor_ang(:,6))-3
plot(Wol.wavelengths,Wol.white_spectral_brdf{output(i),2},'color',mycolors(i,:),'DisplayName',sprintf('%.2f',ref_angles(i)),linewidth = 2);
hold on
end
hold off
title('Spectral BRDF white Lambertian reflector',fontsize = 20)
ylim([0,0.35])
xlim([310,960])
xlabel('Wavelength $\lambda$[nm] ','FontSize',20)
ylabel('$B_\lambda(\theta = 45^\circ)$[1/sr]' ,'FontSize',20)
set(gca,'LineWidth',2,'FontSize',20);
grid()
lgd = legend('Location','eastoutside');
title(lgd,'$\alpha[^\circ]$')



figure;
for i=1:numel(Wol.white_rel_sensor_ang(1,:))
Wol.white_meas_radiance(find(Wol.white_meas_radiance(:,i)./1000 < 0.6),i) = NaN;
plot(Wol.white_rel_sensor_ang(:,i),Wol.white_meas_radiance(:,i)./1000,'color',mycolors(i,:),'Marker','o','DisplayName',sprintf('%.2f',Wol.white_sample_ang(1,i)),linewidth = 2)
hold on
end
hold off
ylim([0,15])
xlim([-90,90])
xlabel('$\alpha [^\circ]$ ','FontSize',20)
ylabel('$L_({\Omega,n}) \mathrm{[kW/m^2/sr]}$' ,'FontSize',20)
set(gca,'LineWidth',2,'FontSize',20);
grid()
lgd = legend('Location','eastoutside');
title(lgd,'$\theta[^\circ]$')

%% White measurement encaps


ref_angles = [];

for i=4:3:numel(Encaps.white_rel_sensor_ang(:,1))/2
    ref_angles =[ref_angles,Encaps.white_rel_sensor_ang(i,1),-Encaps.white_rel_sensor_ang(i,1)]; 
end
ref_angles = [ref_angles,0];
ref_angles = ref_angles(end:-1:1);
output = [];
for n=1:numel(ref_angles)
output = [output,find(Encaps.white_rel_sensor_ang(:,1)==ref_angles(n))];
end

mycolors = [0,0,0;1,0,0;1,0.75,0;0,1,0;0,0.5,0;0,0,1;0,0,0.5;1,1,0;0.5,0.5,0;1,0,1;0.5,0,0.5];
figure(10)
for i=1:numel(ref_angles)%4:3:numel(white_rel_sensor_ang(:,6))-3
output(i)
plot(Wol.wavelengths,Encaps.white_spectral_brdf{output(i),5},'color',mycolors(i,:),'DisplayName',sprintf('%.2f',ref_angles(i)),linewidth = 2);
hold on
end
hold off
title('Spectral BRDF white Lambertian reflector',fontsize = 20)
ylim([0,0.35])
xlim([310,960])
xlabel('Wavelength $\lambda$[nm] ','FontSize',20)
ylabel('$B_\lambda(\theta = 0^\circ)$[1/sr]' ,'FontSize',20)
set(gca,'LineWidth',2,'FontSize',20);
grid()
lgd = legend('Location','eastoutside');
title(lgd,'$\alpha[^\circ]$')

figure;
for i=1:numel(ref_angles)%4:3:numel(white_rel_sensor_ang(:,6))-3
plot(Wol.wavelengths,Encaps.white_spectral_brdf{output(i),2},'color',mycolors(i,:),'DisplayName',sprintf('%.2f',ref_angles(i)),linewidth = 2);
hold on
end
hold off
title('Spectral BRDF white Lambertian reflector',fontsize = 20)
ylim([0,0.35])
xlim([310,960])
xlabel('Wavelength $\lambda$[nm] ','FontSize',20)
ylabel('$B_\lambda(\theta = 45^\circ)$[1/sr]' ,'FontSize',20)
set(gca,'LineWidth',2,'FontSize',20);
grid()
lgd = legend('Location','eastoutside');
title(lgd,'$\alpha[^\circ]$')



figure;
for i=1:numel(Encaps.white_rel_sensor_ang(1,:))
Encaps.white_meas_radiance(find(Encaps.white_meas_radiance(:,i)./1000 < 0.6),i) = NaN;
plot(Encaps.white_rel_sensor_ang(:,i),Encaps.white_meas_radiance(:,i)./1000,'color',mycolors(i,:),'Marker','o','DisplayName',sprintf('%.2f',Encaps.white_sample_ang(1,i)),linewidth = 2)
hold on
end
hold off
ylim([0,15])
xlim([-90,90])
xlabel('$\alpha [^\circ]$ ','FontSize',20)
ylabel('$L_({\Omega,n}) \mathrm{[kW/m^2/sr]}$' ,'FontSize',20)
set(gca,'LineWidth',2,'FontSize',20);
grid()
lgd = legend('Location','eastoutside');
title(lgd,'$\theta[^\circ]$')


%% Comparison Spectral BRDF for all samples.
%Wol.sample_files
[Wol_col,Wol_Str,Wol_keys] = create_clabels(Wol.sample_files);
[Lenz_col,Lenz_Str,Lenz_keys] = create_clabels(Lenz.sample_files);

figure;
ax1 = subplot(2,1,1);
hold on
for n=1:numel(Wol_Str)
plot(wavelengths,Wol.BRDF_spectral_brdf{zero_point,2,n},'DisplayName',Wol_Str(n),'color',Wol_col{Wol_keys(n)},linewidth = 2)
end
hold off
ylim([0,0.18])
xlim([310,980])
xlabel('Wavelength $\lambda$ [nm]','FontSize',20)
ylabel('$B_\lambda(\alpha = 0,\theta = 15^\circ)$ [1/sr]' ,'FontSize',20)
set(gca,'LineWidth',2,'FontSize',20);
grid()
legend('Location','northeastoutside')
ax2 = subplot(2,1,2);
hold on
for n=1:numel(Lenz_Str(1:6))
plot(wavelengths,Lenz.BRDF_spectral_brdf{zero_point,2,n},'DisplayName',Lenz_Str(n),'color',Lenz_col{Lenz_keys(n)},linewidth = 2)
end
hold off
linkaxes([ax1,ax2],'x')
ylim([0,0.18])
xlim([310,980])
xlabel('Wavelength $\lambda$ [nm]','FontSize',20)
ylabel('$B_\lambda(\alpha = 0,\theta = 15^\circ)$ [1/sr]' ,'FontSize',20)
set(gca,'LineWidth',2,'FontSize',20);
grid()
legend('Location','northeastoutside')


figure;
ax1 = subplot(2,1,1);
hold on
for n=1:numel(Wol_Str)
plot(wavelengths,Wol.BRDF_spectral_brdf{zero_point,3,n},'DisplayName',Wol_Str(n),'color',Wol_col{Wol_keys(n)},linewidth = 2)
end
hold off
ylim([0,0.06])
xlim([310,980])
xlabel('Wavelength $\lambda$ [nm]','FontSize',20)
ylabel('$B_\lambda(\alpha = 0,\theta = 30^\circ)$ [1/sr]' ,'FontSize',20)
set(gca,'LineWidth',2,'FontSize',20);
grid()
legend('Location','northeastoutside')
ax2 = subplot(2,1,2);
hold on
for n=1:numel(Lenz_Str(1:6))
plot(wavelengths,Lenz.BRDF_spectral_brdf{zero_point,3,n},'DisplayName',Lenz_Str(n),'color',Lenz_col{Lenz_keys(n)},linewidth = 2)
end
hold off
linkaxes([ax1,ax2],'x')
ylim([0,0.06])
xlim([310,980])
xlabel('Wavelength $\lambda$ [nm]','FontSize',20)
ylabel('$B_\lambda(\alpha = 0,\theta = 30^\circ)$ [1/sr]' ,'FontSize',20)
set(gca,'LineWidth',2,'FontSize',20);
grid()
legend('Location','northeastoutside')
%% Wolbring BRDF

figure;
%ax1 = subplot(2,1,1);
hold on
for n=1:numel(Wol_Str)
plot(wavelengths,Wol.BRDF_spectral_brdf{zero_point,2,n},'DisplayName',Wol_Str(n),'color',Wol_col{Wol_keys(n)},linewidth = 2)
end
hold off
ylim([0,0.18])
xlim([310,980])
xlabel('Wavelength $\lambda$ [nm]','FontSize',24)
ylabel('$B_\lambda(\alpha = 0,\theta_e = 15^\circ)$ [1/sr]' ,'FontSize',24)
set(gca,'LineWidth',2,'FontSize',20);
grid()
legend('Location','northwest')


%%

figure;
%ax1 = subplot(2,1,1);
hold on
for n=1:numel(Wol_Str)
plot(alpha,Wol.BRDF_brdf(:,2,n),'DisplayName',Wol_Str(n),'color',Wol_col{Wol_keys(n)},linewidth = 2)
end
hold off
%ylim([0,0.18])
xlim([-90,90])
xlabel('Relative Sensor Angle $\alpha [^\circ]$','FontSize',24)
ylabel('$B_\lambda(\theta_e = 15^\circ)$ [1/sr]' ,'FontSize',24)
set(gca,'LineWidth',2,'FontSize',20,'YScale','log');
grid()
legend('Location','northwest')



%% Comparison of BRDF same color ( Green ) 

pos_alpha = Wol.white_rel_sensor_ang(zero_point:3:end-1,6);
count = 1;
figure;
ax1 = subplot(2,1,1);
hold on
for i=zero_point:3:numel(Wol.white_rel_sensor_ang(:,6))-1
    plot(wavelengths,Wol.BRDF_spectral_brdf{i,1,2},'DisplayName',sprintf('%.2f',pos_alpha(count)),linewidth = 2)
    count = count + 1;
end
hold off
ylim([0,0.08])
xlim([310,960])
xlabel('Wavelength $\lambda$[nm] ','FontSize',20)
ylabel('$B_\lambda(\theta = 0)$[1/sr]' ,'FontSize',20)
set(gca,'LineWidth',2,'FontSize',20);
grid()
%lgd = legend;
%title(lgd,'$\alpha[^\circ]$')

% Next plot
ax2 = subplot(2,1,2);
hold on
count = 1;
for i=zero_point:3:numel(Wol.white_rel_sensor_ang(:,6))-1
    plot(wavelengths,Lenz.BRDF_spectral_brdf{i,1,1},'DisplayName',sprintf('%.2f',pos_alpha(count)),linewidth = 2)
    count = count + 1;
end
hold off
ylim([0,0.08])
xlim([310,960])
xlabel('Wavelength $\lambda$[nm] ','FontSize',20)
ylabel('$B_\lambda(\theta = 0)$[1/sr]' ,'FontSize',20)
set(gca,'LineWidth',2,'FontSize',20);
grid()
lgd = legend('Location','northeastoutside');
lgd.Position(1:2) = [.85,.3];
title(lgd,'$\alpha[^\circ]$')
linkaxes([ax1,ax2],'x')


%% For encaps

pos_alpha = Encaps.white_rel_sensor_ang(zero_point:3:end-1,6);
count = 1;
Encaps.sample_files.name
figure;
ax1 = subplot(2,1,1);
hold on
for i=zero_point:3:numel(Encaps.white_rel_sensor_ang(:,1))-1
    plot(wavelengths,Encaps.BRDF_spectral_brdf{i,2,2},'DisplayName',sprintf('%.2f',pos_alpha(count)),linewidth = 2)
    count = count + 1;
end
hold off
%ylim([0,0.08])
xlim([310,960])
xlabel('Wavelength $\lambda$[nm] ','FontSize',20)
ylabel('$B_\lambda(\theta = 0)$[1/sr]' ,'FontSize',20)
set(gca,'LineWidth',2,'FontSize',20);
grid()
%lgd = legend;
%title(lgd,'$\alpha[^\circ]$')

% Next plot
ax2 = subplot(2,1,2);
hold on
count = 1;
for i=zero_point:3:numel(Wol.white_rel_sensor_ang(:,6))-1
    plot(wavelengths,Lenz.BRDF_spectral_brdf{i,2,1},'DisplayName',sprintf('%.2f',pos_alpha(count)),linewidth = 2)
    count = count + 1;
end
hold off
ylim([0,0.08])
xlim([310,960])
xlabel('Wavelength $\lambda$[nm] ','FontSize',20)
ylabel('$B_\lambda(\theta = 0)$[1/sr]' ,'FontSize',20)
set(gca,'LineWidth',2,'FontSize',20);
grid()
lgd = legend('Location','northeastoutside');
lgd.Position(1:2) = [.85,.3];
title(lgd,'$\alpha[^\circ]$')
linkaxes([ax1,ax2],'x')

%% Polar plot and colors.
% Initializing color variables. 
Freq = 3;
 S = Wol.sample_rgb(:,:,2); % Green = 2, Yellow = 3, Terracotta = 4, Blue = 1
S_L = Lenz.sample_rgb(:,:,1); % Green = 1, Gold = 2, Terracotta = 4, Reference = 3
[L_Wol,a_Wol,b_Wol,rgb_Wol] = LAB_3D(Wol.sample_Lab(:,:,2),Wol.sample_rgb(:,:,2)); % Green
E_00_M_Wol = create_cmatrix(Wol.sample_Lab,Freq,2,'Single');
[L_Lenz,a_Lenz,b_Lenz,rgb_Lenz] = LAB_3D(Lenz.sample_Lab(:,:,1),Lenz.sample_rgb(:,:,1)); % Green 
E_00_M_Lenz = create_cmatrix(Lenz.sample_Lab,Freq,1,'Single');

%numericMatrix = size(cell2mat(Wol.sample_rgb(:,:,2)));
%rgb_Wol = reshape(numericMatrix, [], 3);


%rgb_Wol(size(Wol.sample_rgb(:,:,2))) = Wol.sample_rgb{:,:,2}
figure;
polarscatter(repmat(deg2rad(Wol.white_rel_sensor_ang(:,1)),numel(Wol.f_ellipse_angles(1,:)),1),reshape(repmat((Wol.f_ellipse_angles),numel(Wol.white_rel_sensor_ang(:,1)),1),[numel(rgb_Wol(:,1)),1]),50,abs(rgb_Wol),'filled')
ax = gca;
ax.ThetaLim = [-90,90];
thetaTicks = (-90:15:90);
ax.ThetaZeroLocation = 'Top';
ax.ThetaDir = 'clockwise';
rticks(0:15:60)
rticklabels({'\theta_e = 0\circ','\theta_e = 15\circ','\theta_e = 30\circ','\theta_e = 45\circ','\theta_e = 60\circ'})
ax.ThetaAxis.Label.String = 'Sensor angle $\alpha$';
ax.ThetaAxis.Label.Position = [0,75];
ax.FontSize = 16;

th_angles = repmat(deg2rad(Lenz.white_rel_sensor_ang(:,1)),numel(Lenz.f_ellipse_angles(1,:)),1);
ah_angles = reshape(repmat((Lenz.f_ellipse_angles),numel(Lenz.white_rel_sensor_ang(:,1)),1),[numel(rgb_Lenz(:,1)),1]);
figure;
polarscatter(th_angles(1:1:end,1),ah_angles(1:1:end,1),50,abs(rgb_Lenz(1:1:end,:)),'filled')
ax = gca;
ax.ThetaLim = [-90,90];
thetaTicks = (-90:15:90);
ax.ThetaZeroLocation = 'Top';
ax.ThetaDir = 'clockwise';
rticks(0:15:60)
rticklabels({'\theta_e = 0\circ','\theta_e = 15\circ','\theta_e = 30\circ','\theta_e = 45\circ','\theta_e = 60\circ'})
ax.ThetaAxis.Label.String = 'Sensor angle $\alpha$';
ax.ThetaAxis.Label.Position = [0,75];
ax.FontSize = 16;
%%
ah_angles = repmat((Wol.white_rel_sensor_ang(:,1)),numel(Wol.f_ellipse_angles(1,:)),1);
  
   l = numel(S(1,:)); % Length of Theta 
    w = numel(S(:,1)); % Length of Alpha 
    rbg_new = ones(w,l,3);
    rgb_new_Lenz = ones(w,l,3);
    c = 1;
    for i=1:w
        for n=1:l
            %log = rbg{i,n} > 1;
             if S{i,n}(1) > 1 || S{i,n}(2) > 1 || S{i,n}(3) > 1

                 %{
                    S12 = ones(1,3);
                    siin = S{i,n};
                    [S_max,id_max] = max(S{i,n});
                    S12(1,id_max) = 1;
                    indices_not_max = find(S{i,n} ~= S_max);
                    %S12 = siin(indices_not_max);
                    S12(1,indices_not_max) = siin(indices_not_max)/S{i,n}(id_max);
                    
                    
                    rbg_new(i,n,1) = S12(1) ;
                    rbg_new(i,n,2) = S12(2) ;
                    rbg_new(i,n,3) = S12(3) ;
                 %}
                 
             else  
                    rbg_new(i,n,1) = S{i,n}(1) ;
                    rbg_new(i,n,2) = S{i,n}(2) ;
                    rbg_new(i,n,3) = S{i,n}(3) ;
             end

             if   S_L{i,n}(1) > 1 || S_L{i,n}(2) > 1 || S_L{i,n}(3) > 1
                  
                    rbg_new_Lenz(i,n,1) = 1;%S_L{i,n}(1);
                    rbg_new_Lenz(i,n,2) = 1;%S_L{i,n}(2);
                    rbg_new_Lenz(i,n,3) = 1;%S_L{i,n}(3);
                 
             else
                    rbg_new_Lenz(i,n,1) = S_L{i,n}(1);
                    rbg_new_Lenz(i,n,2) = S_L{i,n}(2);
                    rbg_new_Lenz(i,n,3) = S_L{i,n}(3);
             end
        end
    end
q =  ismember(Lenz.white_rel_sensor_ang(:,1),ah_angles(1:Freq:37));    
idx_angles = find(q);

result = [];
v = rbg_new;
for row = 1:size(v, 2)
    % Initialize a temporary array for the current row
    temp = [];
    
    % Loop through the elements of the current row with a step of 15
    for i = 1:15:size(v, 1)
        % Take the next 5 values
        temp = [temp; v(i:min(i+4, size(v, 1)), row)];
    end  
    % Append the temporary array to the result
    result = [result, temp];
end
% Loop through the vector with a step of 15 (5 values + 10 skipped)
%for i = 1:15:length(rbg_new)
%    % Take the next 5 values
%    result = [result, rbg_new(i:min(i+4, length(rbg_new)))];
%end
%%
figure;
imagesc(E_00_M_Wol)
for i = 1:size(E_00_M_Wol, 1)
    for j = 1:size(E_00_M_Wol, 2)
        color_condition = squeeze(abs(rbg_new(idx_angles(i), j, :)));
        % Draw a rectangle for each cell with the specified color
        rectangle('Position', [j-0.5, i-0.5, 1, 1], 'FaceColor', squeeze(abs(rbg_new(idx_angles(i), j, :))), 'EdgeColor', 'none');
        if all(color_condition == 1)
            text_color = 'black';
        else
            text_color = 'white';
        end
        text(j, i, num2str(E_00_M_Wol(i, j), '%.1f'), 'HorizontalAlignment', 'center', 'Color', text_color);
       
    end
end
hold off;
xlabel('Sample Angle $\theta_e [^\circ]$','FontSize',18);
ylabel('Relative Sensor Angle $\alpha [^\circ]$','FontSize',18);
xlim([0.5, size(E_00_M_Wol, 2) + 0.5]);
ylim([0.5, size(E_00_M_Wol, 1) + 0.5]);
xticks(1:size(E_00_M_Wol, 2));
yticks(1:size(E_00_M_Wol, 1));
xticklabels(cellstr(num2str(Wol.f_ellipse_angles(:))));
yticklabels(cellstr(num2str(Wol.white_rel_sensor_ang(1:Freq:end,1))));


%% Lenzing heatmap 

figure;
imagesc(E_00_M_Lenz)
for i = 1:size(E_00_M_Lenz, 1)
    for j = 1:size(E_00_M_Lenz, 2)
        % Draw a rectangle for each cell with the specified color
        color_condition = squeeze(abs(rbg_new_Lenz(idx_angles(i), j, :)));
        rectangle('Position', [j-0.5, i-0.5, 1, 1], 'FaceColor', squeeze(abs(rbg_new_Lenz(idx_angles(i), j, :))), 'EdgeColor', 'none');
        % Check the condition and set the text color
    if all(color_condition == 1)
        text_color = 'black';
    else
        text_color = 'white';
    end
        text(j, i, num2str(E_00_M_Lenz(i, j), '%.1f'), 'HorizontalAlignment', 'center', 'Color', text_color);
    end
end
hold off;
xlabel('Sample Angle $\theta_e [^\circ]$','FontSize',18);
ylabel('Relative Sensor Angle $\alpha [^\circ]$','FontSize',18);
xlim([0.5, size(E_00_M_Lenz, 2) + 0.5]);
ylim([0.5, size(E_00_M_Lenz, 1) + 0.5]);
xticks(1:size(E_00_M_Lenz, 2));
yticks(1:size(E_00_M_Lenz, 1));
xticklabels(cellstr(num2str(Lenz.f_ellipse_angles(:))));
yticklabels(cellstr(num2str(Lenz.white_rel_sensor_ang(1:Freq:end,1))));



% Set axis properties
%axis equal tight;
%h = heatmap(Wol.f_ellipse_angles,Wol.white_rel_sensor_ang(1:Freq:end,1),E_00_M_Wol,'Colormap',abs(result),'XLabel','Sample angle \theta_e[\circ]','YLabel','Sensor angle \alpha[\circ]');
%h.ColorbarVisible = 'off';
%h.XLabel = strcat('\fontsize{20}',h.XLabel);
%h.YLabel = strcat('\fontsize{18}',h.YLabel);
%h.XDisplayLabels = strcat('\fontsize{15}',h.XDisplayLabels);
%h.YDisplayLabels = strcat('\fontsize{12}',h.YDisplayLabels);
%h.Title= 'CIE color difference \DeltaE_{00}';
%%

fig = figure;
ax1 = subplot(2,1,1);
scatter(a_Wol,L_Wol,60,abs(rgb_Wol),'filled')
ylabel('$L^\star$',FontSize=20)
grid on
xlabel('$a^\star$',Fontsize = 20 )
%xlim([-30,10])
ax2 = subplot(2,1,2);
scatter(b_Wol,L_Wol,60,abs(rgb_Wol),'filled')
ylabel('$L^\star$',FontSize=24)
xlabel('$b^\star$',Fontsize = 24)
%xlim([-10,15])
grid on
ax1.YAxis.FontSize = 18;
ax1.XAxis.FontSize = 18;
ax1.LabelFontSizeMultiplier = 1.8;
ax2.YAxis.FontSize = 18;
ax2.XAxis.FontSize = 18;
ax2.LabelFontSizeMultiplier = 1.8;

fig = figure;
ax1 = subplot(2,1,1);
scatter(a_Lenz,L_Lenz,60,abs(rgb_Lenz),'filled')
ylabel('$L^\star$',FontSize=20)
grid on
xlabel('$a^\star$',Fontsize = 20 )
%xlim([-30,10])
ax2 = subplot(2,1,2);
scatter(b_Lenz,L_Lenz,60,abs(rgb_Lenz),'filled')
ylabel('$L^\star$',FontSize=24)
xlabel('$b^\star$',Fontsize = 24)
%xlim([-10,15])
grid on
ax1.YAxis.FontSize = 18;
ax1.XAxis.FontSize = 18;
ax1.LabelFontSizeMultiplier = 1.8;
ax2.YAxis.FontSize = 18;
ax2.XAxis.FontSize = 18;
ax2.LabelFontSizeMultiplier = 1.8;


%ax2.TickLength = [0.2, 0.2];
%{
fig = figure;
ax1 = subplot(2,2,1);
scatter(a_Wol,L_Wol,20,abs(rgb_Wol),'filled')
ylabel('$L^\star$',FontSize=20)
grid on
xlabel('$a^\star$',Fontsize = 20 )
xlim([-30,10])
ax2 = subplot(2,2,3);
scatter(b_Wol,L_Wol,20,abs(rgb_Wol),'filled')
ylabel('$L^\star$',FontSize=20)
xlabel('$b^\star$',Fontsize = 20)
xlim([-10,15])
grid on
ax3 = subplot(2,2,2);
scatter(a_Lenz,L_Lenz,20,abs(rgb_Lenz),'filled')
xlabel('$a^\star$',Fontsize = 20 )
xlim([-30,10])
grid on
ax4 = subplot(2,2,4);
scatter(b_Lenz,L_Lenz,20,abs(rgb_Lenz),'filled')
grid on
xlim([-10,15])
linkaxes([ax1,ax2],'y')
linkaxes([ax3,ax4],'y')
xlabel('$b^\star$',Fontsize = 20)
%}
%% Encaps / Lenz comparison

[L_Enc,a_Enc,b_Enc,rgb_Enc] = LAB_3D(Encaps.sample_Lab(:,:,1),Encaps.sample_rgb(:,:,1));
E_00_M_Enc = create_cmatrix(Encaps.sample_Lab,Freq,2,'Single');
E00_compare = create_ccomparematrix(Wol.sample_Lab,Lenz.sample_Lab,Freq,2,'Full'); %create_ccomparematrix(Encaps.sample_Lab,Lenz.sample_Lab,Freq,2,'Full');


figure;
polarscatter(repmat(deg2rad(Encaps.white_rel_sensor_ang(:,1)),numel(Encaps.f_ellipse_angles(1,:)),1),reshape(repmat((Encaps.f_ellipse_angles),numel(Encaps.white_rel_sensor_ang(:,1)),1),[numel(rgb_Enc(:,1)),1]),80,abs(rgb_Enc),'filled')
ax = gca;
ax.ThetaLim = [-90,90];
thetaTicks = (-90:15:90);
ax.ThetaZeroLocation = 'Top';
ax.ThetaDir = 'clockwise';
rticks(0:15:60)
rticklabels({'\theta = 0\circ','\theta = 15\circ','\theta = 30\circ','\theta = 45\circ','\theta = 60\circ'})
ax.ThetaAxis.Label.String = 'Sensor angle $\alpha$';
ax.ThetaAxis.Label.Position = [0,75];
ax.FontSize = 16;

figure;
polarscatter(repmat(deg2rad(Lenz.white_rel_sensor_ang(:,1)),numel(Lenz.f_ellipse_angles(1,:)),1),reshape(repmat((Lenz.f_ellipse_angles),numel(Lenz.white_rel_sensor_ang(:,1)),1),[numel(rgb_Lenz(:,1)),1]),80,abs(rgb_Lenz),'filled')
ax = gca;
ax.ThetaLim = [-90,90];
thetaTicks = (-90:15:90);
ax.ThetaZeroLocation = 'Top';
ax.ThetaDir = 'clockwise';
rticks(0:15:60)
rticklabels({'\theta = 0\circ','\theta = 15\circ','\theta = 30\circ','\theta = 45\circ','\theta = 60\circ'})
ax.ThetaAxis.Label.String = 'Sensor angle $\alpha$';
ax.ThetaAxis.Label.Position = [0,75];
ax.FontSize = 16;
%%
% Heatmap compare.
figure;
h = heatmap(Encaps.f_ellipse_angles,Encaps.white_rel_sensor_ang(1:Freq:end,1),E00_compare,'Colormap',copper,'XLabel','Sample angle \theta_e[\circ]','YLabel','Sensor angle \alpha[\circ]');
%h.Title= 'CIE color difference \DeltaE_{00} with/without glass';
h.XLabel = strcat('\fontsize{20}',h.XLabel);
h.YLabel = strcat('\fontsize{18}',h.YLabel);
h.XDisplayLabels = strcat('\fontsize{15}',h.XDisplayLabels);
h.YDisplayLabels = strcat('\fontsize{12}',h.YDisplayLabels);
annotation('textbox',[0.82,0.81,0.2,0.2],'string','\DeltaE_{00}','FitBoxToText','on','FontSize',17,'EdgeColor','none');
%%
fig = figure;
ax1 = subplot(2,2,1);
scatter(a_Enc,L_Enc,20,abs(rgb_Enc),'filled')
ylabel('$L^\star$',FontSize=20)
grid on
xlabel('$a^\star$',Fontsize = 20 )
xlim([-30,10])
ax2 = subplot(2,2,3);
scatter(b_Enc,L_Enc,20,abs(rgb_Enc),'filled')
ylabel('$L^\star$',FontSize=20)
xlabel('$b^\star$',Fontsize = 20)
xlim([-10,15])
grid on
ax3 = subplot(2,2,2);
scatter(a_Lenz,L_Lenz,20,abs(rgb_Lenz),'filled')
xlabel('$a^\star$',Fontsize = 20 )
xlim([-30,10])
grid on
ax4 = subplot(2,2,4);
scatter(b_Lenz,L_Lenz,20,abs(rgb_Lenz),'filled')
grid on
xlim([-10,15])
linkaxes([ax1,ax2],'y')
linkaxes([ax3,ax4],'y')
xlabel('$b^\star$',Fontsize = 20)

%% a/b plots and colimetry comparisons 
labels = {"Wolbring","Lenzing","Encapsulant"};
colors = {'blue','magenta','Yellow'};
a = {a_Wol,a_Lenz,a_Enc};
b = {b_Wol,b_Lenz,b_Enc};
rgb = {rgb_Wol,rgb_Lenz,rgb_Enc};
h = gobjects(1,3);
figure;
for i=1:3
ax =  subplot(3,1,i);
h(1,i) = scatter(a{i},b{i},28,abs(rgb{i}),'filled');
%text([.1,.1,.1],[.66;.5;.33],{'1' '2' '3'},'VerticalAlignment', 'Bottom')
set(h(1,i), {'DisplayName'},compose('%s',labels{i}))
set(h(1,i), {'MarkerEdgeColor'},compose('%s',colors{i}))
grid on
ylim([-10,10])
xlim([-25,5])
%{
grid on
ax2 = subplot(3,1,2);
scatter(a_Lenz,b_Lenz,20,abs(rgb_Lenz),'filled')
%ylabel('$b^\star$',fontsize = 20)
grid on
ax3 = subplot(3,1,3);
scatter(a_Enc,b_Enc,20,abs(rgb_Enc),'filled','DisplayName',labels)
grid on
%xlabel('$a^\star$',fontsize = 20)
%}
end
%linkaxes([ax1,ax2,ax3],'xy')
set(h(1,:), {'MarkerEdgeAlpha'},compose('%s',0.15))
lgd = legend(h(:),'Location','northeastoutside');
xlabel('$a^\star$',fontsize = 26)
yl = ylabel('$b^\star$',fontsize = 26);
yl.Position(1:2) = [-25.5,28.8];
lgd.Position(1:2) = [.85,.5];
title(lgd,'PV type')
%linkprop([ax1 ax2,ax3], {'GridColor','GridLineStyle','GridAlpha'});


%% Chroma / hue plots
C_ab = cell(1,3);
h_ab = cell(1,3);
for i=1:3
C_ab{i} = sqrt(a{i}.^2 + b{i}.^2);% Chroma in LAB colorspace
h_ab{i} = atan2d(b{i} , a{i});
q = find(h_ab{i} < 0);
h_ab{i}(q) = abs(h_ab{i}(q)) + 180;
end
%%
fig = figure;
ax1 = subplot(2,3,1);
scatter(C_ab{1},L_Wol,28,abs(rgb_Wol),'filled')
ylabel('$L^\star$',FontSize=20)
grid on
xlabel('$C_{ab}^\star$',Fontsize = 20 )
xlim([0,30])
ax2 = subplot(2,3,4);
scatter(h_ab{1},L_Wol,28,abs(rgb_Wol),'filled')
ylabel('$L^\star$',FontSize=20)
xlabel('$h_{ab}$',Fontsize = 20)
xlim([0,360])
grid on
ax3 = subplot(2,3,2);
scatter(C_ab{2},L_Lenz,28,abs(rgb_Lenz),'filled')
xlabel('$C_{ab}^\star$',Fontsize = 20 )
xlim([0,30])
grid on
ax4 = subplot(2,3,5);
scatter(h_ab{2},L_Lenz,28,abs(rgb_Lenz),'filled')
grid on
xlim([0,360])
xlabel('$h_{ab}$',Fontsize = 20)
ax5 = subplot(2,3,3);
scatter(C_ab{3},L_Enc,28,abs(rgb_Enc),'filled')
xlim([0,30])
xlabel('$C^\star_{ab}$',Fontsize = 20)
grid on 
ax6 = subplot(2,3,6);
scatter(h_ab{3},L_Enc,28,abs(rgb_Enc),'filled')
xlim([0,360])
xlabel('$h_{ab}$',Fontsize = 20)
grid on
linkaxes([ax1,ax2],'y')
linkaxes([ax3,ax4],'y')
linkaxes([ax5,ax6],'y')



%% Angular colorimetry 

for i=1:3
ax =  subplot(3,1,i);
hold on
for n=1:5
C_temp = reshape(C_ab{i},37,5);
h_temp = reshape(h_ab{i},37,5);
plot(h_temp(:,n),C_temp(:,n),'o','DisplayName',sprintf('%.f',Lenz.f_ellipse_angles(1,n)),'MarkerFaceColor',mycolors(n,:),'MarkerEdgeColor',mycolors(n,:),linewidth = 2);
grid on
ylim([0,30])
xlim([0,360])
set(gca,'FontSize',18)
if i == 2
    ylabel('Chroma $C_{ab}^\star$',fontsize = 26);
end
end
end
hold off
%set(h(1,:), {'MarkerEdgeAlpha'},compose('%s',0.15))
lgd = legend('Location','northeastoutside');
%ylabel('$C_{ab}^\star$',fontsize = 26);
xlabel('Hue $h_{ab} [^\circ]$',fontsize = 26);
%y_l.Position(1:2) = [.85,.5];
lgd.Position(1:2) = [.88,.45];
title(lgd,'$\theta_e [^\circ]$',Fontsize = 20)
lgd.FontSize = 18;
annotation('textbox',[0.2,0.5,0.1,0.1],'String','Lenzing Plastics','FontSize',16,'FitBoxToText','on')
annotation('textbox',[0.2,0.8,0.1,0.1],'String','Ceramic Colors Wolbring','FontSize',16,'FitBoxToText','on')
annotation('textbox',[0.2,0.2,0.1,0.1],'String','Encapsulant','FontSize',16,'FitBoxToText','on')

%%
mycolors = [0,0,0;1,0,0;1,0.75,0;0,1,0;0,0.5,0;0,0,1;0,0,0.5;1,1,0;0.5,0.5,0;1,0,1;0.5,0,0.5;0.5,0.5,0.5;1,0.5,1];
% ,'MarkerFaceColor',mycolors(n,:),'MarkerEdgeColor',mycolors(n,:),
for i=1:3
ax =  subplot(3,1,i);
hold on
count = 1;
for n=19:Freq:numel(h_temp(:,1))
randcolor = [rand,rand,rand];
C_temp = reshape(C_ab{i},37,5);
h_temp = reshape(h_ab{i},37,5);
plot(h_temp(n,:),C_temp(n,:),'o','DisplayName',sprintf('%.f',Lenz.white_rel_sensor_ang(n,1)),'MarkerFaceColor',mycolors(count,:),'MarkerEdgeColor',mycolors(count,:),linewidth = 2);
count = count + 1;
set(gca,'FontSize',18)
grid on
ylim([0,30])
xlim([0,360])
if i == 2
    ylabel('Chroma $C_{ab}^\star$',fontsize = 26);
end
end
end
hold off
%set(h(1,:), {'MarkerEdgeAlpha'},compose('%s',0.15))
lgd = legend('Location','northeastoutside');
%ylabel('$C_{ab}^\star$',fontsize = 26);
xlabel('Hue $h_{ab} [^\circ]$',fontsize = 26);
%y_l.Position(1:2) = [.85,.5];
lgd.Position(1:2) = [.88,.45];
title(lgd,'$\alpha [^\circ]$',Fontsize = 20)
lgd.FontSize = 14;
annotation('textbox',[0.2,0.5,0.1,0.1],'String','Lenzing Plastics','FontSize',16,'FitBoxToText','on')
annotation('textbox',[0.2,0.8,0.1,0.1],'String','Ceramic Colors Wolbring','FontSize',16,'FitBoxToText','on')
annotation('textbox',[0.2,0.2,0.1,0.1],'String','Encapsulant','FontSize',16,'FitBoxToText','on')

%%

figure;
plot(atan2d)

%% Functions
[L_Wol,a_Wol,b_Wol,rgb_Wol] = LAB_3D(Wol.sample_Lab(:,:,2),Wol.sample_rgb(:,:,2)); % Green
E_00_M_Wol = create_cmatrix(Wol.sample_Lab,Freq,4,'Single')

%E_00_M_Wol = create_cmatrix(Wol.sample_Lab,Freq,1,'Single')
%size(rgb_Wol)
function [col,Str,k] = create_clabels(sample_files)
name_idx = 29;
dic_color = dictionary;
color = [];
RGB = {[0,1,0],[0,0,1],[0.8706,0.8706,0.1294],[1,0,0],[0.4000,0.0745,0.1373],[0,0.60,0],[1,1,0],[193,84,2]./255,[0.5,0.5,0.5],[0,0,1]}; %{[0,0,1],[0,1,0],[0.8706,0.8706,0.1294],[1,0,0],[0.4000,0.0745,0.1373],[0,0.60,0]};
col = dictionary;
colors = "_" + ["Lightgreen","Ref","Gold","Red","Terracotta","Green",'Yellow','Terracotta','Grey','Blue'];
for i=1:numel(sample_files)
    for q=1:numel(colors)
        if contains(lower(sample_files(i).name),lower(colors(q))) == 1
            color = colors(q);
            break
        else
            continue
        end
    end
    key = sample_files(i).name(name_idx:name_idx+3);
   
    dic_color(key) = color;
    col(key) = RGB(q);
end
 k = keys(dic_color);
[u,v] = sort(str2double(k));
 Str = strrep("LP" + k + dic_color(k),'_'," ");

%k = keys(dic_color);
%Str = strrep(k + dic_color(k),'_'," ");
end



function M = create_cmatrix(color_data,freq,PV_cell,compare_type)
M = zeros(numel(color_data(1:freq:end,1,1)),numel(color_data(1,:,1)));
for i=1:numel(color_data(1,:,1))
    E00 = calc_E00(color_data(:,i,PV_cell),color_data(:,1,PV_cell),compare_type); % Change single or full here.
    M(:,i) = E00(1:freq:end);
end
end

function M = create_ccomparematrix(color_data,color_data_ref,freq,PV_cell,compare_type)
M = zeros(numel(color_data(1:freq:end,1,1)),numel(color_data(1,:,1)));
for i=1:numel(color_data(1,:,1))
    E00 = calc_E00(color_data(:,i,PV_cell),color_data_ref(:,i,PV_cell),compare_type); % Change single or full here.
    M(:,i) = E00(1:freq:end);
end

end

function [T,col] = create_ctable(color_data,color_type,RGB,freq)
    if color_type == 'LAB'
        var = ["$a^\star$","$b^\star$","$L^\star$"];
        [L,a,b,col] = LAB_3D(color_data,RGB);
        a = a(1:freq:end);
        b = b(1:freq:end);
        L = L(1:freq:end);
        col = col(1:freq:end,:);
        T = table(a,b,L);
        %T.Properties.VariableNames = var;
    elseif color_type == 'DEGREE'
        var = {'$\alpha$','$\theta$','$\Delta _{00}$'};

    else 
        error('Function does not support this type')
    end
end

function E_00 = calc_E00(LAB_data,LAB_ref,compare_type,k_L,k_C,k_H) % Inserting cell specifying which PV-cell and and theta angle we want.
    if nargin == 3
        k_L = 1;
        k_C = 1;
        k_H = 1;

    elseif nargin < 5
        error('Only specify LAB data to get default values of weighting functions')
    end
    %initialze inputs.
    data = numel(LAB_data(:,1));
    L_star = zeros(data,1); % h2 in the formulars.
    a_star = zeros(data,1);
    b_star = zeros(data,1);
   % L_star_ref = LAB_ref{midpoint}(1);
   % a_star_ref = LAB_ref{midpoint}(2);
   % b_star_ref = LAB_ref{midpoint}(3);
    L_star_ref = zeros(data,1);
    a_star_ref = zeros(data,1); % h1 in the formulars. Reference is the color reference.
    b_star_ref = zeros(data,1);
    
    if strcmp(compare_type,'Full') == 1

    for i=1:data
    L_star(i) = LAB_data{i}(1);
    a_star(i) = LAB_data{i}(2);
    b_star(i) = LAB_data{i}(3);
    L_star_ref(i) = LAB_ref{i}(1);
    a_star_ref(i) = LAB_ref{i}(2);
    b_star_ref(i) = LAB_ref{i}(3);
    end

    
    elseif strcmp(compare_type,'Single') == 1

    for i=1:data
    L_star(i) = LAB_data{i}(1);
    a_star(i) = LAB_data{i}(2);
    b_star(i) = LAB_data{i}(3);
    L_star_ref(i) = LAB_ref{19}(1);
    a_star_ref(i) = LAB_ref{19}(2);
    b_star_ref(i) = LAB_ref{19}(3);
    end
    
    end

    %Lightness parameters
    Delta_L_prime = L_star-L_star_ref;
    L_bar = (L_star + L_star_ref)/2;

    S_L = 1 + (0.015*(L_bar-50).^2) ./ sqrt(20+(L_bar-50).^2);

    %Chroma parameters
    C_star = sqrt(a_star.^2 + b_star.^2);
    C_star_ref = sqrt(a_star_ref.^2 + b_star_ref.^2);
    C_bar = (C_star+C_star_ref)/2;
    a_prime = a_star + (a_star/2) .*(1-sqrt(C_bar.^7 ./ (C_bar.^7+25^7)));
    a_prime_ref = a_star_ref + (a_star_ref/2) .* (1-sqrt(C_bar.^7 ./ (C_bar.^7+25^7)));
    C_prime = sqrt(a_prime.^2 + b_star.^2);
    C_prime_ref = sqrt(a_prime_ref.^2 + b_star_ref.^2);
    Delta_C_prime = C_prime-C_prime_ref;
    C_bar_prime = (C_prime_ref + C_prime)/2;
    
    S_C = 1 + 0.045.*C_bar_prime;
    %hue parameters
    h_prime = mod(atan2d(b_star,a_prime),360);
    h_prime_ref = mod(atan2d(b_star_ref,a_prime_ref),360);
    Delta_h_prime = calc_h(h_prime,h_prime_ref); % Refer to the function calc_h for the piecewise function. 
    Delta_H_prime = 2*sqrt(C_prime.*C_prime_ref).*sind(Delta_h_prime/2);
    H_bar_prime = calc_H(h_prime,h_prime_ref);
    T = 1-0.17*cosd(H_bar_prime-30)+0.24*cosd(2*H_bar_prime)+0.32*cosd(3*H_bar_prime+6)-0.20*cosd(4*H_bar_prime-63);

    S_H = 1+0.015*C_bar_prime.*T;

    %Combined parameters
    R_T = -2*sqrt( (C_bar_prime.^7 ./ (C_bar_prime + 25^7)) ) .* sind(60*exp(-((H_bar_prime-275)/25).^2));
    term_L = (Delta_L_prime./(k_L*S_L)).^2;
    term_C = (Delta_C_prime./(k_C*S_C)).^2;
    term_H = (Delta_H_prime./(k_H*S_H)).^2;
    rest_term = R_T.*(Delta_C_prime./(k_C*S_C)).*(Delta_H_prime./(k_H*S_H));

    % Final calculation
    E_00 = sqrt(term_L + term_C + term_H + rest_term);
    
end
function Delta_h = calc_h(h2,h_ref) % input are two hue vectors that needs to be calculated piecewise.
    L = numel(h2);
    Delta_h = zeros(L,1);
    for i=1:L
        if abs(h_ref(i)-h2(i))<= 180
            Delta_h(i) = h2(i)-h_ref(i);
        elseif abs(h_ref(i)-h2(i)) > 180 && h2(i) <= h_ref(i)
            Delta_h(i) = h2(i)-h_ref(i) + 360;
        elseif abs(h_ref(i)-h2(i)) > 180 && h2(i) > h_ref(i)
            Delta_h(i) = h2(i) - h_ref(i) - 360;
        end
    end
end

function H_bar_prime = calc_H(h2,h_ref)
    L = numel(h2);
    H_bar_prime = zeros(L,1);
    for i=1:L
        if abs(h_ref-h2) > 180
            H_bar_prime(i) = (h2(i) + h_ref(i) + 360)/2;
        elseif abs(h_ref-h2) <= 180
            H_bar_prime(i) = (h2(i)+h_ref(i))/2;
        end
    end

end

function [L,a,b,rbg] =  LAB_3D(S,col)
    l = numel(S(1,:)); % Length of Theta 
    w = numel(S(:,1)); % Length of Alpha 
    L = zeros(l*w,1);
    a = zeros(l*w,1);
    b = zeros(l*w,1);
    rbg = zeros(l*w,3);
    size(rbg)
    c = 1;
    for n=1:l
        for i=1:w
            %log = rbg{i,n} > 1;
             if col{i,n}(1) > 1 || col{i,n}(2) > 1 || col{i,n}(3) > 1
                 continue
             else
                    L(c) = S{i,n}(1);
                    a(c) = S{i,n}(2);
                    b(c) = S{i,n}(3);
                    rbg(c,:) = col{i,n}  ;
                    c = c + 1;
             end
        end
    end
end


 