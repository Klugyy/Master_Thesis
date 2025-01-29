close all;clc

BRDF_colour_analysis_Lenz;

%% BRDF spectral brdf comparison between textured and non textured.
alpha = numel(BRDF_spectral_brdf(:,1,1)); %Sensor angles -90 to 90 at 5 degree increments
theta = numel(BRDF_spectral_brdf(1,:,1)); % Sample angles Number of total angles = nangles in orignal script.
midpoint = round(alpha/2,0); % Point when alpha = 0
name_idx = 29;
% Spectrums of theta = 0, alpha = 0 for all samples divided in textured and
% non textured.
dic_color = dictionary;
color = [];
RBG = {[0,1,0],[0,0,1],[0.8706,0.8706,0.1294],[1,0,0],[0.4000,0.0745,0.1373],[0,0.60,0]}; %{[0,0,1],[0,1,0],[0.8706,0.8706,0.1294],[1,0,0],[0.4000,0.0745,0.1373],[0,0.60,0]};
colorization = dictionary;
colors = "_" + ["Lightgreen","Ref","Gold","Red","Brown","Green"];
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
    colorization(key) = RBG(q);
end
k = keys(dic_color);
[u,v] = sort(str2double(k));
Str = strrep("LP" + k + dic_color(k),'_'," ");


color_compare = '_LightGreen';
color_ID = [];
for ii=1:nsamples
    hjal = lower(values(dic_color));
    if hjal(ii) == lower(color_compare)
        color_ID = [color_ID,ii];
    end
end

for ii=1:nsamples
    n = v(ii);
   % n
    if n<=6
   figure(1)
   plot(wavelengths,BRDF_spectral_brdf{midpoint,2,n},'DisplayName',Str(n),'color',colorization{k(n)},linewidth = 2)
   hold on
   %if n==6
 %   hold off
    ylim([0,0.18])
    xlim([310,980])
    %title('Non-textured of #' + Str(n),fontsize=20)
    xlabel('Wavelength $ \lambda$ [nm]','FontSize',20)
    ylabel('$B_\lambda(\alpha = 0,\theta_e = 15^\circ)$ [1/sr]' ,'FontSize',20)
    set(gca,'LineWidth',2,'FontSize',20);
    grid('on')
    legend('Location','northwest')
  % end
    else
   figure(2)
   plot(wavelengths,BRDF_spectral_brdf{midpoint,2,n},'DisplayName',Str(n),'color',colorization{k(n)},linewidth = 2)
   hold on
    end
end
hold off
%title('Textured of #' + Str(n),fontsize = 20)
ylim([0,0.18])
xlim([310,980])
xlabel('Wavelength $ \lambda$ [nm]','FontSize',20)
ylabel('$B_\lambda(\alpha = 0,\theta_e = 15^\circ)$ [1/sr]' ,'FontSize',20)
set(gca,'LineWidth',2,'FontSize',20);
grid()
legend('Location','northwest')

%% Full BRDF


for ii=1:nsamples
    n = v(ii);
   % n
    if n<=6
       % BRDF_brdf(:,2,n)'
   figure(30)
   plot(white_rel_sensor_ang(:,1),BRDF_brdf(:,4,n),'-o','DisplayName',Str(n),'color',colorization{k(n)},linewidth = 2)
   hold on
   %if n==6
 %   hold off
    ylim([0.001,5])
   xlim([-90,90])
    %title('Non-textured of #' + Str(n),fontsize=20)
    xlabel('Relative Sensor Angle $ \alpha [^\circ]$ ','FontSize',20)
    ylabel('$B(\theta_e = 45^\circ)$ [1/sr]' ,'FontSize',20)
    set(gca,'LineWidth',2,'FontSize',20,'YScale','log');
    grid('on')
    legend('Location','northwest')
  % end
    else
   figure(31)
   plot(white_rel_sensor_ang(:,1),BRDF_brdf(:,4,n),'-o','DisplayName',Str(n),'color',colorization{k(n)},linewidth = 2)
   hold on
    end
end
hold off
%title('Textured of #' + Str(n),fontsize = 20)
ylim([0.001,5])
xlim([-90,90])
xlabel('Relative Sensor Angle $ \alpha [^\circ]$ ','FontSize',20)
ylabel('$B(\theta_e = 45^\circ)$ [1/sr]' ,'FontSize',20)
set(gca,'LineWidth',2,'FontSize',20,'YScale','log');
grid()
legend('Location','northwest')



%% Reference cell comparison
figure(3)
for i=1:nangles
    plot(white_rel_sensor_ang(:,1),BRDF_brdf(:,i,3),'Marker','o','DisplayName',sprintf('%.2f',f_ellipse_angles(i)) ,linewidth = 2)
    hold on
end
hold off
title('Non-Textured',fontsize = 20)
ylim([0,6])
xlim([-90,90])
xlabel('Sensor angle $\alpha [^\circ]$ ','FontSize',20)
ylabel('B[1/sr]' ,'FontSize',20)
set(gca,'LineWidth',2,'FontSize',20);
grid()
lgd = legend('Location','northeastoutside');
title(lgd,'$\theta[^\circ]$')
%hleg.Title.NodeChildren.Position = [0.7,0,0]
figure(4)
for i=1:nangles
    plot(white_rel_sensor_ang(:,1),BRDF_brdf(:,i,7),'Marker','o','DisplayName',sprintf('%.2f',f_ellipse_angles(i)),linewidth = 2)
    hold on
end
hold off
title('Textured',fontsize = 20)
ylim([0,0.2])
xlim([-90,90])
xlabel('Sensor angle $\alpha [^\circ]$ ','FontSize',20)
ylabel('B[1/sr]' ,'FontSize',20)
set(gca,'LineWidth',2,'FontSize',20);
grid()
lgd = legend('Location','northeastoutside');
title(lgd,'$\theta[^\circ]$')
%ylim([0,6])
pos_alpha = white_rel_sensor_ang(midpoint:3:end-1,6);

count = 1;
numel(BRDF_spectral_brdf{i,1,1})
figure(5)
for i=midpoint:3:numel(white_rel_sensor_ang(:,6))-1
    plot(wavelengths,BRDF_spectral_brdf{i,1,3},'DisplayName',sprintf('%.2f',pos_alpha(count)),linewidth = 2)
    count = count + 1;
    hold on
end
hold off
title('Non-Textured',fontsize = 20)
ylim([0,0.03])
xlim([310,960])
xlabel('Wavelength $\lambda$[nm] ','FontSize',20)
ylabel('$B_\lambda(\theta = 0)$[1/sr]' ,'FontSize',20)
set(gca,'LineWidth',2,'FontSize',20);
grid()
lgd = legend('Location','northeastoutside');
title(lgd,'$\alpha[^\circ]$')

count = 1;

figure(6)
for i=midpoint:3:numel(white_rel_sensor_ang(:,6))-1
    plot(wavelengths,BRDF_spectral_brdf{i,1,7},'DisplayName',sprintf('%.2f',pos_alpha(count)),linewidth = 2)
    count = count + 1;
    hold on
end
hold off
title('Textured',fontsize = 20)
ylim([0,0.04])
xlim([310,960])
xlabel('Wavelength $\lambda$[nm] ','FontSize',20)
ylabel('$B_\lambda(\theta = 0)$[1/sr]' ,'FontSize',20)
set(gca,'LineWidth',2,'FontSize',20);
grid()
lgd = legend('Location','northeastoutside');
title(lgd,'$\alpha[^\circ]$')

%% alpha theta contour plot of the two green pieces
size(white_rel_sensor_ang(:,1))
M_Green = Contour_Matrix(BRDF_spectral_brdf,white_rel_sensor_ang(:,1),f_ellipse_angles,'alpha',color_ID(1),1);
M_wave = Contour_Matrix(BRDF_brdf,white_rel_sensor_ang(:,1),f_ellipse_angles,'wavelength',color_ID(1),1);
M_wave_text = Contour_Matrix(BRDF_brdf,white_rel_sensor_ang(:,1),f_ellipse_angles,'wavelength',1,1);

figure(7)

contour(wavelengths, white_rel_sensor_ang(:,1),M_Green.')
colorbar()
xlim([310,950])

figure(8)
contourf(white_rel_sensor_ang(:,1),f_ellipse_angles,M_wave.')
hcb = colorbar();
xlim([0,90])
xlabel('Sensor angle $\alpha[^\circ]$',fontsize = 20)
ylabel('Sample angle $\theta[^\circ]$',fontsize = 20)
ylabel(hcb,'B[1/sr]','Fontsize',16,'Rotation',270)

figure(9)

contourf(white_rel_sensor_ang(:,1),f_ellipse_angles,M_wave_text.')
hcb = colorbar();
xlim([0,90])
xlabel('Sensor angle $\alpha[^\circ]$',fontsize = 20)
ylabel('Sample angle $\theta[^\circ]$',fontsize = 20)
ylabel(hcb,'B[1/sr]','Fontsize',16,'Rotation',270)

%% White Reference plots

ref_angles = [];

for i=4:3:numel(white_rel_sensor_ang(:,1))/2
    ref_angles =[ref_angles,white_rel_sensor_ang(i,1),-white_rel_sensor_ang(i,1)]; 
end
ref_angles = [ref_angles,0];
ref_angles = ref_angles(end:-1:1);
output = [];
for n=1:numel(ref_angles)
output = [output,find(white_rel_sensor_ang(:,1)==ref_angles(n))];
end

mycolors = [0,0,0;1,0,0;1,0.75,0;0,1,0;0,0.5,0;0,0,1;0,0,0.5;1,1,0;0.5,0.5,0;1,0,1;0.5,0,0.5];
%colororder(mycolors)
figure(10)
for i=1:numel(ref_angles)%4:3:numel(white_rel_sensor_ang(:,6))-3
output(i)
plot(wavelengths,white_spectral_brdf{output(i),5},'color',mycolors(i,:),'DisplayName',sprintf('%.2f',ref_angles(i)),linewidth = 2);
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

figure(11)
for i=1:numel(ref_angles)%4:3:numel(white_rel_sensor_ang(:,6))-3
plot(wavelengths,white_spectral_brdf{output(i),2},'color',mycolors(i,:),'DisplayName',sprintf('%.2f',ref_angles(i)),linewidth = 2);
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


%%
figure(13)
for i=1:numel(white_rel_sensor_ang(1,:))
white_meas_radiance(find(white_meas_radiance(:,i)./1000 < 0.01),i) = NaN;
plot(white_rel_sensor_ang(:,i),white_meas_radiance(:,i)./1000,'color',mycolors(i,:),'Marker','o','DisplayName',sprintf('%.2f',white_sample_ang(1,i)),linewidth = 2)
hold on
end
hold off
ylim([0,15])
xlim([-90,90])
title('Radiance white Lambertian reflector',fontsize = 20)
xlabel('Sensor angle $\alpha [^\circ]$ ','FontSize',20)
ylabel('$L_({\Omega,n}) \mathrm{[kW/m^2/sr]}$' ,'FontSize',20)
set(gca,'LineWidth',2,'FontSize',20);
grid()
lgd = legend('Location','eastoutside');
title(lgd,'$\theta[^\circ]$')




%% Colimetry plots

S = sample_Lab(:,:,8);
col = sample_rgb(:,:,5);
 
[L,a,b,rgb] = LAB_3D(S,col);
Mat = [a,b,L];
o_size = 20*ones(numel(L(:,1)),1);
%colororder(abs(rgb))
%colororder('reef')
% o_size,abs(rgb)
figure(12)
scatter3(a,b,L,o_size,abs(rgb),'filled')
xlabel('$a^*$',fontsize = 20)
ylabel('$b^*$',fontsize = 20)
zlabel('$L^*$',fontsize = 20)


fig = figure;
ax1 = subplot(2,1,1);
scatter(L,a,20,abs(rgb),'filled')
grid on
ylabel('$a^\star$',Fontsize = 20 )
ax2 = subplot(2,1,2);
scatter(L,b,20,abs(rgb),'filled')
grid on
linkaxes([ax1,ax2],'x')
xlabel('$L^\star$',FontSize=20)
ylabel('$b^\star$',Fontsize = 20)

Freq = 3;

% polarscatter(repmat(deg2rad(white_rel_sensor_ang(:,1)),numel(f_ellipse_angles(1,:)),1),reshape(repmat((f_ellipse_angles),numel(white_rel_sensor_ang(:,1)),1),[numel(rgb(:,1)),1]),50,abs(rgb),'filled')
figure(15)
polarscatter(repmat(deg2rad(white_rel_sensor_ang(1:Freq:end,1)),numel(f_ellipse_angles(1,:)),1),reshape(repmat((f_ellipse_angles),numel(white_rel_sensor_ang(1:Freq:end,1)),1),[numel(rgb(1:Freq:end,1)),1]),50,abs(rgb(1:Freq:end,:)),'filled')
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

Freq = 1;
E_00_M = create_cmatrix(sample_Lab,Freq,5);

figure(16)
polarscatter(repmat(deg2rad(white_rel_sensor_ang(:,1)),numel(f_ellipse_angles(1,:)),1),reshape(E_00_M,[numel(rgb(:,1)),1]),50,abs(rgb),'filled')
ax = gca;
ax.ThetaLim = [-90,90];
thetaTicks = (-90:15:90);
ax.ThetaZeroLocation = 'Top';
ax.ThetaDir = 'clockwise';
%ax.RLim = [0,10]
rticks(0:15:max(reshape(E_00_M,[numel(rgb(:,1)),1])))
%rticklabels({'\Delta E_{00} = '})
ax.ThetaAxis.Label.String = 'Sensor angle $\alpha$';
ax.RAxis.Label.String = 'Color difference $\Delta E_{00}$';
ax.ThetaAxis.Label.Position = [0,max(reshape(E_00_M,[numel(rgb(:,1)),1]))];
ax.FontSize = 16;



%{
figure(14); hold on ; box on 
scatter(a,b,20,L,'filled')
caxis([min(L), max(L)])
colorMap = abs(rgb);
colormap(colorMap)
ylabel('$a^*$','FontSize',20); xlabel('$b^*$','FontSize',20,'FontName','times')
grid()
c = colorbar;
%}
%z = repmat(L,1,numel(a));

%mdl = scatteredInterpolant(a(:,1),b(:,1),L(:,1));
%[ag,bg] = meshgrid(unique(a),unique(b));
%zg = mdl(ag,bg);
%D = diag(L,0);
%[aq,bq] = meshgrid(-10:2:10,-10:.2:10);
%Lq = griddata(a,b,L,aq,bq);
%% with LAB values.

%[T,col] = create_ctable(sample_Lab(:,:,8),'LAB',sample_rgb(:,:,8),10);% LAB/angle data, which kind of plot is wished to get made into a table, frequency of table inputs.
%figure(14)
%D = diag(T.L);
%heatmap(T.a,T.b,D)
%h = heatmap(T,"a","b",'ColorVariable',"L",'Colormap',abs(col),'ColorMethod','min'); % Columns = 'a',Rows = 'b'
%h.Colormap(abs(col))
%xlabel('$a^*$',fontsize = 20)
%ylabel('$b^*$',fontsize = 20)
%zlabel('$L^*$',fontsize = 20)
%colorbar()

%% With angular values
%,'XLabel','Sample angle \theta[\circ]','YLabel','Sensor angle \alpha[\circ]''XLabel','Sample angle \theta[\circ]','YLabel','Sensor angle \alpha[\circ]'

col = rgb(1:Freq:end,:);
figure;
h = heatmap(f_ellipse_angles,white_rel_sensor_ang(1:Freq:end,1),E_00_M,'Colormap',abs(col),'XLabel','Sample angle \theta[\circ]','YLabel','Sensor angle \alpha[\circ]');
h.ColorbarVisible = 'off'
%axs = gca;
%axs.XLabel.FontSize = 20;
%plot(axs,f_ellipse_angles(2:end))
axs = gca;
%axs.XLabel.String = 'Sample angle';
%axs.XLabel.Fontsize = 18;
%axs.YLabel.Fontsize = 18;
%h.Label('Sample angle',fontsize = 20)

h.Title= 'CIE color difference \DeltaE_{00}';

%cb.Title.String = '\DeltaE_{00}';
%get(c,'Title') % c.Title.String = '$\DeltaE_{00}$';
%,'Colormap',abs(rgb)
% Contourplot showing theta, alpha , E_00 distance from theta=0

%% Functions 

function M = create_cmatrix(color_data,freq,PV_cell)

M = zeros(numel(color_data(1:freq:end,1,1)),numel(color_data(1,:,1)));
for i=1:numel(color_data(1,:,1))
    E00 = calc_E00(color_data(:,i,PV_cell),color_data(:,1,PV_cell));
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

function E_00 = calc_E00(LAB_data,LAB_ref,k_L,k_C,k_H) % Inserting cell specifying which PV-cell and and theta angle we want.
    if nargin == 2
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
    for i=1:data
    L_star(i) = LAB_data{i}(1);
    a_star(i) = LAB_data{i}(2);
    b_star(i) = LAB_data{i}(3);
    L_star_ref(i) = LAB_ref{19}(1);
    a_star_ref(i) = LAB_ref{19}(2);
    b_star_ref(i) = LAB_ref{19}(3);
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
                    rbg(c,:) = col{i,n};
                   
                    c = c + 1;
             end
        end
    end
end

function M = Contour_Matrix(BRDF,alpha,theta,type,PV_nr,const)
L = 1043;
if strcmp('theta',type) == 1
    M = zeros(L,numel(theta));
    for i=1:numel(theta)
     M(:,i) = BRDF{const,i,PV_nr}; %[M,BRDF{const,i,PV_nr}];
    end
elseif strcmp('alpha',type) == 1
    M = zeros(L,numel(alpha));
    size(M)
    for i=1:numel(alpha)
     %size(BRDF{const,i,PV_nr})
     M(:,i) = BRDF{i,const,PV_nr}; %[M,BRDF{i,const,PV_nr}];
     
    end
elseif strcmp('wavelength',type) == 1
    M = zeros(numel(alpha),numel(theta));
    for i=1:numel(alpha)
        for n=1:numel(theta)
            
            M(i,n) = BRDF(i,n,PV_nr);
        end
    end
end

end

