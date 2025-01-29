%% BRDF measurements analysis
% for coloured samples
addpath(genpath('utils'))
rmpath(genpath('utils/dtu-fotonik-photometry-toolbox/.git'))

%% Setup and information
set(0,'defaultTextInterpreter','latex')
set(0,'defaultLegendInterpreter','latex')

%custom functions
extractStructfield = @(x,fieldname,ndata,nsample) cell2mat(reshape({x.(fieldname)},[],ndata,nsample));
extractStructfieldCell = @(x,fieldname,ndata,nsample) reshape({x.(fieldname)},[],ndata,nsample);

%% import reference data and reference calculations



% calibration file path
calibration_path = fullfile('utils','reference_data','20190816_calibration.txt');

% spectralon reflectance
spectralon_raw = importdata(fullfile('utils','reference_data','spectralon_reflectance.csv'));
spectralon = spectralon_raw.data;

% sensor divergence
sensor_divergence = 0.72;

% definition integration range
int_range = [303,980];
data_date = '20240909';


% white reference import
% make sure files are converted to using '.' as decimal separator
% run sed_gui.py or sed_comma_period.py on files before importing
white_files =dir(fullfile('..\',append(data_date,'_BRDF_Albert'),'*_SP_COUNTS_Ref*.txt')); %dir(fullfile('*_SP_COUNTS_Ref*.txt')); %
darkfile =fullfile('..\',append(data_date,'_BRDF_Albert'),'*_DRK_*.txt'); %fullfile('*_DRK_*.txt'); %
white_data = importReflectance_separate(fullfile({white_files.folder}',{white_files.name}'),darkfile,calibration_path,'singledark',true);
white_data = white_data([9:-1:6,1,2:5]); %select correct order of measurements
white_measurements = extractStructfield(white_data,'measurements',numel(white_data),1);
white_info = extractStructfield(white_measurements,'info',numel(white_data),1);
white_rel_sensor_ang = extractStructfield(white_info,'relative_Sensor_Angle_deg',numel(white_data),1); %correct for angle mismatch
white_sample_ang = extractStructfield(white_info,'Sample_Angle_deg',numel(white_data),1);

% area correction factor
%R1 = ones(size(white_measurements,1),1).*19.05E-3;
%R2 = ones(size(white_measurements,1),1).*((11E-3/2)+tand(sensor_divergence)*0.28);
f_ellipse = nan(size(white_measurements));
for idx = 1:size(white_measurements,2)
    f_ellipse(:,idx) = calcOverlapFactor_v2(white_sample_ang(:,idx),white_rel_sensor_ang(:,idx),'sensorspot_overlap_small.mat');
end

% apparent sample area and solid angle
solidangle = ones(size(white_rel_sensor_ang)).*tand(sensor_divergence)^2*pi;
app_area = ((11E-3/2)+tand(sensor_divergence)*0.28)^2*pi./cosd(white_rel_sensor_ang);

% white reference calculations
white_ref.spectral_flux  = [white_measurements(white_rel_sensor_ang==5&white_sample_ang==0).spectral_flux(:,1),mean([white_measurements(white_rel_sensor_ang==5&white_sample_ang==0).spectral_flux(:,2),white_measurements(white_rel_sensor_ang==-5&white_sample_ang==0).spectral_flux(:,2)],2)]; % 3.2
white_ref.counts = [white_measurements(white_rel_sensor_ang==5&white_sample_ang==0).counts(:,1),mean([white_measurements(white_rel_sensor_ang==5&white_sample_ang==0).counts(:,2),white_measurements(white_rel_sensor_ang==-5&white_sample_ang==0).counts(:,2)],2)];
white_ref_counts = white_ref.counts;
sample_spectral_flux = [white_ref.spectral_flux(:,1),pi/solidangle(white_rel_sensor_ang==0&white_sample_ang==0)*white_ref.spectral_flux(:,2)./interp1(spectralon(:,1),spectralon(:,2)./100,white_ref.spectral_flux(:,1))]; % Equation 3.2
sample_radiant_flux = trapz(sample_spectral_flux(sample_spectral_flux(:,1)>=int_range(1)&sample_spectral_flux(:,1)<=int_range(2),1),sample_spectral_flux(sample_spectral_flux(:,1)>=int_range(1)&sample_spectral_flux(:,1)<=int_range(2),2)); % Integration of 3.2
white_spectral_flux = [white_ref.spectral_flux(:,1),pi/solidangle(white_rel_sensor_ang==0&white_sample_ang==0)*white_ref.spectral_flux(:,2)];
white_radiant_flux_sensor = trapz(white_ref.spectral_flux(white_ref.spectral_flux(:,1)>=int_range(1)&white_ref.spectral_flux(:,1)<=int_range(2),1),white_ref.spectral_flux((white_ref.spectral_flux(:,1)>=int_range(1)&white_ref.spectral_flux(:,1)<=int_range(2)),2));
white_radiant_flux_sample = pi./solidangle(white_rel_sensor_ang==0&white_sample_ang==0).*white_radiant_flux_sensor; % Equation 3.7

sample_spectral_irradiance = [sample_spectral_flux(:,1),sample_spectral_flux(:,2)./app_area(white_rel_sensor_ang==0&white_sample_ang==0)]; 
sample_irradiance = sample_radiant_flux./app_area(white_rel_sensor_ang==0&white_sample_ang==0); 

wavelengths = white_ref_counts(:,1);
int_range_logical = wavelengths>=int_range(1)&wavelengths<=int_range(2);

%% calculate white reference measurements

white_meas_radiant_flux = zeros(size(white_measurements));
for idx1 = 1:size(white_measurements,1)
    for idx2 = 1:size(white_measurements,2)
        white_meas_radiant_flux(idx1,idx2) = trapz(white_measurements(idx1,idx2).spectral_flux(int_range_logical,1),white_measurements(idx1,idx2).spectral_flux(int_range_logical,2));
    end
end

white_meas_radiance_uncorr = white_meas_radiant_flux./solidangle./(f_ellipse.*app_area)./cosd(white_rel_sensor_ang); %Equation 3.9 in project.

%make sure to select central measurement:
f_sensorang = white_meas_radiant_flux(1:end,ceil(size(white_measurements,2)/2))./mean([white_meas_radiant_flux(1:end,ceil(size(white_measurements,2)/2)),flipud(white_meas_radiant_flux(1:end,ceil(size(white_measurements,2)/2)))],2); % Equation 3.12
%f_sensorang = white_meas_radiant_flux(1:end,6)./mean([white_meas_radiant_flux(1:end,6),flipud(white_meas_radiant_flux(1:end,6))],2);
f_sampleang = white_meas_radiant_flux(ceil(size(white_measurements,1)/2),:)./mean([white_meas_radiant_flux(ceil(size(white_measurements,1)/2),:);fliplr(white_meas_radiant_flux(ceil(size(white_measurements,1)/2),:))],1); % Equation 3.13
%f_sampleang = white_meas_radiant_flux(19,:)./mean([white_meas_radiant_flux(19,:);fliplr(white_meas_radiant_flux(19,:))],1);

white_meas_radiance = white_meas_radiant_flux./solidangle./(f_ellipse.*app_area)./cosd(white_rel_sensor_ang)./f_sensorang./f_sampleang;
white_brdf = white_meas_radiance./(sample_irradiance.*cosd(white_sample_ang));

white_spectral_brdf = cell(size(white_measurements));
for idx1 = 1:size(white_measurements,1)
    for idx2 = 1:size(white_measurements,2)
        white_spectral_brdf{idx1,idx2} = white_measurements(idx1,idx2).spectral_flux(:,2)./solidangle(idx1,idx2)./(f_ellipse(idx1,idx2).*app_area(idx1,idx2))./cosd(white_rel_sensor_ang(idx1,idx2))./f_sensorang(idx1)./f_sampleang(idx2)./(sample_spectral_irradiance(:,2).*cosd(white_sample_ang(idx1,idx2))); % Equation 3.14
    end
end

%% plot correction factors

fig = figure;
hold on
for idx = 5:size(white_measurements,2)
plot(white_rel_sensor_ang(:,idx),f_ellipse(:,idx),'.-','MarkerSize',18,'LineWidth',2)
end
hold off
grid on
grid minor
xlim([-90,90])
xlabel('Relative Sensor Angle $\alpha [^\circ]$','FontSize',18)
ylabel('Area Correction Factor $k_A$','FontSize',18)
lgd = legend(strcat(num2str(white_sample_ang(1,5:end)','%d'),'$^\circ$'),'Location','eastoutside'); % white_sample_ang(1,6:-1:1)'
title(lgd,'Sample Angle')

fig = figure;
plot(white_rel_sensor_ang(2:end-1,6),f_sensorang(2:end-1),'k.--','MarkerSize',14,'LineWidth',1)
grid on
grid minor
xlim([-90,90])
xlabel('Relative Sensor Angle $\alpha [^\circ]$','FontSize',18)
ylabel('Sensor Angle Correction $k_\alpha$','FontSize',18)

fig = figure;
plot(white_sample_ang(19,:),f_sampleang,'k.--','MarkerSize',14,'LineWidth',1)
grid on
grid minor
xlim([-90,90])
xlabel('Sample Angle $\theta_e [^\circ]$','FontSize',18)
xtickformat('%d')
ylabel('Sample Angle Correction $k_{\theta_e}$','FontSize',18)

%% import sample data
% import only data with '.' as decimal separator
% run sed_gui.py or sed_comma_period.py on files before importing
% select files to import using wildcards
sample_files = dir(fullfile('..\*_BRDF_Albert','*_Lenz_*.txt'));
sample_files = sample_files([3:7,2,8:11,1,12:end]);
sample_data = importReflectance_separate(fullfile({sample_files.folder}',{sample_files.name}'),darkfile,calibration_path,'singledark',true);

% set number of samples to reshape data:
nsamples = 12;
BRDF_data = reshape(vertcat(sample_data),1,[],nsamples);

%% BRDF calculations

f_ellipse_angles = [0,15,30,45,60];
nangles = numel(f_ellipse_angles);
f_ellipse = nan(size(white_measurements,1),numel(f_ellipse_angles));
for idx = 1:size(f_ellipse,2)
    f_ellipse(:,idx) = calcOverlapFactor_v2(ones(37,1).*f_ellipse_angles(idx),[-90:5:90]','sensorspot_overlap_small.mat');
end

BRDF_measurements = extractStructfield(BRDF_data,'measurements',nangles,nsamples);
BRDF_info = extractStructfield(BRDF_data,'info',nangles,nsamples);
BRDF_names = repmat(reshape({BRDF_info.SAMPLE_NUMBER},1,nsamples,[]),37,1,1);
BRDF_measinfo = extractStructfield(BRDF_measurements,'info',nangles,nsamples);
BRDF_saturated = reshape([BRDF_measinfo.Spectrum_Saturated],37,nsamples,[])==1;
BRDF_abs_sensor_ang = extractStructfield(BRDF_measinfo,'Sensor_Angle_deg',nangles,nsamples);
BRDF_rel_sensor_ang = extractStructfield(BRDF_measinfo,'relative_Sensor_Angle_deg',nangles,nsamples);
BRDF_sample_ang = extractStructfield(BRDF_measinfo,'Sample_Angle_deg',nangles,nsamples);

% brdf calculations
BRDF_spectral_radiance = cell(size(BRDF_measurements));
BRDF_spectral_brdf = cell(size(BRDF_measurements));
BRDF_meas_radiant_flux = nan(size(BRDF_measurements));
BRDF_radiance = nan(size(BRDF_measurements));
BRDF_brdf = nan(size(BRDF_measurements));
BRDF_spectral_exitance_0ang = nan(numel(wavelengths),size(BRDF_measurements,3));
for idx3 = 1:size(BRDF_measurements,3)
    for idx2 = 1:size(BRDF_measurements,2)
        for idx1 = 1:size(BRDF_measurements,1)
            BRDF_meas_radiant_flux(idx1,idx2,idx3) = trapz(BRDF_measurements(idx1,idx2,idx3).spectral_flux(int_range_logical,1),BRDF_measurements(idx1,idx2,idx3).spectral_flux(int_range_logical,2));
            BRDF_radiance(idx1,idx2,idx3) = BRDF_meas_radiant_flux(idx1,idx2,idx3)./solidangle(idx1,idx2)./cosd(BRDF_rel_sensor_ang(idx1,idx2,idx3))./(f_ellipse(idx1,idx2).*app_area(idx1,idx2));
            BRDF_brdf(idx1,idx2,idx3) = BRDF_radiance(idx1,idx2,idx3)./f_sensorang(idx1)./interp1(white_sample_ang(19,:),f_sampleang,BRDF_sample_ang(idx1,idx2,idx3))./(sample_irradiance.*cosd(BRDF_sample_ang(idx1,idx2,idx3)));
            if BRDF_abs_sensor_ang(idx1,idx2,idx3) == 0
                BRDF_brdf(idx1,idx2,idx3) = nan;
            end
            BRDF_spectral_radiance{idx1,idx2,idx3} = BRDF_measurements(idx1,idx2,idx3).spectral_flux(:,2)./solidangle(idx1,idx2)./cosd(BRDF_rel_sensor_ang(idx1,idx2,idx3))./(f_ellipse(idx1,idx2).*app_area(idx1,idx2));
            BRDF_spectral_radiance{idx1,idx2,idx3}(isnan(BRDF_spectral_radiance{idx1,idx2,idx3})) = 0;
            BRDF_spectral_brdf{idx1,idx2,idx3} = BRDF_spectral_radiance{idx1,idx2,idx3}./f_sensorang(idx1)./interp1(white_sample_ang(19,:),f_sampleang,BRDF_sample_ang(idx1,idx2,idx3))./(sample_spectral_irradiance(:,2).*cosd(BRDF_sample_ang(idx1,idx2,idx3)));
        end
    end
    BRDF_spectral_exitance_0ang(:,idx3) = 2*pi*trapz(deg2rad(BRDF_rel_sensor_ang(1:ceil(size(BRDF_rel_sensor_ang,1)/2),1,idx3)),[BRDF_spectral_radiance{1:ceil(size(BRDF_rel_sensor_ang,1)/2),1,idx3}].*abs(sind(2.*BRDF_rel_sensor_ang(1:ceil(size(BRDF_rel_sensor_ang,1)/2),1))/2)',2);
end
BRDF_spectral_reflectance = BRDF_spectral_exitance_0ang./sample_spectral_irradiance(:,2);


%% colorimetry calculations
sample_colorimetry = cellfun(@(x) calcReflectanceColorimetry([wavelengths,x]),BRDF_spectral_brdf);
sample_Lab = extractStructfieldCell(sample_colorimetry,'Lab',size(sample_colorimetry,2),size(sample_colorimetry,3));
sample_RAL_colors = extractStructfieldCell(sample_colorimetry,'RAL_colors',size(sample_colorimetry,2),size(sample_colorimetry,3));
sample_rgb = cellfun(@(x) lab2rgb(x),sample_Lab,'UniformOutput',false);

Luv_params_full = extractStructfieldCell(sample_colorimetry,'LCh_uv',size(sample_colorimetry,2),size(sample_colorimetry,3));
s_uv = extractStructfield(sample_colorimetry,'s_uv',size(sample_colorimetry,2),size(sample_colorimetry,3));
Luv_params_av = zeros(size(Luv_params_full,3),3);
Luv_params_std = zeros(size(Luv_params_full,3),3);
s_uv_av = zeros(size(s_uv,3),1);
s_uv_std = zeros(size(s_uv,3),1);
for idx = 1:size(Luv_params_full,3)
    Luv_params_av(idx,:) = mean(vertcat(Luv_params_full{:,:,idx}),1,'omitnan');
    Luv_params_std(idx,:) = std(vertcat(Luv_params_full{:,:,idx}),1,'omitnan');
    hue_weight = reshape(cellfun(@(x) x(2),Luv_params_full(:,:,idx)),[],1,1);
    hue_weight(hue_weight<10) = 0;
    Luv_params_av(idx,3) = sum(reshape(cellfun(@(x) x(3),Luv_params_full(:,:,idx)),[],1,1).*hue_weight,'omitnan')./sum(hue_weight,'omitnan');
    Luv_params_std(idx,3) = std(reshape(cellfun(@(x) mod(x(3)+180,180),Luv_params_full(:,:,idx)),[],1,1),hue_weight,'omitnan');
    s_uv_av(idx) = mean(s_uv(:,:,idx),'all','omitnan');
    s_uv_std(idx) = std(s_uv(:,:,idx),1,'all','omitnan');
end
%Match 