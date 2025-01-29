clear all
close all

% File takes brdf data and restructures it to zemax standards. Generating a
% useable .txt file to use as scattering function in zemax. 



file_dir = dir(fullfile('..\','Processed_data/'));
Files = extractfield(file_dir,'name');
Files(1:2) = [];

Sim = load(fullfile('..\','Processed_data/',Files{5}));

avelengths = Sim.wavelengths;
zero_point = round(numel(Sim.white_rel_sensor_ang(:,1))/2);
theta = Sim.f_ellipse_angles; % Incidence angle/sample angle.
alpha = Sim.white_rel_sensor_ang(:,1); % Called emergince angle. Angle between specular reflection and surface normal. 
brdf = Sim.BRDF_brdf;
white_brdf = Sim.white_brdf;

set(0,'defaultTextInterpreter','latex')
set(0,'defaultLegendInterpreter','latex')

brdf_silicon = brdf(:,:,8);


filename = "Silicon_TIS_v3.txt";

% Getting correct formatting of data.
Source = 'Measured';
Symmetry = 'PlaneSymmetrical';
Spectralcontent = 'Monochrome';
ScatterType = 'BRDF';
SampleRotation = 90;
AngleOfIncidence = theta;
ScatterAzimuth = [0,180];
ScatterRadial = alpha(19:end);%linspace(0,150,150/5 +1 );%alpha(19:end);
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
brdf_silicon(isnan(brdf_silicon(:,m)),m) = 0;
TIS =  tis(brdf_silicon(19:end,m),deg2rad(5)) %brdf_silicon(~isnan(brdf_silicon(:,m))
data = zeros(1,numel(ScatterRadial));
ID = find( ScatterRadial >= 0+idx & ScatterRadial <= 90+idx);
writelines("TIS " + num2str(TIS),filename,'WriteMode','append')
data(1,ID) = round(brdf_silicon(19:end,m),3).';
data(1,isnan(data(1,:))) = 0;%data(1,i)
writematrix(data,filename,'Delimiter','tab',WriteMode = 'append')
data(1,ID) = flip(round(brdf_silicon(1:19,m),3).');
data(1,isnan(data(1,:))) = 0;%data(1,i)
writematrix(data,filename,'Delimiter','tab',WriteMode = 'append')
idx = idx + spacing;
%end
end

writelines("DataEnd",filename,'WriteMode','append')

%%
filename = "ABG_Combine.ABGF";

DIR=fopen(filename,'wt+'); % Clear and open file. 
% Making doc header from variables. 
writelines("2  "+ num2str(1),filename,WriteMode='append')
writelines("ABG_COEFFS_BACKWARD  "+ num2str(1),filename,WriteMode='append')
writelines("ABG_COEFFS_FORWARD  "+ num2str(1),filename,WriteMode='append')

%% Functions 
function TIS = tis(B,dth)
th = 0:dth:pi/2;
%phi = 0:dphi:2*pi;
% Calc of eachx and product of integration of TIS.
for i=1:length(th)
    %for n=1:length(phi)
       % x_tis(i,n) = sqrt((sin(th(i))*cos(phi(n))-sin(incidence))^2 + (sin(th(i))*sin(phi(n)))^2 );
        cossin(i) = cos(th(i))*sin(th(i));
       
    %end
end
%sum(A./(B+abs(x_tis).^g))
%B+abs(x_tis).^g
% Integration of the resulting arrays.
 %size(cossin)
 %size(B)
% B.*cossin
 TIS = 2*pi*(sum(B.*cossin'))*dth;%*dphi;
 %TIS = sum(A./(B+x_tis.^g).*cossin)*dth*phi;

end
