%% non-sequential raytracing using ZOS-API connection to OpticStudio
%ZOS_system.Tools.CurrentTool.Close()
% figure formatting
clear c 
close all
set(0,'defaultTextInterpreter','latex');
set(0,'defaultLegendInterpreter','latex');

%% initialize connection
instance_num = 1; %instance number as shown in OpticStudio when starting "Interactive Extension" mode

ZOS_instance = MATLABZOSConnection(instance_num); % Application imported. 
%ZOS_instance.CreateNewSystem('C:\Users\s194086\Documents\Zemax\Samples\PV_BSDF_analysis.ZMX',false);

%% System and analysis initialzing 

%ZOS_system = ZOS_instance.CreateNewSystem(ZOSAPI.SystemType.NonSequential);
%ZOS_system.LoadFile('C:\Users\s194086\Documents\Zemax\Samples\PV_BSDF_analysis.ZMX',false)


import ZOSAPI.Tools.RayTrace.*;
import BatchRayTrace.*;
ZOS_system = ZOS_instance.PrimarySystem;

SampleDir = ZOS_instance.SamplesDir;
testFile = SampleDir.Concat(SampleDir,'PV_BSDF_analysis_ver2.zmx'); % Lens file
%ZOS_system.New(testFile,false)
ZOS_system.LoadFile(testFile,false);
ZOS_instance.LoadNewSystem('PV_BSDF_analysis_ver2.zmx')
LDE = ZOS_system.LDE;
%LDE.GetSurfaceAt(3)
Nce = ZOS_system.NCE;

%source = Nce.GetObjectAt(2);
%source.GetObjectCell(12).IntegerValue = 1000;
ZOS_system.SystemFile
%ZOS_system.SystemData.Wavelengths.NumberOfWavelengths;
%ZOS_system.SystemFile.TrimEnd('.zos')

%ZOS_system.Tools.CurrentTool.Close()
%% Wavelength processing
samples = {'reference'};%,'white','blue','green','red','yellow','terracotta'};
transm = cell(numel(samples),1);
refl = cell(numel(samples,1),1);
Radiance = cell(numel(samples,1),1);
Logic = {'False', 'True'};
keys = ["Ray_nr","Seg_nr","Wavelength","x","y","z","AOI","ObjNr"]; % 'L','M','N','Nx','Ny','Nz'

Max_rays = 5000;
ZOS_system.NCE.GetObjectAt(2).GetObjectCell(ZOSAPI.Editors.NCE.ObjectColumn.Par2).IntegerValue = Max_rays;
RayTrace_dir = dictionary(keys,[1,2,3,4,5,6,7,8]);
nsur = ZOS_system.LDE.NumberOfSurfaces;


%%


if ~isempty(ZOS_system.Tools.CurrentTool)
    ZOS_system.Tools.CurrentTool.Close();
end

%for s=1:numel(samples)
 % initialization
    wavelengths = 0.55;%linspace(0.30,0.9,24); %length must be divisible by 24
  %  ZOS_system.Wavelengths.GetWavelength(1).Wavelength = 0.55
    tic
    
    for batch = 1:(numel(wavelengths))
        ZOS_system.SystemData.Wavelengths.GetWavelength(1).Wavelength = wavelengths(batch);
%{
           % insert wavelengths
            for idx = 1:ZOS_system.SystemData.Wavelengths.NumberOfWavelengths
                ZOS_system.SystemData.Wavelengths.RemoveWavelength(1);
            end
            for idx = [1:24] +(batch-1)*24
                ZOS_system.SystemData.Wavelengths.AddWavelength(wavelengths(idx),1);
            end
%}
     incidence_array = [0]; %linspace(0,60,60/3 +1);
     N = cell(numel(incidence_array),3);
     div = [];
     P_Reflective = [];
     source_size = [0.75];%linspace(0.01,5,30);
     coordinates = cell(3,numel(source_size));
     col = cell(1,numel(source_size));
     for F= 1:numel(source_size)
      theta_i = incidence_array(1,1);
     % Change detector angle and define incident theta angle. Object nr3
     ZOS_system.NCE.GetObjectAt(3).GetObjectCell(ZOSAPI.Editors.NCE.ObjectColumn.TiltY).DoubleValue = -theta_i;%-theta_i;
     ZOS_system.NCE.GetObjectAt(3).GetObjectCell(ZOSAPI.Editors.NCE.ObjectColumn.TiltX).DoubleValue = 0;
     ZOS_system.NCE.GetObjectAt(1).GetObjectCell(ZOSAPI.Editors.NCE.ObjectColumn.TiltY).DoubleValue = theta_i;
    % ZOS_system.NCE.GetObjectAt(2).GetObjectCell(ZOSAPI.Editors.NCE.ObjectColumn.TiltY).DoubleValue = 0;%theta_i;
     ZOS_system.NCE.GetObjectAt(3).GetObjectCell(ZOSAPI.Editors.NCE.ObjectColumn.TiltZ).DoubleValue = 0;
     ZOS_system.NCE.GetObjectAt(1).GetObjectCell(ZOSAPI.Editors.NCE.ObjectColumn.TiltZ).DoubleValue = 0;
     ZOS_system.NCE.GetObjectAt(1).GetObjectCell(ZOSAPI.Editors.NCE.ObjectColumn.TiltX).DoubleValue = 0;
     ZOS_system.NCE.GetObjectAt(2).GetObjectCell(ZOSAPI.Editors.NCE.ObjectColumn.Par6).DoubleValue = source_size(F);
     ZOS_system.NCE.GetObjectAt(2).GetObjectCell(ZOSAPI.Editors.NCE.ObjectColumn.Par7).DoubleValue = source_size(F);
    
     

     d = ZOS_system.NCE.GetObjectAt(3).GetObjectCell(ZOSAPI.Editors.NCE.ObjectColumn.Par2).DoubleValue;
     r = ZOS_system.NCE.GetObjectAt(6).GetObjectCell(ZOSAPI.Editors.NCE.ObjectColumn.Par2).DoubleValue;
     max_ang = ZOS_system.NCE.GetObjectAt(6).GetObjectCell(ZOSAPI.Editors.NCE.ObjectColumn.Par1).DoubleValue;
     px_radial = ZOS_system.NCE.GetObjectAt(6).GetObjectCell(ZOSAPI.Editors.NCE.ObjectColumn.Par3).IntegerValue;
     px_azimuth = ZOS_system.NCE.GetObjectAt(6).GetObjectCell(ZOSAPI.Editors.NCE.ObjectColumn.Par4).IntegerValue;
     P = ZOS_system.NCE.GetObjectAt(2).GetObjectCell(ZOSAPI.Editors.NCE.ObjectColumn.Par3).DoubleValue;
     
     
     %theta_i = ZOS_system.NCE.GetObjectAt(3).GetObjectCell(ZOSAPI.Editors.NCE.ObjectColumn. TiltY).DoubleValue; 
     a3 = [];

    %---------------------------
    % Create ray trace
    NSCRayTrace = ZOS_system.Tools.OpenNSCRayTrace();
    NSCRayTrace.ClearDetectors(0);
    NSCRayTrace.SplitNSCRays = true;
    NSCRayTrace.ScatterNSCRays = true;
    NSCRayTrace.UsePolarization = true;
    NSCRayTrace.IgnoreErrors = true;
    NSCRayTrace.SaveRays = true;
    NSCRayTrace.SaveRaysFile = 'Lambertian_glass_5000.ZRD';%'One_Ray_No_scat.ZRD';
   

    % Run ray trace
    NSCRayTrace.RunAndWaitForCompletion();
    NSCRayTrace.Close();
%}
    %---------------------------------------
    % read results file
        ZRD_reader = ZOS_system.Tools.OpenRayDatabaseReader();
        ZRD_reader.ZRDFile = 'C:\Users\s194086\Documents\Zemax\Samples\Lambertian_glass_5000.ZRD'; % Ray_1000_const.ZRD % Lambertian_glass_1000.ZRD
        ZRD_reader.Filter = ''; %(H6 & R3 & !H4) H6 & R3 & R4
        ZRD_reader.RunAndWaitForCompletion();
        % Check if reader is sucessful. 
        if ZRD_reader.Succeeded == 0
        disp('Raytracing failed!')
        disp(ZRD_reader.ErrorMessage)
        else
        disp('Raytracing completed!')
        end

        ZRDResult = ZRD_reader.GetResults();
       % ZRDResult.ReadNextResult
        % Get Incident angles from ZOS_reader
        
       
        Data_d8 = {};
        Data_relf = {};
        Hit_count = 0;
        Hit_count_relf = 0;
        
        Azimuth_filter = linspace(0,360,px_azimuth+1);
        Azimuth_filter = Azimuth_filter(1:end-1);
        Polar_filter = linspace(0,90,px_radial);
        c = cell([numel(Polar_filter),numel(Azimuth_filter)]);
        [circ,A_arr,Solid_ang,az_arc] = Solid_ang_calc(r,Azimuth_filter,Polar_filter);
        cc = cell([numel(Polar_filter),numel(Azimuth_filter)]); 
        bin_rad = abs(Polar_filter(1)-Polar_filter(2));
        bin_azi = abs(Azimuth_filter(1)-Azimuth_filter(2));
        [success_NextResult, rayNumber, waveIndex, wlUM, numSegments] = ZRDResult.ReadNextResult();
        while success_NextResult == 1
         %  Segidx = -1;
           preseg_store = [];
           preseg_data = zeros(numSegments,5); % preseg,hitobj,seglvl
           
           % [success_NextSegmentFull,segidx,seglvl,hitObj,~,~,~,x,y,z,l,m,n,~,~,~,~,~,~,intensity,~,~,~,Nx,Ny,Nz,~,~,~,~] = ZRDResult.ReadNextSegmentFull();
            [success_NextSegmentFull, seglvl, preseg, hitobj, hitFace, insideOf, status,...
            x, y, z, l, m, n, exr, exi, eyr, eyi, ezr, ezi, segint, pathLength,...
            xybin, lmbin, Nx, Ny, Nz, index, startingPhase, phaseOf, phaseAt] = ZRDResult.ReadNextSegmentFull();
            j = 1;
            while success_NextSegmentFull == 1
               %[r11,r12,r13,r21,r22,r23,r31,r32,r33,x0,y0,zo] = ZOS_system.LDE.GetGlobalMatrix(1);
              %  Ang = acosd( (l*Nx+m*Ny+n*Nz) / (sqrt(l^2+m^2+n^2)*sqrt(Nx^2+Ny^2+Nz^2)));
               % If n is negative and n_pre is positive = reflection.
               % if hitobj == 6
               % if n_pre > 0 && n  < 0 % 
               % Segidx
               % Ang = acosd( (l*Nx_pre+m*Ny_pre+n*Nz_pre) / (sqrt(l^2+m^2+n^2)*sqrt(Nx^2+Ny^2+Nz^2)));
                % Set the normal plane as glass surface.
               % Nx = 0.5;
               % Ny = 0.0;
               % Nz = 0.8660;
               % elseif n_pre < 0 && n < 0 % Transmitting through surface.
               % Ang = acosd( (l*Nx_pre+m*Ny_pre+n*Nz_pre) / (sqrt(l^2+m^2+n^2)*sqrt(Nx_pre^2+Ny_pre^2+Nz_pre^2)));
               % end
                %end
             %   Hit_count = Hit_count + 1;
             %   Data_d8(Hit_count,:) = {[rayNumber,hitobj,numSegments,seglvl,preseg],Ang,segint,[x,y,z],[l,m,n,Nx,Ny,Nz],Segidx};
                
               % sprintf(' Segment %d',(Segidx))
               % t_ang = acosd(Nx*l+Ny*m+Nz*n)
                P_Reflective = [P_Reflective,segint];
                
                preseg_data(j,:) = [preseg,hitobj,seglvl,j-1,sprintf("%f",segint)];
                j = j+1;
                if hitobj == 6 %&& segint >= (P/(1000*Max_rays)) %0.1% Energy to avoid getting stray rays.
                %preseg_store = [preseg_store,preseg];
                rp = cross([r,r,r],[l,m,n])/(sqrt(l^2+m^2+n^2));
                r_prime = [x,y,z];
                r_inc = r_prime + pathLength*([l,m,n]*(-1));
               % Ang_ = acosd(n/(sqrt(l^2+m^2+n^2)));
                Ang_pol = acosd(z/(sqrt(x^2+y^2+z^2)));
              %  Ang_pol_inc =  acosd(r_inc(3)/(sqrt(r_inc(1)^2+r_inc(2)^2+r_inc(3)^2)));
               % Ang_pol = acosd((z-r_inc(3))/(sqrt((x-r_inc(1))^2+(y-r_inc(2))^2+(z-r_inc(3))^2)))
             %   Ang_pol = Ang_pol - Ang_pol_inc;
                phi = atan2d(y,x);
              %  phi = sign(y)*acosd(x/(sqrt(x^2+y^2)));
                Ang = acosd( (l*Nx+m*Ny+n*Nz) / (sqrt(l^2+m^2+n^2)*sqrt(Nx^2+Ny^2+Nz^2)));
                %Ang =abs(180- Ang_pol-theta_i) ;
               % r*sin(Ang)*cos(phi)
               % q = cross([x,y,z],[0,0,-1]);
               % acosd( (l*q(1)+m*q(2)+n*q(3)) / (sqrt(l^2+m^2+n^2)*sqrt(q(1)^2+q(2)^2+q(3)^2)));
               if Ang > 90
                   Ang = 180-Ang;
               end
               if phi < 0 
                   phi = 180-abs(phi)+180;
               elseif isnan(phi)
                   phi = 0;     
               end
              if Ang_pol > 90
                   Ang_pol = 180- Ang_pol;
              %    Ang_pol
                % Ang_pol = (Ang_pol-90);
                % Ang_pol = Ang_pol  +(theta_i/2)
              end
                rs = cosd(Ang_pol)*r;
                curv = 1/rs;
               % Ang_pol = Ang_pol - curv*100;
               %{
               nsc = acosd( (x*Nx+y*Ny+z*Nz) / (sqrt(x^2+y^2+z^2)*sqrt(Nx^2+Ny^2+Nz^2)));
               if nsc > 90
                   nsc = 180-nsc;
               end
               %}
               % Ang_pol = Ang_pol-zug;
                %Ang_pol = Ang_pol - (theta_i/2);
                      
              % Spatial filter
              % Ang_pol
              % idx_polar= floor((0.5+round(Ang_pol)) / bin_rad)+1; % Rad_col
              % idx_azi = floor((0.5+round(phi)) / bin_azi);
              % if idx_azi ==0 
              %  idx_azi = 1;
              % elseif  idx_polar == 0
               %    idx_polar = 1;
               %end
               %Azi_col
               idx_azi = filter(phi,Azimuth_filter);
               idx_polar = filter(Ang_pol,Polar_filter);
               
               I = find(preseg == preseg_data(:,4));
               preseg_cur = preseg_data(I,:);
               hit_array = zeros(3,1); % zero if obj not hit, 1 if object is hit for the ray path.
               if preseg_cur(1,1) == 0
                   hit_array = [1,0,0].'; 
               end
               while preseg_cur(1,1) ~= 0
               if preseg_cur(1,2) == 3
                    hit_array(1) = 1;
               elseif preseg_cur(1,2) == 4 % Referring to hitobj 4/5.
                    hit_array(2) = 1;
               elseif preseg_cur(1,2) == 5
                    hit_array(3) = 1;
                end
               I = find(preseg_cur(1,1) == preseg_data(:,4));
               preseg_cur = preseg_data(I,:);
               end
               
               if  Ang <= 85 % Ang >= 0 &&

               if isequal([1,0,0].',hit_array)
                   Ray_hit = 3;
                   dist = 0;
                   pred = 0;
               elseif isequal([1,1,0].',hit_array)
                   Ray_hit = 4;
                   dist = 2*d*tand(theta_i);
                   %dist_2 = sqrt((r + dist)^2 + dist^2 - 2*(r + dist)*dist*cosd(2*theta_i)); %sqrt(d^2+(dist/2)^2);
                   dist_2 = sqrt(r^2+dist^2-2*dist*r*cosd(90-Ang)); %sqrt(r^2+dist^2);
                   pred = (180 - acosd((dist^2-dist_2^2-r^2)/(2*r*dist_2)))/2;
                  % sqrt((r + dist)^2 + dist^2 - 2*(r + dist)*dist*cosd(2*theta_i))
                  % (2*theta_i + acosd((dist^2-(r+dist)^2-dist_2^2) /(2*dist_2*(r+dist))))
                   % a/sin(A) = b/sin(B)
                   %(90-atand(r/dist))/2
               elseif isequal([1,1,1].',hit_array)
                   dist = 2*2*d*tand(theta_i);
                   dist_2 = sqrt(r^2+dist^2-2*dist*r*cosd(90-Ang));
                   %dist_2 = sqrt(r^2+dist^2-2*dist*r*cosd(90-Ang)); %sqrt(r^2+dist^2);
                   pred = (180 - acosd((dist^2-dist_2^2-r^2)/(2*r*dist_2)))/2;
                   %dist_2 = sqrt((r + dist)^2 + dist^2 - 2*(r + dist)*dist*cosd(2*theta_i));
                   %180 - (2*theta_i + acosd(sqrt(dist^2-(r+dist)^2-dist_2^2+2*dist_2*(r+dist))))
                   Ray_hit = 5;
               end
               
               c{idx_polar,idx_azi} = [c{idx_polar,idx_azi},[Ang,segint,Ang_pol,phi,Ray_hit,dist,pred].'];
               cc{idx_polar,idx_azi} = [cc{idx_polar,idx_azi},[x,y,z].'];
               Hit_count_relf = Hit_count_relf + 1;
                
                %Ang = acosd( (l*Nx+m*Ny+n*Nz) / (sqrt(l^2+m^2+n^2)*sqrt(Nx^2+Ny^2+Nz^2)));
                Data_relf(1,Hit_count_relf) = {[x,y,z,r_inc]}; %{[rayNumber,wlUM,hitobj,numSegments,seglvl,preseg,hitFace,Ang,segint]};
                Data_relf(2,Hit_count_relf) = {hit_array};
                Hit_count = Hit_count + 1;
                Data_d8(Hit_count,:) = {[rayNumber,hitobj,numSegments,seglvl,preseg],[Ang,Ang_pol,phi],segint,[l,m,n,Nx,Ny,Nz]};
                
               end
                
              %  if Nz < 0 
              %  Ang = Ang_pol - Ang_in;
              %  elseif Nz > 0
              %  Ang = Ang_pol + (180 - Ang_in);
              %  end
                
                %{                
                polo = zeros(1,numel(Data_d8(:,2)));
                for i=1:numel(Data_d8(:,2))
                
                M =  Data_d8{i,4};
                MM = Data_d8{i,2};
                polo(i) = MM(2);
                l = M(1,1);
                m = M(1,2);
                n = M(1,3);
                Nx = M(1,4);
                Ny = M(1,5);
                Nz = M(1,6);
                end
                 %}
           %     N(F,1) = {[Nx,Ny,Nz,Ang_pol,theta_i,nsc,Ang_]};
           %     N(F,2) = {[x,y,z]};
           %     N(F,3) = {r_inc};
               % elseif hitobj == 9 
               % hitFace;
                %{
               % Angie_4  =  acosd( (l*Nx+m*Ny+n*Nz) / (sqrt(l^2+m^2+n^2)*sqrt(Nx^2+Ny^2+Nz^2)));    
                v_1 = r_inc + pathLength*([l,m,n]);
                
                % Alternative calculation of theta and azimuth.
                th = rad2deg(v_1(3)*(-1))/(sqrt(0^2+0^2+(-1)^2)*sqrt(v_1(1)^2+v_1(2)^2+v_1(3)^2));
                if l < 0
                ph_x = 180-rad2deg(v_1(1)*(1))/(sqrt(0^2+0^2+(-1)^2)*sqrt(v_1(1)^2+v_1(2)^2+v_1(3)^2));
                elseif l > 0
                ph_x = rad2deg(v_1(1)*(1))/(sqrt(0^2+0^2+(-1)^2)*sqrt(v_1(1)^2+v_1(2)^2+v_1(3)^2));
                
                else
                    ph_x = 0;
                end

                if m < 0
                offset = 180-ph_x;
                ph_x = ph_x + offset; 
               
            

               % ph_y = rad2deg(v_1(2)*(1))/(sqrt(0^2+0^2+(-1)^2)*sqrt(v_1(1)^2+v_1(2)^2+v_1(3)^2));
                end

                elseif hitobj == 3 
                    t_ang = acosd(Nx*l+Ny*m+Nz*n);
                   % if t_ang > 90
                   % t_ang = 180-t_ang;
                   % end
                    
                    a3 = [a3,t_ang]; %[a3,acosd((r_inc(1)*l+r_inc(2)*m+r_inc(3)*n)/(sqrt(r_inc(1)^2+r_inc(2)^2+r_inc(3)^2)))];
                
             
                %}
               end
                
                 [success_NextSegmentFull, seglvl, preseg, hitobj, hitFace, insideOf, status,...
            x, y, z, l, m, n, exr, exi, eyr, eyi, ezr, ezi, segint, pathLength,...
            xybin, lmbin, Nx, Ny, Nz, index, startingPhase, phaseOf, phaseAt] = ZRDResult.ReadNextSegmentFull();
              %  fprintf(['Ray number ',num2str(rayNumber),' Segment ',num2str(numSegments),' at level ',num2str(seglvl),' coming from segment ',num2str(preseg+1),' hits object ',num2str(hitobj),' with intensity ',num2str(segint),' after travelling ',num2str(pl),' units; Electric field: ',num2str([exr,eyr,ezr],'%f, %f, %f')]);
               % [success_NextSegmentFull,seglvl,preseg,hitobj,hitFace,in,X,Y,Z,l,m,n,Nx,Ny,Nz,exr,exi,eyr,eyi,ezr,ezi,segint,pl] = ZRDResult.ReadNextSegmentFull();
                %[success_NextSegmentFull,Seg_Nr,~,hitObj,~,~,~,x,y,z,l,m,n,~,~,~,~,~,~,intensity,~,~,~,Nx,Ny,Nz,~,~,~,~] = ZRDResult.ReadNextSegmentFull();
            end
        [success_NextResult, rayNumber, waveIndex, wlUM, numSegments] = ZRDResult.ReadNextResult();
        end
        
        ZRD_reader.Close();

        DataFlag_Power = ZOSAPI.Editors.NCE.PolarDetectorDataType.Power;
        Power = ZOS_system.NCE.GetAllPolarDetectorDataSafe(6,DataFlag_Power).double;

        MM = [];
        for i=1:numel(Data_d8(:,2))
                M = Data_d8{i,2};
                MM = [MM,M(1,2)];
         end
        div = [div,abs(max(MM)-min(MM))];
        col_v = zeros(numel(Data_relf(1,:)),3);
        x_v = zeros(1,numel(Data_relf(1,:)));
        y_v = zeros(1,numel(Data_relf(1,:)));
        z_v = zeros(1,numel(Data_relf(1,:)));
        for i=1:numel(Data_relf(1,:))
       
        k = Data_relf{1,i};
        kk = Data_relf{2,i};
        x_v(1,i) = k(1);
        y_v(1,i) = k(2);
        z_v(1,i) = k(3); 
 
        col_v(i,:) = kk;
        end
        
        

        %Re_Power = zeros(size(Power));
        
     
       
        % Ingoing irradiance on surface object. Using detector incident
        % aligned with surface.
        
        %ZOS_system.NCE.GetAllPolarDetectorData

       % d6 = ZOS_system.Analyses.New_Analysis(ZOSAPI.Analysis.AnalysisIDM.DetectorViewer);
       % d6_settings = d6.GetSettings();
       % d6.Detector.SetDetectorNumber(6);
       % d8_settings.ShowAs = ZOSAPI.Analysis.DetectorViewerShowAsTypes.FalseColor;
       % d8_settings.DataType =  ZOSAPI.Analysis.DetectorViewerShowDataTypes.;
       
       % d8_settings.Scale = ZOSAPI.Analysis.Settings.DetectorViewerScaleTypes.Linear;
       % d8.ApplyAndWaitForCompletion();
       % d8_Results = d8.GetResults();
       % matrixData = d8_Results.GetDataGrid;
       % d8_Irradiance = flipud(matrixData);
       % matrix_data

       
        % analyze results
        %dataReader = ReadZRDData(ZRDResult);
        %data = dataReader.InitializeOutput(10000);
        %dataReader.ReadNextBlock(data)

%mean(polo)
    
    coordinates(1,F) = {x_v};
    coordinates(2,F) = {y_v};
    coordinates(3,F) = {z_v};
    dist_ref = [x_v(end),y_v(end),z_v(end)];
    end
    toc




end



%% Calculating the BRDF

% Detector angle data
np = numel(Power(1,:));

azimuth = linspace(0,358,180);
polar = linspace(0,90,np).'; %reshape(repmat(linspace(0,90,np),numel(azimuth(1,:),1)),numel(azimuth),1);

pA = reshape(repmat(azimuth,[np,1]),[],1);
pP = repmat(polar,numel(azimuth),1);
val_c = reshape(Power.',[],1);


figure(1)
polarscatter(deg2rad(pA), pP,20,val_c*1000,'filled')
set(gca,"CLim",[0,0.0005])  
%colormap(jet(max(val_c)))
cb = colorbar();
title(cb,'Power [mW]','Fontsize',16)
%% Testing
%x_v =zeros(1,numel(Data_relf));
%y_v = zeros(1,numel(Data_relf));
%z_v = zeros(1,numel(Data_relf));

%for i=1:numel(Data_relf(1,:))
%k = Data_relf{1,i};
%x_v(1,i) = k(1);
%y_v(1,i) = k(2);
%z_v(1,i) = k(3); 
%end

%[au,ia,ic] = unique(y_v,'stable');

figure(10)
scatter3(x_v(1,:),y_v(1,:),z_v(1,:),20,'filled')
xlabel('x [mm]',fontsize = 20)
ylabel('y [mm]',fontsize = 20)
zlabel('z [mm]',fontsize = 20)

%Z = diag(z_v);
t1 = atan2d(y_v,x_v);
t2 = acosd(z_v./sqrt(x_v.^2+y_v.^2+z_v.^2));
t3 = sqrt(x_v.^2+y_v.^2+z_v.^2);
figure(11)
scatter3(t3,t2,t1)
ylabel('$\theta$',fontsize = 20)
zlabel('$\varphi$',fontsize = 20)
xlabel('r [mm]',fontsize = 20)


%% Create detector sphere

[X,Y,Z] = sphere(double(px_azimuth));

X = X(double(px_azimuth)/2:end,:)*r;
Y = Y(double(px_azimuth)/2:end,:)*r;
Z = Z(double(px_azimuth)/2:end,:)*r;

z_offset = 5;
str_legend = 'Spotarea = %.2f [%s]';
%{
figure(12)
s = surf(X,Y,-Z);
colormap gray
axis equal
hold on
scatter3(coordinates{1,10},coordinates{2,10},coordinates{3,10},20,'filled',MarkerEdgeColor='blue',MarkerFaceColor='blue')
scatter3(coordinates{1,round(20)},coordinates{2,round(20)},coordinates{3,round(20)},20,'filled',MarkerEdgeColor='green',MarkerFaceColor='green')
scatter3(x_v,y_v,z_v,20,'filled',MarkerEdgeColor='red',MarkerFaceColor='red')
%scatter3(x_v,y_v,z_v,20,'filled','DisplayName',sprintf('Spotradius %.2f [mm]',source_size(end)),MarkerEdgeColor='red',MarkerFaceColor='red')
%scatter3(coordinates{1,10},coordinates{2,10},coordinates{3,10},20,'filled','DisplayName',sprintf('Spotradius %.2f [mm]',source_size(10)),MarkerEdgeColor='blue',MarkerFaceColor='blue')
%scatter3(coordinates{1,round(20)},coordinates{2,round(20)},coordinates{3,round(20)},20,'filled','DisplayName',sprintf('Spotradius %.2f [mm]',source_size(20)),MarkerEdgeColor='green',MarkerFaceColor='green')
xlabel('X [mm]',fontsize = 20)
ylabel('Y [mm]',fontsize = 20)
zlabel('Z [mm]',fontsize = 20)
hold off
legend({'Detector',sprintf(str_legend,source_size(10)^2*pi,'$\mathrm{mm^2}$'),sprintf(str_legend,source_size(11)^2*pi,'$\mathrm{mm^2}$'),sprintf(str_legend,source_size(end)^2*pi,'$\mathrm{mm^2}$')},'interpreter','latex',Fontsize = 22)

%}
%rotate(s,[0,1,0],-90,[0,0,0])

figure(12)
s = surf(X,Y,-Z);
colormap gray
axis equal
hold on
scatter3(coordinates{1,1},coordinates{2,1},coordinates{3,1},20,'filled',MarkerEdgeColor='blue',MarkerFaceColor='blue')
%scatter3(coordinates{1,round(20)},coordinates{2,round(20)},coordinates{3,round(20)},20,'filled',MarkerEdgeColor='green',MarkerFaceColor='green')
%scatter3(x_v,y_v,z_v,20,'filled',MarkerEdgeColor='red',MarkerFaceColor='red')
%scatter3(x_v,y_v,z_v,20,'filled','DisplayName',sprintf('Spotradius %.2f [mm]',source_size(end)),MarkerEdgeColor='red',MarkerFaceColor='red')
%scatter3(coordinates{1,10},coordinates{2,10},coordinates{3,10},20,'filled','DisplayName',sprintf('Spotradius %.2f [mm]',source_size(10)),MarkerEdgeColor='blue',MarkerFaceColor='blue')
%scatter3(coordinates{1,round(20)},coordinates{2,round(20)},coordinates{3,round(20)},20,'filled','DisplayName',sprintf('Spotradius %.2f [mm]',source_size(20)),MarkerEdgeColor='green',MarkerFaceColor='green')
xlabel('X [mm]',fontsize = 20)
ylabel('Y [mm]',fontsize = 20)
zlabel('Z [mm]',fontsize = 20)
hold off

%{
str_legend = '%s = %.2f [%s]';
figure(12)
s = surf(X,Y,-Z,'DisplayName','Detector');
colormap gray
axis equal
hold on
scatter3(coordinates{1,1},coordinates{2,1},coordinates{3,1},20,'filled',MarkerEdgeColor='blue',MarkerFaceColor='blue')
scatter3(coordinates{1,11},coordinates{2,11},coordinates{3,11},20,'filled',MarkerEdgeColor='green',MarkerFaceColor='green')
scatter3(x_v,y_v,z_v,20,'filled',MarkerEdgeColor='red',MarkerFaceColor='red')
%scatter3(coordinates{1,1},coordinates{2,1},coordinates{3,1},20,'filled','DisplayName',sprintf('Incidence angle:  %.2f [Deg]',incidence_array(1)),MarkerEdgeColor='blue',MarkerFaceColor='blue')
%scatter3(coordinates{1,11},coordinates{2,11},coordinates{3,11},20,'filled','DisplayName',sprintf('Incidence angle: %.2f [Deg]',incidence_array(11)),MarkerEdgeColor='green',MarkerFaceColor='green')
%scatter3(x_v,y_v,z_v,20,'filled','DisplayName',sprintf('Incidence angle: %.2f [Deg]',incidence_array(end)),MarkerEdgeColor='red',MarkerFaceColor='red')

xlabel('X [mm]',fontsize = 20)
ylabel('Y [mm]',fontsize = 20)
zlabel('Z [mm]',fontsize = 20)
hold off
legend({'Detector',sprintf(str_legend,'$\theta_i$',incidence_array(1),'$^\circ$'),sprintf(str_legend,'$\theta_i$',incidence_array(11),'$^\circ$'),sprintf(str_legend,'$\theta_i$',incidence_array(end),'$^\circ$')},'interpreter','latex',Fontsize = 22)
%}

%{
figure(13)
s = surf(X,Y,-Z,'DisplayName','Detector');
colormap gray
axis equal
hold on
scatter3(coordinates{1,1},coordinates{2,1},coordinates{3,1},20,col_v,'filled','DisplayName',sprintf('Incidence angle:  %.2f [Deg]',incidence_array(1)))
xlabel('X [mm]',fontsize = 20)
ylabel('Y [mm]',fontsize = 20)
zlabel('Z [mm]',fontsize = 20)
hold off
annotation('textbox',[0,.9,.3,.1],'String','Glass layer','BackgroundColor','red','FitBoxToText','on')
annotation('textbox',[0,.83,.3,.1],'String','Encapsulant layer','BackgroundColor','yellow','FitBoxToText','on')
annotation('textbox',[0,.76,.3,.1],'String','Silicon cell layer','BackgroundColor','white','FitBoxToText','on')
%annotation('rectangle',[0,.9,.3,.1],'Color','red')

%legend()
%}
dist_alt = sqrt((coordinates{1,1}-dist_ref(1)).^2 + (coordinates{2,1}-dist_ref(2)).^2 + (coordinates{3,1}-dist_ref(3)).^2);



%% Angular calc
polo = zeros(1,numel(Data_d8(:,2)));
for i=1:numel(Data_d8(:,2))

M =  Data_d8{i,4};
MM = Data_d8{i,2};
polo(i) = MM(2);

%MM = Data_d8{i,4};
l = M(1,1);
m = M(1,2);
n = M(1,3);
Nx = M(1,4);
Ny = M(1,5);
Nz = M(1,6);

%x = MM(1,1);
%y = MM(1,2);
%z = MM(1,3);
%asind(norm(cross([l,m,n],[Nx,Ny,Nz])));

%rp = cross([r,r,r],[l,m,n])/(sqrt(l^2+m^2+n^2));

end


%mean(polo)
%size(Data_relf)
Ang_6 = zeros(1,numel(Data_d8(:,2)));
polar = zeros(1,numel(Data_d8(:,2)));
azimuth = zeros(1,numel(Data_d8(:,2)));
Intensity = zeros(1,numel(Data_d8(:,2)));

for i=1:numel(Data_d8(:,2))
Q = Data_d8{i,2};
Ang_6(1,i) = Q(1);
%Ang_refl(1,i) =Data_relf{i,2}; 
Intensity(1,i) = Data_d8{i,3};
end
%{
Angles = linspace(0,180,180/2 + 1);

brdf = zeros(1,numel(Angles-1,1));
for i=1:numel(Angles)-1
brdf(i) = sum(Intensity(Ang_6 >= Angles(i) & Ang_6 < Angles(i+1)))/P;
end

figure(2)
p = plot(Angles(1:numel(Angles)-1),brdf,'-o','color','red','Linewidth',3);
p.MarkerFaceColor = 'red';
p.MarkerEdgeColor = 'red';
xlabel('Incidence Angle on detector [$^\circ$]','Interpreter','latex',fontsize = 22)
ylabel('$B_{\lambda = 0.55 \mathrm{\mu m}}(\theta = 45^\circ)$ [1/sr]',fontsize = 22)
grid()
%}
%{
        for q=1:numel(Angles)-1
        p_ang(q) = sum(val(2,val(1,:) >= Polar_filter(q) & val(1,:) < Polar_filter(q+1)))/P;
        end
        M_power{i,n} = brdf.';
        %}
pxl = [];
M_power = zeros(numel(Polar_filter),numel(Azimuth_filter));
bin_rad = abs(Polar_filter(1)-Polar_filter(2));
bin_azi = abs(Azimuth_filter(1)-Azimuth_filter(2));
%Angles = linspace(0,90,90/2 + 1);
% Row 1 is the incidence angle and the second is the power of the ray.
M_Ang = cell(numel(Polar_filter),numel(Azimuth_filter));
M_3 = zeros(numel(Polar_filter),numel(Azimuth_filter));
M_4 = zeros(numel(Polar_filter),numel(Azimuth_filter));
M_5 = zeros(numel(Polar_filter),numel(Azimuth_filter));
deg_diff = [];
Ray_hit = [];
r_test = cell(numel(Polar_filter),numel(Azimuth_filter));
r_val = zeros(numel(Polar_filter),1);
for i=1:numel(Polar_filter)
    for n=1:numel(Azimuth_filter)
        val = c{i,n};
        c_test = cc{i,n};
        M_Ang{i,n} = [Polar_filter(i),Azimuth_filter(n)]; 
        if isempty(val)
            continue
        end
        for q=1:3
        x(i,n) = mean(c_test(1,:));
        y(i,n) = mean(c_test(2,:));
        z(i,n) = mean(c_test(3,:));
        %r_test{i,n} = [x,y,z]; 
        end
        M_power(i,n) = sum(val(2,:));
       % sum(val(2,1))
        P_123 = zeros(3,1);
        for ii=1:numel(val(5,:))
        if val(5,ii) == 3
            P_123(1) = P_123(1) + val(2,ii);
            deg_diff = [deg_diff,(abs(val(3,1)-30))-val(7,1)];
            Ray_hit = [Ray_hit,val(5,1)];
        elseif val(5,ii) == 4
             P_123(2) = P_123(2) + val(2,ii);
            deg_diff = [deg_diff,(abs(val(3,1)-30))-val(7,1)];
            Ray_hit = [Ray_hit,val(5,1)];
        elseif val(5,ii) == 5
            P_123(3) = P_123(3) + val(2,ii);
           deg_diff =  [deg_diff,(abs(val(3,1)-30))-val(7,1)];
           Ray_hit = [Ray_hit,val(5,1)];
       
        end
        end
        M_3(i,n) = P_123(1);
        M_4(i,n) = P_123(2);
        M_5(i,n) = P_123(3);
        if sum(P_123) ~= M_power(i,n)
         disp('Power Mismatch')
        end
        %sum(P_123)
        %qw = M_power(i,n)
        Rad_col = floor((0.5+Polar_filter(i)) / bin_rad)+1;
        Azi_col = floor((0.5+Azimuth_filter(n)) / bin_azi);
        pxl = [pxl,Azi_col*px_radial + Rad_col]; 

    end
end
P_Ref = round(sum(sum(M_power)),5);
P_compare = ([sum(sum(M_3))/P,sum(sum(M_4))/P,sum(sum(M_5))/P,P-P_Ref]);
figure(3)
polarscatter(deg2rad(pA), pP,20,reshape(M_power.*1000,[],1),'filled')
% 
%colormap(jet(max(val_c)))
cb = colorbar('Position',[0.75,0.25,0.08,0.6]);
title(cb,'Flux \phi [mW]','Fontsize',18)
rticks(0:15:90)
rticklabels({'\theta = 0\circ','\theta = 15\circ','\theta = 30\circ','\theta = 45\circ','\theta = 60\circ','\theta = 75\circ','\theta = 90\circ'}')
%pax = gca;
set(gca,"CLim",[0,0.002]) 
set(gca,'Layer','Top')
set(gca,'ThetaColor','blue','Rcolor','red','FontSize',18,'FontWeight','bold','LineWidth',3)
%pax.ThetaColor = 'blue';
%pax.RColor = [0 .5 0];

figure(7)
heatmap(Azimuth_filter,Polar_filter, M_power)

%X = categorical({'Glass' 'Encapsulant' 'Silicon'});
%X = reordercats(X,{'Glass' 'Encapsulant' 'Silicon'});
figure(8) 
bar(0,P_compare.*100,'stacked')
grid()
set(gca,FontSize = 16)
xlabel('Reflection layer',Fontsize = 20)
ylabel('Relative power reflected from surface [\%]',FontSize = 20)
%bar(0,Y(1,:),Y(2,1),'stacked')

%figure;
%plot(Ray_hit(1:end-1),deg_diff(1:end-1))
%xlabel('Object hit',FontSize= 20)
%ylabel('Polar Angle spread difference [Deg]',FontSize = 20)


%% Solid angle / Pxl area calculation 
A_sph = (4*pi*r^2)/2; % Area of half spehere. 
A_avg = A_sph/double(pxl(end)); % Area pr pxl in mm.
x_avg = mean(x,2);
y_avg = mean(y,2);
z_avg = mean(z,2);

[circ,A_arr,Solid_ang,az_arc] = Solid_ang_calc(r,Azimuth_filter,Polar_filter);

%for i=1:numel(Polar_filter)-1
%r_val(i,1) = sqrt((x_avg(i)-x_avg(i+1))^2 + (y_avg(i)-y_avg(i+1))^2 + (z_avg(i)-z_avg(i+1))^2);
%ABBA(i,1) = az_arc*r_val(i,1);
%end


A_source = (source_size)^2*pi;

PP_avg = mean(Power,1).';


%for i=1:numel(Polar_filter)-1
%pol_filtered = polo(polo >= Polar_filter(i) & polo < Polar_filter(i+1))

%end


%%
figure(4)
plot(Polar_filter,circ/numel(Polar_filter),'linewidth',2,'color','red')
grid()
xlabel('Polar angle $\theta [^\circ]$',fontsize = 20)
ylabel('Arclength [mm]',fontsize = 20)

figure(5)
plot(Polar_filter,(Solid_ang),'linewidth',2,'color','red')
grid()
ax = gca; % Get current axes
ax.XAxis.FontSize = 14; % Set the font size for X-axis tick labels
ax.YAxis.FontSize = 14; % Set the font size for Y-axis tick labels
ax.XTick = 0:10:90;
ax.XTickLabel = {'0', '10', '20', '30', '40', '50', '60', '70', '80', '90'};
xlabel('Scattering Angle $\theta_s [^\circ]$',fontsize = 24)
ylabel('Solid Angle $\Omega$ [sr]',fontsize = 24)
xlim([0,90])

figure(18)
plot(Polar_filter,A_arr,'linewidth',2,'color','red')
hold on
plot(Polar_filter,ones(numel(Polar_filter),1)*A_source,'linewidth',2,'color','black')
grid()
xlabel('Polar angle $\theta [^\circ]$',fontsize = 20)
ylabel('Area $[\mathrm{mm^2}] $ ',fontsize = 20)

%figure(6)
%hold on;
% Plot and fill the first polygon
%fill(circ(1:15:end,1)./91, az_arc*ones(numel(Polar_filter(1,1:15:end)),1), 'cyan', 'FaceAlpha', 0.3);
%text(mean(circ(1:15:end,1)./91), mean(circ(1:15:end,1)./91), ['Area: ', num2str(area1)], 'HorizontalAlignment', 'center', 'FontSize', 12, 'BackgroundColor', 'white');

% Plot and fill the second polygon
%fill(x2, y2, 'magenta', 'FaceAlpha', 0.3);
%text(mean(x2), mean(y2), ['Area: ', num2str(area2)], 'HorizontalAlignment', 'center', 'FontSize', 12, 'BackgroundColor', 'white');

%hold off;

%Y = zeros(2,numel(Polar_filter(1,1:15:end))); % [circ;az_arc*ones(numel(Polar_filter),1)]
%Y(1,:) = circ(1:15:end,1)./91;
%Y(2,:) = az_arc*ones(numel(Polar_filter(1,1:15:end)),1);
%I = linspace(0,90,7);
%figure(6)
%bar(0,Y(1,:),Y(2,1),'stacked')
%Format_spec = 'Mesh Area = %.3f %s for %s = %d';
%xlabel('Arclength of azimuth direction [mm]',FontSize = 20)
%ylabel('Arclength in polar direction [mm]',FontSize = 20)
%legend({sprintf(Format_spec,A_arr(I(1)+1),'$\mathrm{mm^2}$','$\theta$',Polar_filter(I(1)+1)),sprintf(Format_spec,A_arr(I(2)+1),'$\mathrm{mm^2}$','$\theta$',Polar_filter(I(2)+1)),sprintf(Format_spec,A_arr(I(3)+1),'$\mathrm{mm^2}$','$\theta$',Polar_filter(I(3)+1)),sprintf(Format_spec,A_arr(I(4)+1),'$\mathrm{mm^2}$','$\theta$',Polar_filter(I(4)+1)),sprintf(Format_spec,A_arr(I(5)+1),'$\mathrm{mm^2}$','$\theta$',Polar_filter(I(5)+1)),sprintf(Format_spec,A_arr(I(6)+1),'$\mathrm{mm^2}$','$\theta$',Polar_filter(I(6)+1)),sprintf(Format_spec,A_arr(end),'$\mathrm{mm^2}$','$\theta$',Polar_filter(end))},'FontSize',18,'Interpreter','latex')
%xticks(linspace(-1,1,21))
%area(circ,az_arc*ones(numel(Polar_filter),1))

%{
figure(5)
plot(Polar_filter,M_arc,'linewidth',2,'color','red')
grid()
xlabel('Polar angle $\theta [^\circ]$',fontsize = 20)
ylabel('Arclength [mm]',fontsize = 20)
%}


figure(10) % Polar angle difference as function of source area.
plot(source_size.^2*pi,div,'linewidth',2,'color','red')
grid('on')
xlabel('Spot area [$\mathrm{mm^2}$]',fontsize = 20)
ylabel('$\mathrm{|\theta_{max}-\theta_{min}|} [^\circ]$',fontsize = 20)
set(gca,FontSize = 22)
%}

%{
figure(10) % Polar angle difference as function of source area.
plot(incidence_array,div,'-o','linewidth',3,'color','red')
grid('on')
xlabel('Incidence angle $[^\circ]$',fontsize = 20)
ylabel('$\mathrm{|\theta_{max}-\theta_{min}|} [^\circ]$',fontsize = 20)
set(gca,FontSize = 22);
%set(gca,MarkerFaceColor,'blue')
%}


%% brdf calculation

A_source = (source_size)^2*pi;
A_n = A_source / cosd(theta_i);
M_Solid_Ang = repmat((Solid_ang),[1,numel(M_power(1,:))]);
Irradiance_in = P./ A_n; %./ source_size^2*pi;

Irradiance_out =  M_power ./ (A_n .* cosd(Polar_filter).');%(arc_pol*az_arc); Sensor Area/ cos(theta) / cos(theta_i)M_power
Irradiance_avg = mean(Irradiance_out,2);

brdf_M = (Irradiance_out ./ (M_Solid_Ang)) / Irradiance_in';
brdf = (Irradiance_avg./(Solid_ang))./ (Irradiance_in.'); %(Irradiance_in).'; %  W/m^2/sr / W/m^2 ./(Solid_ang)

%dA = flip(A_arr);
%domega = flip(Solid_ang);
%{
figure(1)
yyaxis left 
plot(Polar_filter,Irradiance_avg,'-o','Color','black')
ylim([0,0.000001])
yyaxis right
plot(Polar_filter,Irradiance_in,'-o','Color','red')
ylim([0,6])

%set(gca,'Yscale','log')
%}
brdf_slice_az = brdf(30,:);
brdf_slice_pol =  brdf(:,1);
avg_pol = mean(brdf,2);
avg_pow = mean(M_power,2);

%L = avg_pow(1:end-1,1) ./ (flip(dA).*flip(Solid_ang(2:end)))  ;

%Lamb = zeros(90,1);

%%for i=1:numel(Polar_filter)-1
  %  dA0 = dA(2);
  %  dOmega0 = domega(3);

   % Lamb(i) =  (avg_pow(i)*domega(i)*dA(i)) / (dA0*dOmega0);
%end
%figure;
%plot(Polar_filter,((1/pi)*P_Ref) ./ avg_pol)

figure(17)
plot(Polar_filter,Irradiance_avg./(Solid_ang))



figure(15)
plot(Polar_filter,PP_avg,'LineWidth',2,'Color','red','Marker','o','LineStyle','-')
hold on
%plot(Polar_filter,avg_pow./cosd(linspace(0,90,91)).')
plot(Polar_filter,avg_pow,'LineWidth',2,'Color','black','Marker','square','LineStyle','none')
xlabel('Scatter Angle $\theta_s$ [$^\circ$]',FontSize = 20)
ylabel('Radiant Flux $\Phi$[W]',FontSize = 20)
grid('on')
legend({'$\Phi_{Detector}$','$\Phi_{calc}$'},'Interpreter','latex','FontSize',16)
set(gca,'YScale','log')

figure(16)
plot(Polar_filter(2:end),avg_pol(2:end),'Color','black','Marker','square','LineStyle','--','LineWidth',2)
hold on
plot(Polar_filter(1:end),brdf_M(:,1),'Color','blue','Marker','square','LineStyle','none','LineWidth',2)
plot(-Polar_filter(1:end),flip(brdf_M(:,91)),'Color','red','Marker','square','LineStyle','none','LineWidth',2)
grid('on')
legend({'$B_{\mathrm{mean}}$','$B_{\varphi = 0^\circ}$','$B_{\varphi = 180^\circ}$'},'Interpreter','latex','FontSize',16)
set(gca,'YScale','log')
xlim([-90,90])

%plot(Polar_filter,(avg_pow./(Solid_ang))./ (P.*cosd(Polar_filter).'),'-o','Color','blue')
%plot(Polar_filter,(1/pi)*P_Ref*ones(1,numel(Polar_filter)),'Color','red','LineWidth',2)
%plot(Polar_filter,mean(avg_pol(2:end-1)).*ones(1,numel(Polar_filter)),'Color','blue','LineWidth',2)
%plot(Polar_filter,domega)
%plot(Polar_filter(1:90),Lamb)
xlabel('Scatter Angle $\theta_s$ [$^\circ$]',FontSize = 20)
ylabel('B[1/sr]',FontSize = 20)
grid('on')
set(gca,'Yscale','log')
%legend({'$L_{\Omega,out} / I_{in}$','$\frac{1}{\pi} \cdot P_{Ref}$','$\mathrm{mean}(L_{\Omega,out} / I_{in})$'},'Interpreter','latex','FontSize',16)

%'$[P_{\Omega,out}/d\Omega_s$] / [$cos(\theta_s) \cdot P_{in}$]'

figure(18)
plot(Polar_filter,Irradiance_avg ./(Irradiance_in*P_Ref),'-o','Color','black')
xlabel('Polar Angle [Deg]',FontSize = 20)
ylabel('brdf [1/sr]',FontSize = 20)
grid('on')
set(gca,'Yscale','log')
%legend({'$L_{\Omega,out} / I_{in}$','$[P_{\Omega,out}/d\Omega_s$] / [$cos(\theta_s) \cdot P_{in}$]','$\frac{1}{\pi} \cdot P_{Ref}$'},'Interpreter','latex','FontSize',16)

%% Save file data


dlgtitle = 'Saving test data';
dlgQ = 'Want to save test variables as file';
choice = questdlg(dlgQ,dlgtitle,'Yes','No','Yes');
% Handling response
switch choice
    case 'Yes'
        Prompt = 'Specify the PV type for the data:'; % Use manufactorer Lenzing, Wolbring etc.
        index = input(Prompt,"s");
        file_name = index;
      %  file_name = file_name(1:10); % Find the date of the data file. 
        folder = 'C:\Users\s194086\Desktop\Simulation_plots';
        if exist(folder,'dir') == 0 % Make Folder if it does not exist.
            mkdir(folder)
        end
        Full_file_name = fullfile(folder,append(file_name,'_',index));
        save(Full_file_name,'avg_pol','Power','M_power','Solid_ang','Polar_filter','P_Ref','M_3','M_4','M_5','brdf_M')
        disp('Data saved to file')
    case 'No'
        disp('Data is not being saved.')
end


%% Functions
%filter(1,Azimuth_filter)


function idx = filter(val,array)
if numel(array) ~=180
max_ang = 90;
inc = max_ang/(numel(array)-1);
r_blob = zeros(numel(array),2);
for j=1:numel(array)
r_blob(j,:) = [inc*(j-2)+(1/2),inc*(j-1 + (1/2))].';
r_blob(1,1) = 0;
r_blob(end,2) = 90; 
if val >= r_blob(j,1) && val < r_blob(j,2)
    idx = j;
end
end
else
max_ang = 360;
inc = max_ang/(numel(array));
r_blob = zeros(numel(array),2);

for j=1:numel(array)
r_blob(j,:) = [(j-1)*inc,inc*j].';
r_blob(end,2) = 0; 

if val >= r_blob(j,1) && val < r_blob(j,2)
    idx = j;
elseif val >= 358
    idx = numel(array);
    break

end
end


end








%[~,idx] = min(abs(val-array));


%{
for i=1:numel(array)-1
if val >= array(i) && val < array(i+1)
idx = i;
elseif val <=360 && val > 358
    idx = i;
elseif val <= array(end) && val >= array(end-1)
    idx = i;
end
end
%}
end



function [circ,A_arr,Solid_ang,az_arc] = Solid_ang_calc(r,Azimuth,Polar)
circ = zeros(numel(Polar),1);
M_arc = zeros(numel(Polar),1);
az_arc  = ((2*pi)*r)/numel(Azimuth); % Assumed to cirumference of half cirle. But full for azimuth. 
% Area of patches are equal for same azimuth angle. 
Solid_ang = zeros(numel(Polar),1);
A_arr = zeros(numel(Polar),1);
%arc = zeros(1,numel())
for i=1:numel(Polar)
    % Init coordinates: (theta,phi)
     %   v1 = M_Ang{1,1};
     %   v2 = M_Ang{i+1,2};
    % Calculate Arc lenth
   % q = cosd(v1(1))*cosd(v2(1))+sind(v1(1))*sind(v2(1))*cosd(0);
    rs = cosd(Polar(i))*r;
    circ(i) = pi*rs; % Circumference for a quater of the sphere from 0-90 degree polar angle. 
%    arc = r*acosd(q);
%    M_arc(i,1) = arc;
    A = (circ(i)/(2*numel(Polar)))*az_arc; % Assumed squared. 
    A_arr(i) = A;
    Solid_ang(i,1) = A/r^2;
end
Solid_ang = flip(Solid_ang);
A_arr = flip(A_arr);
circ = flip(circ);
end

