
 a = [2.2356,-6.85*10^(-3),8.8453*10^(-3),1.286*10^(-3),-1.715*10^(-4),7.6109*10^(-6);];

 wvl = linspace(300,900,600)./1000;
 rn = Schott(a,wvl);


figure(1)
plot(wvl,rn)

 function  rn  = Schott(a,wvln)
% SCHOTT  
% 
%    SCHOTT(A,WVLN) returns a matrix whose elements are the
%                   refractive indices calculated using the Schott
%                   dispersion formula.
%
%     SCHOTT(A)     returns the refractive index at 0.58756 (n_d) 
%  
%          A     vector of Schott coefficients
%       WVLN     vector of wavelengths (microns)
%
%
 if (nargin<2) 
    wvln = [0.58756];
 end
 ws = wvln.*wvln;
 rn = sqrt(a(1)+ws.*a(2)+ (a(3)+(a(4)+(a(5)+a(6)./ws)./ws)./ws)./ws);
end
