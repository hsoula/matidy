% solution stationnaire
%clear all;
%close all;

r0 = 15.0; % mum
vl = 1e6; % from mol to mum
a = 0.5; %0.3; % mol.mum^-2 h^-1       %% need a larger a to have bimodal
b = 0.27; % mol.mum^-2 h^-1
B = 125;  % mol.h^-1
ka = 200; % mum
na = 3; % hill radius
kt = 0.01; %  mol
kL = 0.1; % mol
D = 50*1e3;
%u0 = exp(-(x-mu).^2 * si)
mu = 45.0;
si = 0.01;
 
r_max = 300;

N = integral( @(x) exp(-(x-mu).^2 * si), r0, r_max ); %number of cells (constant)


%for L = 0.2:0.1:2
L = .1716;
dr = 0.1;
r = r0:dr:r_max;
equi = zeros(size(r));
for i=1:length(r)
    equi(i) = exp(1/D * integral( @(r) Tau_R(r, L, r0, a, ka, na, kL, B, b, kt, vl),r0,r(i)) );
end
int_equi = (sum(equi) - 0.5*(equi(1)+equi(end)))*dr;

uStat = (N/int_equi) .* equi;
rst = r;

%plot(r, u)
%drawnow
%end

%plot(r, Tau_R(r,  L, r0, a, ka, na, kL, B, b, kt, vl))

function Tau_r = Tau_R(r, L, r0, a, ka, na, kL, B, b, kt, vl)
% tau (r, L_inf)
    v = 4 / 3 * pi * r .^3;
    v0 = 4 / 3 * pi * r0 .^3;
    
    Tau_r = (a /(4*pi) * 1./(1 + (r/ka).^na) * L/(kL+L)) - (B + b * r.^2)./(4*pi * r.^2) .* (v - v0)./(v - v0 + kt*vl);
    Tau_r = vl * Tau_r;
end
% u = N/
% 
% 
% 
% function solveL = ??
% 
% L0 + U0 - L - 
% 
% 
% end
% 

