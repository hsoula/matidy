
a = 0.5; %0.3; % mol.mum^-2 h^-1       %% need a larger a to have bimodal
b = 0.27; % mol.mum^-2 h^-1
B = 125;  % mol.h^-1
ra = 200; % mum
na = 3; % hill radius
kl = 0.01; %  mol
KL = 0.1; % mol
tg0 = 3;
D = 50*1e3;
p0 = [a, ra, kl, KL,tg0,D];

nx = 500;
r_max = 300; % mum
r_min = 20; % mum

adipocytes = load('../data/Hys-D0-1-1.txt');

dx = r_max/(nx-1);
r = r_min + (dx * (1:nx)-dx);
size(r)

data = histogram(adipocytes, [r r(end)+dx]);
u0 = data.Values; 

my_xhi2 = @(p) xhi2_pde(p, u0, nx, r_max, r_min) ;

popt = fminsearch(my_xhi2, p0)
