%function [r, u] = pders 
% main pde
clear all;
close all;

r0 = 15; % mum
vl = 1e6; % from mol to mum
a = 0.5; %0.3; % mol.mum^-2 h^-1       %% need a larger a to have bimodal
b = 0.27; % mol.mum^-2 h^-1
B = 125;  % mol.h^-1
ra = 200; % mum
na = 3; % hill radius
kl = 0.01; %  mol
KL = 0.1; % mol

tg0 = 3; % initial external lipid (in mol)

% pas en r + r max
nx = 500;
r_max = 300; % mum
dx = r_max/(nx-1);
rmin = r0 - dx;
r = rmin + (dx * (1:nx)-dx);

% diffusion parameter
D = 50*1e3; %% needed to be large : units? not sure it make sens

% initial density of cells : gaussian
mu = 45.0;
si = 0.01;

u0 = exp(-(r-mu).^2 * si);
u = u0; % density vector

% boundary conditions
u(1) = 0; 
u(end) = 0;

% volumes
v = 4 / 3 * pi * r.^3;
v0 = 4 / 3 * pi * r0^3;
Ltotal = tg0 +  dx* sum( (v-v0).*u .*(4*pi .*r.^2 ./ vl.^2)); % total lipid intra+extra cellular (in mol)

L = tg0; %% variable extra cellular lipid (in mol)

tmax = 1*100000;
Ls = zeros(tmax,3); %time, store external lipid, lipid in cell 

t =  0.0;

figure()

plot(r,u0)

for(K = 1: tmax)
%norm(u_old(:) - u(:))/norm(u(:))
    u_old(:) = u(:);
    
    [lg, lp] = drr1(r,v0,L, a,ra,na,b,kl,vl, KL, B); % compute velocity at current L for all l and r
    
    dp = vl*(lg-lp);%
    mdp = max(0, dp);
    idp = min(0, dp);
  
    dt = 0.4*dx*dx/(max(abs(dp))*dx + D); % cfl for realtime dt adjustment : not sur 0.2 is needed
    br = D*dt/(dx*dx);
    
    % start of the pde scheme
    Flux=mdp.*u ; % compute right (c>0) flux
    
    Flux(1:nx-1)=Flux(1:nx-1)+idp(1:nx-1).*u(2:nx); % add left (c<0) flux
    Flux(1:nx-1)=Flux(1:nx-1)-(D/dx)*(u(2:nx) - u(1:nx-1)); %% diffusion part
    
    Flux(nx)=0; % no flux at boundary
    Flux(1)=0; % no flux at boundary 
    
    % we can also test 
    %Flux(1) = 100;
    %Flux(1) = 100*L;
    
    %u(2:nx-1)=u(2:nx-1)-(dt/dx)*(Flux(2:nx-1)-Flux(1:nx-2)) + 0.5*br*(u(1:nx-2)+u(3:nx)-2*u(2:nx-1)); % update density  
    u(2:nx-1)=u(2:nx-1)-(dt/dx)*(Flux(2:nx-1)-Flux(1:nx-2)); % update density
 
    % since they are never updated u(1) = u(nx) = 0
    L = Ltotal - dx* sum( (v-v0).*u .* (4 * pi .*r.^2 ./ vl^2)); % update external lipid content
    if L <0
        L=0;
    end
    
    t = t+dt; % update time
    Ut = dx* sum( (v-v0).*u .* (4 * pi .*r.^2 ./ vl^2)); % total intracell lipid
    Ls(K,:) = [t, L, Ut]; % store lipid + cell lipid
    
    %test les zeros :
    %if(mod(K, 1000) == 0)
    %    g = @(r) A1(r,a, ra, KL, B, b, kl, v0, vl, na, L);
    %    data = [];  
    %    for rr=r0:r_max
    %        data = [data; fzero(g, rr)];
    %    end;
        %unique(data)
        %plot(r,u)
        %drawnow
    %end
    
    %plot(r,u,r,u0); drawnow
% plot result
end
Tmax = t
norm(u_old(:) - u(:))/norm(u(:))

%plot(r,u,r,u0)
%figure(2);plot(Ls(:,1), Ls(:,2), 'r', Ls(:,1), Ls(:,3),'b');% lipid evolution


%% calcul integral rmin to rmax of drr1 : stationnary solution
Lfin = Ls(end,2);
mass = integral( @(x) exp(-(x-mu).^2 * si), r0, r_max )
Tr = @(x) A1(x,a,ra, KL, B, b, kl, v0, vl, na, Lfin);

equi = zeros(size(r));
for i=1:length(r)
    equi(i) = exp(1/D * integral(Tr, r0, r(i)) );
end

int_equi = dx*(sum(equi) - 0.5*(equi(1)+equi(end)));
C = mass / int_equi; 

figure(3);
plot(r, u, r, C*equi);
