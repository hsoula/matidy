function u = compute_pde(p, u0, nx, r_max, r_min)
%function [r, u] = pders 

r0 = r_min; % mum
vl = 1e6; % from mol to mum
a = p(1);
ra = p(2);
kl = p(3);
KL = p(4);
tg0 = p(5);
D = p(6);
na = 3;
b = 0.27; % mol.mum^-2 h^-1
B = 125;  % mol.h^-1


% pas en r + r max
dx = r_max/(nx-1);
r = r0 + (dx * (1:nx)-dx);

size(r)
size(u0)

% initial density of cells : gaussian
%u0 = interp1(data(:,1), data(:,2), r);

%u0 = exp(-(r-mu).^2 * si);
u = u0; % density vector

% boundary conditions
u(1) = 0; 
u(end) = 0;

% volumes
v = 4 / 3 * pi * (r + r0).^3;
v0 = 4 / 3 * pi * r0^3;
Ltotal = tg0 + sum( v.*u)/sum(u) / vl; % total lipid intra+extra cellular (in mol)

L = tg0; %% variable extra cellular lipid (in mol)

tmax = 100000;
Ls = zeros(tmax,3); %time, store external lipid, lipid in cell 

t =  0.0;

%while(norm(u_old(:) - u(:))/norm(u(:)) > 1e-6)
for K = 1: tmax    
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
    %Flux(1) = 0;
    %Flux(1) = 100*L;
%  u(2:nx-1)=u(2:nx-1)-(dt/dx)*(Flux(2:nx-1)-Flux(1:nx-2)) + 0.5*br*(u(1:nx-2)+u(3:nx)-2*u(2:nx-1)); % update density  
    
    u(2:nx-1)=u(2:nx-1)-(dt/dx)*(Flux(2:nx-1)-Flux(1:nx-2)); % update density
 
    % since they are never updated u(1) = u(nx) = 0
   
    L = Ltotal - sum(v.*u)/sum(u)/ vl; % update external lipid content
    if L <0
        L=0;
    end
    
     %test les zeros :
    if(mod(K, 1000) == 0)
        plot(r,u, r, u0)
        drawnow
    end
    t = t+dt; % update time
    Ls(K,:) = [t, L, sum(v.*u)/sum(u)/ vl]; % store lipid + cell lipid       
end


