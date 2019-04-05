function khi2 = xhi2_pde(p, u0, nx, r_max, r_min)
p
dx = r_max/(nx-1);
r = r_min + (dx * (1:nx)-dx);
u = compute_pde(p, u0, nx, r_max, r_min);
khi2 = sum((u-u0).^2)
plot(r,u, r, u0)
