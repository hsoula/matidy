function y = A1(r,a, kr, kL, B, b, kt, V0, Vl, n, L0)

l = (4*pi/3 * r.^3 - V0)/Vl;
y = a * r.^2 * L0 ./(L0 + kL) .* 1./(1+(r/kr).^n) - (B + b * r.^2).* l ./(kt + l) ;
 
 
 