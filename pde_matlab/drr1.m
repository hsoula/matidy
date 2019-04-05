function [dri, dro] = drr1(r,v0, L, a, ka, na, b, kt, vl, kL, B)

v = 4 / 3 * pi * r .^3;

dri = (a /(4*pi) * 1./(1 + (r/ka).^na) * L/(kL+L));
dro = (B + b * r.^2)./(4*pi * r.^2) .* (v - v0)./(v - v0 + kt*vl);
