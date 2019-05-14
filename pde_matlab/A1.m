function y = A1(r,a, kr, kL, B, b, kt, v0, vl, na, L0)

    v = 4 / 3 * pi * r .^3;
    
    Tau_r = (a /(4*pi) * 1./(1 + (r/kr).^na) * L0/(kL+L0)) - (B + b * r.^2)./(4*pi * r.^2) .* (v - v0)./(v - v0 + kt*vl);
    y = vl * Tau_r;
end

 
 
 