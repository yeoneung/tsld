function[hess_sum] = hess_log_gm(pi,data,gm_a,r,n,m,lam)

hess_sum = -lam*eye((n+m)*n);
for transition = 1:length(data)
    x = data{transition}{1};
    u = data{transition}{2};
    x_prime = data{transition}{3};
    
    z = cat(2, x', u')';
    
    theta = tr_phi_to_theta(pi,n,m);
    w=x_prime-theta'*z;
    
    hess_log_phi = -kron((1/r)*eye(n)-4*(1/(r^2))*(gm_a*gm_a')*(exp(2*w'*gm_a/r)/(1+exp(2*w'*gm_a/r))^2),z*z');

    hess_sum = hess_sum+hess_log_phi;
    
end
