function[hess_sum] = hess_log_as(pi,data,mean,M,L,alpha,beta,n,m,lam)

hess_sum = -lam*eye((n+m)*n);
for transition = 1:length(data)
    x = data{transition}{1};
    u = data{transition}{2};
    x_prime = data{transition}{3};
    
    z = cat(2, x', u')';
    
    theta = tr_phi_to_theta(pi,n,m);
    w=x_prime-theta'*z;
    
    hess_log_phi = kron(hess_log(w+mean,M,L,alpha,beta,n),z*z');

    hess_sum = hess_sum+hess_log_phi;
    
end
