function[grad_sum] = grad_log_as(pi,data,mean,M,L,alpha,beta,n,m,lam)

grad_sum = -lam*(pi-0.5*ones((n+m)*n,1));
for transition = 1:length(data)
    x = data{transition}{1};
    u = data{transition}{2};
    x_prime = data{transition}{3};
    
    z = cat(2, x', u')';
    
    theta = tr_phi_to_theta(pi,n,m);
    w=x_prime-theta'*z;
    
    grad_log_phi = grad_log(w+mean,M,L,alpha,beta,n,1)*z;
    for j=2:n
        grad_log_phi = cat(2, grad_log_phi',(grad_log(w+mean,M,L,alpha,beta,n,j)*z)')';
    end

    grad_sum = grad_sum+grad_log_phi;
    
end
