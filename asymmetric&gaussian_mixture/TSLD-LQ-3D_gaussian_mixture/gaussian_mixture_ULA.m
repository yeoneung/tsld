
function [theta] = ULA_Gaussian_mixture(gm_a, r, theta, lam, phi_U, data, n, m, t_k)

preconditioner = lam*eye((n+m)*n);

%Initialize phi
phi = phi_U;

for k = 1:length(data)
    x = data{k}{1};
    u = data{k}{2};    
    z = cat(2, x',u')';
    zeta = z* z';
    for i=2:n
        zeta = blkdiag(zeta, z*z');
    end
    preconditioner = preconditioner + zeta;
end

inv_preconditioner = 1*(preconditioner)\eye((n+m)*n);

%strong log-concavity and Lipschitz smoothness
M = 1/r*(1-1/r*norm(gm_a)^2);
L = 1/r;


%calculate step size and number of step iteration for preconditioned ULA
step_size = (M*min(eig(preconditioner)))/(16*L^2*max(t_k,min(eig(preconditioner))));
step_iteration = ceil(4*(log2(max(t_k,min(eig(preconditioner)))/min(eig(preconditioner)))/(M*step_size)));



%Langevin iteration
for iteration = 1: step_iteration
    %sum of gradient of log-likelihood
    grad_prior = -lam*(phi-0.5*ones(18,1));
    grad_sum=0;

    for transition = 1:length(data)
        x = data{transition}{1};
        u = data{transition}{2};
        x_prime = data{transition}{3};
        
        z = cat(2, x', u')';
        
        theta = tr_phi_to_theta(phi,n,m);
        w=x_prime-theta'*z;
        
        grad_log_phi = 1/r*(w(1)-gm_a(1)+2*gm_a(1)/(1+exp(1)^(2/r*w'*gm_a)))*z;
        for j=2:n
            grad_log_phi = cat(2, grad_log_phi',(1/r*(w(j)-gm_a(j)+2*gm_a(j)/(1+exp(1)^(2/r*w'*gm_a)))*z)')';
        end

        grad_sum = grad_sum+grad_log_phi;
        
    end
    
    grad_U = -grad_sum-grad_prior;
    scaled_grad_U = inv_preconditioner*grad_U;
    
    phi = mvnrnd(phi - step_size*scaled_grad_U, 2*(step_size)*inv_preconditioner*eye((n+m)*n))';
   
end

theta = tr_phi_to_theta(phi,n,m);

end
