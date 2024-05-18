
function [] = PSRL_LQ()
n = 3;
m = 3;
simul = 100;
time_horizon = 2000;
iteration = zeros(time_horizon,simul);
A = [[0.3 0.1 0.2];[0.1 0.4 0];[0 0.7 0.6]];
B = [[0.5 0.4 0.5];[0.6 0.3 0];[0.3 0 0.2]];
Q = 2*eye(n);
R = eye(m);

W = eye(n);

%admissible condition
theta_bound = 20;
J_bound = 20000;
rho = 0.99;

%prior condition
lam = 5;
theta_mean_ = 0.5*ones(n+m,n);


for simulation = 1:simul
    disp(simulation)
    t=1;
    k=1;
    x=zeros(n,1);
    theta= zeros((n+m),n);
    data={};
    while 1
        disp(t)
        T = k+1;
        t_k = t;
        while 1
            [arbi,step_iteration]= ULA_Gaussian(theta, lam, theta_mean_, data, n, m, t_k);
            disp(arbi)
            disp(step_iteration)

            a=arbi((1:n),:)';
            b=arbi((n+1:n+m),:)';
            [S,Gain,l] = idare(a,b,Q,R);
            if (norm(arbi)<theta_bound)&&(trace(W*S)<J_bound)&&((sqrt((max((eig((a-b*Gain)'*(a-b*Gain)))))) < rho))                    
                break
            else
                disp('rejection')
            end
        end

        while t<=t_k+T-1
            u=-Gain*x;
            
            iteration(t,simulation) = step_iteration;
            
            if t==time_horizon
                break
            end
                
            

            x_prime = A*x + B*u + mvnrnd(zeros(n,1),eye(n))';
            
            data{end+1} = {x;u;x_prime};
        
            x = x_prime;
        
            t = t+1;
        end
        k=k+1;
        if t == time_horizon
            break
        end
    end
end
%preconditioner
writematrix(iteration,'preconditioned_ULA_iteration.csv');
%naive ULA
%writematrix(iteration,'ULA_iteration.csv');
end


function [theta,step_iteration] = ULA_Gaussian(theta, lam, theta_mean_, data, n, m, t_k)
preconditioner = lam*eye((n+m)*n);

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
M = 1;
L = 1;

%calculate the mean of posterior(closed form)
theta_mean= zeros((n+m),n);
precision = lam*eye(n+m); %inverse of covariance
theta_mean_cal = lam*theta_mean_;
for k = 1:length(data)
    x = data{k}{1};
    u = data{k}{2};
    x_prime = data{k}{3};
    z = cat(2, x',u')';
    zeta = z* z';
    for i =1:n
        theta_mean_cal(:,i) = theta_mean_cal(:,i)+  x_prime(i)*z;

    end
    precision = precision + zeta;
end
sigma = 1*(precision)\eye(n+m); %covariance
for l =1:n
    theta_mean(:,l) = sigma*theta_mean_cal(:,l);
end

%vectorization of mean of posterior
phi = tr_theta_to_phi(theta_mean,n,m);

%preconditioned ULA
step_size = (M*min(eig(preconditioner)))/(16*L^2*max(t_k,min(eig(preconditioner))));
step_iteration = ceil(4*(log2(max(t_k,min(eig(preconditioner)))/min(eig(preconditioner)))/(M*step_size)));

%naive ULA
%step_size = (M*min(eig(preconditioner)))/(16*L^2*(max(eig(preconditioner)))^2);
%step_iteration = ceil(64*(max(eig(preconditioner)))^2/(min(eig(preconditioner)))^2);

disp(max(eig(preconditioner)))
disp(min(eig(preconditioner)))
disp(max(eig(preconditioner))/min(eig(preconditioner)))

s1 = 0;
s2 = 0;
s_prior=-lam*(phi-0.5*ones(18,1));
for transition = 1:length(data)
    
    
    x = data{transition}{1};
    u = data{transition}{2};
    x_prime = data{transition}{3};
    
    z = cat(2, x', u')';
    zeta = z * z';
    for i=2:n
        zeta = blkdiag(zeta, z*z');
    end
    
    grad_log_phi2 = x_prime(1)*z;
    for j=2:n
        grad_log_phi2 = cat(2, grad_log_phi2',(x_prime(j)*z)')';
    end
    s1 = s1 -zeta;
    s2 = s2 + grad_log_phi2;
end


for iteration = 1: step_iteration
    
    s = s1 * phi + s2;
    grad_U = -s-s_prior;
    %In case of using preconditioner
    scaled_grad_U = inv_preconditioner*grad_U;
    %In case of using ULA
    %scaled_grad_U = grad_U;
    
    phi = mvnrnd(phi - step_size*scaled_grad_U, 2*(step_size)*inv_preconditioner*eye((n+m)*n))';
    
end

for i=1:n
    theta(:,i) = phi((n+m)*(i-1)+1:(n+m)*i);
end

end

    