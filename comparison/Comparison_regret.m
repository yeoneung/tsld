
function [] = PSRL_LQ()
n = 3;
m = 3;
simul = 100;
time_horizon = 50000;
regret_ULA = zeros(time_horizon,simul);
regret = zeros(time_horizon,simul);
A = [[0.3 0.1 0.2];[0.1 0.4 0];[0 0.7 0.6]];
B = [[0.5 0.4 0.5];[0.6 0.3 0];[0.3 0 0.2]];
Q = 2*eye(n);
R = eye(m);
 
W = 1*eye(n);

%admissible condition
theta_bound = 20;
J_bound = 20000;
rho = 0.99;

%prior condition
lam = 5;
theta_mean_ = 0.5*ones(n+m,n);
sigma_ = (1\lam)*eye((n+m));


%PSRL-LQ
for simulation=1:simul
    disp(simulation)
    t = 1;
    t_k = 0;

    theta_mean = theta_mean_;
    sigma = sigma_;
    
    x = zeros(n,1);
    
    while 1
        disp(t)
            
        T = t - t_k;
        t_k = t;
        sigma_k = sigma;
        
        while 1
                
            arbi = zeros(n+m,n);
            for i =1:n
                arbi(:,i) = mvnrnd(theta_mean(:,i),sigma);
                   
            end
                
            a = arbi([1,2,3],:)';
            b = arbi([4,5,6],:)';
            [S,Gain,L] = idare(a,b,Q,R);
            
            if (norm(arbi)<theta_bound)&&(trace(W*S)<J_bound)&&((sqrt((max((eig((a-b*Gain)'*(a-b*Gain)))))) < rho))
                break
            else
                disp('rejection')
            end
            
            
        end
        

        while (t <= t_k+T)&&(det(sigma)>=(1/2)*det(sigma_k))
            u = -Gain*x;
               
                
            z = cat(2, x',u')';
                
            zeta = z*z';
                
            cost = x'*Q*x + u'*R*u;
            j = trace(W*S);
                
            regret(t,simulation) = cost - j;
            
            if t == time_horizon
                break
            end
                
            x = A*x + B*u + mvnrnd(zeros(n,1),1*eye(n))';
            
            for i =1:n
                theta_mean(:,i) = theta_mean(:,i) + sigma*z*(x(i)-theta_mean(:,i)'*z)/(1+z'*sigma*z);
    
            end
            sigma = sigma -sigma*zeta*sigma/(1+z'*sigma*z);
            sigma = (sigma+sigma')/2;
            
            t = t+1;
        end
        if t == time_horizon
            break
        end
    end
end
cumreg_PSRL_LQ = cumsum(regret);
writematrix(cumreg_PSRL_LQ,'PSRL.csv');





%ULA-TSLD-LQ

for simulation = 1:simul
    disp(simulation)
    t=1;
    k=1;
    x=zeros(n,1);
    theta= zeros((n+m),n);
    data={};
    theta_mean = theta_mean_;
    sigma = sigma_;
    while 1
        disp(t)
        T = k+1;
        t_k = t;
        while 1
            arbi = ULA_Gaussian(theta,lam,theta_mean_, data, n, m, t_k);
            disp(arbi)
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
            
            if t == t_k+T-1
                u = -Gain*x+mvnrnd(zeros(n,1),0.0001*eye(n))';
            else
                u=-Gain*x;
            end
                
            cost = x'*Q*x+ u'*R*u;

            %Take expectation with repect to exact posterior
            while 1        
                arbi_ = zeros(n+m,n);
                for i =1:n
                    arbi_(:,i) = mvnrnd(theta_mean(:,i),sigma);
                       
                end
                    
                a_ = arbi_([1,2,3],:)';
                b_ = arbi_([4,5,6],:)';
                
                [S_,Gain,L] = idare(a_,b_,Q,R);
                
                if (norm(arbi_)<theta_bound)&&(trace(W*S_)<J_bound)&&((sqrt((max((eig((a_-b_*Gain)'*(a_-b_*Gain)))))) < rho))
                    break
                else
                    disp('rejection')
                end
                 
            end
            
            j = trace(W*S_);
                
            regret_ULA(t,simulation) = cost-j;
           
            
            if t==time_horizon
                break
            end

            x_prime = A*x + B*u + mvnrnd(zeros(n,1),eye(n))';
            
            data{end+1} = {x;u;x_prime};
                  
            x = x_prime;

            for i =1:n
                theta_mean(:,i) = theta_mean(:,i) + sigma*z*(x(i)-theta_mean(:,i)'*z)/(1+z'*sigma*z);
    
            end
            sigma = sigma -sigma*zeta*sigma/(1+z'*sigma*z);
            sigma = (sigma+sigma')/2;
        
            t = t+1;
        end
        k=k+1;
        if t == time_horizon         
            break
        end
    end
end
cumreg_TSLD = cumsum(regret_ULA);
writematrix(cumreg_TSLD,'TSLD.csv');

end


function [theta] = ULA_Gaussian(theta, lam, theta_mean_, data, n, m, t_k)

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

step_size = (M*min(eig(preconditioner)))/(16*L^2*max(t_k,min(eig(preconditioner))));
step_iteration = ceil(4*(log2(max(t_k,min(eig(preconditioner)))/min(eig(preconditioner)))/(M*step_size)));

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
    scaled_grad_U = inv_preconditioner*grad_U;
    
    phi = mvnrnd(phi - step_size*scaled_grad_U, 2*(step_size)*inv_preconditioner*eye((n+m)*n))';
    
end

theta = tr_phi_to_theta(phi,n,m);

end

    