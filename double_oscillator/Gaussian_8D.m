function [] = PSRL_LQ()
n = 8;
m = 8;
simul = 10;
time_horizon = 1000;
regret_ULA = zeros(time_horizon,simul);
step_iter = zeros(time_horizon,simul);
compute_time = zeros(1,simul);

A = [[0.45 0.2 0 0 0 0 0 0];
    [0.2 0.45 0.2 0 0 0 0 0];
    [0 0.2 0.45 0.2 0 0 0 0];
    [0 0 0.2 0.45 0.2 0 0 0];
    [0 0 0 0.2 0.45 0.2 0 0];
    [0 0 0 0 0.2 0.45 0.2 0];
    [0 0 0 0 0 0.2 0.45 0.2];
    [0 0 0 0 0 0 0.2 0.45]];

B=[[0.35 0.35 0 0 0 0 0 0];
   [0.35 0.35 0.35 0 0 0 0 0];
   [0 0.35 0.35 0.35 0 0 0 0];
   [0 0 0.35 0.35 0.35 0 0 0];
   [0 0 0 0.35 0.35 0.35 0 0];
   [0 0 0 0 0.35 0.35 0.35 0];
   [0 0 0 0 0 0.35 0.35 0.35];
   [0 0 0 0 0 0 0.35 0.35]];
Q = eye(n);
R = eye(m);

%covariance of noise
W = 0.5*eye(n);

%admissible condition
theta_bound = 2000;
J_bound = 20000;
rho = 0.99;


%prior of PSRL-LQ
lam = 10;
theta_mean_ = 0.3*ones(n+m,n);
sigma_ = (1\lam)*eye((n+m));

%ULA-TSLD-LQ

for simulation = 1:simul
    tic
    disp(simulation)
    t=1;
    k=1;
    x=zeros(n,1);
    theta= zeros((n+m),n);
    data={};
    theta_mean = theta_mean_;
    while 1
        disp(t)
        T = k+1;
        t_k = t;
        while 1
            [arbi,step_iteration] = ULA_Gaussian(theta,theta_mean,lam, data, n, m, t_k);
            %disp(arbi)
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
            
            z = cat(2, x',u')';  

            cost = x'*Q*x+ u'*R*u;

            j=trace(W*S);
            regret_ULA(t,simulation) = cost-j;
            step_iter(t,simulation) = step_iteration;
            
            if t==time_horizon
                break
            end
                

            x_prime = A*x + B*u + mvnrnd(zeros(n,1),0.5*eye(n))';
            
            data{end+1} = {x;u;x_prime};
                  
            x = x_prime;
          
            t = t+1;
        end
        k=k+1;
        if t == time_horizon         
            break
        end
    end
    ti=toc;
    disp(ti)
    compute_time(1,simulation)=ti;
end
cumreg_PSRL_LQ_ULA = cumsum(regret_ULA);

writematrix(cumreg_PSRL_LQ_ULA,['6d-regret' '.csv']);
writematrix(compute_time,['6d-time' '.csv']);
writematrix(step_iter,['6d-iter' '.csv']);
figure
hold on
x = 1:time_horizon;
y = cumreg_PSRL_LQ_ULA(1:time_horizon);

%confidence interval
ts = tinv([0.025  0.975],size(y,1)-1);
err = ts(2)*std(cumreg_PSRL_LQ_ULA,0,2)/sqrt(size(y,1));
errorbar(x,y,err)

plot(cumreg_PSRL_LQ_ULA(1:time_horizon),'r-')
leg = legend('ULA-PSRL-LQ');
set(leg,'Fontsize',10)
xlabel('Horizon','Fontsize',16)
ylabel('Cumulative Regret','Fontsize',16)

end


function [theta,step_iteration] = ULA_Gaussian(theta,theta_mean, lam, data, n, m, t_k)

preconditioner = lam*eye((n+m)*n);

Theta_mean = theta_mean;
precision = lam*eye(n+m);
Theta_mean_cal = lam*Theta_mean;
for k = 1:length(data)
    x = data{k}{1};
    u = data{k}{2};
    x_prime = data{k}{3};
    z = cat(2, x',u')';
    zeta = z* z';
    for i =1:n
            Theta_mean_cal(:,i) = Theta_mean_cal(:,i)+  x_prime(i)*z;    
    end
    precision = precision + zeta;

    for i=2:n
        zeta = blkdiag(zeta, z*z');
    end
    preconditioner = preconditioner + zeta;
end
sigma = 1*(precision)\eye(n+m); %covariance
for l =1:n
    Theta_mean(:,l) = sigma*Theta_mean_cal(:,l);
end

inv_preconditioner = 1*(preconditioner)\eye((n+m)*n);

M = 1;
L = 1;

phi = tr_theta_to_phi(Theta_mean,n,m);
minimum_eig = min(eig(preconditioner));

step_size = (M*min(eig(preconditioner)))/(16*L^2*max(t_k,min(eig(preconditioner))));

step_iteration = ceil(4*(log2(max(t_k,min(eig(preconditioner)))/min(eig(preconditioner)))/(M*step_size)));

s1 = 0;
s2 = 0;

phi_mean = tr_theta_to_phi(theta_mean,n,m);
s_prior=-lam*(phi-phi_mean);

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

for i=1:n
    theta(:,i) = phi((n+m)*(i-1)+1:(n+m)*i);
end

end
    