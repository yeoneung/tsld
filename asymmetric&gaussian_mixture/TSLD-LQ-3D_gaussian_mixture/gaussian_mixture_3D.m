
    n=3;
    m=3;
    simul=1;
    
    time_horizon=2000;
    
    regret_ULA = zeros(time_horizon,simul);
    
    A = [[0.3 0.1 0.2];[0.1 0.4 0];[0 0.7 0.6]];
    B = [[0.5 0.4 0.5];[0.6 0.3 0];[0.3 0 0.2]];
    Q = 2*eye(n);
    R = eye(m);
    
    %admissible condition
    theta_bound = 20;
    J_bound = 20000;
    rho = 0.99;

    data = {};
    
    % Initial prior
    lam = 5;
    phi_mean=zeros([(n+m)*n 1]);
    phi_var=(1\lam)*eye((n+m)*n);   
       
    % Gaussian mixture setting
    gm_a = [1/2,1/2,1/2]';
    r=1;
    mu = [gm_a';-gm_a'];
    cov = [r,r,r];
    gm = gmdistribution(mu,cov);
    W = r*eye(n) + gm_a*gm_a';    

    for simulation = 1:simul
        disp(simulation)
        t=1;
        t_k=0;
        k=1;
        x=zeros(n,1);
        theta= zeros((n+m),n);
        while 1
            disp(t)
            T = k+1;
            t_k = t;
            
            %argmax of U
            phi_U = zeros((n+m)*n,1);
            phi_U = newton_method(phi_U,data,gm_a,r,n,m,lam,1000);
                        
            while 1
                arbi = gaussian_mixture_ULA(gm_a, r, theta, lam, phi_U, data, n, m, t_k);
                disp(arbi)
                a=arbi((1:n),:)';
                b=arbi((n+1:n+m),:)';
                [S,Gain,l] = idare(a,b,Q,R);
                if (norm(arbi)<theta_bound)&&(trace(W*S)<J_bound)&&(max(eig((a-b*Gain)'*(a-b*Gain)))<rho)                   
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
                z=[x',u']';
                    
                cost = x'*Q*x+ u'*R*u;
                j = trace(W*S);
                    
                regret_ULA(t,simulation) = cost-j;

                if t==time_horizon
                    break
                end
                
                x_prime = A*x + B*u + random(gm)';
                
                data{end+1} = {x;u;x_prime};
            
                x = x_prime;
            
                t = t+1;
            end
            k=k+1;
            if t == time_horizon
                break
            end
        end
        cumreg_PSRL_LQ_ULA = cumsum(regret_ULA);
        writematrix(cumreg_PSRL_LQ_ULA,['3D-gaussian_mixture-regret' '.csv']);
    end

    figure
    hold on
    x = 1:time_horizon;
    y = ave_PSRL_LQ_ULA(1:time_horizon);
    
    %confidence interval
    ts = tinv([0.025  0.975],size(y,1)-1);
    err = ts(2)*std(cumreg_PSRL_LQ_ULA,0,2)/sqrt(size(y,1));
    errorbar(x,y,err)
    
    plot(cumreg_PSRL_LQ_ULA(1:time_horizon),'r-')
    leg = legend('ULA-PSRL-LQ');
    set(leg,'Fontsize',10)
    xlabel('Horizon','Fontsize',16)
    ylabel('Cumulative Regret','Fontsize',16)

    function [minimizer] =newton_method(pi,data,gm_a,r,n,m,lam,N)
    for i =1:N
        pi = pi - (gaussian_mixture_hess_log(pi,data,gm_a,r,n,m,lam)\eye((n+m)*n))*gaussian_mixture_grad_log(pi,data,gm_a,r,n,m,lam);
    end
    minimizer = pi;
    end



            
               
                
                
                
                
                
                
                
            
