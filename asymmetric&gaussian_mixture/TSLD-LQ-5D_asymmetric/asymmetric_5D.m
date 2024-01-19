
    n=5;
    m=5;
    simul=1;
    
    time_horizon=2000;
    
    regret_ULA = zeros(time_horizon,simul);
    
    A = [[0.3 0.6 0.2 0.3 0.1];
        [0 0.1 0.4 0 0.6];
        [0.1 0.5 0.3 0 0.2];
        [0.4 0 0.3 0.3 0];
        [0.3 0.3 0.1 0.4 0.4]];
    B = [[0.5 0.4 0.2 0.5 0.4];
        [0.6 0 0.3 0.1 0.3];
        [0.5 0 0 0.1 0.2];
        [0.1 0.5 0 0.2 0.4];
        [0.2 0.1 0.6 0 0]];
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
    

    %Asymmetric setting
    M=1;
    L=10;
    alpha = 0;
    beta = 40;
    noise = readmatrix('asymmetric_noise_5D.csv');
    mean = [0 0 0 0 -0.0471]';
    W = [[1 0 0 0 0];[0 1 0 0 0];[0 0 1 0 0];[0 0 0 1 0];[0 0 0 0 0.9293]];
    
    
    % Asymmetric noise setting
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
            phi_U = newton_method(phi_U,data,mean,M,L,alpha,beta,n,m,lam,1000);

            while 1
                arbi = asymmetric_ULA(mean, theta, lam, phi_U, data, M, L, n, m, t_k,alpha,beta);
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

                rn=randi(100000);
                as_noise = noise(rn,:)'-mean;
                x_prime = A*x + B*u + as_noise;
                
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
        writematrix(cumreg_PSRL_LQ_ULA,['5D-asymmetric-regret' '.csv']);
    end
    
    
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

    function [minimizer] =newton_method(pi,data,mean,M,L,alpha,beta,n,m,lam,N)
    for i =1:N
        pi = pi - (asymmetric_hess_log(pi,data,mean,M,L,alpha,beta,n,m,lam)\eye((n+m)*n))*asymmetric_grad_log(pi,data,mean,M,L,alpha,beta,n,m,lam);
    end
    minimizer = pi;
    end



            
               
                
                
                
                
                
                
                
            
