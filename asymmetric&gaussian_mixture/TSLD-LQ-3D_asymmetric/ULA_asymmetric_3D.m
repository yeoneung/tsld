%sampling from asymmetric noise distribution using ULA algorithm
n=3; %3-dimension
alpha = 0;
beta = 40;
k=1;
K=10;
x=zeros(n,1);
h=(k)/(32*K^2); %ULA step size
step_iteration = ceil(640*K^2/k^2); %ULA number of step iteration
ULA(k,K,x,n,h,step_iteration)

function [data] = ULA(k,K,x, n, h, step_iteration)
data = {};
alpha = 0;
beta = 40;
C_1 = (K-k)/2*alpha^2/(beta-alpha);
D_1 = -(K-k)/2*(alpha+beta);
C_2 = -(K-k)/6*alpha^3/(beta-alpha);
D_2 = (K-k)/6*(beta^3-alpha^3)/(beta-alpha);
writematrix(zeros(n,1)','asymmetric_noise_3D.csv')
for iteration = 1:step_iteration+1000000
    
    grad_U = k*x;

    if (alpha < x(n) <beta)
        grad_U(n) = ((K-k)/(2*(beta-alpha))*(x(n))^2+(k-(K-k)*alpha/(beta-alpha))*x(n)+C_1);

    elseif (x(n)>=beta)
        grad_U(n) = (K*x(n)+D_1);
  
    end
    
    x = mvnrnd(x - h*grad_U, 2*h*eye(n))';

    
    if iteration > step_iteration
        
        writematrix(x','asymmetric_noise_3D.csv','WriteMode','append')
        data{end+1} = {x};
    end
end


end
