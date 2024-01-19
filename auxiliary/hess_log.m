function [a] = hess(w,M,L,alpha,beta,n)

%C_1 = (L-M)/2*alpha^2/(beta-alpha);
%D_1 = -(L-M)/2*(alpha+beta);

hess_log_phi = M*eye(n);

if (alpha < w(n) <beta)
    hess_log_phi(n,n) = (L-M)/(beta-alpha)*(w(n))+(M-(L-M)*alpha/(beta-alpha));

elseif (w(n)>=beta)
    hess_log_phi(n,n) = L;

end

a = -hess_log_phi;
