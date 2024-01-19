
function [a] = grad(w,M,L,alpha,beta,n,i)
C_1 = (L-M)/2*alpha^2/(beta-alpha);
D_1 = -(L-M)/2*(alpha+beta);

grad_log_phi = M*w;

if (alpha < w(n) <beta)
    grad_log_phi(n) = ((L-M)/(2*(beta-alpha))*(w(n))^2+(M-(L-M)*alpha/(beta-alpha))*w(n)+C_1);

elseif (w(n)>=beta)
    grad_log_phi(n) = (L*w(n)+D_1);

end

a = grad_log_phi(i);
