function [phi]=tr_t(theta_,n,m)
    phi_=[];
    for j=1:n
        for i=1:n+m
            phi_(end+1)=theta_(i,j);
        end
    end
    phi=phi_';

end