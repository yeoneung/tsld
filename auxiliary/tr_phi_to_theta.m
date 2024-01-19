function [theta]=tr_p(phi,n,m)
    theta=zeros((n+m),n);
    for i=1:n
        theta(:,i) = phi((n+m)*(i-1)+1:(n+m)*i);
    end

end