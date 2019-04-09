function [y] = tranxz(x)
%3D_TRAN Summary of this function goes here
%   Detailed explanation goes here
y=zeros(size(x,3),size(x,2),size(x,1));
for i=1:size(x,3)
    for j=1:size(x,2)
        for k=1:size(x,1)
            y(i,j,k)=x(k,j,i);
        end
    end
end
end

