function[v] = Qu(Q,u,srcnum)
% [v] = Qu(Q,u,srcnum)
%Selects a subset of data from the entire potential field.
%Adam Pidlisecky, modified November 2005

v = [];

for i = 1:size(u,2)
    %find q cells related to the source config
    j = find(srcnum == i)  ;
    vv = Q(j,:)*u(:,i);
    v = [v;vv];
end

