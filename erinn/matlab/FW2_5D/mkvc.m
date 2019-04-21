function[v] = mkvc(A)
% v = mkvc(A)
%Rearrange a matrix into a vector. 
%Can be substituted for v = v(:)
v = reshape(A,size(A,1)*size(A,2)*size(A,3),1);