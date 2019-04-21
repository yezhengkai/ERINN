function [dobs,U] = dcfw2_5D(s,Para);
%[U] = dcfw2d(s,Para);
%Solves the equation (-Div*sigma*grad)*u =q;
%2.5D forward model using fourier cosine transform in y direction to
%produce an approximation to 3D current flow, while assuming cthe
%conductivity structure in the y direction is invarient.

%% s is the conductivity structure in 2d with dimensions size[nx, nz];
%% This means imagesc(s') would correspond to a crossection view.

%% Para is a matlab structure that contains all the information required for 
%% forward modelling. this sturcture is generated using get_2_5Dpara.


%Code written by Adam Pidlisecky, July 2005; last update Aug 2005;

dx =Para.dx;
dz =Para.dz;
%%make the sigma matrix
R = massf2d(1./s,dx,dz);
S = spdiags(1./diag(R),0,size(R,1),size(R,2));
%put it all together to create operator matrix
A = -Para.D*S*Para.G;

%initialize the solution vector
U = zeros(size(Para.b));

%%Enter a loop to solve the forward problem for all fourier coeffs
for i = 1:length(Para.k);
         
    %%Now we solve the forward problem                     
        
    %modify the operator based on the fourier transform
    L = A +(Para.k(i)^2)*spdiags(mkvc(s),0,size(A,1),size(A,2));
   
    %now integrate for U;
    U = U+Para.g(i)*(L\(0.5*Para.b));
  
end;
  disp('Finished forward calc  ');

 %%see if the Q is around and pick data otherwise return an empty vector
 try
 dobs= Qu(Para.Q,U,Para.srcnum);
 catch
     dobs = [];
 end
 