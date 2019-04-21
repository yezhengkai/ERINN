function [b] = boundary_correction_2_5(dx,dz,s,srcterm,k,g);
%%function [b] = boundary_correction_2_5(dx,dz,srcterm,k,g);
%% dx is the x direction cell discretization, a vector length [nx,1];
%% dz is the z direction cell discretization, a vector length [nz,1];
%% srcterm is matrix size [Number of RHS, 4] the rows contain the coords
%of the positive and negative sources [xp zp xn zn] Sources do not need to be
%located at cell centers. This code assumes that x = 0 is the center of the
%model space and the z = 0 is the surface, and z is positve down.
%k is the fourier coeffs we are using in the 2.5D approximation
%g is the weighting coeffs we are using in the 2.5D approximation

%%Removes singualrity and decreases boundary effects by modifying source
%%term based on the anaylitical solution for a homogeneous halfspace
%%written by Adam Pidlisecky June 2005; Last modified Feb 2006.


%Initialize a few quantities for later
nx = length(dx);
nz = length(dz);
FOR = zeros(nx*nz,nx*nz);

%%First Create a grid in realspace

%build the 2d grid - numbered fromn 0 to maximum extent
z(1) = 0; for i=1:length(dz); z(i+1) = z(i)+dz(i); end;
x(1) = 0; for i=1:length(dx); x(i+1) = x(i)+dx(i); end;

%Center the grid about zero
x = shiftdim(x) - max(x)/2;

%Set surface to Z = 0
z= shiftdim(z);

%find the cell centers
xc = x(1:end-1) +dx/2;
zc = z(1:end-1) +dz/2;

%Make a couple matrices so we don't have to loop through each location
%below
[X,Z] = ndgrid(xc,zc);

U = zeros(nx*nz,size(srcterm,1));
%solve for u on this grid using average mref;

%Now we need to average the conductivity structure
area = dx(:)*dz(:)';

savg = area.*s;
savg = sum(savg(:))./sum(area(:));

%turn the warning off b/c we know there is a divide by zero, we will fix it
%later.
warning('off');

%loop over all sources
for i = 1:size(srcterm,1);

%norm of positive current electrode and 1st potential electrode    
pve1 = ((X - srcterm(i,1)).^2 + (Z - srcterm(i,2)).^2).^0.5;
%norm of negative current electrode and 1st potential electrode
nve1 = ((X - srcterm(i,3)).^2 + (Z - srcterm(i,4)).^2).^0.5;
%norm of imaginary positive current electrode and 1st potential electrode
pveimag1 = ((X - srcterm(i,1)).^2 + (Z + srcterm(i,2)).^2).^0.5;
%norm of imaginary negative current electrode and 1st potential electrode
nveimag1 = ((X - srcterm(i,3)).^2 + (Z + srcterm(i,4)).^2).^0.5;
U(:,i) = reshape(1/(savg*4*pi)*(1./pve1-1./nve1+1./pveimag1-1./nveimag1),nx*nz,1);

end
warning('on');

%%now check for singularites due to the source being on a node
for i = 1:size(srcterm,1);
    I  = find(isinf(U(:,i)));
    
    if max(size(I)) > 0;
        for j = 1:length(I);
            [a,c] = ind2sub([nx,nz], I(j));
            %%Check to see if this a surface electrode
            if c ==1
                %if it is average over 3 cells
                         U(I(j),i) = mean(U(sub2ind([nx,nz],a+1,c),i) + U(sub2ind([nx,nz],a,c+1),i)...
                             +U(sub2ind([nx,nz],a-1,c),i));
            else
                %otherwise average over 4 cells
            U(I(j),i) = mean(U(sub2ind([nx,nz],a+1,c),i) + U(sub2ind([nx,nz],a,c+1),i)...
                +  U(sub2ind([nx,nz],a-1,c),i) +  U(sub2ind([nx,nz],a,c-1),i));
            end;
        end;
    end;
end;

%now that we have the "true" potentials, we need to crank that through our
%forward operator so we can create a corrected source term
D = div2d(dx,dz);
G = grad2d(dx,dz);

%% Assemble a homogeneous forward operator
%put it all together to create operator matrix

savg = savg*ones(size(s));

R = massf2d(1./savg,dx,dz);
S = spdiags(1./diag(R),0,size(R,1),size(R,2));
%put it all together to create operator matrix

Ahomo = -D*S*G;
        
%Now we form the operator that will yield our new RHS
for j = 1:length(k);
%             
    %modify the operator based on the fourier transform
    L = Ahomo +(k(j)^2)*spdiags(mkvc(savg),0,size(Ahomo,1),size(Ahomo,2));
            
    %Assemble the forwad operator, including the Fourier
    %integration
    FOR = FOR + 0.5*g(j)*inv(L);
                  
end;


%now we get bnew by solving bnew = FOR*U;
b = FOR\U;
disp('Finished Source correction ');
       