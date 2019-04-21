function[q] = calcRHS_2_5(dx,dz,src);
% [q] = calcRHS(dx,dz,src)
%% dx is the x direction cell discretization, a vector length [nx,1];
%% dz is the z direction cell discretization, a vector length [nz,1];
%% src is matrix size [Number of RHS, 4] the rows contain the coords
%of the positive and negative sources [xp zp xn zn] Sources do not need to be
%located at cell centers. This code assumes that x = 0 is the center of the
%model space and the z = 0 is the surface, and z is positve down.

% this algoritm interpolates the source locations onto the grid defined by
% dx and dz, then it assembles things into an RHS vector for computation of
% the potential field in dcfw2dmod
% Adam Pidlisecky, Aug 2005

nx = length(dx);
nz = length(dz);
%find the area of all the cells by taking the kronecker product
da = dx(:)*dz(:)';
da =da(:);

%%%%%%%% Loop over sources - where sources are a dipole %%%%%%%%%%%
% allocate space
nh = nx*nz;
q = spalloc(nh,size(src,1),size(src,1)*16 );

fprintf('   Calc RHS (for source)   ');

%if there is more than one source   
for k=1:size(src,1);
    %interpolate the location of the sources to the nearest cell nodes
    Q= interpmat_N2_5(dx,dz,src(k,1),src(k,2)); %%%%%
    Q = Q - interpmat_N2_5(dx,dz,src(k,3),src(k,4));
    
    q(:,k) = mkvc(Q)./da;
 
end;
disp('Done ');