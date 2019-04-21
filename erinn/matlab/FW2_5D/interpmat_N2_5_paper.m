function [Q] = interpmat_N2_5_paper(dx,dz,xr,zr)
% [Q] = interpmat(dx,dz,xr,yr,zr)
%
% Interpolation program for creating interpoaltion operator
% dx is the x direction cell discretization, a vector length [nx,1];
% dz is the z direction cell discretization, a vector length [nz,1];
% xr x- xlocation of the source, assuming dx is centered about zero
% zr z-location of the source assuming surface is zero, and z is positive
% down.
%Will blow up if reciever is located on the outside cell
% modified july 2005 - Adam Pidlisecky


dx = shiftdim(dx);
dz = shiftdim(dz);

%build the 3d grid - numbered fromn 0 to maximum extent
z(1) = 0; for i=1:length(dz); z(i+1) = z(i)+dz(i); end;
x(1) = 0; for i=1:length(dx); x(i+1) = x(i)+dx(i); end;

%Center the grid about zero
x = shiftdim(x) - max(x)/2;

%Set surface to Z = 0
z= shiftdim(z);

%find the cell centers
xc = x(1:end-1) +dx/2;
zc = z(1:end-1) +dz/2;

%take care of surface sources by shifting them to first cell centre
I = find(zr < min(zc));
zr(I) = min(zc);

%call linear interp scheme below

[Q] = linint(xc,zc,xr,zr);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function[Q] = linint(x,z,xr,zr)
%
% This function does local linear interpolation
% computed for each receiver point in turn
%
% calls mkvc
%
% [Q] = linint(x,z,xr,zr)
% Interpolation matrix 
%

 
nx = length(x) ;
nz = length(z) ;

np = length(xr);
Q = sparse(np,nx*nz);

for i = 1:np,

       % fprintf('Point %d\n',i); 

        [dd,im] = min(abs(xr(i)-x));
        
        if  xr(i) - x(im) >= 0,  % Point on the left 
                 ind_x(1) = im;
                 ind_x(2) = im+1;
        elseif  xr(i) - x(im) < 0,  % Point on the right
                 ind_x(1) = im-1; 
                 ind_x(2) = im;
       end;
       dx(1) = xr(i) - x(ind_x(1));
       dx(2) = x(ind_x(2)) - xr(i);

       
        [dd,im] = min(abs(zr(i) - z));
        if  zr(i) -z(im) >= 0,  % Point on the left
                 ind_z(1) = im;
                 ind_z(2) = im+1;
       elseif  zr(i) -z(im) < 0,  % Point on the right
                 ind_z(1) = im-1;
                 ind_z(2) = im;
       end;
       dz(1) = zr(i) - z(ind_z(1)); 
       dz(2) = z(ind_z(2)) - zr(i);      

       Dx =  (x(ind_x(2)) - x(ind_x(1)));
       Dz =  (z(ind_z(2)) - z(ind_z(1)));  

      
      % Build the row for the Q matrix
      v = zeros(nx, nz);

      v( ind_x(1), ind_z(1)) = (1-dx(1)/Dx)*(1-dz(1)/Dz);
      v( ind_x(1), ind_z(1)) = (1-dx(1)/Dx)*(1-dz(1)/Dz);
      v( ind_x(2), ind_z(1)) = (1-dx(2)/Dx)*(1-dz(1)/Dz);
      v( ind_x(2), ind_z(1)) = (1-dx(2)/Dx)*(1-dz(1)/Dz);
      v( ind_x(1), ind_z(2)) = (1-dx(1)/Dx)*(1-dz(2)/Dz);
      v( ind_x(1), ind_z(2)) = (1-dx(1)/Dx)*(1-dz(2)/Dz);
      v( ind_x(2), ind_z(2)) = (1-dx(2)/Dx)*(1-dz(2)/Dz);
      v( ind_x(2), ind_z(2)) = (1-dx(2)/Dx)*(1-dz(2)/Dz);

     % Insert Row into Q matrix
      Q(i,:) = mkvc(v)';

end;

