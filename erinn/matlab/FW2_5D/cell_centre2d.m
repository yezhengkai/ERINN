function [xc,zc] = cell_centre2d(dx,dz);
%%[xc,zc] = cell_centre(dx,dz);
%%Finds the LH coordsystem cartisien coords of the cell centers;
%%%Adam Pidlisecky, Modified July 2004

%calculate the cartesian cell centered grid for inversion
dx = mkvc(dx);
dz = mkvc(dz);
%build the 3d grid - numbered fromn 0 to maximum extent
z = [0; cumsum(dz)];
x = [0; cumsum(dx)];

%Center the grid about zero
x = shiftdim(x) - max(x)/2;

%z = shiftdim(z) - max(z)/2;
%Set surface to Z = 0
z= shiftdim(z);

%find the cell centers
xc = x(1:end-1) +dx/2;
zc = z(1:end-1) +dz/2;