function[G,Gx,Gz] = grad2d(dx,dz)
% [G,Gx,Gz] = grad2d(dx,dz)
%% dx is the x direction cell discretization, a vector length [nx,1];
%% dz is the z direction cell discretization, a vector length [nz,1];
%Adam Pidlisecky 2005; based on an implementation of E.Haber 1999 for the
%3D problem

dx = shiftdim(dx);
dz = shiftdim(dz);

Nx = length(dx)-2;
Nz = length(dz)-2;

% Number the phi grid 
np = (Nx+2)*(Nz+2);
GRDp = reshape(1:1:np,Nx+2,Nz+2);

% Number the Ax grid
nax = (Nx+1)*(Nz+2); 
GRDax = reshape(1:1:nax, (Nx+1),(Nz+2));

% Number the Az grid
naz = (Nx+2)*(Nz+1); 
GRDaz = reshape(1:1:naz, (Nx+2),(Nz+1));

%%%%   Generate d/dx  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lx = []; jx = []; kx = [];

% Generate grid
ex = ones(Nx+2,1);
ez = ones(Nz+2,1);
Dx =kron(dx',ez)';
% Entries (l,j,k)

lx = mkvc(GRDax);
jx = mkvc(GRDp(1:end-1,:));
kx = mkvc(-2./(Dx(1:end-1,:) + Dx(2:end,:)));

% Entries (l+1,j,k)

lx = [lx; lx];
jx = [jx;mkvc(GRDp(2:end,:))];
kx = [kx;-kx];


%%%%   Generate d/dz  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lz = []; jz = []; kz = [];
Dz = kron(ex',dz)';
% Entries (l,j,k)

lz = mkvc(GRDaz);
jz = mkvc(GRDp(:,1:end-1));
kz = mkvc(-2./(Dz(:,1:end-1) + Dz(:,2:end)));;

% Entries (l,j,k+1)

lz = [lz; lz];
jz = [jz;mkvc(GRDp(:,2:end))];
kz = [kz; -kz];

Gx = sparse(lx,jx,kx,nax,np);
Gz = sparse(lz,jz,kz,naz,np);

G = [Gx;Gz];

