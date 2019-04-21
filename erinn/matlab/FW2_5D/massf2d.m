function[S] = massf2d(s,dx,dz)
% [S] = massf(s,dx,dz)
%% s is the conductivity structure in 2d with dimensions size[nx nz];
%% dx is the x direction cell discretization, a vector length [nx,1];
%% dz is the z direction cell discretization, a vector length [nz,1];

%Adam Pidlisecky 2005; based on an implementation of E.Haber 1999 for the
%3D problem

dx = shiftdim(dx);
dz = shiftdim(dz);

Nx = length(dx)-2;
Nz = length(dz)-2;

% Number the Ax grid
nax = (Nx+1)*(Nz+2); 
GRDax = reshape(1:1:nax, (Nx+1),(Nz+2));


% Number the Az grid
naz = (Nx+2)*(Nz+1); 
GRDaz = reshape(1:1:naz, (Nx+2),(Nz+1));

% Generates the 2D grid
ex = ones(Nx+2,1);
ez = ones(Nz+2,1);

Dx =kron(dx',ez)';
Dz = kron(ex',dz)';

dA = Dx.*Dz;

%%%% Generate x Coeficients %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l = 2:Nx+2; k = 1:Nz+2;

% Avarage rho on x face
rhof = (dA(l,k).*s(l,k) + dA(l-1,k).*s(l-1,k))/2;

dVf = (dA(l,k) + dA(l-1,k))/2; 

rhof = rhof./dVf;
                         
lx = []; jx = []; kx = []; rx = [];

%% Coef (i,j,k)
lx = mkvc(GRDax); 
jx = mkvc(GRDax);
kx = mkvc(rhof);

Sx = sparse(lx,jx,kx);

%%%% Generate z Coeficients %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


l = 1:Nx+2; k = 2:Nz+2;

% Avarage rho on z face
rhof = (dA(l,k-1).*s(l,k-1) + dA(l,k).*s(l,k))/2;

dVf = (dA(l,k-1) + dA(l,k))/2;

rhof = rhof./dVf;

lz = []; jz = []; kz = [];

%% Coef (i,j,k)
lz = mkvc(GRDaz);
jz = mkvc(GRDaz);
kz = mkvc(rhof);

Sz = sparse(lz,jz,kz);
%%%% Assemble Matrix  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
Oxz = sparse(nax,naz);

S = [Sx,  Oxz; ...
     Oxz',  Sz];
