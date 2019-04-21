function[D,Dx,Dz] = div2d(dx,dz)
%[D,Dx,Dz] = div2d(dx,dz)
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

% Generates the grid
ex = ones(Nx+2,1);
ez = ones(Nz+2,1);

Dx =kron(dx',ez)';
Dz = kron(ex',dz)';



%%%%   Generate d/dx  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lx = []; jx = []; kx = [];

% Entries (l,j,k)

lx = mkvc(GRDp(2:end-1,:));
jx = mkvc(GRDax(1:end-1,:));
kx = mkvc(-1./Dx(2:end-1,:));

% Entries (l+1,j,k)

lx = [lx;lx];
jx = [jx;mkvc(GRDax(2:end,:))];
kx = [kx;-kx];

% BC at x = 0

lx = [lx;mkvc(GRDp(1,:))];
jx = [jx;mkvc(GRDax(1,:))];
kx = [kx;mkvc(1./Dx(1,:))];

% BC at x = end

lx = [lx;mkvc(GRDp(end,:))];
jx = [jx;mkvc(GRDax(end,:))];
kx = [kx;mkvc(-1./Dx(end,:))];


%%%%   Generate d/dz  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lz = []; jz = []; kz = [];

% Entries (l,j,k)

lz = mkvc(GRDp(:,2:end-1));
jz = mkvc(GRDaz(:,1:end-1));
kz = mkvc(-1./Dz(:,2:end-1));

% Entries (l+1,j,k)

lz = [lz;lz];
jz = [jz;mkvc(GRDaz(:,2:end))];
kz = [kz;-kz];

% BC on z = 0
lz = [lz; mkvc(GRDp(:,1))];
jz = [jz; mkvc(GRDaz(:,1))];
kz = [kz; mkvc(1./Dz(:,1))];

% BC on z = end
lz = [lz; mkvc(GRDp(:,end))];
jz = [jz; mkvc(GRDaz(:,end))];
kz = [kz; mkvc(-1./Dz(:,end))];

%%%%%%%% Generate the div %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Dx = sparse(lx,jx,kx,np,nax);
Dz = sparse(lz,jz,kz,np,naz);

D = [Dx,Dz];
