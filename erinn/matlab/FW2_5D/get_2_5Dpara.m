function [Para] = get_2_5Dpara(srcloc,dx,dz,BC_cor,num,recloc,srcnum);
%%function [Para] = get_2_5Dpara(srcloc,dx,dz,BC_cor,num,recloc,srcnum);
%%Function generates the structure required for calculating the potential field 
%%using dcfw_2_5d.m%

 
%%srcloc is matrix size [Number of RHS, 4] the rows contain the coords
%of the positive and negative sources [xp zp xn zn]
%%Sources and recievers do not need to be located at cell centers.
%%This code assumes that x = 0 is the center of the
%%model space and the z = 0 is the surface, and z is positve down.

%dx is the x direction cell discretization, a vector length [nx,1];

%dz is the z direction cell discretization, a vector length [nz,1];

%Bc_cor enter '[]' for no correction, pass in the s matrix to apply correction.

%num - a scalar containing the number of fourier coefficents you want to
%use. Enter '0' to use the default parameters (from Xu et al, 2000)

%recloc - Enter [] if you do not want to calculate data at receiver
%locations
%%for a pole-dipole survey this is an [ndata, 2] matrix
%%containing the x,z coords of the +ve reciever
%%for a dipole-dipole survey this is an [ndata, 4] matrix
%%containing the x,z coords of the +ve and -ve reciever locations

%srcnum - a vector length(recloc) that lists the source term number that
%a given reciever corresponds to. e.g. if we have 4 receiver locations and
%the first 2 receivers correspond to the first source term, and the second
%receivers correspond to the second source term, then srcnum = [1 1 2 2].


%%Adam Pidlisecky Created Nov, 2005.

nx = length(dx);
nz = length(dz);

%%Assign grid information
Para.dx = dx;
Para.dz = dz;
Para.nx = nx;
Para.nz = nz;

%%Create Diveragnce and Gradient operators once so we don't need to calculate them
%%again
Para.D = div2d(dx,dz);
Para.G = grad2d(dx,dz);

%%optimize k and g for the given survey geometry.
if num == 0;
    disp('Using default Fourier Coeffs');
    Para.k = [0.0217102 .2161121 1.0608400 5.0765870];
    Para.g = [0.0463660 0.2365931 1.0382080 5.3648010];
 else           
    disp('Optimizing for Fourier Coeffs');
    [k,g,obj,err] = get_k_g_opt(dx,dz,srcloc,num);
    %%Assign the k and g values to Para    
    Para.k = k;
    Para.g = g;
end;


%%Create the right hand side of the forward modeling equation
%%See if we are applying the BC correction

if isempty(BC_cor);
    %no correctoion, so we interpolate the src locations onto the grid
    disp('Interpolating source locations');
    Para.b = calcRHS_2_5(dx,dz,srcloc);  
    
else           
    %Calculate the RHS with a BC coorection applied
    disp('Applying BC/Singularity correction');
    Para.b = boundary_correction_2_5(dx,dz,BC_cor,srcloc,Para.k,Para.g);
end;



%%See if we are creating a reciever term
try
%%Get the Q matrix for the observation points
Para.Q = interpmat_N2_5(dx,dz,recloc(:,1),recloc(:,2)); %%%%%

%% See if it is a dipole survey - if not the other electrode is assumed to
%% be at infinitey
    try
        
    Para.Q = Para.Q - interpmat_N2_5(dx,dz,recloc(:,3),recloc(:,4));
   
    end;

end;
%%Assign srcnumbers (empty vector if no receivers were supplied)
Para.srcnum=srcnum;
