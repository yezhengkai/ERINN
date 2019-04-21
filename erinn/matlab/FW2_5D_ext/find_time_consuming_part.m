addpath(genpath('F:/python_package'));

load srcloc.mat
load dxdz.mat
load s.mat
load recloc.mat
load srcnum.mat

s2 = ones(size(s));
s2(18:23, 7:22) = 0.000001;
s2(35:40, 2:30) = 0.001;
[Para1] = get_2_5Dpara(srcloc,dx,dz,s,4,recloc,srcnum);
[dobs1,U1] = dcfw2_5D(s,Para1);
[Para2] = get_2_5Dpara(srcloc,dx,dz,s2,4,recloc,srcnum);
[dobs2,U2] = dcfw2_5D(s2,Para2);

isequal(Para1.dx, Para2.dx) % same
isequal(Para1.dz, Para2.dz) % same
isequal(Para1.nx, Para2.nx) % same
isequal(Para1.nz, Para2.nz) % same
isequal(Para1.D, Para2.D) % same
isequal(Para1.G, Para2.G) % same
isequal(Para1.k, Para2.k) % same
isequal(Para1.g, Para2.g) % same
isequal(Para1.b, Para2.b) % slightly different (+- 10^-15)
isequal(Para1.Q, Para2.Q) % same
isequal(Para1.srcnum, Para2.srcnum) % same

figure; imagesc(Para1.b - Para2.b)
% ========
% In demo mode, time-consuming ranking in get_2_5Dpara
% 1. boundary_correction_2_5 => Para.b
% 2. get_k_g_opt => Para.k, Para.g
% 3. interpmat_N2_5 => Para.b
% 4. div2d => Para.D
% 5. grad2d => Para.G
% maybe we can pre-calculate all Para not only Para.Q
rmpath(genpath('F:/python_package'));