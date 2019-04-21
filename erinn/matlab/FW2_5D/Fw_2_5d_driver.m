%%Demo driver for testing the FW_2_5d package
%%The driver creates the Para structure for a borehole ert survey.
%%The included data files can be modified to suit the users input.


load srcloc.mat
load dxdz.mat
load s.mat


%First we run the code for no receiver locations, no BC correction and
%default fourier parameters
tic;
[Para] = get_2_5Dpara(srcloc,dx,dz,[],0,[],[]);% 
toc;
%%Note you can save Para, and then you need not recreate it for different
%%conductivity fields.

[dobs1,U1] = dcfw2_5D(s,Para);
%%note U1 will be a matrix dimensions [length(dx)*length(dz), #of srcterms];
%% To visualize the potential field for any source term u = reshape(U1(:,i),length(dx),length(dz))
%%u is now a 2D matrix. To plot in mapview (ie. x-axis is horizontal) use
%%imagesc(u');

%Add Fourier parameters
tic;
[Para] = get_2_5Dpara(srcloc,dx,dz,[],4,[],[]);% 
toc;
%%Note you can save Para, and then you need not recreate it for different
%%conductivity fields.

[dobs2,U2] = dcfw2_5D(s,Para);

%Now add the BC correction
tic;
[Para] = get_2_5Dpara(srcloc,dx,dz,s,4,[],[]);% 
toc;
%Run the forward model
[dobs3,U3] = dcfw2_5D(s,Para);


%Finally add the receiver locations
load recloc.mat
load srcnum.mat

tic;
[Para] = get_2_5Dpara(srcloc,dx,dz,[],4,recloc,srcnum);% 
toc;

%Run the forward model
[dobs4,U4] = dcfw2_5D(s,Para);




 