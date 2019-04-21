function [k,g,obj,err] = get_k_g_opt(dx,dz,srcterm,num);
%function [k,g,obj,err]= get_k_g_opt(dx,dz,srcterm,num);
%%Function for generating a selection optimized wavenumbers
%%and weighting coeff's for use with dcfw_2_5d.m. 

%dx is the x direction cell discretization, a vector length [nx,1];
%dz is the z direction cell discretization, a vector length [nz,1]; 

%%srcterm is matrix size [Number of RHS, 4] the rows contain the coords
%of the positive and negative sources [xp zp xn zn]
%%Sources and recievers do not need to be located at cell centers.
%%This code assumes that x = 0 is the center of the
%%model space and the z = 0 is the surface, and z is positve down.

%%num - scalar value indicating the number of Fourier coeffs we solve for.

%%Adam Pidlisecky, Created Nov 2005, Last Modified Feb 2006.


%set the maximum number of iterations for the optimization routine
itsmax = 25;

%Max number of radii to search over
Max_num =2000;

%Number of linesearch steps
lsnum = 10;
%Line Search parameters
%lower bound
ls_low_lim = 0.01;
%upper bound
ls_up_lim =1;


%Define observation distances
rpos =[]; rneg =[]; rpos_im = []; rneg_im = [];


%hard wired search radius for determining k and g.
Xradius = [0.1 0.5 1 5 10 20 30];
Xradius = [zeros(size(Xradius)) Xradius]';
Zradius = flipud(Xradius);



for i = 1:size(srcterm,1);
    Xr = Xradius+srcterm(i,1);
    Zr = Zradius+srcterm(i,2);
    
%norm of positive current electrode and 1st potential electrode
rpost = ((Xr-srcterm(i,1)).^2 + (Zr-srcterm(i,2)).^2).^.5;

%norm of negative current electrode and 1st potential electrode
rnegt = ((Xr-srcterm(i,3)).^2 + (Zr-srcterm(i,4)).^2).^.5;

%norm of imaginary positive current electrode and 1st potential electrode
rpos_imt = ((Xr-srcterm(i,1)).^2 + (Zr+srcterm(i,2)).^2).^.5;

%norm of imaginary negative current electrode and 1st potential electrode
rneg_imt = ((Xr-srcterm(i,3)).^2 + (Zr+srcterm(i,4)).^2).^.5;

rpos = [rpos; rpost(:)]; rneg = [rneg; rnegt(:)];
rpos_im = [rpos_im; rpos_imt(:)]; rneg_im = [rneg_im; rneg_imt(:)];
end;


%%Now we remove all non-unique radii
rtot = [rpos rneg rpos_im rneg_im];
rtot = unique(rtot,'rows');


%Trim the number of radii down to the size Max_num
tnum = length(rpos);
if tnum > Max_num
rpos=rpos(1:ceil(tnum./Max_num):end);
rneg=rneg(1:ceil(tnum./Max_num):end);
rpos_im=rpos_im(1:ceil(tnum./Max_num):end);
rneg_im=rneg_im(1:ceil(tnum./Max_num):end);
end

%initialize a starting guess for k0
k0 = logspace(-2,0.5,num);

%Calculate the A matrix
    %Set up a matrix of radii
     rinv = (1./rtot(:,1)-1./rtot(:,2)+1./rtot(:,3)-1./rtot(:,4)).^-1;
     %check for any divide by zeros and remove them
     i = find([ ~isinf(sum(1./rtot,2)+rinv(:))]);
     rtot= rtot(i,:); rinv=rinv(i);
     
       
     %Form matrices for computation
     rinv1 = rinv*ones(1,num);
     rpos1 = rtot(:,1)*ones(1,num);
     rneg1 = rtot(:,2)*ones(1,num);
     rpos_im1 = rtot(:,3)*ones(1,num);
     rneg_im1 = rtot(:,4)*ones(1,num);
     
     %Identity vector
     I = ones(size(rpos1,1),1);
     %K values matrix   
     Km = ones(size(rpos1,1),1) *(k0(:))';

    %Calculate the A matrix
    A = rinv1.*real(besselk(0,rpos1.*Km)-besselk(0,rneg1.*Km)+besselk(0,rpos_im1.*Km)-besselk(0,rneg_im1.*Km));

    %%Estimate g for the given K values
    v = A*((A'*A)\(A'*I));
    %Evaluate the objective function for the initial guess
    obj(1) = (1-v)'*(1-v);
    
    %Start counter and initialize the optimization
    its = 1; %iteration counter
    knew=k0; %updated k vector
    stop =0; %Stopping toggle incase A becomes illconditioned
    reduction = 1; %Variable for ensure sufficent decrease between iterations
                    % Optimization terminates if objective function is not
                    % reduced by at least 5% at each iteration

   while obj >1e-5 & its <itsmax & stop ==0 & reduction > 0.05; 
  
    %%Create the derivative matrix
    dvdk = zeros(length(v),num);
    for i = 1:num;
        Ktemp = Km;
        Ktemp(:,i) = 1.05*(Ktemp(:,i));
         %form an new A matrix
         A = rinv1.*real(besselk(0,rpos1.*Ktemp)-besselk(0,rneg1.*Ktemp)...
             +besselk(0,rpos_im1.*Ktemp)-besselk(0,rneg_im1.*Ktemp));
        
             L = A'*A;
        
        %%Estimate g for the given K values
       vT = A*((L)\(A'*I));
       
       %Calculate the derivative for the appropriate column
        dvdk(:,i) = (vT-v)./(Ktemp(:,i)-Km(:,i));
    end;
    
    %Apply some smallness regularization
    h = dvdk'*(I-v)+1e-8*eye(length(knew))*knew(:);
    dk = (dvdk'*dvdk+1e-8*eye(length(knew)))\h;
    
    %Perform a line-search to maximize the descent 
    for j =[1:lsnum];
        warning off;
        ls =linspace(ls_low_lim,ls_up_lim,lsnum);
    ktemp =knew(:) +ls(j)*dk(:);

    Km = ones(size(rpos1,1),1) *(ktemp(:))';
    %Matrix of ones
    %Calculate the A matrix
    A = rinv1.*real(besselk(0,rpos1.*Km)-besselk(0,rneg1.*Km)...
        +besselk(0,rpos_im1.*Km)-besselk(0,rneg_im1.*Km));
     L = A'*A;
    %%Estimate g for the given K values
  
    v = A*(L\(A'*I));
    objt = (1-v)'*(1-v);
    ls_res(j,:) = [objt ls(j)];
  
    warning on;
    end;
   
    %Find the smallest objective function from the line-search
    [b,c] = (min(ls_res(:,1)));
         
    %Create a new guess for k
    knew =knew(:) +ls(c)*dk(:);
    %eval obj funct
    Km = ones(size(rpos1,1),1) *(knew(:))';
    
    %Calculate the A matrix
    A = rinv1.*real(besselk(0,rpos1.*Km)-besselk(0,rneg1.*Km)...
        +besselk(0,rpos_im1.*Km)-besselk(0,rneg_im1.*Km));
    %%Estimate g for the given K values
      
    v = A*((A'*A)\(A'*I));
    obj(its+1)= (1-v)'*(1-v);
    reduction = obj(its)./obj(its+1)-1;         
    its = its+1;
    %Check the conditioning of the matrix
    if rcond(A'*A) < 1e-20;
        knew =knew(:) -ls(c)*dk(:);
        stop = 1;
    end;
end;

%Get the RMS fit 
err = sqrt(obj./length(rpos));
%The final k values
k =abs(knew);
Km = ones(size(rpos1,1),1) *(k(:))';
%Reform A to obtian the final g values
A = rinv1.*real(besselk(0,rpos1.*Km)-besselk(0,rneg1.*Km)...
        +besselk(0,rpos_im1.*Km)-besselk(0,rneg_im1.*Km));
%Calculate g values
g = ((A'*A)\(A'*I));

