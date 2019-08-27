"""The prototype of fw25d is reimplemented in python"""
from __future__ import division, absolute_import, print_function

import warnings

import numpy as np
import scipy.sparse.linalg as spsl
from scipy import sparse
from scipy.special import kv


# MATLAB-NumPy Equivalents
# https://www.numpy.org/devdocs/user/numpy-for-matlab-users.html
# https://cheatsheets.quantecon.org/
# http://mathesaurus.sourceforge.net/matlab-numpy.html


def get_2_5Dpara(srcloc, dx, dz, BC_cor, num, recloc, srcnum):

    nx = np.max(dx.shape)
    nz = np.max(dz.shape)
    # If there are empty lists or tuples, convert them to empty ndarray
    BC_cor = np.array(BC_cor, dtype=np.float64)
    recloc = np.array(recloc, dtype=np.float64)
    srcnum = np.array(srcnum, dtype=np.float64)

    # Assign grid information
    # Create Divergence and Gradient operators once so we don't need to calculate them again
    # Assign source numbers (empty vector if no receivers were supplied)
    Para = {'dx': dx, 'dz': dz, 'nx': nx, 'nz': nz, 'D': div2d(dx, dz)[0], 'G': grad2d(dx, dz)[0], 'srcnum': srcnum}

    # optimize k and g for the given survey geometry.
    if num == 0:
        print('Using default Fourier Coeffs')
        Para['k'] = np.array([0.0217102, 0.2161121, 1.0608400, 5.0765870]).reshape(-1, 1)
        Para['g'] = np.array([0.0463660, 0.2365931, 1.0382080, 5.3648010]).reshape(-1, 1)
    else:
        print('Optimizing for Fourier Coeffs')
        k, g, obj, err = get_k_g_opt(dx, dz, srcloc, num)
        # Assign the k and g values to Para
        Para['k'] = k
        Para['g'] = g

    ## Create the right hand side of the forward modeling equation
    # See if we are applying the BC correction
    if not BC_cor.size:
        # no correction, so we interpolate the src locations onto the grid
        print('Interpolating source locations')
        Para['b'] = calcRHS_2_5(dx, dz, srcloc)
    else:
        # Calculate the RHS with a BC correction applied
        print('Applying BC/Singularity correction')
        Para['b'] = boundary_correction_2_5(dx, dz, BC_cor, srcloc, Para['k'], Para['g'])

    ## See if we are creating a receiver term
    try:
        # Get the Q matrix for the observation points
        Para['Q'] = interpmat_N2_5(dx, dz, recloc[:, 0], recloc[:, 1])
        # See if it is a dipole survey - if not the other electrode is assumed to be at infinity
        try:
            Para['Q'] = Para['Q'] - interpmat_N2_5(dx, dz, recloc[:, 2], recloc[:, 3])
        except:
            pass
    except:
        Para['Q'] = np.array([], np.float64)

    return Para


def dcfw2_5D(s, Para):

    dx = Para['dx']
    dz = Para['dz']
    # make the sigma matrix
    R = massf2d(1 / s, dx, dz)
    S = sparse.spdiags(1 / np.diag(R.toarray()), 0, R.shape[0], R.shape[1])
    # put it all together to create operator matrix
    A = -Para['D'] @ S @ Para['G']

    # initialize the solution vector
    U = np.zeros(Para['b'].shape)

    # Enter a loop to solve the forward problem for all fourier coeffs
    for i in range(np.max(Para['k'].shape)):
        # Now we solve the forward problem
        # modify the operator based on the fourier transform
        L = A + (Para['k'][i, 0] ** 2) * sparse.spdiags(s.flatten(order='F'), 0, A.shape[0], A.shape[1])
        # now integrate for U
        U = U + (Para['g'][i, 0] * spsl.spsolve(L, 0.5 * Para['b'])).toarray()
    print('Finished forward calc')

    # see if the Q is around and pick data otherwise return an empty vector
    try:
        dobs = Qu(Para['Q'], U, Para['srcnum'])
    except:
        dobs = np.array([], dtype=np.float64)

    return dobs, U


def get_k_g_opt(dx, dz, srcterm, num):

    # set the maximum number of iterations for the optimization routine
    itsmax = 25
    # Max number of radii to search over
    Max_num = 2000

    # Number of line search steps
    lsnum = 10
    # Line Search parameters
    # lower bound
    ls_low_lim = 0.01
    # upper bound
    ls_up_lim = 1

    # Define observation distances
    rpos = np.array([], np.float64)
    rneg = np.array([], np.float64)
    rpos_im = np.array([], np.float64)
    rneg_im = np.array([], np.float64)

    # hard wired search radius for determining k and g.
    Xradius = np.zeros((14, 1), np.float64)
    Xradius[7:] = np.array([0.1, 0.5, 1, 5, 10, 20, 30], ndmin=2).T
    Zradius = np.flipud(Xradius)

    for i in range(srcterm.shape[0]):
        Xr = Xradius + srcterm[i, 0]
        Zr = Zradius + srcterm[i, 1]

        # norm of positive current electrode and 1st potential electrode
        rpost = np.sqrt((np.square(Xr - srcterm[i, 0]) + np.square(Zr - srcterm[i, 1])))
        # norm of positive current electrode and 1st potential electrode
        rnegt = np.sqrt((np.square(Xr - srcterm[i, 2]) + np.square(Zr - srcterm[i, 3])))
        # norm of imaginary positive current electrode and 1st potential electrode
        rpos_imt = np.sqrt((np.square(Xr - srcterm[i, 0]) + np.square(Zr + srcterm[i, 1])))
        # norm of imaginary positive current electrode and 1st potential electrode
        rneg_imt = np.sqrt((np.square(Xr - srcterm[i, 2]) + np.square(Zr + srcterm[i, 3])))

        rpos = np.vstack((rpos, rpost.reshape(-1, 1))) if rpos.size else rpost
        rneg = np.vstack((rneg, rnegt.reshape(-1, 1))) if rneg.size else rnegt
        rpos_im = np.vstack((rpos_im, rpos_imt.reshape(-1, 1))) if rpos_im.size else rpos_imt
        rneg_im = np.vstack((rneg_im, rneg_imt.reshape(-1, 1))) if rneg_im.size else rneg_imt

    # Now we remove all non-unique radii
    rtot = np.hstack((rpos, rneg, rpos_im, rneg_im))
    rtot = np.unique(rtot, axis=0)

    # Trim the number of radii down to the size Max_num
    tnum = np.max(rpos.shape)
    if tnum > Max_num:
        step = int(np.ceil(tnum / Max_num))
        rpos = rpos[::step]
        rneg = rneg[::step]
        rpos_im = rpos_im[::step]
        rneg_im = rneg_im[::step]

    # initialize a starting guess for k0
    k0 = np.logspace(-2, 0.5, num).reshape(1, -1)

    ## Calculate the A matrix
    # Set up a matrix of radii
    rinv = 1 / (1 / rtot[:, 0] - 1 / rtot[:, 1] + 1 / rtot[:, 2] - 1 / rtot[:, 3])
    # check for any divide by zeros and remove them
    i = ~np.isinf(np.sum(1 / rtot, axis=1) + rinv)
    rtot = rtot[i, :]
    rinv = rinv[i].reshape(-1, 1)

    # Form matrices for computation
    rinv1 = rinv @ np.ones((1, num))
    rpos1 = rtot[:, 0].reshape(-1, 1) @ np.ones((1, num))
    rneg1 = rtot[:, 1].reshape(-1, 1) @ np.ones((1, num))
    rpos_im1 = rtot[:, 2].reshape(-1, 1) @ np.ones((1, num))
    rneg_im1 = rtot[:, 3].reshape(-1, 1) @ np.ones((1, num))

    # Identity vector
    I = np.ones((rpos1.shape[0], 1))
    # K values matrix
    Km = np.ones((rpos1.shape[0], 1))  @ k0

    # Calculate the A matrix
    A = rinv1 * np.real(kv(0, rpos1 * Km) - kv(0, rneg1 * Km) + kv(0, rpos_im1 * Km) - kv(0, rneg_im1 * Km))

    ## Estimate g for the given K values
    L = A.conj().T @ A
    v = A @ np.linalg.solve(L, A.conj().T @ I)
    # initialize the array of objective function
    obj = np.full(itsmax + 1, np.inf)
    # Evaluate the objective function for the initial guess
    obj[0] = (1 - v).conj().T @ (1 - v)

    # Start counter and initialize the optimization
    its = 1  # iteration counter
    knew = k0  # updated k vector
    stop = 0  # Stopping toggle in case A becomes ill-conditioned
    reduction = 1  # Variable for ensure sufficient decrease between iterations
    # Optimization terminates if objective function is not reduced by at least 5% at each iteration
    while obj[its - 1] > 1e-5 and its < itsmax and stop == 0 and reduction > 0.05:
        ## Create the derivative matrix
        dvdk = np.zeros((np.max(v.shape), num))
        for i in range(num):
            Ktemp = Km.copy()
            Ktemp[:, i] = 1.05 * Ktemp[:, i]
            # form a new A matrix
            A = rinv1 * np.real(kv(0, rpos1 * Ktemp) - kv(0, rneg1 * Ktemp)
                                + kv(0, rpos_im1 * Ktemp) - kv(0, rneg_im1 * Ktemp))
            ## Estimate g for the given K values
            L = A.conj().T @ A
            vT = A @ np.linalg.solve(L, A.conj().T @ I)
            # Calculate the derivative for the appropriate column
            dvdk[:, i] = (vT - v).flatten() / (Ktemp[:, i] - Km[:, i])

        # Apply some smallness regularization
        len_knew = np.max(knew.shape)
        h = dvdk.conj().T @ (I - v) + 1e-8 * np.eye(len_knew) @ knew.reshape(-1, 1)
        dk = np.linalg.solve(dvdk.conj().T @ dvdk + 1e-8 * np.eye(len_knew), h)

        # Perform a line-search to maximize the descent
        for j in range(lsnum):

            ls = np.linspace(ls_low_lim, ls_up_lim, lsnum)
            ktemp = knew.reshape(-1, 1) + ls[j] * dk.reshape(-1, 1)
            # matrix of ones
            Km = np.ones((rpos1.shape[0], 1)) @ ktemp.reshape(1, -1)
            # calculate the A matrix
            A = rinv1 * np.real(kv(0, rpos1 * Km) - kv(0, rneg1 * Km)
                                + kv(0, rpos_im1 * Km) - kv(0, rneg_im1 * Km))

            ## Estimate g for the given K values
            L = A.conj().T @ A
            v = A @ np.linalg.solve(L, A.conj().T @ I)
            objt = (1 - v).conj().T @ (1 - v)
            if j == 0:
                ls_res = np.hstack((objt, ls[j].reshape(-1, 1)))
            else:
                ls_res = np.vstack((ls_res, np.hstack((objt, ls[j].reshape(-1, 1)))))

        # Find the smallest objective function from the line-search
        b, c = ls_res[:, 0].min(0), ls_res[:, 0].argmin(0)

        # Create a new guess for k
        knew = knew.reshape(-1, 1) + ls[c] * dk.reshape(-1, 1)
        # eval obj function
        Km = np.ones((rpos1.shape[0], 1)) @ knew.reshape(1, -1)

        # calculate the A matrix
        A = rinv1 * np.real(kv(0, rpos1 * Km) - kv(0, rneg1 * Km)
                            + kv(0, rpos_im1 * Km) - kv(0, rneg_im1 * Km))
        ## Estimate g for the given K values
        L = A.conj().T @ A
        v = A @ np.linalg.solve(L, A.conj().T @ I)

        obj[its] = (1 - v).conj().T @ (1 - v)
        reduction = obj[its - 1] / obj[its] - 1
        its = its + 1
        # Check the conditioning of the matrix
        if np.linalg.cond(A.conj().T @ A) > 1e+20:
            knew = knew.reshape(-1, 1) - ls[c] @ dk.reshape(-1, 1)
            stop = 1

    # get the RMS fit
    err = np.sqrt(obj / np.max(rpos.shape))
    # the final k values
    k = np.abs(knew)
    Km = np.ones((rpos1.shape[0], 1)) @ knew.reshape(1, -1)
    # Reform A to obtain the final g values
    A = rinv1 * np.real(kv(0, rpos1 * Km) - kv(0, rneg1 * Km)
                        + kv(0, rpos_im1 * Km) - kv(0, rneg_im1 * Km))
    g = np.linalg.solve(A.conj().T @ A, A.conj().T @ I)

    return k, g, obj, err


def calcRHS_2_5(dx, dz, src):

    nx = np.max(dx.shape)
    nz = np.max(dz.shape)

    # find the area of all the cells by taking the kronecker product
    da = dx.reshape(-1, 1) @ dz.reshape(1, -1)
    da = da.reshape(-1, 1, order='F')

    ## Loop over sources - where sources are a dipole
    # allocate space
    nh = nx * nz
    q = sparse.lil_matrix((nh, src.shape[0]), dtype=np.float64)

    print('Calc RHS (for source)')
    # if there id more than one source
    for k in range(src.shape[0]):
        # interpolate the location of the sources to the nearest cell nodes
        Q = interpmat_N2_5(dx, dz, np.array([[src[k, 0]]]), np.array([[src[k, 1]]]))
        Q = Q - interpmat_N2_5(dx, dz, np.array([[src[k, 2]]]), np.array([[src[k, 3]]]))
        q[:, k] = Q.reshape(-1, 1, order='F') / da

    return q.tocsc()


def boundary_correction_2_5(dx, dz, s, srcterm, k, g):

    # Initialize a few quantities for later
    nx = np.max(dx.shape)
    nz = np.max(dz.shape)
    # FOR = np.zeros((nx * nz, nx * nz))
    FOR = sparse.csc_matrix((nx * nz, nx * nz), dtype=np.float64)

    ## First Create a grid in real space
    # build the 2d grid - numbered from 0 to maximum extent
    z = np.insert(dz.reshape(-1, 1).cumsum(), 0, 0)
    x = np.insert(dx.reshape(-1, 1).cumsum(), 0, 0)

    # Center the grid about zero
    tmp_x, _ = shiftdim(x)
    x = tmp_x - x.max() / 2
    # Set surface to Z = 0
    z, _ = shiftdim(z)

    # find the cell centers
    xc = x[:-1, :] + dx / 2
    zc = z[:-1, :] + dz / 2

    # Make a couple matrices so we don't have to loop through each location below
    Z, X = np.meshgrid(zc, xc)
    U = np.zeros((nx * nz, srcterm.shape[0]))
    # solve for u on this grid using average mref

    # Now we need to average the conductivity structure
    area = dx.reshape(-1, 1) @ dz.reshape(1, -1)
    savg = area * s
    savg = savg.sum() / area.sum()

    # turn the warning off b/c we know there is a divide by zero, we will fix it later.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered")
        # loop over all sources
        for i in range(srcterm.shape[0]):
            # norm of positive current electrode and 1st potential electrode
            pve1 = ((X - srcterm[i, 0]) ** 2 + (Z - srcterm[i, 1]) ** 2) ** 0.5
            # norm of negative current electrode and 1st potential electrode
            nve1 = ((X - srcterm[i, 2]) ** 2 + (Z - srcterm[i, 3]) ** 2) ** 0.5
            # norm of imaginary positive current electrode and 1st potential electrode
            pveimag1 = ((X - srcterm[i, 0]) ** 2 + (Z + srcterm[i, 1]) ** 2) ** 0.5
            # norm of imaginary negative current electrode and 1st potential electrode
            nveimag1 = ((X - srcterm[i, 2]) ** 2 + (Z + srcterm[i, 3]) ** 2) ** 0.5
            U[:, i] = np.reshape(1 / (savg * 4 * np.pi) * (1 / pve1 - 1 / nve1 + 1 / pveimag1 - 1 / nveimag1), (nx*nz,), order='F')

    # now check for singularities due to the source being on a node
    for i in range(srcterm.shape[0]):
        row = np.where(np.isinf(U[:, i]))[0]
        if row.size > 0:
            for j in range(row.size):
                a, c = np.unravel_index(row[j], (nx, nz), order='F')
                # Check to see if this a surface electrode
                if c == 1:
                    # if it is average over 3 cells
                    U[row[j], i] = np.mean(U[np.ravel_multi_index([a + 1, c], [nx, nz], order='F'), i]
                                           + U[np.ravel_multi_index([a, c + 1], [nx, nz], order='F'), i]
                                           + U[np.ravel_multi_index([a - 1, c], [nx, nz], order='F'), i])
                else:
                    # otherwise average over 4 cells
                    U[row[j], i] = np.mean(U[np.ravel_multi_index([a + 1, c], [nx, nz], order='F'), i]
                                           + U[np.ravel_multi_index([a, c + 1], [nx, nz], order='F'), i]
                                           + U[np.ravel_multi_index([a - 1, c], [nx, nz], order='F'), i]
                                           + U[np.ravel_multi_index([a, c - 1], [nx, nz], order='F'), i])

    # now that we have the "true" potentials, we need to crank that through our
    # forward operator so we can create a corrected source term
    D, _, _ = div2d(dx, dz)
    G, _, _ = grad2d(dx, dz)

    ## Assemble a homogeneous forward operator
    # put it all together to create operator matrix
    savg = savg * np.ones(s.shape)

    R = massf2d(1 / savg, dx, dz)
    S = sparse.spdiags(1 / np.diag(R.toarray()), 0, R.shape[0], R.shape[1])

    Ahomo = -D @ S @ G

    # Now we form the operator that will yield our new RHS
    for j in range(np.max(k.shape)):
        # modify the operator based on the fourier transform
        L = Ahomo + (k[j, 0] ** 2) * sparse.spdiags(savg.flatten(order='F'), 0, Ahomo.shape[0], Ahomo.shape[1])
        # Assemble the forward operator, including the Fourier integration
        FOR = FOR + 0.5 * g[j, 0] * spsl.inv(L.tocsc())

    # now we get bnew by solving bnew = FOR * U
    b = spsl.spsolve(FOR, sparse.csc_matrix(U))  # should be accelerated
    print('Finished Source correction ')

    return b


def interpmat_N2_5(dx, dz, xr, zr):

    dx, _ = shiftdim(dx)
    dz, _ = shiftdim(dz)

    # build the 3d grid - numbered from 0 to maximum extent
    z = np.vstack((0, dz.cumsum(axis=0, dtype=np.float64)))
    x = np.vstack((0, dx.cumsum(axis=0, dtype=np.float64)))
    # z = np.zeros((np.max(dz.shape) + 1, 1))
    # for i in range(np.max(dz.shape)):
    #     z[i + 1] = z[i] + dz[i]
    # x = np.zeros((np.max(dx.shape) + 1, 1))
    # for i in range(np.max(dx.shape)):
    #     x[i + 1] = x[i] + dx[i]

    # Center the grid about zero
    tmp_x, _ = shiftdim(x)
    x = tmp_x - x.max() / 2

    # Set surface to Z = 0
    z, _ = shiftdim(z)

    # find the cell centers
    xc = x[:-1] + dx / 2
    zc = z[:-1] + dz / 2

    # take care of surface sources by shifting them to first cell centre
    zr[zr < zc.min()] = zc.min()

    # call linear interp scheme
    Q = linint(xc, zc, xr, zr)

    return Q


def linint(x, z, xr, zr):
    """This function does local linear interpolation computed for each receiver point in turn

    Parameters
    ----------
    x : np.ndarray

    z : np.ndarray

    xr : np.ndarray

    zr : np.ndarray


    Returns
    -------
    Q : scipy.sparse.csr_matrix
        Interpolation matrix

    """

    nx = np.max(x.shape)
    nz = np.max(z.shape)
    num_p = np.max(xr.shape)
    Q = sparse.lil_matrix((num_p, nx * nz), dtype=np.float64)
    for i in range(num_p):

        im = np.argmin(abs(xr[i] - x))
        if xr[i] - x[im] >= 0:
            # Point on the left
            ind_x = [im, im + 1]
        elif xr[i] - x[im] < 0:
            # Point on the right
            ind_x = [im - 1, im]
        dx = [xr[i] - x[ind_x[0]], x[ind_x[1]] - xr[i]]

        im = np.argmin(abs(zr[i] - z))
        if zr[i] - z[im] >= 0:
            # Point on the left
            ind_z = [im, im + 1]
        elif zr[i] - z[im] < 0:
            # Point on the right
            ind_z = [im - 1, im]
        dz = [zr[i] - z[ind_z[0]], z[ind_z[1]] - zr[i]]

        Dx = x[ind_x[1]] - x[ind_x[0]]
        Dz = z[ind_z[1]] - z[ind_z[0]]

        # Build the row for the Q matrix
        v = np.zeros((nx, nz))

        v[ind_x[0], ind_z[0]] = (1 - dx[0] / Dx) * (1 - dz[0] / Dz)
        v[ind_x[1], ind_z[0]] = (1 - dx[1] / Dx) * (1 - dz[0] / Dz)
        v[ind_x[0], ind_z[1]] = (1 - dx[0] / Dx) * (1 - dz[1] / Dz)
        v[ind_x[1], ind_z[1]] = (1 - dx[1] / Dx) * (1 - dz[1] / Dz)

        # Insert Row into Q matrix
        Q[i, :] = v.reshape(1, -1, order='F')

    return Q


def div2d(dx, dz):

    dx, _ = shiftdim(dx)
    dz, _ = shiftdim(dz)
    Nx = np.max(dx.shape) - 2
    Nz = np.max(dz.shape) - 2

    # Number the phi grid
    num_p = (Nx + 2) * (Nz + 2)
    tmp = np.arange(0, num_p)
    GRDp = tmp.reshape(Nx + 2, Nz + 2, order='F')

    # Number the Ax grid
    nax = (Nx + 1) * (Nz + 2)
    tmp = np.arange(0, nax)
    GRDax = tmp.reshape(Nx + 1, Nz + 2, order='F')

    # Number the Az grid
    naz = (Nx + 2) * (Nz + 1)
    tmp = np.arange(0, naz)
    GRDaz = tmp.reshape(Nx + 2, Nz + 1, order='F')

    # Generates the grid
    ex = np.ones((Nx + 2, 1))
    ez = np.ones((Nz + 2, 1))
    Dx = np.kron(dx.conj().T, ez).conj().T  # maybe use np.kron(ez.T, dx)
    Dz = np.kron(ex.conj().T, dz).conj().T  # maybe use np.kron(dz.T, ex)


    ## Generate d/dx
    # Entries (l, j, k)
    lx = GRDp[1:-1, :].reshape(-1, 1, order='F')
    jx = GRDax[0:-1, :].reshape(-1, 1, order='F')
    kx = (-1 / Dx[1:-1, :]).reshape(-1, 1, order='F')

    # Entries (l + 1, j, k)
    lx = np.vstack((lx, lx))
    jx = np.vstack((jx, GRDax[1:, :].reshape(-1, 1, order='F')))
    kx = np.vstack((kx, -kx))

    # BC at x = 0
    lx = np.vstack((lx, GRDp[0, :].reshape(-1, 1, order='F')))
    jx = np.vstack((jx, GRDax[0, :].reshape(-1, 1, order='F')))
    kx = np.vstack((kx, (1 / Dx[0, :]).reshape(-1, 1, order='F')))

    # BC at x = end
    lx = np.vstack((lx, GRDp[-1, :].reshape(-1, 1, order='F'))).flatten()
    jx = np.vstack((jx, GRDax[-1, :].reshape(-1, 1, order='F'))).flatten()
    kx = np.vstack((kx, (-1 / Dx[-1, :]).reshape(-1, 1, order='F'))).flatten()


    ## Generate d/dz
    # Entries (l, j, k)
    lz = GRDp[:, 1:-1].reshape(-1, 1, order='F')
    jz = GRDaz[:, 0:-1].reshape(-1, 1, order='F')
    kz = (-1 / Dz[:, 1:-1]).reshape(-1, 1, order='F')

    # Entries (l + 1, j, k)
    lz = np.vstack((lz, lz))
    jz = np.vstack((jz, GRDaz[:, 1:].reshape(-1, 1, order='F')))
    kz = np.vstack((kz, -kz))

    # BC at z = 0
    lz = np.vstack((lz, GRDp[:, 0].reshape(-1, 1, order='F')))
    jz = np.vstack((jz, GRDaz[:, 0].reshape(-1, 1, order='F')))
    kz = np.vstack((kz, (1 / Dz[:, 0]).reshape(-1, 1, order='F')))

    # BC at z = end
    lz = np.vstack((lz, GRDp[:, -1].reshape(-1, 1, order='F'))).flatten()
    jz = np.vstack((jz, GRDaz[:, -1].reshape(-1, 1, order='F'))).flatten()
    kz = np.vstack((kz, (-1 / Dz[:, -1]).reshape(-1, 1, order='F'))).flatten()

    ## Generate the div
    Dx = sparse.csc_matrix((kx, (lx, jx)), shape=(num_p, nax), dtype=np.float64)
    Dz = sparse.csc_matrix((kz, (lz, jz)), shape=(num_p, naz), dtype=np.float64)
    D = sparse.hstack([Dx, Dz])

    return D, Dx, Dz


def grad2d(dx, dz):

    dx, _ = shiftdim(dx)
    dz, _ = shiftdim(dz)
    Nx = np.max(dx.shape) - 2
    Nz = np.max(dz.shape) - 2

    # Number the phi grid
    num_p = (Nx + 2) * (Nz + 2)
    tmp = np.arange(0, num_p)
    GRDp = tmp.reshape(Nx + 2, Nz + 2, order='F')

    # Number the Ax grid
    nax = (Nx + 1) * (Nz + 2)
    tmp = np.arange(0, nax)
    GRDax = tmp.reshape(Nx + 1, Nz + 2, order='F')

    # Number the Az grid
    naz = (Nx + 2) * (Nz + 1)
    tmp = np.arange(0, naz)
    GRDaz = tmp.reshape(Nx + 2, Nz + 1, order='F')

    # Generates the grid
    ex = np.ones((Nx + 2, 1))
    ez = np.ones((Nz + 2, 1))
    Dx = np.kron(dx.conj().T, ez).conj().T  # maybe use np.kron(ez.T, dx)
    Dz = np.kron(ex.conj().T, dz).conj().T  # maybe use np.kron(dz.T, ex)

    ## Generate d/dx
    # Entries (l, j, k)
    lx = GRDax.reshape(-1, 1, order='F')
    jx = GRDp[0:-1, :].reshape(-1, 1, order='F')
    kx = (-2 / (Dx[0:-1, :] + Dx[1:, :])).reshape(-1, 1, order='F')

    # Entries (l + 1, j, k)
    lx = np.vstack((lx, lx)).flatten()
    jx = np.vstack((jx, GRDp[1:, :].reshape(-1, 1, order='F'))).flatten()
    kx = np.vstack((kx, -kx)).flatten()

    ## Generate d/dz
    # Entries (l, j, k)
    lz = GRDaz.reshape(-1, 1, order='F')
    jz = GRDp[:, 0:-1].reshape(-1, 1, order='F')
    kz = (-2 / (Dz[:, 0:-1] + Dz[:, 1:])).reshape(-1, 1, order='F')

    # Entries (l + 1, j, k)
    lz = np.vstack((lz, lz)).flatten()
    jz = np.vstack((jz, GRDp[:, 1:].reshape(-1, 1, order='F'))).flatten()
    kz = np.vstack((kz, -kz)).flatten()

    ## Generate the grad
    Gx = sparse.csc_matrix((kx, (lx, jx)), shape=(nax, num_p), dtype=np.float64)
    Gz = sparse.csc_matrix((kz, (lz, jz)), shape=(naz, num_p), dtype=np.float64)
    G = sparse.vstack([Gx, Gz])

    return G, Gx, Gz


def massf2d(s, dx, dz):

    dx, _ = shiftdim(dx)
    dz, _ = shiftdim(dz)

    Nx = np.max(dx.shape) - 2
    Nz = np.max(dz.shape) - 2

    # Number the Ax grid
    nax = (Nx + 1) * (Nz + 2)
    tmp = np.arange(0, nax)
    GRDax = tmp.reshape(Nx + 1, Nz + 2, order='F')

    # Number the Az grid
    naz = (Nx + 2) * (Nz + 1)
    tmp = np.arange(0, naz)
    GRDaz = tmp.reshape(Nx + 2, Nz + 1, order='F')

    # Generates the 2D grid
    ex = np.ones((Nx + 2, 1))
    ez = np.ones((Nz + 2, 1))
    Dx = np.kron(dx.conj().T, ez).conj().T
    Dz = np.kron(ex.conj().T, dz).conj().T

    dA = Dx * Dz

    #### Generate x coefficients ####
    # l = np.arange(1, Nx + 2)
    # k = np.arange(0, Nz + 2)
    # Average rho on x face
    rhof = (dA[1:, :] * s[1:, :] + dA[:-1, :] * s[:-1, :]) / 2
    dVf = (dA[1:, :] + dA[:-1, :]) / 2
    rhof = rhof / dVf

    ## Coef (i, j, k)
    lx = GRDax.flatten(order='F')
    jx = GRDax.flatten(order='F')
    kx = rhof.flatten(order='F')

    Sx = sparse.csc_matrix((kx, (lx, jx)), dtype=np.float64)

    #### Generate z coefficients ####
    # Average rho on x face
    rhof = (dA[:, :-1] * s[:, :-1] + dA[:, 1:] * s[:, 1:]) / 2
    dVf = (dA[:, :-1] + dA[:, 1:]) / 2
    rhof = rhof / dVf

    ## Coef (i, j, k)
    lz = GRDaz.flatten(order='F')
    jz = GRDaz.flatten(order='F')
    kz = rhof.flatten(order='F')

    Sz = sparse.csc_matrix((kz, (lz, jz)), dtype=np.float64)

    #### Assemble Matrix ####
    # Oxz = sparse.csr_matrix((nax, naz), dtype=np.float64)
    S = sparse.bmat([[Sx, None], [None, Sz]])

    return S


def cell_centre2d(dx, dz):
    """Finds the LH coordsystem cartisien coords of the cell centers

    Parameters
    ----------
    dx : np.ndarray

    dz : np.ndaeeay


    Returns
    -------
    xc : np.ndarray

    zc : np.ndarray

    """

    # calculate the cartesian cell centered grid for inversion
    dx = dx.reshape(-1, 1)
    dz = dz.reshape(-1, 1)

    # build the 3d grid - numbered fromn 0 to maximum extent
    z = np.vstack((0, dz.cumsum(axis=0, dtype=np.float64)))
    x = np.vstack((0, dx.cumsum(axis=0, dtype=np.float64)))

    # Center the grid about zero
    tmp_x, _ = shiftdim(x)
    x = tmp_x - x.max() / 2

    # Set surface to Z = 0
    z, _ = shiftdim(z)

    # find the cell centers
    xc = x[:-1] + dx / 2
    zc = z[:-1] + dz / 2

    return xc, zc


def Qu(Q, u, srcnum):
    """Selects a subset of data from the entire potential field.

    Parameters
    ----------
    Q : sparse matrix

    u : np.ndarray

    srcnum : np.ndarray

    Returns
    -------
    v : np.ndarray

    """

    v = np.array([], dtype=np.float64)
    for i in range(u.shape[1]):
        # find q cells related to the source config
        j = np.flatnonzero(srcnum == i)
        vv = Q[j, :] @ u[:, i].reshape(-1, 1)
        v = np.vstack((v, vv)) if v.size else vv

    return v


def shiftdim(x, n=None, nargout=2):
    # Same function as shiftdim in MATLAB

    size = x.shape
    if n is None:
        for i in range(len(x.shape)):
            n = i
            if x.shape[i] != 1:
                break
    elif n > 0:
        n = np.fmod(n, np.ndim(x))

    nshifts = n
    if n is None or n == 0:
        b = x
        nshift = 0
    elif n > 0:
        b = x.reshape(x.shape[n:])
    else:
        b = x.reshape(tuple(1 for _ in range(-n)) + size)

    b = b.reshape(b.shape[0], 1) if len(b.shape) == 1 else b

    if nargout > 1:
        return b, nshifts
    else:
        return b


if __name__ == '__main__':

    import os
    import scipy.io as sio

    matfile_path = './solution_from_matlab'

    srcloc = sio.loadmat(os.path.join(matfile_path, 'srcloc.mat'))
    dxdz = sio.loadmat(os.path.join(matfile_path, 'dxdz.mat'))
    s = sio.loadmat(os.path.join(matfile_path, 's.mat'))
    recloc = sio.loadmat(os.path.join(matfile_path, 'recloc.mat'))
    srcnum = sio.loadmat(os.path.join(matfile_path, 'srcnum.mat'))
    solution1 = sio.loadmat(os.path.join(matfile_path, 'solution1.mat'))
    solution2 = sio.loadmat(os.path.join(matfile_path, 'solution2.mat'))
    solution3 = sio.loadmat(os.path.join(matfile_path, 'solution3.mat'))
    solution4 = sio.loadmat(os.path.join(matfile_path, 'solution4.mat'))

    srcloc = srcloc['srcloc']
    dx = dxdz['dx']
    dz = dxdz['dz']
    s = s['s']
    recloc = recloc['recloc']
    srcnum = srcnum['srcnum']

    # First we run the code for no receiver locations, no BC correction and default fourier parameters
    # Note you can save Para, and then you need not recreate it for different conductivity fields.
    Para1 = get_2_5Dpara(srcloc, dx, dz, [], 0, [], [])
    dobs1, U1 = dcfw2_5D(s, Para1)
    # note U1 will be a matrix dimensions [dx.size * dz.size, number of source terms]
    # Because dobs1 and U1 are the same as the matlab version, the index order is fortran-style.
    # To visualize the potential field for any source term u = U1[:, i].reshape(np.max(dz.shape), np.max(dx.shape))
    # u is now a 2D matrix. To plot in map view (ie. x-axis is horizontal) use
    # plt.imshow(u)

    # Check if the results of matlab and python are equal
    print('Para1.dx is equal: ', np.array_equal(Para1['dx'], solution1['Para1']['dx'][0, 0]))
    print('Para1.dz is equal: ', np.array_equal(Para1['dz'], solution1['Para1']['dz'][0, 0]))
    print('Para1.D is equal: ', np.array_equal(Para1['D'].toarray(), solution1['Para1']['D'][0, 0].toarray()))
    print('Para1.G is equal: ', np.array_equal(Para1['G'].toarray(), solution1['Para1']['G'][0, 0].toarray()))
    print('Para1.k is equal: ', np.array_equal(Para1['k'], solution1['Para1']['k'][0, 0].T))
    print('Para1.g is equal: ', np.array_equal(Para1['g'], solution1['Para1']['g'][0, 0].T))
    print('Para1.b is equal: ', np.array_equal(Para1['b'].toarray(), solution1['Para1']['b'][0, 0].toarray()))
    if not Para1['srcnum'].size:
        print('Para1.srcnum is empty: ', True if not solution1['Para1']['srcnum'][0, 0].size else False)
    else:
        print('Para1.srcnum is equal: ', np.array_equal(Para1['srcnum'], solution1['Para1']['srcnum'][0, 0]))
    if not dobs1.size:
        print('dobs1 is empty: ', True if not solution1['dobs1'].size else False)
    else:
        print('dobs1 is equal: ', np.array_equal(dobs1, solution1['dobs1']))
        print('dobs1 is close: ', np.allclose(dobs1, solution1['dobs1']))
    print('U1 is equal: ', np.array_equal(U1, solution1['U1']))
    print('U1 is close: ', np.allclose(U1, solution1['U1']))

    # Add Fourier parameters
    Para2 = get_2_5Dpara(srcloc, dx, dz, [], 4, [], [])
    dobs2, U2 = dcfw2_5D(s, Para2)

    # Check if the results of matlab and python are equal
    print('Para2.dx is equal: ', np.array_equal(Para2['dx'], solution2['Para2']['dx'][0, 0]))
    print('Para2.dz is equal: ', np.array_equal(Para2['dz'], solution2['Para2']['dz'][0, 0]))
    print('Para2.D is equal: ', np.array_equal(Para2['D'].toarray(), solution2['Para2']['D'][0, 0].toarray()))
    print('Para2.G is equal: ', np.array_equal(Para2['G'].toarray(), solution2['Para2']['G'][0, 0].toarray()))
    print('Para2.k is equal: ', np.array_equal(Para2['k'], solution2['Para2']['k'][0, 0]))
    print('Para2.k is close: ', np.allclose(Para2['k'], solution2['Para2']['k'][0, 0]))
    print('Para2.g is equal: ', np.array_equal(Para2['g'], solution2['Para2']['g'][0, 0]))
    print('Para2.g is close: ', np.allclose(Para2['g'], solution2['Para2']['g'][0, 0]))
    print('Para2.b is equal: ', np.array_equal(Para2['b'].toarray(), solution2['Para2']['b'][0, 0].toarray()))
    print('Para2.b is close: ', np.allclose(Para2['b'].toarray(), solution2['Para2']['b'][0, 0].toarray()))
    if not Para2['srcnum'].size:
        print('Para2.srcnum is empty: ', True if not solution2['Para2']['srcnum'][0, 0].size else False)
    else:
        print('Para2.srcnum is equal: ', np.array_equal(Para2['srcnum'], solution2['Para2']['srcnum'][0, 0]))
    if not dobs2.size:
        print('dobs2 is empty: ', True if not solution2['dobs2'].size else False)
    else:
        print('dobs2 is equal: ', np.array_equal(dobs2, solution2['dobs2']))
        print('dobs2 is close: ', np.allclose(dobs2, solution2['dobs2']))
    print('U2 is equal: ', np.array_equal(U2, solution2['U2']))
    print('U2 is close: ', np.allclose(U2, solution2['U2']))

    # Add the BC correction
    Para3 = get_2_5Dpara(srcloc, dx, dz, s, 4, [], [])
    dobs3, U3 = dcfw2_5D(s, Para3)

    # Check if the results of matlab and python are equal
    print('Para3.dx is equal: ', np.array_equal(Para3['dx'], solution3['Para3']['dx'][0, 0]))
    print('Para3.dz is equal: ', np.array_equal(Para3['dz'], solution3['Para3']['dz'][0, 0]))
    print('Para3.D is equal: ', np.array_equal(Para3['D'].toarray(), solution3['Para3']['D'][0, 0].toarray()))
    print('Para3.G is equal: ', np.array_equal(Para3['G'].toarray(), solution3['Para3']['G'][0, 0].toarray()))
    print('Para3.k is equal: ', np.array_equal(Para3['k'], solution3['Para3']['k'][0, 0]))
    print('Para3.k is close: ', np.allclose(Para3['k'], solution3['Para3']['k'][0, 0]))
    print('Para3.g is equal: ', np.array_equal(Para3['g'], solution3['Para3']['g'][0, 0]))
    print('Para3.g is close: ', np.allclose(Para3['g'], solution3['Para3']['g'][0, 0]))
    print('Para3.b is equal: ', np.array_equal(Para3['b'].toarray(), solution3['Para3']['b'][0, 0]))
    print('Para3.b is close: ', np.allclose(Para3['b'].toarray(), solution3['Para3']['b'][0, 0]))
    if not Para3['srcnum'].size:
        print('Para3.srcnum is empty: ', True if not solution3['Para3']['srcnum'][0, 0].size else False)
    else:
        print('Para3.srcnum is equal: ', np.array_equal(Para3['srcnum'], solution3['Para3']['srcnum'][0, 0]))
    if not dobs3.size:
        print('dobs3 is empty: ', True if not solution3['dobs3'].size else False)
    else:
        print('dobs3 is equal: ', np.array_equal(dobs3, solution3['dobs3']))
        print('dobs3 is close: ', np.allclose(dobs3, solution4['dobs3']))
    print('U3 is equal: ', np.array_equal(U3, solution3['U3']))
    print('U3 is close: ', np.allclose(U3, solution3['U3']))

    # Add the receiver locations
    Para4 = get_2_5Dpara(srcloc, dx, dz, [], 4, recloc, srcnum - 1)  # The index in python starts from 0
    dobs4, U4 = dcfw2_5D(s, Para4)

    # Check if the results of matlab and python are equal
    print('Para4.dx is equal: ', np.array_equal(Para4['dx'], solution4['Para4']['dx'][0, 0]))
    print('Para4.dz is equal: ', np.array_equal(Para4['dz'], solution4['Para4']['dz'][0, 0]))
    print('Para4.D is equal: ', np.array_equal(Para4['D'].toarray(), solution4['Para4']['D'][0, 0].toarray()))
    print('Para4.G is equal: ', np.array_equal(Para4['G'].toarray(), solution4['Para4']['G'][0, 0].toarray()))
    print('Para4.k is equal: ', np.array_equal(Para4['k'], solution4['Para4']['k'][0, 0]))
    print('Para4.k is close: ', np.allclose(Para4['k'], solution4['Para4']['k'][0, 0]))
    print('Para4.g is equal: ', np.array_equal(Para4['g'], solution4['Para4']['g'][0, 0]))
    print('Para4.g is close: ', np.allclose(Para4['g'], solution4['Para4']['g'][0, 0]))
    print('Para4.b is equal: ', np.array_equal(Para4['b'].toarray(), solution4['Para4']['b'][0, 0].toarray()))
    print('Para4.b is close: ', np.allclose(Para4['b'].toarray(), solution4['Para4']['b'][0, 0].toarray()))
    print('Para4.Q is equal: ', np.array_equal(Para4['Q'].toarray(), solution4['Para4']['Q'][0, 0].toarray()))
    if not Para4['srcnum'].size:
        print('Para4.srcnum is empty: ', True if not solution4['Para4']['srcnum'][0, 0].size else False)
    else:
        print('Para4.srcnum is equal: ', np.array_equal(Para4['srcnum'], solution4['Para4']['srcnum'][0, 0] - 1))
    if not dobs4.size:
        print('dobs4 is empty: ', True if not solution4['dobs4'].size else False)
    else:
        print('dobs4 is equal: ', np.array_equal(dobs4, solution4['dobs4']))
        print('dobs4 is close: ', np.allclose(dobs4, solution4['dobs4']))
    print('U4 is equal: ', np.array_equal(U4, solution4['U4']))
    print('U4 is close: ', np.allclose(U4, solution4['U4']))

    print('Done.')
