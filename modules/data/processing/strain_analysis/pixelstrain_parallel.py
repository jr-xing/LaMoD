import numpy as np
import concurrent.futures
from scipy.signal import convolve2d
from scipy.linalg import eig

def compute_frame(fr, xtrj, ytrj, X, Y, mask, Nneighbor, Isz, ct, st):
    dx = np.zeros((2,4))
    dX = np.zeros((2,4))
    tf = np.zeros(4, dtype=bool)
    strain_frame = {
        'XX': np.full((Isz[0], Isz[1]), np.nan),
        'YY': np.full((Isz[0], Isz[1]), np.nan),
        'XY': np.full((Isz[0], Isz[1]), np.nan),
        'YX': np.full((Isz[0], Isz[1]), np.nan),
        'RR': np.full((Isz[0], Isz[1]), np.nan),
        'CC': np.full((Isz[0], Isz[1]), np.nan),
        'RC': np.full((Isz[0], Isz[1]), np.nan),
        'CR': np.full((Isz[0], Isz[1]), np.nan),
        'p1': np.full((Isz[0], Isz[1]), np.nan),
        'p2': np.full((Isz[0], Isz[1]), np.nan),
        'p1or': np.full((Isz[0], Isz[1]), np.nan)
    }
    for j in range(Isz[1]):
        for i in range(Isz[0]):
            if mask[i,j] and Nneighbor[i,j] > 1:
                dx.fill(0)
                dX.fill(0)
                tf.fill(False)

                if i-1 >= 0 and mask[i-1,j]:
                    tf[0] = True
                    dx[:,0] = [xtrj[i-1,j,fr] - xtrj[i,j,fr], ytrj[i-1,j,fr] - ytrj[i,j,fr]]
                    dX[:,0] = [X[i-1,j] - X[i,j], Y[i-1,j] - Y[i,j]]

                if i+1 < Isz[0] and mask[i+1,j]:
                    tf[1] = True
                    dx[:,1] = [xtrj[i+1,j,fr] - xtrj[i,j,fr], ytrj[i+1,j,fr] - ytrj[i,j,fr]]
                    dX[:,1] = [X[i+1,j] - X[i,j], Y[i+1,j] - Y[i,j]]

                if j-1 >= 0 and mask[i,j-1]:
                    tf[2] = True
                    dx[:,2] = [xtrj[i,j-1,fr] - xtrj[i,j,fr], ytrj[i,j-1,fr] - ytrj[i,j,fr]]
                    dX[:,2] = [X[i,j-1] - X[i,j], Y[i,j-1] - Y[i,j]]

                if j+1 < Isz[1] and mask[i,j+1]:
                    tf[3] = True
                    dx[:,3] = [xtrj[i,j+1,fr] - xtrj[i,j,fr], ytrj[i,j+1,fr] - ytrj[i,j,fr]]
                    dX[:,3] = [X[i,j+1] - X[i,j], Y[i,j+1] - Y[i,j]]

                # Average deformation gradient tensor
                # Fave = dx[:,tf] / dX[:,tf]
                # Fave = np.linalg.solve(dX[:,tf], dx[:,tf])
                Fave = np.linalg.lstsq(dX[:,tf].T, dx[:,tf].T, rcond=None)[0].T


                # X/y strain tensor
                E = 0.5 * (np.dot(Fave.T, Fave) - np.eye(2))

                # Coordinate system rotation matrix
                # (Note this is the transpose of the vector rotation matrix)
                Rot = np.array([[ct[i,j], st[i,j]], [-st[i,j], ct[i,j]]])

                # Radial/circumferential strain tensor
                Erot = np.dot(np.dot(Rot, E), Rot.T)

                # Principal strains
                # v, d = eig(E, overwrite_a=True, check_finite=False)
                d, v = np.linalg.eig(E)
                d = np.diag(d)

                # Record output
                strain_frame['XX'][i,j] = E[0,0]
                strain_frame['XY'][i,j] = E[0,1]
                strain_frame['YX'][i,j] = E[1,0]
                strain_frame['YY'][i,j] = E[1,1]

                strain_frame['RR'][i,j] = Erot[0,0]
                strain_frame['RC'][i,j] = Erot[0,1]
                strain_frame['CR'][i,j] = Erot[1,0]
                strain_frame['CC'][i,j] = Erot[1,1]

                if np.all(d == 0):
                    strain_frame['p1'][i,j] = 0
                    strain_frame['p2'][i,j] = 0
                else:
                    strain_frame['p1'][i,j] = d.flatten()[-1]
                    strain_frame['p2'][i,j] = d.flatten()[0]
                    strain_frame['p1or'][i,j] = np.arctan2(v[1,1], v[0,1])
    # for j in range(Isz[1]):
    #     for i in range(Isz[0]):
    #         if mask[i,j] and Nneighbor[i,j] > 1:
                # (Implementation of the inner loop as before...)
                # (Remember to change `strain['XX'][i,j,fr] = E[0,0]` to `strain_frame['XX'][i,j] = E[0,0]` etc.)

    return strain_frame

def pixelstrain_parallel(**kwargs):
    # Parse application data
    errid = 'invalidInput'

    # Parse input data
    api = {
        'X': kwargs.get('X', None),
        'Y': kwargs.get('Y', None),
        'mask': kwargs.get('mask', None),
        'times': kwargs.get('times', None),
        'method': kwargs.get('method', 'RSTLS'),
        'dXt': kwargs.get('dXt', None),
        'dYt': kwargs.get('dYt', None),
        'Origin': kwargs.get('Origin', None),
        'Orientation': kwargs.get('Orientation', None)
    }

    # Check X/Y/mask
    X = api['X']
    Y = api['Y']
    mask = api['mask']

    # Check times
    time = api['times']

    # Additional parameters
    Ntime = len(time)
    Isz = [mask.shape[0], mask.shape[1]]

    # Pixel trajectories
    xtrj = np.full((Isz[0], Isz[1], Ntime), np.nan)
    ytrj = np.full((Isz[0], Isz[1], Ntime), np.nan)
    for k in range(Ntime):
        dx = api['dXt'][:,:,k]
        dy = api['dYt'][:,:,k]
        xtrj[:,:,k] = X + dx.reshape(Isz)
        ytrj[:,:,k] = Y + dy.reshape(Isz)

    # Parse origin
    origin = api['Origin']
    if origin is None:
        origin = [np.mean(X[mask]), np.mean(Y[mask])]

    # Parse orientation
    theta = api['Orientation']
    if theta is None:
        theta = np.arctan2(Y-origin[1], X-origin[0])

    # Cos/sin calculation (saving computational effort)
    ct = np.cos(theta)
    st = np.sin(theta)

    # Eliminate any invalid mask locations
    h = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    tmp = np.zeros((Isz[0], Isz[1], 4), dtype=bool)
    for k in range(4):
        tmp[:,:,k] = convolve2d(mask.astype(float), h, mode='same') == 2
        h = np.rot90(h)

    mask = np.any(tmp, axis=2) & mask

    # Strain calculation
    h = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    Nneighbor = mask * convolve2d(mask.astype(float), h, mode='same')

    # Initialize output strain structure
    tmp = np.full((Isz[0], Isz[1], Ntime), np.nan)
    strain = {
        'vertices': None,
        'faces': None,
        'orientation': None,
        'maskimage': None,
        'XX': tmp.copy(),
        'YY': tmp.copy(),
        'XY': tmp.copy(),
        'YX': tmp.copy(),
        'RR': tmp.copy(),
        'CC': tmp.copy(),
        'RC': tmp.copy(),
        'CR': tmp.copy(),
        'p1': tmp.copy(),
        'p2': tmp.copy(),
        'p1or': tmp.copy()
    }

    # Strain calculation at each point
    


    # Parallel computation
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(compute_frame, range(Ntime), [xtrj]*Ntime, [ytrj]*Ntime, [X]*Ntime, [Y]*Ntime, [mask]*Ntime, [Nneighbor]*Ntime, [Isz]*Ntime, [ct]*Ntime, [st]*Ntime))

    # Merging results
    for fr, result in enumerate(results):
        for key in result:
            strain[key][:,:,fr] = result[key]

    return strain

from typing import Union
def pixelstrainFromVolParallel(disp_vol: np.ndarray, mask:Union[np.ndarray, None]=None):
    """
    Input: disp_vol: np.ndarray of shape (2, H, W, Nfr)
    """
    if mask is None:
        mask = np.abs(disp_vol[...,0]).sum(axis=0)>1e-5
    mask = mask.astype(bool)
    H, W, Nfr = disp_vol[0].shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    strain = pixelstrain_parallel(
        X=X, 
        Y=Y, 
        dXt=disp_vol[0], 
        dYt=disp_vol[1], 
        mask=mask, times=np.arange(Nfr))
    return strain