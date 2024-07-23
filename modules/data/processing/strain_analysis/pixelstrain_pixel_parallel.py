import numpy as np
import concurrent.futures
from scipy.signal import convolve2d
from scipy.linalg import eig

def compute_pixel(i, j, fr, xtrj, ytrj, X, Y, mask, Nneighbor, ct, st):
    if mask[i, j] and Nneighbor[i, j] > 1:
        strain_pixel = np.full((12,), np.nan)  # Storing 12 outputs: XX, YY, XY, YX, RR, CC, RC, CR, p1, p2, p1or
        dx = np.zeros((2, 4))
        dX = np.zeros((2, 4))
        tf = np.zeros(4, dtype=bool)

        neighbors = [
            ((i-1, j), 0), ((i+1, j), 1),  # Vertical neighbors
            ((i, j-1), 2), ((i, j+1), 3)   # Horizontal neighbors
        ]

        for (ni, nj), idx in neighbors:
            if 0 <= ni < mask.shape[0] and 0 <= nj < mask.shape[1] and mask[ni, nj]:
                tf[idx] = True
                dx[:, idx] = [xtrj[ni, nj, fr] - xtrj[i, j, fr], ytrj[ni, nj, fr] - ytrj[i, j, fr]]
                dX[:, idx] = [X[ni, nj] - X[i, j], Y[ni, nj] - Y[i, j]]

        if np.any(tf):
            Fave = np.linalg.lstsq(dX[:, tf].T, dx[:, tf].T, rcond=None)[0].T
            E = 0.5 * (np.dot(Fave.T, Fave) - np.eye(2))
            Rot = np.array([[ct[i, j], st[i, j]], [-st[i, j], ct[i, j]]])
            Erot = np.dot(np.dot(Rot, E), Rot.T)
            d, v = np.linalg.eig(E)

            strain_pixel[:4] = E.flatten()
            strain_pixel[4:8] = Erot.flatten()
            d_sorted = np.sort(np.diag(d))
            strain_pixel[8:10] = d_sorted[::-1]
            strain_pixel[10] = np.arctan2(v[1, 1], v[0, 1])

        return (i, j, fr, strain_pixel)
    else:
        return (i, j, fr, None)

def pixelstrain_full_parallel(X, Y, xtrj, ytrj, mask, Nneighbor, ct, st):
    Isz = X.shape
    Ntime = xtrj.shape[2]
    results = np.full((Isz[0], Isz[1], Ntime, 12), np.nan)  # 12 for each result type

    tasks = [(i, j, fr) for fr in range(Ntime) for j in range(Isz[1]) for i in range(Isz[0])]
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(compute_pixel, i, j, fr, xtrj, ytrj, X, Y, mask, Nneighbor, ct, st) for (i, j, fr) in tasks]
        for future in concurrent.futures.as_completed(futures):
            i, j, fr, result = future.result()
            if result is not None:
                results[i, j, fr, :] = result

    return results

# Call pixelstrain_full_parallel with appropriate parameters.
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
    # tmp = np.full((Isz[0], Isz[1], Ntime), np.nan)
    # strain = {
    #     'vertices': None,
    #     'faces': None,
    #     'orientation': None,
    #     'maskimage': None,
    #     'XX': tmp.copy(),
    #     'YY': tmp.copy(),
    #     'XY': tmp.copy(),
    #     'YX': tmp.copy(),
    #     'RR': tmp.copy(),
    #     'CC': tmp.copy(),
    #     'RC': tmp.copy(),
    #     'CR': tmp.copy(),
    #     'p1': tmp.copy(),
    #     'p2': tmp.copy(),
    #     'p1or': tmp.copy()
    # }

    results = pixelstrain_full_parallel(X, Y, xtrj, ytrj, mask, Nneighbor, ct, st)
    return results

from typing import Union
def pixelstrainFromVolParallelPixel(disp_vol: np.ndarray, mask:Union[np.ndarray, None]=None):
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