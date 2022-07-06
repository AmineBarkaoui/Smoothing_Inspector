"""Whittaker filter L-curve optimization of S and asymmetric weights."""
# pyright: reportGeneralTypeIssues=false
from math import log

import numpy
from numba import guvectorize, jit
from numba.core.types import float64, int16

#from scipy import optimize

from ._helper import lazycompile
from .ws2d import ws2d


@lazycompile(
    guvectorize(
        [(float64[:], float64, float64, float64[:], int16[:], float64[:])],
        "(n),(),(),(m) -> (n),()",
        nopython=True,
    )
)
def ws2doptlp(y, nodata, p, llas, out, lopt):
    """
    Whittaker filter L-curve optimization of S and asymmetric weights.

    Args:
        y (numpy.array): raw data array (1d, expected in float64)
        nodata (double, int): nodata value
        p (float): Envelope value for asymmetric weights
        llas (numpy.array): 1d array of s values to use for optimization
    """
    m = y.shape[0]
    w = numpy.zeros(y.shape, dtype=float64)


    n = 0
    for ii in range(m):
        if y[ii] == nodata:
            w[ii] = 0
        else:
            n += 1
            w[ii] = 1
    if n > 1:
        def curvature(llas,y,w,p):
            m = y.shape[0]
            m1 = m - 1
            m2 = m - 2
            nl = len(llas)
            i = 0
            j = 0
            p1 = 1 - p
            
            fits = numpy.zeros(nl)
            pens = numpy.zeros(nl)
            z = numpy.zeros(m)
            znew = numpy.zeros(m)
            diff1 = numpy.zeros(m1)
            wa = numpy.zeros(m)
            ww = numpy.zeros(m)
            
            # Compute L-curve
            for lix in range(nl):
                lmda = pow(10, llas[lix])
    
                for i in range(10):
                    for j in range(m):
                        y_tmp = y[j]
                        z_tmp = z[j]
                        if y_tmp > z_tmp:
                            wa[j] = p
                        else:
                            wa[j] = p1
                        ww[j] = w[j] * wa[j]
    
                    znew[:] = ws2d(y, lmda, ww)
                    z_tmp = 0.0
                    j = 0
                    for j in range(m):
                        z_tmp += abs(znew[j] - z[j])
    
                    if z_tmp == 0.0:
                        break
    
                    z[0:m] = znew[0:m]
    
                for i in range(m):
                    w_tmp = w[i]
                    y_tmp = y[i]
                    z_tmp = z[i]
                    fits[lix] += pow(w_tmp * (y_tmp - z_tmp), 2)
                fits[lix] = log(fits[lix])
    
                for i in range(m1):
                    z_tmp = z[i]
                    z2 = z[i + 1]
                    diff1[i] = z2 - z_tmp
                for i in range(m2):
                    z_tmp = diff1[i]
                    z2 = diff1[i + 1]
                    pens[lix] += pow(z2 - z_tmp, 2)
                pens[lix] = log(pens[lix])
                    
                llastep = llas[1] - llas[0]
              
                ## Compute first and second derivatives
                fits_first = ((fits[2]-fits[0])/(2*llastep))*(llas[0]-llastep-llas[1]) + fits[1]
                fits_last = ((fits[-1]-fits[-3])/(2*llastep))*(llas[-1]+llastep-llas[-2]) + fits[-2]
                fits_temp = numpy.concatenate((numpy.array([fits_first]),fits,numpy.array([fits_last])))
                fits_diff1 = (fits_temp[2:] - fits_temp[:-2])/(2*llastep)
                fits_diff2 = (fits_temp[2:] - fits + fits_temp[:-2])/(pow(llastep,2))
                
                pens_first = ((pens[2]-pens[0])/(2*llastep))*(llas[0]-llastep-llas[1]) + pens[1]
                pens_last = ((pens[-1]-pens[-3])/(2*llastep))*(llas[-1]+llastep-llas[-2]) + pens[-2]
                pens_temp = numpy.concatenate((numpy.array([pens_first]),pens,numpy.array([pens_last])))
                pens_diff1 = (pens_temp[2:] - pens_temp[:-2])/(2*llastep)
                pens_diff2 = (pens_temp[2:] - pens + pens_temp[:-2])/(pow(llastep,2))
            
                ## Curvature
                curv = numpy.divide((fits_diff1 * pens_diff2 - fits_diff2 * pens_diff1), (fits_diff1**2 + pens_diff1**2)**(1.5))
                
                return curv
        
        ## Maximum Curvature
        curv = curvature(llas,y,w,p)
        curv_id = numpy.argmax(curv)
        x1 = llas[int(numpy.amin([curv_id, len(curv)]))]
        x2 = llas[int(numpy.amax([curv_id, 0]))]
        lopt[0] = (x1 + x2) / 2#optimize.fminbound(curvature, x1, x2, args = (y, fits, pens), full_output=False, disp=False)
        lopt[0] = pow(10, lopt[0])

        z = numpy.zeros(m)
        znew = numpy.zeros(m)
        wa = numpy.zeros(m)
        ww = numpy.zeros(m)
        p1 = 1 - p

        for i in range(10):
            for j in range(m):
                y_tmp = y[j]
                z_tmp = z[j]

                if y_tmp > z_tmp:
                    wa[j] = p
                else:
                    wa[j] = p1
                ww[j] = w[j] * wa[j]

            znew[0:m] = ws2d(y, lopt[0], ww)
            z_tmp = 0.0
            j = 0
            for j in range(m):
                z_tmp += abs(znew[j] - z[j])

            if z_tmp == 0.0:
                break

            z[0:m] = znew[0:m]

        z = ws2d(y, lopt[0], ww)
        numpy.round_(z, 0, out)

    else:
        out[:] = y[:]
        lopt[0] = 0.0


@jit(nopython=True)
def _ws2doptlp(y, w, p, llas):
    """
    Whittaker filter L-curve optimization of S and asymmetric weights.

    Args:
        y (numpy.array): raw data array (1d, expected in float64)
        w (numpy.array): weights same size as y
        p (float): Envelope value for asymmetric weights
        llas (numpy.array): 1d array of s values to use for optimization
    """
    m = y.shape[0]
    def curvature(llas,y,w,p):  
        m = y.shape[0]
        m1 = m - 1
        m2 = m - 2
        nl = len(llas)
        i = 0
        j = 0
        p1 = 1 - p
    
        fits = numpy.zeros(nl, dtype=float64)
        pens = numpy.zeros(nl, dtype=float64)
        z = numpy.zeros(m, dtype=float64)
        znew = numpy.zeros(m, dtype=float64)
        diff1 = numpy.zeros(m1, dtype=float64)
        wa = numpy.zeros(m, dtype=float64)
        ww = numpy.zeros(m, dtype=float64)
    
        # Compute L-curve
        for lix in range(nl):
            lmda = pow(10, llas[lix])
    
            for i in range(10):
                for j in range(m):
                    y_tmp = y[j]
                    z_tmp = z[j]
                    if y_tmp > z_tmp:
                        wa[j] = p
                    else:
                        wa[j] = p1
                    ww[j] = w[j] * wa[j]
    
                znew[:] = ws2d(y, lmda, ww)
                z_tmp = 0.0
                j = 0
                for j in range(m):
                    z_tmp += abs(znew[j] - z[j])
    
                if z_tmp == 0.0:
                    break
    
                z[0:m] = znew[0:m]
    
            for i in range(m):
                w_tmp = w[i]
                y_tmp = y[i]
                z_tmp = z[i]
                fits[lix] += pow(w_tmp * (y_tmp - z_tmp), 2)
            fits[lix] = log(fits[lix])
    
            for i in range(m1):
                z_tmp = z[i]
                z2 = z[i + 1]
                diff1[i] = z2 - z_tmp
            for i in range(m2):
                z_tmp = diff1[i]
                z2 = diff1[i + 1]
                pens[lix] += pow(z2 - z_tmp, 2)
            pens[lix] = log(pens[lix])
    
            llastep = llas[1] - llas[0]
              
            ## Compute first and second derivatives
            fits_first = ((fits[2]-fits[0])/(2*llastep))*(llas[0]-llastep-llas[1]) + fits[1]
            fits_last = ((fits[-1]-fits[-3])/(2*llastep))*(llas[-1]+llastep-llas[-2]) + fits[-2]
            fits_temp = numpy.concatenate((numpy.array([fits_first]),fits,numpy.array([fits_last])))
            fits_diff1 = (fits_temp[2:] - fits_temp[:-2])/(2*llastep)
            fits_diff2 = (fits_temp[2:] - fits + fits_temp[:-2])/(pow(llastep,2))
            
            pens_first = ((pens[2]-pens[0])/(2*llastep))*(llas[0]-llastep-llas[1]) + pens[1]
            pens_last = ((pens[-1]-pens[-3])/(2*llastep))*(llas[-1]+llastep-llas[-2]) + pens[-2]
            pens_temp = numpy.concatenate((numpy.array([pens_first]),pens,numpy.array([pens_last])))
            pens_diff1 = (pens_temp[2:] - pens_temp[:-2])/(2*llastep)
            pens_diff2 = (pens_temp[2:] - pens + pens_temp[:-2])/(pow(llastep,2))
            
            ## Curvature
            curv = numpy.divide((fits_diff1 * pens_diff2 - fits_diff2 * pens_diff1), (fits_diff1**2 + pens_diff1**2)**(1.5))
            
            return curv
    
    ## Maximum Curvature
    curv = curvature(llas,y,w,p)
    curv_id = numpy.argmax(curv)
    x1 = llas[int(numpy.amin([curv_id, len(curv)]))]
    x2 = llas[int(numpy.amax([curv_id, 0]))]
    lopt = (x1 + x2) / 2#optimize.fminbound(curvature, x1, x2, args = (y, fits, pens), full_output=False, disp=False)
    lopt = pow(10, lopt[0])

    z = numpy.zeros(m)
    znew = numpy.zeros(m)
    wa = numpy.zeros(m)
    ww = numpy.zeros(m)
    p1 = 1 - p
    
    for i in range(10):
        for j in range(m):
            y_tmp = y[j]
            z_tmp = z[j]

            if y_tmp > z_tmp:
                wa[j] = p
            else:
                wa[j] = p1
            ww[j] = w[j] * wa[j]

        znew[0:m] = ws2d(y, lopt, ww)
        z_tmp = 0.0
        j = 0
        for j in range(m):
            z_tmp += abs(znew[j] - z[j])

        if z_tmp == 0.0:
            break

        z[0:m] = znew[0:m]

    z = ws2d(y, lopt, ww)
    return z, lopt
