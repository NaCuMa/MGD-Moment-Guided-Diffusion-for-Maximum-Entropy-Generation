    ## ----------------------------------------------------- Entropy related funtions -----------------------------------------------------    
import numpy as np
import torch
import matplotlib.pyplot as plt



def kl_divergence(p, q, n_bins, bins = None, epsilon=1e-5):
   
    """Histogram estimate of the KL divergence ``KL(p || q)`` (``p`` reference).
 
    Both samples are binned into densities (with ``epsilon`` floor) and the discrete
    KL is integrated against the bin widths. If ``bins`` is None, equal-count edges
    on ``(p + q) / 2`` are used (:func:`histedges_equalN`).
 
    Parameters
    ----------
    p, q : array_like
        Samples; ``p`` is the reference distribution.
    n_bins : int
        Number of bins when ``bins`` is None.
    bins : array_like, optional
        Explicit bin edges.
    epsilon : float, optional
        Density floor, by default 1e-5.
 
    Returns
    -------
    float
        Estimated KL divergence.
    """

    
    if bins is not None:
        p = np.histogram(p, bins, range=None, density=True, weights=None)[0]+epsilon
        q = np.histogram(q, bins, range=None, density=True, weights=None)[0]+epsilon
        d_bins = bins[1:]-bins[:-1]
    else:
        #minus = min(np.min(p),np.min(q))
        #maxus = max(np.max(p),np.max(q))
        #bins = np.linspace(minus,maxus,n_bins)
        #d_bins = (maxus-minus)/n_bins

        bins = histedges_equalN((p+q)/2, n_bins)
        d_bins = bins[1:]-bins[:-1]
        
        p = np.histogram(p, bins, range=None, density=True, weights=None)[0]+epsilon
        q = np.histogram(q, bins, range=None, density=True, weights=None)[0]+epsilon
        
    return np.sum(np.where(p != 0, p * np.log(p / q), 0)*d_bins)

def entropy(p, n_bins, bins = None, epsilon=1e-5):
   
    """Histogram estimate of the differential entropy of ``p``.
 
    Bins ``p`` into a density (with ``epsilon`` floor) and integrates ``-p log p``
    against the bin widths. Equal-count edges are used when ``bins`` is None.
 
    Parameters
    ----------
    p : array_like
        Samples.
    n_bins : int
        Number of bins when ``bins`` is None.
    bins : array_like, optional
        Explicit bin edges.
    epsilon : float, optional
        Density floor, by default 1e-5.
 
    Returns
    -------
    float
        Estimated differential entropy.
    """

    
    if bins is not None:
        p = np.histogram(p, bins, range=None, density=True, weights=None)[0]+epsilon
    else:
        bins =  histedges_equalN(p, n_bins)
        p = np.histogram(p, bins, range=None, density=True, weights=None)[0]+epsilon

    d_bins = bins[1:]-bins[:-1]
    
    return np.sum(np.where(p != 0,  -np.log(p)*p, 0)*d_bins)

def histedges_equalN(x, nbin):
    """Bin edges placing (approximately) equal sample counts per bin.
 
    Quantile binning: interpolates the sorted samples at evenly spaced ranks.
 
    Parameters
    ----------
    x : array_like
        Samples.
    nbin : int
        Number of bins.
 
    Returns
    -------
    numpy.ndarray
        ``nbin + 1`` bin edges.
    """

    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))

def resample_spectrum(s, n, dim=-1):
    L = s.shape[dim]
    u = torch.arange(n, device=s.device) * (L / n)
    i0 = u.floor().long(); f = (u - i0).to(s.dtype)
    shape = [1] * s.dim(); shape[dim] = n; f = f.reshape(shape)
    return s.index_select(dim, i0 % L) * (1 - f) + s.index_select(dim, (i0 + 1) % L) * f

def compute_gaussian_entropy(x1, interpolant, t, L=None):
    if x1.dim() == 4:                       # (N, 1, H, W) -> (N, H, W)
        x1 = x1.squeeze(1)
    x1 = x1 - x1.mean(0, keepdim=True)      # dataset-mean centering
    full = x1.shape[1:]
    d = int(np.prod(full))

    if L is None:                                                          # plain full-FFT estimate
        xf = torch.fft.fftn(x1, dim=tuple(range(1, x1.dim())), norm="ortho")
        spectrum_x1 = (xf.abs()**2).mean(0)
    else:                                                                  # Welch estimate
        w = torch.hann_window(L, device=x1.device, dtype=x1.dtype)
        if x1.dim() == 2:
            xf = torch.fft.fft(x1.unfold(-1, L, L // 2) * w, norm="ortho")
            spectrum_x1 = (xf.abs()**2).mean((0, 1)) / (w**2).mean()
        else:
            w2 = w[:, None] * w[None]
            xf = torch.fft.fft2(x1.unfold(1, L, L // 2).unfold(2, L, L // 2) * w2, norm="ortho")
            spectrum_x1 = (xf.abs()**2).mean((0, 1, 2)) / (w2**2).mean()
        for dim, n in enumerate(full):                                     # back to full-FFT shape
            spectrum_x1 = resample_spectrum(spectrum_x1, n, dim=dim)
    spectrum_x1 = spectrum_x1.reshape(-1).cpu().clamp_min(1e-12)           # (d,)

    H_p_0 = (np.log(2 * np.pi) + 1) * d / 2
    t = t[:, None].to(spectrum_x1)
    a, b = {
        'Linear':     (1 - t,           t),
        'VarPreserv': ((1 - t).sqrt(),  t.sqrt()),
        'Sqrt':       (1 - t.sqrt(),    t.sqrt()),
        'Cos':        (torch.cos(torch.pi * t / 2), torch.sin(torch.pi * t / 2)),
    }[interpolant]
    return H_p_0 + torch.log(a**2 + b**2 * spectrum_x1[None]).sum(1) / 2
