
import numpy as np
from astropy.io import fits
from astropy.table import Table

# Create dummy data
w = np.linspace(1.0, 12.0, 1000)
# Blackbody-ish continuum (just a curve)
continuum = 100 * (1 / w**2) * np.exp(-1/w)
# Absorption feature at 1.4um (H2O)
flux = continuum.copy()
mask = (w > 1.35) & (w < 1.45)
flux[mask] *= 0.8 # 20% drops

# Add noise
noise = np.random.normal(0, 0.05 * flux, size=len(w))
flux += noise
err = 0.05 * flux

# Create Table
t = Table([w, flux, err], names=('WAVELENGTH', 'FLUX', 'ERROR'))

# Write to FITS
hdu = fits.BinTableHDU(t)
hdu.writeto('dummy.fits', overwrite=True)
print("Created dummy.fits")
