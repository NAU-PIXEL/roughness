"""Topography and shadowing module."""
import numpy as np

def make_zsurf(m,H,Lx,qr):

    """
    Return a synthetic surface with self-affine roughness.
    Much of this is detailed in this paper:
    Tevis D B Jacobs et al 2017 Surf. Topogr.: Metrol. Prop. 5 013001
    https://doi.org/10.1088/2051-672X/aa51f8

    Parameters
    ----------
    m: Size for x,y dimensions.
    H: Hurst exponent.
    Lx: Image width (m).
    qr: Rolloff wavelength in units of q (wavevector; 1/m).
    noiseflag: Set to 1 to use a precanned noise field. This is for debugging.

    Returns
    -------
    z(height) (size x size, arr): Artifical z surface with self-affine roughness.
    """
    # Use modulo to make sure the dimensions are even. If they aren't, subtract 1. Even is easier for symmetry.
    if m%2 != 0:
        m=m-1
    n = m

    # Detemine image x/y lengths. This implementation is finite but can be modified for continuous wavevectors where q->infinity
    PixelWidth = Lx/m
    Lx = m * PixelWidth  # image length in x direction
    Ly = m * PixelWidth  # image length in y direction

    # Build the x and y direction wavevectors
    qx = np.zeros((m))
    for k in range(0,m):
            qx[k] = (2*np.pi/m)*k

    # 1D FFT shift then use unwrap to adjust for dincontinuities in 2pi phase.
    qx = (np.unwrap(np.fft.fftshift(qx))-2*np.pi)/PixelWidth

    qy = np.zeros((n))
    for k in range(0,n):
            qy[k] = (2*np.pi/n)*k
    qy = (np.unwrap(np.fft.fftshift(qy))-2*np.pi)/PixelWidth

    # Create a 2D mesh of the x and y wavevectors and convert to polar coordinates.
    qxx,qyy = np.meshgrid(qx,qy)
    rho,phi = cart2pol(qxx,qyy)

    # Check if a rolloff wavelength is declared. If not call it 0.
    try:
        qr
    except NameError:
        qr=0

    # Make a 2D PSD matrix of all zeros and fill it with the polar rho values from the 2D wavevectors.
    # First line account for the rolloff by making it constant.
    # This also properly accounts for Hurst exponent by raising the wavevectors to (-2*(H+1))
    Cq=np.zeros((m,n))
    Cq[rho < qr]=qr**(-2*(H+1))
    Cq[rho >= qr]=rho[rho >= qr]**(-2*(H+1))
    # The center of the PSD is the mean. We set this to 0. This is important later when we apply conjugate symmetry.
    Cq[int(m/2),int(n/2)]=0


    # This section calculates the C1D+ PSD.
    rhof = np.floor(rho)
    J = 200 # resolution in q space (increase if you want)
    # Calculate the minimum and maximum frequencies.
    qrmin = np.log10(np.sqrt((((2*np.pi)/Lx)**2+((2*np.pi)/Ly)**2)))
    qrmax = np.log10(np.sqrt(qx[-1]**2 + qy[-1]**2)) # Nyquist
    # Make an array for q (wavevector frequency).
    q = np.floor(10**np.linspace(qrmin,qrmax,num=J))
    # preallocate a C array (spectral density) to plot against q.
    C_AVE = np.zeros(len(q))

    #Loop through and find where the rho is ~= the q and calculate the mean of the 2D PSD (Cq) at those locations.
    for j in range(0,len(q)):
        C_AVE[j]=np.nanmean(Cq[np.logical_and(rhof>q[j],rhof<=(q[j]+1))])

    # We only want data that is a number to get rid of NAN.
    ind=np.isfinite(C_AVE)
    C=C_AVE[ind]
    q=q[ind]


    # This section is transposing certain parts of the magnitude and phase to satisfy conjugate symmetry for input to ifft
    Bq = np.sqrt(Cq/(PixelWidth**2/((n*m)*((2*np.pi)**2))))

    Xquad1=0
    Xquad2=int(n/2)
    Yquad1=0
    Yquad2=int(m/2)

    Bq[Xquad1,Yquad1] = 0
    Bq[Xquad1,Yquad2] = 0
    Bq[Xquad2,Yquad2] = 0
    Bq[Xquad2,Yquad1] = 0

    Bq[1:n,1:int(m/2)] = np.rot90(Bq[1:n,int(m/2+1):m],2)
    Bq[0,1:int(m/2)] = np.rot90(Bq[0,int(m/2+1):m].reshape(-1,1),2).reshape(-1)
    Bq[int(n/2+1):n,0] = np.rot90(Bq[1:int(n/2),0].reshape(-1,1),2).reshape(-1)
    Bq[int(n/2+1):n,int(m/2)] = np.rot90(Bq[1:int(n/2),int(m/2)].reshape(-1,1),2).reshape(-1)

    # Here is where we introduce the random gaussian portion.
    noise=np.random.rand(m,n)

    Phi = (2*np.pi)*noise-np.pi

    Phi[Xquad1,Yquad1] = 0
    Phi[Xquad1,Yquad2] = 0
    Phi[Xquad2,Yquad2] = 0
    Phi[Xquad2,Yquad1] = 0

    Phi[1:n,1:int(m/2)] = -np.rot90(Phi[1:n,int(m/2+1):m],2)
    Phi[0,1:int(m/2)] = -np.rot90(Phi[0,int(m/2+1):m].reshape(-1,1),2).reshape(-1)
    Phi[int(n/2+1):n,0] = -np.rot90(Phi[1:int(n/2),0].reshape(-1,1),2).reshape(-1)
    Phi[int(n/2+1):n,int(m/2)] = -np.rot90(Phi[1:int(n/2),int(m/2)].reshape(-1,1),2).reshape(-1)

    # Now we convert back to cartesian coordinates.
    a,b = pol2cart(Bq,Phi)

    # Time to invert the FFT and generate our topography
    # Need to allocate an empty complex array similar in size to our Bq.
    out=np.empty_like(Bq,dtype='complex128')
    # NumPy complex arrays allow explicit definition of the real and imaginary components.
    out.real=a
    out.imag=b

    # Here we call the inverse FFT shift to put our low frequency stuff to the margins.
    out=np.fft.ifftshift(out)

    # And finally we call the inverse FFT.
    # Because we went through all that trouble to make the array symmetric this should return a real, float array.
    # This should also make iFFT run faster.
    outsurf = np.fft.ifft2(out)
    # But it won't do that because of rounding so we use real_if_close (imaginary is tiny).
    outsurf = np.real_if_close(outsurf)

    # Voila. A real array of height values with self-affine roughness.
    # Send it!
    return outsurf

def gradSlope(dem):
    # We don't always need slope and aspect and aspect is expensive.
    dz_dy,dz_dx = np.gradient(dem)
    return np.rad2deg(np.arctan(np.sqrt(dz_dx**2. + dz_dy**2.)))

def gradSlopeAspect(dem):
    # We use the gradient function to return the dz_dx and dz_dy.
    dz_dy,dz_dx = np.gradient(dem)
    # I'm am going to flatten the arrays to make them easily iterable for the aspect crap.
    # First I am storing the original dimensions so I can reshape upon return.
    dims=dem.shape
    dz_dy=dz_dy.flatten()
    dz_dx=dz_dx.flatten()

    # Slope is returned in degrees through simple calculation. No windowing is done (e.g. Horn's method).
    slope=np.rad2deg(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    # Preallocate an array for the aspect.
    aspect=np.rad2deg(np.arctan(dz_dy/dz_dx))

    # Loop though and deal with all of the conditions to convert the arctan output to azimuth in degrees relative to North.
    for i in range(0,len(dz_dx)):
        # If there is no slope we don't need aspect. Set to 0 because 360 is North.
        if (slope[i] == 0.0):
            aspect[i] = 0
            continue

        # Aspect is initially from -pi to pi clockwise about the x-axis. We need to translate to clockwise about Y.
        # To avoid dividing by 0 we're just going to explicitly set these to North or South based on the sign of dz_dy.
        if (dz_dx[i]) == 0:
            if (dz_dy[i]) < 0:
                aspect[i] = 180.
                continue
            else:
                aspect[i] = 360.
                continue

        # Here we make our offsets to account for counterclockwise about y axis and 0-360.
        if (dz_dx[i]) > 0:
            aspect[i] = 270. - aspect[i]
            continue
        else:
            aspect[i] = 90. - aspect[i]
            continue

    return slope.reshape(dims), aspect.reshape(dims)

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def rms(y):
    return np.sqrt(np.mean(y**2))
