from functools import lru_cache
from scipy.integrate import quad
from evolution import generate_schedule
import numpy as np
import qibo

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! set this !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
flav = 'multi-gauss'
hep = False
if hep:
    fit_name = f'hep/{flav}'
else:
    fit_name = flav
path = f"../../../fit-results/{fit_name}/"


def load_scheduling(path):
    best_p = np.load(path+"best_p.npy")
    finalT = np.load(path+"finalT.npy")
    return best_p, finalT

best_p, finalT = load_scheduling(path)
poly, derpoly = generate_schedule(best_p)

@lru_cache
def sched(t):
    """The schedule at a time t is poly(t/finalT)"""
    return poly(t/finalT)

@lru_cache
def eigenval(t):
    """Compute the eigenvalue of the Hamiltonian at a time t"""
    s = sched(t)
    return np.sqrt(s**2 + (1-s)**2)

@lru_cache
def integral_eigen(t, n=int(1e6)):
    """Compute the integral of eigenval from 0 to t
    """
    res = quad(eigenval, 0, t)
    return res[0]

@lru_cache
def u00(t, swap = 1.0):
    """Compute the value of u00 (real and imag) at a time T"""
    if t == finalT:
        t -= 1e-2
    integral = integral_eigen(t)
    l = eigenval(t)
    s = sched(t)
    
    # Normalization for the change of basis matrix P^{-1}HP = H_diagonal so that PP^-1 = I
    normalize = 2.0/(1-s)*np.sqrt(l*(l-s))
    fac = swap*(l-s)/(1-s)
    
    # (the multiplication by t not sure where does it come from)
    ti = t/finalT
    real_part = swap*np.cos(integral)*(1+fac)/normalize
    imag_part = np.sin(integral)*(1-fac)/normalize
    
    return real_part, imag_part

@lru_cache
def u10(t):
    """Compute the value of u10 (real and imag), the offdiagonal term"""
    pr, pi = u00(t, swap=-1.0)
    return pr, pi

@lru_cache
def old_rotation_angles(t):
    x, y = u00(t)
    u, z = u10(t)

    a = x + 1j*y
    b = -u + 1j*z

    arga = np.angle(a)
    moda = np.absolute(a)
    argb = np.angle(b)
    
    # Another interpretation
    integral = I = integral_eigen(t)
    l = eigenval(t)
    s = sched(t)
    
    fac = (l-s)/(1-s)
    inside = 1 +  fac**2 + 2*fac*np.cos(2*integral)
    norma = (1-s)/2/np.sqrt(l*(l-s))
    
    new_moda = np.sqrt(inside)*norma
    
    theta = - 2 * np.arccos(moda) 
    psi = - 0.5 * np.pi - arga + argb
    phi = - arga + np.pi * 0.5 - argb
    return psi, theta, phi


@lru_cache
def sched_p(t):
    """The schedule at a time t is poly(t/finalT)"""
    return derpoly(t/finalT)/finalT

@lru_cache
def eigenvalp(l, s, sp):
    return sp*(2*s - 1)/l

@lru_cache
def n(l,s):
    return (1-s)/2/np.sqrt(l*(l-s))

@lru_cache
def nder(l, s, lp, sp):
    roote = l*(l-s)
    upder = (1-s)*(2*lp*l - lp*s - sp*l)
    inter = -sp - upder/2/roote
    return 1/2/np.sqrt(roote)*inter

@lru_cache
def f(l,s):
    return (l-s)/(1-s)

@lru_cache
def fp(l, s, lp, sp):
    return (lp-sp-lp*s+sp*l)/(1-s)**2

@lru_cache
def rotation_angles(t):
    I = integral_eigen(t)
    l = eigenval(t)
    s = sched(t)
        
    fac = f(l,s)
    
    norma = n(l,s)
    inside00 = gt = 1 +  fac**2 + 2*fac*np.cos(2*I)
    
    absu00 = norma*np.sqrt(inside00)    
    
    upf = (1-l)
    dpf = (1+l-2*s)
    sinIt = np.sin(I)
    cosIt = np.cos(I)
    
    arga = np.arctan2(sinIt*upf, cosIt*dpf)
    argb = np.arctan2(sinIt*dpf, -cosIt*upf)

    theta = -2 * np.arccos(absu00)
    psi = - 0.5 * np.pi - arga + argb
    phi = - arga + np.pi * 0.5 - argb
    
    return phi, theta, psi

@lru_cache
def derivative_rotation_angles(t):
    s = sched(t)
    l = eigenval(t)
    I = integral_eigen(t)
    
    derI = l
    sp = sched_p(t) 
    lp = eigenvalp(l, s, sp)
    
    nt = n(l, s)
    ntp = nder(l, s, lp, sp)
        
    ft = f(l,s)
    ftp = fp(l, s, lp, sp)
        
    gt = 1 + ft**2 + 2*ft*np.cos(2*I)    
    
    # Terms of the final sum for the derivative of the theta
    x1 = ntp*np.sqrt(gt)
    y1 = 2*ft*ftp
    
    y2 = 2*ftp*np.cos(2*I)
    y3 = -2*ft*np.sin(2*I)*(2*derI)
    
    dgt = (y1+y2+y3)
    
    absu00 = nt*np.sqrt(gt)
    dabsu = x1 + nt/np.sqrt(gt)*(dgt)/2.0
    darcos = 2.0/np.sqrt(1 - absu00**2)
    dtheta = darcos * dabsu
    
    # Let's do the derivative of the phi,psi
    upf = (1-l)
    dpf = (1+l-2*s)
    tanI = np.tan(I)
    
    dtan = l/np.cos(I)**2
    dfrac_01 = 2*(lp - sp + sp*l - s*lp)/upf**2
    dfrac_00 = -2*(lp - sp + sp*l - s*lp)/dpf**2
    
    inside_arga = tanI*(upf/dpf)
    inside_argb = -tanI*(dpf/upf)
    
    dinside_arga = dtan*(upf/dpf) + tanI*dfrac_00
    darga = dinside_arga / (1 + inside_arga**2)
    
    dinside_argb = -dtan*(dpf/upf) - tanI*dfrac_01
    dargb = dinside_argb / (1 + inside_argb**2)
    
    dpsi = -darga + dargb
    dphi = -darga - dargb
    
    return dphi, dtheta, dpsi

@lru_cache
def rotations_circuit(t):

    psi, theta, phi = rotation_angles(t)

    c = qibo.models.Circuit(1)
    r1 = qibo.gates.RZ(q=0, theta=psi)
    r2 = qibo.gates.RX(q=0, theta=theta)
    r3 = qibo.gates.RZ(q=0, theta=phi)
    c.add([r1, r2, r3])
    c.add(qibo.gates.M(0))

    matrix = np.dot(r3.matrix, np.dot(r2.matrix, r1.matrix))

    return c

@lru_cache
def numeric_derivative(t, h=1e-7):
    # Do the derivative with 4 points
    a1, b1, c1 = rotation_angles(t + 2*h)
    a2, b2, c2 = rotation_angles(t + h)
    a3, b3, c3 = rotation_angles(t - h)
    a4, b4, c4 = rotation_angles(t - 2*h)

    dd1 = (-a1 + 8*a2 - 8*a3 + a4)/12/h
    dd2 = (-b1 + 8*b2 - 8*b3 + b4)/12/h
    dd3 = (-c1 + 8*c2 - 8*c3 + c4)/12/h
    return dd1, dd2, dd3
    


if True:
    verbose = False
    # interesting_range = np.random.rand()*finalT
    interesting_range = np.linspace(1, 45, int(1e2))
    
    for tt in interesting_range:
        if verbose:
            print("Resultados:")
            print(rotation_angles(tt))
            print(old_rotation_angles(tt))
            print("\nDerivatives")
            
        dd1, dd2, dd3 = numeric_derivative(tt, h=1e-7)
        r1, r2, r3 = derivative_rotation_angles(tt)
        
        m1, m2, m3 = dd1/r1, dd2/r2, dd3/r3
        if verbose:
            print(m1, m2, m3)
            
        if not np.allclose([m1, m2, m3], 1.0, atol=1e-4):
            print(f"For {tt=:.3}: {m1:.8}, {m2:.8}, {m3:.8} we find instabilities in the analatic/numeric ratio!")