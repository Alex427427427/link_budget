'''
Created by alexander Li on 2025/04/29
'''
import numpy as np
import matplotlib.pyplot as plt

rain_coeffs = {
    "k_h": {
        "a": [-5.33980, -0.35351, -0.23789, -0.94158],
        "b": [-0.10008, 1.26970, 0.86036, 0.64552],
        "c": [1.13098, 0.45400, 0.15354, 0.16817],
        "mk": -0.18961,
        "ck": 0.71147
    },
    "k_v": {
        "a": [-3.80595, -3.44965, -0.39902, 0.50167],
        "b": [0.56934, -0.22911, 0.73042, 1.07319],
        "c": [0.81061, 0.51059, 0.11899, 0.27195],
        "mk": -0.16398,
        "ck": 0.63297
    },
    "alpha_h": {
        "a": [-0.14318, 0.29591, 0.32177, -5.37610, 16.1721],
        "b": [1.82442, 0.77564, 0.63773, -0.96230, -3.29980],
        "c": [-0.55187, 0.19822, 0.13164, 1.47828, 3.43990],
        "ma": 0.67849,
        "ca": -1.95537
    },
    "alpha_v": {
        "a": [-0.07771, 0.56727, -0.20238, -48.2991, 48.5833],
        "b": [2.33840, 0.95545, 1.14520, 0.791669, 0.791459],
        "c": [-0.76284, 0.54039, 0.26809, 0.116226, 0.116479],
        "ma": -0.053739,
        "ca": 0.83433
    }
}

def bilinear_interp(x1, x2, y1, y2, q11, q12, q21, q22, x, y):
    """
    Perform bilinear interpolation for a given point (x, y) within the rectangle defined by (x1, y1), (x2, y2).
    
    Parameters:
    x1, x2 : float
        The x-coordinates of the rectangle's corners.
    y1, y2 : float
        The y-coordinates of the rectangle's corners.
    q11, q12, q21, q22 : float
        The values at the corners of the rectangle.
    x : float
        The x-coordinate of the point to interpolate.
    y : float
        The y-coordinate of the point to interpolate.

    Returns:
    float
        The interpolated value at (x, y).
    """
    if x1 == x2 or y1 == y2:
        raise ValueError("x1, x2 and y1, y2 must be different")
    if x < x1 or x > x2 or y < y1 or y > y2:
        raise ValueError("x and y must be within the rectangle defined by (x1, y1), (x2, y2)")
    
    
    result = ((x2-x)*(y2-y))/((x2-x1)*(y2-y1)) * q11 + \
                ((x-x1)*(y2-y))/((x2-x1)*(y2-y1)) * q21 + \
                ((x2-x)*(y-y1))/((x2-x1)*(y2-y1)) * q12 + \
                ((x-x1)*(y-y1))/((x2-x1)*(y2-y1)) * q22
    
    return result 

def k_calc(f, polarisation):
    '''
    calculate a parameter k in the rain attenuation model
    f: frequency in GHz
    polarisation: str, either 'h' for horizontal or 'v' for vertical polarisation 
    '''
    if polarisation not in ['h', 'v']:
        raise ValueError("Polarisation must be either 'h' or 'v'")
    if f < 0:
        raise ValueError("Frequency must be positive")
    # read coefficients from the dictionary
    if polarisation == 'h':
        a_list = np.array(rain_coeffs["k_h"]["a"])
        b_list = np.array(rain_coeffs["k_h"]["b"])
        c_list = np.array(rain_coeffs["k_h"]["c"])
        mk = rain_coeffs["k_h"]["mk"]
        ck = rain_coeffs["k_h"]["ck"]
    elif polarisation == 'v':
        a_list = np.array(rain_coeffs["k_v"]["a"])
        b_list = np.array(rain_coeffs["k_v"]["b"])
        c_list = np.array(rain_coeffs["k_v"]["c"])
        mk = rain_coeffs["k_v"]["mk"]
        ck = rain_coeffs["k_v"]["ck"]
    # calculate
    l10k = ck + mk*np.log10(f) + np.sum(
        a_list*np.exp(-((np.log10(f)-b_list)/c_list)**2)
    )
    k = 10**l10k
    return k

def alpha_calc(f, polarisation):
    '''
    calculate a parameter alpha in the rain attenuation model
    f: frequency in GHz
    polarisation: str, either 'h' for horizontal or 'v' for vertical polarisation 
    '''
    if polarisation not in ['h', 'v']:
        raise ValueError("Polarisation must be either 'h' or 'v'")
    if f < 0:
        raise ValueError("Frequency must be positive")
    # read coefficients from the dictionary
    if polarisation == 'h':
        a_list = np.array(rain_coeffs["alpha_h"]["a"])
        b_list = np.array(rain_coeffs["alpha_h"]["b"])
        c_list = np.array(rain_coeffs["alpha_h"]["c"])
        ma = rain_coeffs["alpha_h"]["ma"]
        ca = rain_coeffs["alpha_h"]["ca"]
    elif polarisation == 'v':
        a_list = np.array(rain_coeffs["alpha_v"]["a"])
        b_list = np.array(rain_coeffs["alpha_v"]["b"])
        c_list = np.array(rain_coeffs["alpha_v"]["c"])
        ma = rain_coeffs["alpha_v"]["ma"]
        ca = rain_coeffs["alpha_v"]["ca"]
    # calculate
    alpha = ca + ma*np.log10(f) + np.sum(
        a_list*np.exp(-((np.log10(f)-b_list)/c_list)**2)
    )
    return alpha

def rain_spec_atten(f, rr, elev_angle=np.pi/2, p_angle=np.pi/4):
    """
    Calculate the specific attenuation due to rain for a given frequency and rain rate.
    
    Parameters:
    f : float
        Frequency in GHz.
    rr : float
        Rain rate in mm/h.
    p_angle : float
        Polarization angle in radians (default is pi/4 for circular polarization). Must be between 0 and pi/2.
    elev_angle : float
        Elevation angle in radians (default is pi/2 for vertical polarization). Must be between 0 and pi/2.

    Returns:
    float
        Specific attenuation in dB/km.
    """
    if f < 0 or rr < 0:
        raise ValueError("Frequency and rain rate must be positive")
    if p_angle < 0 or p_angle > np.pi/2:
        raise ValueError("Polarization angle must be between 0 and pi/2")
    if elev_angle < 0 or elev_angle > np.pi/2:
        raise ValueError("Elevation angle must be between 0 and pi/2")
    if elev_angle == 0:
        raise ValueError("Elevation angle must be greater than 0")
    
    kh = k_calc(f, 'h')
    kv = k_calc(f, 'v')
    alpha_h = alpha_calc(f, 'h')
    alpha_v = alpha_calc(f, 'v')
    k = (kh + kv + (kh-kv)*np.cos(elev_angle)**2 * np.cos(2*p_angle))/2
    alpha = (kh*alpha_h + kv*alpha_v + (kh*alpha_h - kv*alpha_v)*np.cos(elev_angle)**2 * np.cos(2*p_angle))/(2*k)
    # Calculate specific attenuation
    gamma = k * rr**alpha
    return gamma

'''
# visualise the bilinear interpolation
x1 = 0
x2 = 10
y1 = 0
y2 = 5
q11 = 1
q12 = 1
q21 = 1
q22 = 4
x = np.linspace(x1, x2, 100)
y = np.linspace(y1, y2, 100)
X, Y = np.meshgrid(x, y)
print(X)
Z = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = bilinear_interp(x1, x2, y1, y2, q11, q12, q21, q22, X[i, j], Y[i, j])
plt.figure(figsize=(10, 10))
plt.contourf(X, Y, Z, levels=100, cmap='viridis')
plt.colorbar(label='Interpolated Value')
plt.title('Bilinear Interpolation')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.scatter([x1, x2], [y1, y2], color='red', label='Corners')
plt.scatter([x1, x2], [y1, y2], color='red', label='Corners')
plt.legend()
plt.show()
'''
