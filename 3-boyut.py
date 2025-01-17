import numpy as np
from scipy.integrate import tplquad
from scipy.optimize import root
import plotly.graph_objects as go

# Small epsilon to avoid division by zero
EPSILON = 1e-8

a = 0
b = 1
c = 0
d = 1
e = 0
f = 2

def rho(x, y, z):
    return np.sqrt(5 * x * y * z * z)

# Kısmi türevler
def partial_xf_integrand(x, y, z, xf, yf, zf):
    denom = np.sqrt((xf - x) ** 2 + (yf - y) ** 2 + (zf - z) ** 2 + EPSILON)
    return rho(x, y, z) * (xf - x) / denom

def partial_yf_integrand(x, y, z, xf, yf, zf):
    denom = np.sqrt((xf - x) ** 2 + (yf - y) ** 2 + (zf - z) ** 2 + EPSILON)
    return rho(x, y, z) * (yf - y) / denom

def partial_zf_integrand(x, y, z, xf, yf, zf):
    denom = np.sqrt((xf - x) ** 2 + (yf - y) ** 2 + (zf - z) ** 2 + EPSILON)
    return rho(x, y, z) * (zf - z) / denom

# Integral hesaplamaları
def partial_xf(xf, yf, zf):
    return tplquad(
        lambda z, y, x: partial_xf_integrand(x, y, z, xf, yf, zf),
        a, b,  # x limitleri
        lambda x: c, lambda x: d,  # y limitleri
        lambda x, y: e, lambda x, y: f  # z limitleri
    )[0]

def partial_yf(xf, yf, zf):
    return tplquad(
        lambda z, y, x: partial_yf_integrand(x, y, z, xf, yf, zf),
        a, b,  # x limitleri
        lambda x: c, lambda x: d,  # y limitleri
        lambda x, y: e, lambda x, y: f  # z limitleri
    )[0]

def partial_zf(xf, yf, zf):
    return tplquad(
        lambda z, y, x: partial_zf_integrand(x, y, z, xf, yf, zf),
        a, b,  # x limitleri
        lambda x: c, lambda x: d,  # y limtleri
        lambda x, y: e, lambda x, y: f  # z limitleri
    )[0]

# f(xf, yf, zf)'nin gradyanı
def gradient(xf, yf, zf):
    grad_xf = partial_xf(xf, yf, zf)
    grad_yf = partial_yf(xf, yf, zf)
    grad_zf = partial_zf(xf, yf, zf)
    return np.array([grad_xf, grad_yf, grad_zf])

# Kök bulma fonksiyonu
def gradient_root(vars):
    xf, yf, zf = vars
    return gradient(xf, yf, zf)

# İlk tahmin
initial_guess = [0.64, 0.64, 1.1]

# Gradyanın sıfır olduğu yer için çöz
result = root(gradient_root, initial_guess)

# Sonucu çıktılama
ft_point = result.x
print("Çözüm (xf, yf, zf):", ft_point)
print("Çözümde gradyan:", gradient_root(ft_point))

# Plotly ile görselleştirme
n = 50

x = np.linspace(a, b, n)
y = np.linspace(c, d, n)
z = np.linspace(e, f, n)

X, Y, Z = np.meshgrid(x, y, z)

X_flat = X.flatten()
Y_flat = Y.flatten()
Z_flat = Z.flatten()
density_flat = rho(X, Y, Z).flatten()

fig = go.Figure(data=go.Volume(
    x = X_flat,
    y = Y_flat,
    z = Z_flat,
    value = density_flat,
    isomin = density_flat.min(),
    isomax = density_flat.max(),
    opacity = 0.1,
    surface_count = 30,
    colorscale = 'Viridis',
))

# Fermat-Torricelli noktasında kırmızı nokta
fig.add_trace(go.Scatter3d(
    x = [ft_point[0]], 
    y = [ft_point[1]], 
    z = [ft_point[2]],
    mode = 'markers+text',
    marker = dict(size=10, color='red', symbol = 'circle'),
    text = [f"F({ft_point[0]:.2f}, {ft_point[1]:.2f}, {ft_point[2]:.2f})"],
    textposition = "top center",
    textfont = dict(color = "red", size = 12),
    name = 'Fermat-Torricelli Noktası'
))

fig.update_layout(
    scene=dict(
        xaxis_title = 'x',
        yaxis_title = 'y',
        zaxis_title = 'z',
    ),
    title='Değişken Yoğunluklu 3-Boyutlu Cisim',
)

fig.show()