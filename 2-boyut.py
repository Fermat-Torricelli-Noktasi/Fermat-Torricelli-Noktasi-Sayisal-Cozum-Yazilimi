import numpy as np
from scipy.integrate import dblquad
from scipy.optimize import root
import matplotlib.pyplot as plt

# Sıfırla bölmeyi engellemek için
EPSILON = 1e-8

a = 0
b = 1
c = 0
d = 1

def rho(x, y):
    return np.sqrt(2 * x * y * y)

# Kısmi türevler
def partial_xf_integrand(x, y, xf, yf):
    denom = np.sqrt((xf - x) ** 2 + (yf - y) ** 2 + EPSILON)
    return rho(x, y) * (xf - x) / denom

def partial_xy_integrand(x, y, xf, yf):
    denom = np.sqrt((xf - x) ** 2 + (yf - y) ** 2 + EPSILON)
    return rho(x, y) * (yf - y) / denom

# Integral hesaplamaları
def partial_xf(xf, yf):
    return dblquad(
        lambda y, x: partial_xf_integrand(x, y, xf, yf),
        a, b,  # x limitleri
        lambda x: c, lambda x: d  # y limitleri
    )[0]

def partial_xy(xf, yf):
    return dblquad(
        lambda y, x: partial_xy_integrand(x, y, xf, yf),
        a, b,  # x limitleri
        lambda x: c, lambda x: d  # y limitleri
    )[0]

# f(xf, yf)'nin gradyanı
def gradient(xf, yf):
    return np.array([partial_xf(xf, yf), partial_xy(xf, yf)])

# Kök bulma fonksiyonu
def gradient_root(vars):
    xf, yf = vars
    return gradient(xf, yf)

# İlk tahmin
init = [0.5, 0.5]

# Gradyanın sıfır olduğu yer için çöz
result = root(gradient_root, init)

# Sonucu çıktılama
ft_point = result.x
print("Çözüm F(xf, yf):", ft_point)
print("Çözümde gradyan:", gradient_root(ft_point))

# pcolormesh ile görselleştirme
n = 500

x = np.linspace(a, b, n)
y = np.linspace(c, d, n)

X, Y = np.meshgrid(x, y)

F_x, F_y = ft_point[0], ft_point[1]

plt.figure(figsize=(8, 6))

density_plot = plt.pcolormesh(X, Y, rho(X, Y), shading='auto', cmap='viridis')
# Yoğunluk için renk barı
plt.colorbar(density_plot, label=r'$\rho(x, y) = \sqrt{2xy}$')
# Fermat-Torricelli noktasında kırmızı nokta
plt.scatter(F_x, F_y, color='red', s=100, label=f'F Noktası ({F_x:.4f}, {F_y:.4f})')

plt.xlabel('x')
plt.ylabel('y')
plt.title('$\\rho(x, y) = \\sqrt{2xy^2}$ Yoğunluklu Karesel Alan')
plt.legend()
plt.show()