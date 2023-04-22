import numpy as np
from math import exp, pi, sqrt, cos, sin, log
import matplotlib.pyplot as plt

x_a = 1
print("x_a:")
print(x_a)
print("--------")
x_b = exp(pi/sqrt(3))
print("x_b:")
print(x_b)
print("--------")
def C(x,y):
    C1 = cos(sqrt(3)/2 * log(x))/sqrt(x)
    C2 = cos(sqrt(3)/2 * log(y))/sqrt(y)
    C3 = sin(sqrt(3)/2 * log(x))/sqrt(x)
    C4 = sin(sqrt(3)/2 * log(y))/sqrt(y)
    return np.matrix('{} {};{} {}'.format(C1, C3, C2, C4))

matrix = C(x_a, x_b)
print("matrix:")
print(matrix)
print("--------")
Y_a = 1
Y_b = 2
def Y(x,y):
    Y1 = x
    Y2 = y
    return np.matrix('{};{}'.format(Y1, Y2))

Y = Y(Y_a, Y_b)

inverse_array = np.linalg.inv(matrix)
print("inverse_array:")
print(inverse_array)
print("--------")
C = np.dot(inverse_array, Y)
print("C:")
print(C)
print("--------")
C1 = 1
#Проверяем наш C2
C2 = 2 * exp(pi/(2*sqrt(3)))
print("C2:")
print(C2)
print("--------")
k = 1/(exp(pi/sqrt(3))-1)
print("k:")
print(k)
print("--------")
b = 1 - k
print("b:")
print(b)
print("--------")
W = pi/(exp(pi/sqrt(3))-1)
print("W:")
print(W)
print("--------")
B = pi - W
print("B:")
print(B)


print("--------")
X_1 = x_a + (x_b - x_a)/6
X_2 = X_1 + (x_b - x_a)/6
X_3 = X_2 + (x_b - x_a)/6
X_4 = X_3 + (x_b - x_a)/6
X_5 = X_4 + (x_b - x_a)/6
X_6 = X_5 + (x_b - x_a)/6
print(X_1)
print(X_2)
print(X_3)
print(X_4)
print(X_5)
print(X_6)
print("--------")
A11 = sin(1 * (W * X_1 + B)) * (1 / (X_1 * X_1) - 1 * 1 * W * W)
A12 = sin(1 * (W * X_2 + B)) * (1 / (X_2 * X_2) - 1 * 1 * W * W)
A13 = sin(1 * (W * X_3 + B)) * (1 / (X_3 * X_3) - 1 * 1 * W * W)
A14 = sin(1 * (W * X_4 + B)) * (1 / (X_4 * X_4) - 1 * 1 * W * W)
A15 = sin(1 * (W * X_5 + B)) * (1 / (X_5 * X_5) - 1 * 1 * W * W)
A21 = sin(2 * (W * X_1 + B)) * (1 / (X_1 * X_1) - 2 * 2 * W * W)
A22 = sin(2 * (W * X_2 + B)) * (1 / (X_2 * X_2) - 2 * 2 * W * W)
A23 = sin(2 * (W * X_3 + B)) * (1 / (X_3 * X_3) - 2 * 2 * W * W)
A24 = sin(2 * (W * X_4 + B)) * (1 / (X_4 * X_4) - 2 * 2 * W * W)
A25 = sin(2 * (W * X_5 + B)) * (1 / (X_5 * X_5) - 2 * 2 * W * W)
A31 = sin(3 * (W * X_1 + B)) * (1 / (X_1 * X_1) - 3 * 3 * W * W)
A32 = sin(3 * (W * X_2 + B)) * (1 / (X_2 * X_2) - 3 * 3 * W * W)
A33 = sin(3 * (W * X_3 + B)) * (1 / (X_3 * X_3) - 3 * 3 * W * W)
A34 = sin(3 * (W * X_4 + B)) * (1 / (X_4 * X_4) - 3 * 3 * W * W)
A35 = sin(3 * (W * X_5 + B)) * (1 / (X_5 * X_5) - 3 * 3 * W * W)
A41 = sin(4 * (W * X_1 + B)) * (1 / (X_1 * X_1) - 4 * 4 * W * W)
A42 = sin(4 * (W * X_2 + B)) * (1 / (X_2 * X_2) - 4 * 4 * W * W)
A43 = sin(4 * (W * X_3 + B)) * (1 / (X_3 * X_3) - 4 * 4 * W * W)
A44 = sin(4 * (W * X_4 + B)) * (1 / (X_4 * X_4) - 4 * 4 * W * W)
A45 = sin(4 * (W * X_5 + B)) * (1 / (X_5 * X_5) - 4 * 4 * W * W)
A51 = sin(5 * (W * X_1 + B)) * (1 / (X_1 * X_1) - 5 * 5 * W * W)
A52 = sin(5 * (W * X_2 + B)) * (1 / (X_2 * X_2) - 5 * 5 * W * W)
A53 = sin(5 * (W * X_3 + B)) * (1 / (X_3 * X_3) - 5 * 5 * W * W)
A54 = sin(5 * (W * X_4 + B)) * (1 / (X_4 * X_4) - 5 * 5 * W * W)
A55 = sin(5 * (W * X_5 + B)) * (1 / (X_5 * X_5) - 5 * 5 * W * W)
A_matrix = np.matrix('{} {} {} {} {};{} {} {} {} {};{} {} {} {} {};{} {} {} {} {};{} {} {} {} {}'.format(A11, A21, A31, A41, A51, A12, A22, A32, A42, A52, A13, A23, A33, A43, A53, A14, A24, A34, A44, A54, A15, A25, A35, A45, A55))
print(A_matrix)

print("--------")
inverse_matrix_A = np.linalg.inv(A_matrix)
print("inverse_matrix_A:")
print(inverse_matrix_A)


def K(x, y, z):
    k = -(x/y + z/(y*y))
    return k
print("--------")
K_matrix = np.matrix('{};{};{};{};{}'.format(K(k, X_1, b), K(k, X_2, b), K(k, X_3, b), K(k, X_4, b), K(k, X_5, b)))
print("K_matrix:")
print(K_matrix)
print("--------")
A = np.dot(inverse_matrix_A, K_matrix)
print("A:")
print(A)
print("--------")
vals = np.arange(x_a, x_b, 0.1)
kollokacii, tochnoe, Eps = [], [], []
for i in vals:
    fi_0 = k*i+b
    fi_1 = (A[0]*sin(1*(W*i+B)))[0, 0]
    fi_2 = (A[1]*sin(2*(W*i+B)))[0, 0]
    fi_3 = (A[2]*sin(3*(W*i+B)))[0, 0]
    fi_4 = (A[3]*sin(4*(W*i+B)))[0, 0]
    kol = fi_0 + fi_1 + fi_2 + fi_3 + fi_4
    kollokacii.append(kol)
    toch = C1 * cos(sqrt(3) / 2 * log(i))/sqrt(i) + C2 * sin(sqrt(3) / 2 * log(i))/sqrt(i)
    tochnoe.append(toch)
    eps = tochnoe[-1] - kollokacii[-1]
    Eps.append(eps)
    print(kol, " ", toch, " ", eps)
plt.title("График сходимости и погрешности")
plt.plot(vals, kollokacii, label='метод коллокаций')
plt.plot(vals, tochnoe, label='точное решение')
plt.plot(vals, Eps, label='Погрешность')
plt.legend()
plt.savefig('saved.png')


