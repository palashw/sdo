import numpy as np
import matplotlib.pyplot as plt
import sympy as spy
from datetime import datetime

E  = 150e9
v = 0.3
thickness  = 5e-2
load  = -2e9
length = 0.5
breadth = 0.2

# discretization
m = 25
n = 10

dx = length/m
dy = breadth/n

# element nodes matrix ,row is - node number, nodes in the element, coordinates of nodes
elements_nodes = np.zeros([m*n,13])

count = -1
for i in range(n):

    count += 1

    for j in range(m):
        elements_nodes[i * (m) + j, 0] = i * (m) + (j + 1)

        # nodes associated with the element
        elements_nodes[i * (m) + j, 1] = count + elements_nodes[i * (m) + j, 0]
        elements_nodes[i * (m) + j, 2] = count + elements_nodes[i * (m) + j, 0] + 1
        elements_nodes[i * (m) + j, 3] = count + elements_nodes[i * (m) + j, 0] + m + 1
        elements_nodes[i * (m) + j, 4] = count + elements_nodes[i * (m) + j, 0] + m + 2

        # nodal coordinates of each element
        elements_nodes[i * (m) + j, 5] = j * dx
        elements_nodes[i * (m) + j, 6] = breadth - i * dy
        elements_nodes[i * (m) + j, 7] = (j + 1) * dx
        elements_nodes[i * (m) + j, 8] = breadth - i * dy
        elements_nodes[i * (m) + j, 9] = j * dx
        elements_nodes[i * (m) + j, 10] = breadth - (i + 1) * dy
        elements_nodes[i * (m) + j, 11] = (j + 1) * dx
        elements_nodes[i * (m) + j, 12] = breadth - (i + 1) * dy

print(elements_nodes)

# defining shape functions

xi, eta = spy.symbols('xi, eta')
N1 = (1-xi)*(1+eta)*0.25
N2 = (1+xi)*(1+eta)*0.25
N3 = (1-xi)*(1-eta)*0.25
N4 = (1+xi)*(1-eta)*0.25

# differentials of shape functions wrt xi and eta
dN1dxi = spy.diff(N1,xi)
dN1deta = spy.diff(N1,eta)
dN2dxi = spy.diff(N2,xi)
dN2deta = spy.diff(N2,eta)
dN3dxi = spy.diff(N3,xi)
dN3deta = spy.diff(N3,eta)
dN4dxi = spy.diff(N4,xi)
dN4deta = spy.diff(N4,eta)

# finding jacobian

Jac = np.array([[dx*0.5,0],
                [0, dy*0.5]])

# finding differentials of shape functions wrt x and y - 2X1 matrices

dN1xy = np.dot(np.linalg.inv(Jac), np.array([[dN1dxi],[dN1deta]]))
dN2xy = np.dot(np.linalg.inv(Jac), np.array([[dN2dxi],[dN2deta]]))
dN3xy = np.dot(np.linalg.inv(Jac), np.array([[dN3dxi],[dN3deta]]))
dN4xy = np.dot(np.linalg.inv(Jac), np.array([[dN4dxi],[dN4deta]]))

# finding B matrix 3X8 matrix

Bmat = np.array([[dN1xy[0][0],0,dN2xy[0][0],0,dN3xy[0][0],0,dN4xy[0][0],0],
              [0,dN1xy[1][0],0,dN2xy[1][0],0,dN3xy[1][0],0,dN4xy[1][0]],
              [dN1xy[1][0],dN1xy[0][0],dN2xy[1][0],dN2xy[0][0],dN3xy[1][0],dN3xy[0][0],dN4xy[1][0],dN4xy[0][0]]])

# finding E matrix 3X3 matrix

Emat = (E/(1-v**2))*np.array([[1,v,0],
                              [v,1,0],
                              [0,0,0.5*(1-v)]])

# finding K matrix = 8X3*3X3*3X8 = 8X8
Kmat = Bmat.transpose() @ Emat @ Bmat
Kint = np.zeros_like(Kmat)

# performing gauss integration
gauss_points = [ 1/np.sqrt(3), -1/np.sqrt(3)]
for i in range(8):
    for j in range(8):
        Kint[i][j] = Kmat[i][j].subs([(xi,-gauss_points[0]),(eta,gauss_points[0])])+Kmat[i][j].subs([(xi,-gauss_points[0]),(eta,gauss_points[1])])+Kmat[i][j].subs([(xi,-gauss_points[1]),(eta,gauss_points[0])])+Kmat[i][j].subs([(xi,-gauss_points[1]),(eta,gauss_points[1])])

Kint_new = thickness*np.linalg.det(Jac)*Kint

# Global stiffness matrix

n_nodes = (m + 1) * (n + 1)
n_elements = m * n

global_stiff = np.zeros([2*n_nodes,2*n_nodes],dtype=float)
global_p = np.zeros(2*n_nodes)

# #integrating element stiffenss matrix into global stiffness matrix

for elem in range(n_elements):

    #     adding elements from local stiffness matrix to global stiffness matrix
    [i, j, k, l] = elements_nodes[elem, 1:5]

    i = int(i - 1)
    j = int(j - 1)
    k = int(k - 1)
    l = int(l - 1)

    index = np.ix_([2 * i, 2 * i + 1, 2 * j, 2 * j + 1, 2 * k, 2 * k + 1, 2 * l, 2 * l + 1],
                   [2 * i, 2 * i + 1, 2 * j, 2 * j + 1, 2 * k, 2 * k + 1, 2 * l, 2 * l + 1])

    global_stiff[index] = global_stiff[index] + Kint_new
    #     k_global[index] += k_elem

    p = np.zeros(8)

    if elements_nodes[elem, 10] == 0 and elements_nodes[elem, 9] >= 0.4:
        f3y = 0.5 * (1 - xi) * thickness* load * Jac[0, 0]
        f4y = 0.5 * (1 + xi) * thickness* load * Jac[0, 0]

        p[5] = 2*f3y.subs(xi,0)
        p[7] = 2*f4y.subs(xi,0)

        global_p[2 * k + 1] += p[5]
        global_p[2 * l + 1] += p[7]

print("global stiffness = ",global_stiff)
print("gloabl stiffness size = ", global_stiff.shape)
print("global load",(global_p))

# extracting free and fixed dof

u_fixed = []

for e in range(n_elements):

    if elements_nodes[e, 5] == 0:
        u_fixed.append(2 * (elements_nodes[e, 1] - 1))
        u_fixed.append(2 * (elements_nodes[e, 1] - 1) + 1)
        u_fixed.append(2 * (elements_nodes[e, 3] - 1))
        u_fixed.append(2 * (elements_nodes[e, 3] - 1) + 1)

u_fixed = [int(i) for i in set(u_fixed)]
print("fixed nodes = ", u_fixed)

# deleting the rows and columns corresponding to the fixed DoF
global_stiff_new = np.delete(global_stiff, u_fixed, 0)
global_stiff_new = np.delete(global_stiff_new, u_fixed, 1)

global_p_new = np.delete(global_p, u_fixed)

# calculating the displacements
disp = np.dot(np.linalg.inv(global_stiff_new),global_p_new)

for i in u_fixed:
    disp = np.insert(disp, i, 0)

print(" displacement = ",(disp))

u_undeformed = []

count = -1

for i in range(n+1):
    count+=1
    for j in range(m+1):
        u_undeformed.append(dx*j)
        u_undeformed.append(breadth - i*dy)

u_undeformed = np.array(u_undeformed)

u_undeformed = np.reshape(u_undeformed,(n_nodes,2))
u_delta = np.reshape(disp,(n_nodes,2))

u_deformed = u_undeformed+u_delta

start_time = datetime.now()
# direct method
Lmat = np.zeros(2*n_nodes)
Lmat[2*n_nodes-1] = 1
print(Lmat)

Umat = disp
print(Umat)

Func = Lmat.T*Umat
# print(Func)

indexfixed = np.ix_(u_fixed,u_fixed)
Kpp = global_stiff[indexfixed]

nodes = [i for i in range(2*n_nodes)]
nodes = np.array(nodes)

u_free = np.delete(nodes,u_fixed)

indexpf = np.ix_(u_fixed,u_free)

Kpf = global_stiff[indexpf]
print("Kpf = ", Kpf)

indexff = np.ix_(u_free,u_free)
Kff = global_stiff[indexff]
print("Kff= ",Kff)

drbydy = np.block([[-np.identity(len(u_fixed)),Kpf],
                  [np.zeros([len(u_free),len(u_fixed)]),Kff]])

print("drbydy = ",drbydy)
sens = np.zeros(n_elements)

for elem in range(n_elements):
    Kelem = np.zeros([2*n_nodes,2*n_nodes])
    [i, j, k, l] = elements_nodes[elem, 1:5]

    i = int(i - 1)
    j = int(j - 1)
    k = int(k - 1)
    l = int(l - 1)

    index = np.ix_([2 * i, 2 * i + 1, 2 * j, 2 * j + 1, 2 * k, 2 * k + 1, 2 * l, 2 * l + 1],
                   [2 * i, 2 * i + 1, 2 * j, 2 * j + 1, 2 * k, 2 * k + 1, 2 * l, 2 * l + 1])

    Kelem[index] = (1/thickness)*Kint_new
    sens[elem] = (-np.linalg.inv(drbydy)@Kelem@disp)[-1]
end_time = datetime.now()

print("sensitivity by direct method and time = ",sens, 'Duration: {}'.format(end_time - start_time))

# adjoint method
start_time = datetime.now()

psi = np.zeros(2*n_nodes)
Lmat_free = Lmat[np.ix_(u_free)]

print("L free=", Lmat_free)
psif = -(np.linalg.inv(Kff)).T@Lmat_free.T

print("psi free =", psif)

psi[np.ix_(u_free)] = psif

print("psi complete",psi)

sens_adj = np.zeros(n_elements)

for elem in range(n_elements):
    [i, j, k, l] = elements_nodes[elem, 1:5]

    i = int(i - 1)
    j = int(j - 1)
    k = int(k - 1)
    l = int(l - 1)

    index = np.ix_([2 * i, 2 * i + 1, 2 * j, 2 * j + 1, 2 * k, 2 * k + 1, 2 * l, 2 * l + 1],
                   [2 * i, 2 * i + 1, 2 * j, 2 * j + 1, 2 * k, 2 * k + 1, 2 * l, 2 * l + 1])

    psi_elem = psi[np.ix_([2 * i, 2 * i + 1, 2 * j, 2 * j + 1, 2 * k, 2 * k + 1, 2 * l, 2 * l + 1])]
    disp_elem = disp[np.ix_([2 * i, 2 * i + 1, 2 * j, 2 * j + 1, 2 * k, 2 * k + 1, 2 * l, 2 * l + 1])]

    Kelem_adjoint = (1 / thickness) * Kint_new

    sens_adj[elem] = psi_elem.T@Kelem_adjoint@disp_elem

end_time = datetime.now()

print("sensitivity by adjoint method and time = ",sens_adj, 'Duration: {}'.format(end_time - start_time))

