import numpy as np
import matplotlib.pyplot as plt
import sympy as spy
from datetime import datetime
m = 10
n = 4
def FEA2Dplane(elem_selected,h):

    E  = 150e9
    v = 0.3
    load  = -2e9
    length = 0.5
    breadth = 0.2
    # discretization
    m = 10
    n = 4

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

        thickness = 0.05

        if elem_selected != -1 and elem == elem_selected:
            thickness = thickness + h

        Kint_new = thickness * np.linalg.det(Jac) * Kint

        global_stiff[index] = global_stiff[index] + Kint_new

        p = np.zeros(8)

        if elements_nodes[elem, 10] == 0 and elements_nodes[elem, 9] >= 0.4:
            f3y = 0.5 * (1 - xi) * 0.05* load * Jac[0, 0]
            f4y = 0.5 * (1 + xi) * 0.05* load * Jac[0, 0]

            p[5] = 2*f3y.subs(xi,0)
            p[7] = 2*f4y.subs(xi,0)

            global_p[2 * k + 1] += p[5]
            global_p[2 * l + 1] += p[7]

    # extracting free and fixed dof

    u_fixed = []

    for e in range(n_elements):

        if elements_nodes[e, 5] == 0:
            u_fixed.append(2 * (elements_nodes[e, 1] - 1))
            u_fixed.append(2 * (elements_nodes[e, 1] - 1) + 1)
            u_fixed.append(2 * (elements_nodes[e, 3] - 1))
            u_fixed.append(2 * (elements_nodes[e, 3] - 1) + 1)

    u_fixed = [int(i) for i in set(u_fixed)]

    # deleting the rows and columns corresponding to the fixed DoF
    global_stiff_new = np.delete(global_stiff, u_fixed, 0)
    global_stiff_new = np.delete(global_stiff_new, u_fixed, 1)

    global_p_new = np.delete(global_p, u_fixed)

    # calculating the displacements
    disp = np.dot(np.linalg.inv(global_stiff_new),global_p_new)

    for i in u_fixed:
        disp = np.insert(disp, i, 0)

    return disp[-1]

h = 10e-7
start_time = datetime.now()

disppure = FEA2Dplane(-1,h)

sens = []
for elem in range(m*n):
    dispelem = FEA2Dplane(elem,h)
    sens.append((dispelem - disppure)/h)

end_time = datetime.now()

print(sens,'Duration: {}'.format(end_time - start_time))


