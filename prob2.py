import numpy as np
import matplotlib.pyplot as plt

# material props
import scipy.optimize
NFeval = 1
converge = []

def truss(A):
    E = 120e9

    rho = 800

    # total 11 nodes
    nodes = [[0,0],[4,0],[8,0],[12,0],[16,0],[20,0],[2,3.464],[6,3.464],[10,3.464],[14,3.464],[18,3.464]]

    # total 19 elements
    elements = [[0,6],[6,7],[1,6],[0,1],[1,7],[1,2],[7,8],[2,7],[2,3],
                [2,8],[8,9],[3,8],[3,4],[3,9],[9,10],[4,9],[4,5],[4,10],[5,10]]

    nodes = np.array(nodes).astype(float)
    elements = np.array(elements).astype(float)

    # input loads
    P = np.zeros(22)

    global_stiff = np.zeros([22,22])

    local_stiffness_list = []

    for elem in range(len(elements)):

        # identifying nodes of the element
        node1 = int(elements[elem][0])
        node2 = int(elements[elem][1])
        # print("nodes = ",node1,node2)

        # getting x,y coordinates
        [x1,y1] = nodes[node1]
        [x2,y2] = nodes[node2]

        #getting lenght, sin and cos
        length = np.sqrt(np.power(y2-y1,2)+np.power(x2-x1,2))
        s = (y2-y1)/length
        c = (x2-x1)/length

        # getting local stiff of this element
        local_stifflist = np.array([[c**2,c*s,-c**2,-c*s],
                       [c*s, s**2,-c*s, -s**2],
                       [-c**2,-c*s,c**2,c*s],
                       [-c*s,-s**2,c*s,s**2]])

        local_stiff = E*A[elem]*0.25*local_stifflist
        local_stiffness_list.append(local_stiff/A[elem])

        # placing local stiffness elements in global stiffness
        aux = 2*elements[elem,:]
        index = np.r_[aux[0]:aux[0]+2,aux[1]:aux[1]+2].astype(int)
        global_stiff[np.ix_(index,index)] = global_stiff[np.ix_(index,index)]+local_stiff

        m = rho*9.8*length*A[elem]
        P[2*node1 + 1 ] +=  m*-0.5
        P[2*node2 + 1 ] += m*-0.5

    # force applied to node 3 in y direction. Numbering in python starts from 0 hence index 5 for 6th entry of vector
    P[5] += -1e6

    # print("load vector = ",P)

    disp1 = np.zeros(22)
    non_zero_indices = [2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21]

    # print("global stiffness = ",global_stiff, global_stiff.shape)

    #ignoring disp at 0,1 and 11 - node 1 fixed and node 5 x y dir
    new_global_stiff = global_stiff[np.ix_(non_zero_indices,non_zero_indices)]
    # print("new global stiffness = ",new_global_stiff, new_global_stiff.shape)

    new_P = P[non_zero_indices]

    # disp = np.dot(np.linalg.inv(new_global_stiff),new_P)
    disp = np.linalg.solve(new_global_stiff,new_P)

    disp1[non_zero_indices] += disp

    # adjoint method
    u_fixed = [0,1,11]
    Lmat = np.zeros(22)
    Lmat[5] = 1

    indexfixed = np.ix_(u_fixed, u_fixed)
    u_free = np.array([2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21])
    indexff = np.ix_(u_free, u_free)
    Kff = global_stiff[indexff]
    # print("K free free= ",Kff)

    Lmat_free = Lmat[np.ix_(u_free)]
    psif = -(np.linalg.inv(Kff)).T @ Lmat_free.T

    # print("psi f = ",psif)

    psi = np.zeros(22)

    psi[np.ix_(u_free)] = psif

    # print("psi complete = ", psi)

    sens_adj = np.zeros(19)

    for e in range(len(elements)):
        elem = elements[e]

        i = int(elem[0])
        j = int(elem[1])
        ind = np.array([2*i,2*i+1,2*j,2*j+1])
        psi_elem = psi[ind]
        disp_elem = disp1[ind]

        local_stiff_ = local_stiffness_list[e]
        sens_adj[e] = psi_elem.T @ local_stiff_ @ disp_elem
    return disp1[5]

def truss_sens(A):
    E = 120e9

    rho = 800

    # total 11 nodes
    nodes = [[0,0],[4,0],[8,0],[12,0],[16,0],[20,0],[2,3.464],[6,3.464],[10,3.464],[14,3.464],[18,3.464]]

    # total 19 elements
    elements = [[0,6],[6,7],[1,6],[0,1],[1,7],[1,2],[7,8],[2,7],[2,3],
                [2,8],[8,9],[3,8],[3,4],[3,9],[9,10],[4,9],[4,5],[4,10],[5,10]]

    nodes = np.array(nodes).astype(float)
    elements = np.array(elements).astype(float)

    # input loads
    P = np.zeros(22)

    global_stiff = np.zeros([22,22])

    local_stiffness_list = []

    for elem in range(len(elements)):

        # identifying nodes of the element
        node1 = int(elements[elem][0])
        node2 = int(elements[elem][1])
        # print("nodes = ",node1,node2)

        # getting x,y coordinates
        [x1,y1] = nodes[node1]
        [x2,y2] = nodes[node2]

        #getting lenght, sin and cos
        length = np.sqrt(np.power(y2-y1,2)+np.power(x2-x1,2))
        s = (y2-y1)/length
        c = (x2-x1)/length

        # getting local stiff of this element
        local_stifflist = np.array([[c**2,c*s,-c**2,-c*s],
                       [c*s, s**2,-c*s, -s**2],
                       [-c**2,-c*s,c**2,c*s],
                       [-c*s,-s**2,c*s,s**2]])

        local_stiff = E*A[elem]*0.25*local_stifflist
        local_stiffness_list.append(local_stiff/A[elem])

        # placing local stiffness elements in global stiffness
        aux = 2*elements[elem,:]
        index = np.r_[aux[0]:aux[0]+2,aux[1]:aux[1]+2].astype(int)
        global_stiff[np.ix_(index,index)] = global_stiff[np.ix_(index,index)]+local_stiff

        m = rho*9.8*length*A[elem]
        P[2*node1 + 1 ] +=  m*-0.5
        P[2*node2 + 1 ] += m*-0.5

    # force applied to node 3 in y direction. Numbering in python starts from 0 hence index 5 for 6th entry of vector
    P[5] += -1e6

    # print("load vector = ",P)

    disp1 = np.zeros(22)
    non_zero_indices = [2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21]

    # print("global stiffness = ",global_stiff, global_stiff.shape)

    #ignoring disp at 0,1 and 11 - node 1 fixed and node 5 x y dir
    new_global_stiff = global_stiff[np.ix_(non_zero_indices,non_zero_indices)]
    # print("new global stiffness = ",new_global_stiff, new_global_stiff.shape)

    new_P = P[non_zero_indices]

    # disp = np.dot(np.linalg.inv(new_global_stiff),new_P)
    disp = np.linalg.solve(new_global_stiff,new_P)

    disp1[non_zero_indices] += disp

    # adjoint method
    u_fixed = [0,1,11]
    Lmat = np.zeros(22)
    Lmat[5] = 1

    indexfixed = np.ix_(u_fixed, u_fixed)
    u_free = np.array([2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21])
    indexff = np.ix_(u_free, u_free)
    Kff = global_stiff[indexff]
    # print("K free free= ",Kff)

    Lmat_free = Lmat[np.ix_(u_free)]
    psif = -(np.linalg.inv(Kff)).T @ Lmat_free.T

    # print("psi f = ",psif)

    psi = np.zeros(22)

    psi[np.ix_(u_free)] = psif

    # print("psi complete = ", psi)

    sens_adj = np.zeros(19)

    for e in range(len(elements)):
        elem = elements[e]

        i = int(elem[0])
        j = int(elem[1])
        ind = np.array([2*i,2*i+1,2*j,2*j+1])
        psi_elem = psi[ind]
        disp_elem = disp1[ind]

        local_stiff_ = local_stiffness_list[e]
        sens_adj[e] = psi_elem.T @ local_stiff_ @ disp_elem
    return sens_adj

A0 = np.zeros(19)
for i in range(19):
    A0[i] = 5e-4
print(truss_sens(A0))
#
def cons(A):
    return 30.4 - 3200*sum(A)

def obj(A):
    return -truss(A)

cons = ({'type':'ineq', 'fun': cons})

bounds = scipy.optimize.Bounds(1e-9*np.ones(19),10*np.ones(19))

def callbackF(A):
    global NFeval
    global converge
    NFeval+=1
    converge.append(obj(A))

res = scipy.optimize.minimize(obj,A0,constraints=cons, method='SLSQP',callback=callbackF,options={'disp':True},bounds=bounds)

print(res.x)
print(-res.fun)

print(3200*sum(res.x))
print(converge)
n = np.linspace(1,38,38)

plt.plot(n,converge)
plt.show()
