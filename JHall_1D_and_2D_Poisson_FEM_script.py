# Import needed packages
import numpy as np
import scipy as sp
import scipy.integrate as integrate
import dolfin as fn

# MyA : L ---> A
# L is the list with elements #s
# A is the induced "stiffness" matrix
# from discretizing the weak problem

def MyA(L):
    
    # If we have a 1D problem...
    if np.shape(L)[0]== 1:
        
        # Grab dimension
        n = L[0]
        
        # Initialize empty A
        A = np.zeros((n-1,n-1))
        
        # Write A_ij as derived in HW2
        for i in range(0,n-1):
            
            A[i,i] = 2
            
            if i > 0:
            
                A[i,i-1] = -1
                A[i-1,i] = -1
        
        # Scale by n (see HW2)
        A = (n)*A
        
        return A
    
    # If we have a 2D problem...
    if np.shape(L)[0]== 2:
        
        # Grab dimensions
        n = L[0]
        m = L[1]
        s = (n-1)*(m-1)
        
        # Initialize empty A
        A = np.zeros((s,s))
        
        # Write A_ij as derived in HW2
        for i in range(0,s):
            
            A[i,i] = 4
            
            if i>0:
                
                A[i,i-1] = -1
                A[i-1,i] = -1
                
            if i>2:
                
                A[i,i-3] = -1
                A[i-3,i] = -1
        
        return A

# MyF : L,f ---> F, PHI
# L is the list with elements #s
# f is the forcing function
# F is the induced "loading" vector
# from discretizing the weak problem
# PHI contains the basis elements created
# in this process

def MyF(L, f):
    
    # If we have a 1D problem...
    if np.shape(L)[0]== 1:
        
        # Grab dimension
        n = L[0]
        
        # Make mesh for ref and comp
        mesh1_1D = fn.UnitIntervalMesh(n)
        V1_1D = fn.FunctionSpace(mesh1_1D, "Lagrange", 1)
        
        # Make a class on the fly for building interpolants
        class unit_dof_1D(fn.UserExpression):
            def __init__(self, x_pt, **kwargs):
                self.x0 = x_pt[0]
                super().__init__(**kwargs)
            def eval(self, v, x):
                v[0] = 0
                if (abs(x[0]-self.x0)< 0.0001):
                    v[0] = 1
                return v
            def value_shape(self):
                return ()

        # Get the interior coordinates extracted
        dofs_coordinates_V1_1D = V1_1D.tabulate_dof_coordinates()
        corrected_coords = dofs_coordinates_V1_1D[::-1]
        interior_coords=corrected_coords[1:-1]
        
        # Define and save our basis fcns
        PHI = []
        for i in range(1,n):
            PHI.append(fn.interpolate(unit_dof_1D([interior_coords[i-1, 0]]), V1_1D))
        
        # Make loading vector, and use scipy to integrate for inner products
        F = np.zeros((n-1,1))
        for i in range(0,n-1):
            
            def needs_integration(*args):
                return PHI[i](*args) * f(*args) 
            
            F[i] = sp.integrate.quad(needs_integration, 0, 1)[0]
        
        return F, PHI
    
    # If we have a 2D problem...
    if np.shape(L)[0]== 2:
        
        # Grab dim's
        n = L[0]
        m = L[1]
        s = (n-1)*(m-1)
        
        # Make meshes
        mesh1 = fn.UnitSquareMesh(n, m, 'right')
        V1 = fn.FunctionSpace(mesh1, "Lagrange", 1)
        
        # Make a class on the fly so we can interpolate things
        class unit_dof(fn.UserExpression):
            def __init__(self, x_pt, **kwargs):
                self.x0 = x_pt[0]
                self.x1 = x_pt[1]
                super().__init__(**kwargs)
            def eval(self, v, x):
                v[0] = 0
                if (abs(x[0]-self.x0)< 0.0001) & (abs(x[1]-self.x1)< 0.0001):
                    v[0] = 1
                return v
            def value_shape(self):
                return ()
        
        
        # Now grab our interior nodes in the order we want them
        dofs_coordinates_V1 = V1.tabulate_dof_coordinates()
        c = 0
        coords = mesh1.coordinates()
        int_node_list = np.zeros(((n-1)*(m-1),1))
        for i in range(0,(n+1)*(m+1)):
            if coords[i][0]>0 and coords[i][0]<1 and coords[i][1]>0 and coords[i][1]<1:
                
                int_node_list[c]=i
                c=c+1
        
        # Get the basis functions + store them
        PHI = []
        for i in range(0,(n-1)*(m-1)):
            PHI.append(fn.interpolate(unit_dof([coords[int(int_node_list[i][0]), 0], coords[int(int_node_list[i][0]), 1]]), V1))
        
        
        # Compute loading!
        F = np.zeros(((n-1)*(m-1),1))
        for i in range(0,(n-1)*(m-1)):
    
            def needs_integration(*args):
                return PHI[i](*args) * f(*args) # generalized f needs to go here, not square_func
    
            F[i] = sp.integrate.dblquad(needs_integration, 0, 1, lambda x: 0, lambda x: 1)[0]
        
        return F, PHI
