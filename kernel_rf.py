import time
import tensorflow as tf
import numpy as np
import scipy
import scipy.sparse as sparse
import gpflow
from gpflow.inducing_variables import InducingPoints
from gpflow.base import TensorLike
from gpflow.utilities import to_default_float
from gpflow import covariances as cov
from gpflow.config import default_float


class InducingGraphVariables(InducingPoints):
    pass
#kernel with remove features
class GraphKernelMultihop(gpflow.kernels.Kernel):
    
    def __init__(self,features, base_kernel, adj_matrix,X_train,hops=1):
        super().__init__()
        self.base_kernel=base_kernel
        adj_matrix=adj_matrix+np.eye(adj_matrix.shape[0])
        adj_matrix_hop=adj_matrix.copy()
        if(hops>1):
            adj_matrix_hop =sparse.csr_matrix(adj_matrix)
            for i in range(hops-1):
                adj_matrix_hop =adj_matrix_hop.dot(adj_matrix)
                adj_matrix_hop[ adj_matrix_hop>0]=1
        self.p_matrix=self.p_matrix(adj_matrix_hop)
        self.tr_features, self.tr_masks, self.tr_masks_counts = self._diag_tr_helper(features, adj_matrix, X_train)
        self.features=features
    def K(self,X,X2=None):
        p_matrix_red,features_red,X_new =self.remove_features(self.p_matrix,self.features,X)
        if X2 is not None:
            assert("Full Covariance not Implemented")
        X_cov= self.base_kernel.K(features_red)
        PX_cov =tf.linalg.matmul(p_matrix_red,X_cov)
        PXP_cov=tf.linalg.matmul(PX_cov,p_matrix_red,transpose_b=True)
        X_int =tf.cast(X_new,dtype =tf.int32)
        PXP_cov=tf.gather(PXP_cov,X_int,axis=0)
        PXP_cov=tf.gather(PXP_cov,X_int,axis=1)
        return PXP_cov
    def K_diag(self,X):
        diag_cov=tf.linalg.diag_part(self.K(X))
        return diag_cov
    #
    def _diag_tr_helper(self, node_features, adj_mat, x_tr):
        z = np.asarray([node_features[a == 1.] for a in adj_mat[x_tr.flatten()]])
        max_n = np.max([t.shape[0] for t in z])
        out = np.zeros((len(z), max_n, node_features.shape[1]))
        masks = np.zeros((len(z), max_n, max_n))
        for i in range(len(z)):
            out[i,:len(z[i]),:] = z[i]
            masks[i, :len(z[i]), :len(z[i])] = 1
        return out, masks, np.sum(np.sum(masks, 2), 1)
    def diag_tr(self):
        base_k_mat=self.tr_masks*self.base_kernel.K(self.tr_features)
        return tf.reduce_sum(base_k_mat, [1, 2])/self.tr_masks_counts
    
    
    #removes the vertices that cannot be reached in 1-hop to speed up computation  
    def remove_features(self,adj_matrix,features,pos):
        pos=pos.reshape(pos.shape[0],)
        #start_time = time.time() #for performance check
        adj_red = adj_matrix.copy()
        features_red =features.copy()
        new_pos=pos.copy().astype('int64')
        index_set = set(range(adj_matrix.shape[0]))-set(new_pos)
        remove_set =[]
        for i in index_set:# find the rows and columns that need to be removed
            if np.array_equal(np.zeros(pos.shape[0]),adj_red[new_pos,i]):
                    remove_set.append(i)
        for j,_ in enumerate(new_pos):#calculate the new positions
            new_pos[j]-= len([i for i in remove_set if i < new_pos[j]]) 
        adj_red= np.delete(adj_red,remove_set,0)
        adj_red= np.delete(adj_red,remove_set,1)
        features_red= np.delete(features_red,remove_set,0)
        #print("remove features takes --- %s seconds ---" % (time.time() - start_time)) # for performance check
        return adj_red,features_red,new_pos.astype('float64')    
    
    def nodedegree(self,A, inv=False):
        if(inv):
            degree=1/np.sum(A,axis=0)
        else:
            degree=np.sum(A,axis=0)
        return degree;


    def p_matrix(self,A):
        D=np.copy(self.nodedegree(A,inv=True))
        A=A.copy()
        for i in range(A.shape[0]):
            A[i,:]=D[i]*A[i,:]
        return(A)
    
@cov.Kuu.register(InducingGraphVariables,GraphKernelMultihop)
def Kuu_GraphKernel(feat, kernel,jitter=0.1):
    return kernel.base_kernel.K(feat.Z)+ jitter * tf.eye(len(feat), dtype=default_float())


@cov.Kuf.register(InducingGraphVariables,GraphKernelMultihop,TensorLike)
def Kuf_GraphKernel(feat, kernel,X):
    p_matrix_red,features_red,X_new = kernel.remove_features(kernel.p_matrix,kernel.features,X)
    uX_cov=kernel.base_kernel.K(features_red,feat.Z)
    uXP_cov= tf.linalg.matmul(p_matrix_red,uX_cov)
    X_int=tf.cast(X_new,dtype =tf.int32)
    return tf.gather(tf.transpose(uXP_cov),X_int,axis=1)