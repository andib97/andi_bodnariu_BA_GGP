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



#kernel without remove features
class GraphKernelMultihop(gpflow.kernels.Kernel):
    
    def __init__(self,features, base_kernel, adjMatrix,X_train,hops=1):
        super().__init__()
        self.base_kernel=base_kernel
        adjMatrix =adj+np.eye(adj.shape[0])
        
        if(hops==1):   
            adjMatrixsparse=sparse.csr_matrix(self.p_matrix(adjMatrix))
            coo = adjMatrixsparse.tocoo()
            indices = np.mat([coo.row, coo.col]).transpose()
            self.sparse_p_matrix=tf.SparseTensor(indices, coo.data, coo.shape)
            print(self.sparse_p_matrix)
        else:
            adjMatrix_hop =sparse.csr_matrix(adjMatrix)
            for i in range(hops-1):
                adjMatrix_hop =adjMatrix_hop.dot(adjMatrix_hop)
                adjMatrix_hop[ adjMatrix_hop>0]=1
            self.p_matrix=self.p_matrix(adjMatrix_hop.toarray())
        self.tr_features, self.tr_masks, self.tr_masks_counts = self._diag_tr_helper(features, adjMatrix, X_train)
        #self.p_matrix=np.eye(adj.shape[0])
        self.features=features
    def K(self,X,X2=None):
        if X2 is not None:
            assert("Full Covariance not Implemented")
        X_cov= self.base_kernel.K(self.features)
        PX_cov =tf.sparse.sparse_dense_matmul(self.sparse_p_matrix,X_cov)
        PXP_cov=tf.sparse.sparse_dense_matmul(self.sparse_p_matrix,X_cov,adjoint_b=True)
        Xint =tf.cast(X,dtype =tf.int32)
        PXP_cov=tf.gather(PXP_cov,Xint,axis=0)
        PXP_cov=tf.gather(PXP_cov,Xint,axis=1)
        return PXP_cov
    def K_diag(self,X):
        diag_cov=tf.linalg.diag_part(self.K(X))
        return diag_cov
    
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
    def remove_features(self,adjMatrix,features,pos):
        pos=pos.reshape(pos.shape[0],)
        start_time = time.time()
        adjRed = adjMatrix.copy()
        featuresRed =features.copy()
        newpos=pos.copy().astype('int64')
        indexSet = set(range(adjMatrix.shape[0]))-set(newpos)
        removeSet =[]
        for i in indexSet:
            if np.array_equal(np.zeros(pos.shape[0]),adjRed[newpos,i]):
                    removeSet.append(i)
        for j,_ in enumerate(newpos):
            newpos[j]-= len([i for i in removeSet if i < newpos[j]]) 
        adjRed= np.delete(adjRed,removeSet,0)
        adjRed= np.delete(adjRed,removeSet,1)
        featuresRed= np.delete(featuresRed,removeSet,0)
        print("remove features takes --- %s seconds ---" % (time.time() - start_time))
        return adjRed,featuresRed,newpos.astype('float64')    
    
    def nodedegree(self,A, inv=False):
        if(inv):
            degree=1/np.sum(A,axis=0)
        else:
            degree=np.sum(A,axis=0)
        return degree;


    def p_matrix(self,A):
        D=np.copy(self.nodedegree(A,inv=True))
        D=D.ravel()
        A=A.copy()
        for i in range(A.shape[0]):
            A[i,:]=D[i]*A[i,:]
        return(A)

@cov.Kuu.register(InducingGraphVariables,GraphKernelMultihop)
def Kuu_GraphKernel(feat, kernel,jitter=0.1):
    return kernel.base_kernel.K(feat.Z)+ jitter * tf.eye(len(feat), dtype=default_float())


@cov.Kuf.register(InducingGraphVariables,GraphKernelMultihop,TensorLike)
def Kuf_GraphKernel(feat, kernel,X):
    uX_cov=kernel.base_kernel.K(kernel.features,feat.Z)
    uXP_cov =tf.sparse.sparse_dense_matmul(kernel.sparse_p_matrix,uX_cov)
    Xint=tf.cast(X,dtype =tf.int32)
    return tf.gather(tf.transpose(uXP_cov),Xint,axis=1)