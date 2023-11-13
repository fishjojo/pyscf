import numpy as np
from pyscf import gto, ao2mo
from pyscf.df import df, thcdf

mol = gto.Mole()
mol.atom = '''
 O   -0.831126382592   -0.800500805235    1.714899834172
 H   -1.791126382592   -0.648500805235    1.904899834172
 H   -0.703126382592   -1.102500805235    0.785899834172
 O   -1.014126382592    2.189499194765   -2.328100165828
 H   -1.234126382592    1.223499194765   -2.335100165828
 H   -0.189126382592    2.152499194765   -2.820100165828
 O    1.647873617408    0.120499194765    1.520899834172
 H    2.192873617408   -0.324500805235    0.863899834172
 H    0.803873617408   -0.392500805235    1.434899834172
 O    0.188873617408   -1.347500805235   -0.770100165828
 H    0.220873617408   -2.228500805235   -0.535100165828
 H    0.834873617408   -1.250500805235   -1.483100165828
'''
mol.basis = 'ccpvdz'
mol.max_memory=4000
mol.verbose = 6
mol.build()

auxbasis='cc-pvdz-ri'

#Gaussian density fit
gdf = df.DF(mol, auxbasis=auxbasis)
gdf.build()
int3c = gdf._cderi
eri1 = np.dot(int3c.T, int3c)
eri1 = ao2mo.addons.restore('1', eri1, mol.nao)

#THC of Gaussian density fit
mydf = thcdf.THCDF(mol, auxbasis=auxbasis, c_isdf=20)
mydf.build()
D, X = mydf._cderi
int3c = np.einsum('PL,Li,Lj->Pij', D, X, X, optimize=True)
eri2 = np.einsum('Pij,Pkl->ijkl', int3c, int3c, optimize=True)

# error
print('THC error: ', np.linalg.norm(eri2 - eri1))
