# Author: Tsung-Han Lee henhans74716@gmail.com
import numpy as np
from math import factorial
from itertools import combinations
from numba import jit

def reverseBits(norb,n):
  strb = '{0:0'+str(norb)+'b}'
  rb = strb.format(n)[::-1]
  return int(rb,2)

@jit(nopython=True)#,cache=True)
def countSetBits(n):
    count = 0
    while (n):
        count += n & 1
        n >>= 1
    return count

#class basis(object):
#  '''
#  class for Fock state basis
#  '''
#  def __init__(self, norb, use_ntot=True, use_sz=True, thermal=False):
#    '''
#    Constructor.
#    Input:
#      norb: number of orbital
#      ntot: total particle number. If 'None', partition the Hibert space without using particle number.
#      sz: total Sz number. If 'None', partition the Hilbert space without using the Sz number.
#      thermal: Ture: return all the basis set. False: return only the specific subspace
#
#    Note: Try to leave some room to implement other symmtery such as rotation and translation ...
#    '''
#    self.norb = norb
#    self.use_ntot = use_ntot
#    self.use_sz = use_sz
#    self.thermal = thermal
#    print 'initialize the Hilbert space for %g orbitals using Ntot (%s), Sz (%s), and thermal (%s)'%(self.norb, self.use_ntot, self.use_sz, self.thermal)
#  
#  def create_basis(self, N, Sz):
#    '''
#    Create many-body basis.
#    Input:
#      N: total particle number
#      Sz: total z-component of spin
#
#    Return:
#      basis: Fock space basis, stored as dictionary
#    '''

def table_ep(nstate,nparticle,dtype=np.int64):
    '''
    This function generates the table of binary representations of a particle-conserved and spin-non-conserved basis.
    '''
    result=np.zeros(factorial(nstate)//factorial(nparticle)//factorial(nstate-nparticle),dtype=dtype)
    buff=combinations(range(nstate),nparticle)
    for i,v in enumerate(buff):
        basis=0
        for num in v:
            basis+=(1<<num)
        result[i]=basis
    result.sort()
    return result

def table_es(nstate,nparticle,spinz,dtype=np.int64):
    '''
    This function generates the table of binary representations of a particle-conserved and spin-conserved basis.
    '''
    n,nup,ndw=nstate//2,(nparticle+int(2*spinz))//2,(nparticle-int(2*spinz))//2
    result=np.zeros(factorial(n)//factorial(nup)//factorial(n-nup)*factorial(n)//factorial(ndw)//factorial(n-ndw),dtype=dtype)
    buff_up=list(combinations(range(1,2*n,2),nup))
    buff_dw=list(combinations(range(0,2*n,2),ndw))
    count=0
    for vup in buff_up:
        buff=0
        for num in vup:
            buff+=(1<<num)
        for vdw in buff_dw:
            basis=buff
            for num in vdw:
                basis+=(1<<num)
            result[count]=basis
            count+=1
    result.sort()
    return result

def table_es_sc(nstate,spinz,dtype=np.int64):
    '''
    This function generates the table of binary representations of a particle-non-conserved and spin-conserved basis.
    '''
    strb = '{0:0'+str(nstate)+'b}'
    tmp = np.arange(0,2**nstate)
    result = []
    for bs in tmp:
        print(bs, strb.format(bs) )
        # Get all even bits of x
        even_bits = bs & 0xAAAAAAAA 
        # Get all odd bits of x
        odd_bits = bs & 0x55555555
        nup = countSetBits(even_bits)
        ndn = countSetBits(odd_bits)
        print(nup,ndn)
        if nup-ndn==spinz:
            result.append(bs)
    result = np.array(result)
    result.sort()
    return result

@jit(nopython=True)#,cache=True)
def residues(norb,determinant):
    ''' Returns list of residues, which is all possible ways to remove two
        electrons from a given determinant with number of orbitals norb
    '''
    residue_list = []
    nonzero = countSetBits(determinant)#bin(determinant).count('1')
    for i in range(norb):
        mask1 = (1 << i)
        for j in range(i):
            mask2 = (1 << j)
            mask = mask1 ^ mask2
            #if bin(determinant & ~mask).count('1') == (nonzero - 2):
            if countSetBits(determinant & ~mask)  == (nonzero - 2):
                residue_list.append(determinant & ~mask)
    return residue_list

@jit(nopython=True)#,cache=True)
def add_particles(norb,residue_list):
    ''' Returns list of determinants, which is all possible ways to add two
        electrons from a given residue_list with number of orbitals norb
    '''
    determinants = []
    for residue in residue_list:
        determinant = residue
        for i in range(norb):
            mask1 = (1 << i)
            if not bool(determinant & mask1):
                one_particle = determinant | mask1
                for j in range(i):
                    mask2 = (1 << j)
                    if not bool(one_particle & mask2):
                        two_particle = one_particle | mask2
                        determinants.append(two_particle)
    #return [format(det,'#0'+str(n_orbitals+2)+'b') for det in list(set(determinants))]
    return list(set(determinants))

@jit(nopython=True)#,cache=True)
def single_and_double_determinants(norb, determinant, use_Sz=False):
    #print("building cisd basis")
    #strb = '{0:0'+str(norb)+'b}'
    result = np.array([i for i in add_particles(norb, residues(norb, determinant))])
    if use_Sz ==True:
        result_sz = []
        for bs in result:
            # Get all even bits of x
            even_bits = bs & 0xAAAAAAAA 
            # Get all odd bits of x
            odd_bits = bs & 0x55555555
            nup = countSetBits(even_bits)
            ndn = countSetBits(odd_bits)
            #print(bs, strb.format(bs), nup,ndn)
            if nup-ndn==0:
                result_sz.append(bs)
                #print(bs, strb.format(bs), nup,ndn)
        result = np.array(result_sz)
    #print("sorting cisd basis")
    #result.sort()
    #print("finished cisd basis")
    return result

def build_no_trial_states(norb, nimp, nelc):
    """ Natural orbital convention as (nimp|empty|inter|filled) for example 
        1-orbital impurity        
        (01|00000|01|11111)
        (10|00000|10|11111)

        3-orbital impurity
        (010101|000000|010101|111111)
        (101010|000000|101010|111111)
    """
    print("building cisd basis")
    strb = '{0:0'+str(norb)+'b}'
    result = []
    for i in range(2):#spin-block
        tmp = 0
        for j in range(nimp//2):# impurity orbital block
            tmp += 1 <<(2*j+i)
        #print('tmp=',strb.format(tmp))
        tmp = tmp << (norb - nimp)
        #print('tmp1=',strb.format(tmp))
        tmp2 = 0
        for j in range(nimp//2):# intermediate obirtal block
            tmp2 += 1 <<(2*j+(1-i))
        tmp2 = tmp2 << (nelc - nimp)
        #print('tmp2=',strb.format(tmp2))
        tmp = tmp | tmp2
        for j in range(nelc-nimp):
            tmp = tmp | (1<<j)
        #print(strb.format(tmp))
        result.append(tmp)
    result = np.array(result)
    return result

if __name__ == "__main__":
  #basis = basis(2, True, True, False)
  #print(table_ep(4,2))
  #print(table_es(4,2,0))
  #print(table_es_sc(6,-2))
  #basis.create_basis()

  #Test postHF CISD basis
  norb = 20
  #print(len(table_ep(norb,norb//2)), table_ep(norb,norb//2))
  #strb = '{0:0'+str(norb)+'b}'
  reference_determinant = int(2**(norb//2) - 1) # reference determinant, lowest nEle orbitals filled
  #basis = np.array([reference_determinant])
  basis = single_and_double_determinants(norb, reference_determinant)
  #print (len(basis), basis)
  #basis = single_and_double_determinants(norb, reference_determinant, use_Sz=True)
  #print (len(basis), basis)

  #Test generate natural orbital
  #build_no_trial_states(24, 4, 12)
