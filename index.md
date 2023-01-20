# RPA計算

## C2_Cuについて

GPAWマニュアルに書いてあるジョブスクリプトをSiのマニュアルに従って分けて計算を進めます。

まず元のC2_Cuのスクリプト

```
from pathlib import Path

from ase import Atoms
from ase.build import fcc111
from gpaw import GPAW, PW, FermiDirac, MixerSum, Davidson
from gpaw.hybrids.energy import non_self_consistent_energy as nsc_energy
from gpaw.mpi import world
from gpaw.xc.rpa import RPACorrelation

# Lattice parametes of Cu:
d = 2.56
a = 2**0.5 * d
slab = fcc111('Cu', a=a, size=(1, 1, 4), vacuum=10.0)
slab.pbc = True

# Add graphite (we adjust height later):
slab += Atoms('C2',
              scaled_positions=[[0, 0, 0],
                                [1 / 3, 1 / 3, 0]],
              cell=slab.cell)


def calculate(xc: str, d: float) -> float:
    slab.positions[4:6, 2] = slab.positions[3, 2] + d
    tag = f'{d:.3f}'
    if xc == 'RPA':
        xc0 = 'PBE'
    else:
        xc0 = xc
    slab.calc = GPAW(xc=xc0,
                     mode=PW(800),
                     basis='dzp',
                     eigensolver=Davidson(niter=4),
                     nbands='200%',
                     kpts={'size': (12, 12, 1), 'gamma': True},
                     occupations=FermiDirac(width=0.05),
                     convergence={'density': 1e-5},
                     parallel={'domain': 1},
                     mixer=MixerSum(0.05, 5, 50),
                     txt=f'{xc0}-{tag}.txt')
    e = slab.get_potential_energy()

    if xc == 'RPA':
        e_hf = nsc_energy(slab.calc, 'EXX').sum()

        slab.calc.diagonalize_full_hamiltonian()
        slab.calc.write(f'{xc0}-{tag}.gpw', mode='all')

        rpa = RPACorrelation(f'{xc0}-{tag}.gpw',
                             ecut=[200],
                             txt=f'RPAc-{tag}.txt',
                             skip_gamma=True,
                             frequency_scale=2.5)
        e_rpac = rpa.calculate()[0]
        e = e_hf + e_rpac

        if world.rank == 0:
            Path(f'{xc0}-{tag}.gpw').unlink()
            with open(f'RPA-{tag}.result', 'w') as fd:
                print(d, e, e_hf, e_rpac, file=fd)

    return e


if __name__ == '__main__':
    import sys
    xc = sys.argv[1]
    for arg in sys.argv[2:]:
        d = float(arg)
        calculate(xc, d)
```

これを`pbe.py`  ,  `pbe_exx.py` ,  `rpa_init.py` ,  `rpa.py`の４つに分けて計算しました。

<br>

### `pbe.py`

```
from pathlib import Path

from ase import Atoms
from ase.build import fcc111
from gpaw import GPAW, PW, FermiDirac, MixerSum, Davidson
from gpaw.hybrids.energy import non_self_consistent_energy as nsc_energy
from gpaw.mpi import world
from gpaw.xc.rpa import RPACorrelation

d = 2.56
a = 2**0.5 * d
slab = fcc111('Cu', a=a, size=(1, 1, 4), vacuum=10.0)
slab.pbc = True

slab += Atoms('C2',
              scaled_positions=[[0, 0, 0],
                                [1 / 3, 1 / 3, 0]],
              cell=slab.cell)

slab.positions[4:6, 2] = slab.positions[3, 2] + d
tag = f'{d:.3f}'
slab.calc = GPAW(xc='PBE',
                 mode=PW(600),
                 eigensolver=Davidson(niter=4),
                 nbands='200%',
                 kpts={'size': (12, 12, 1), 'gamma': True},
                 occupations=FermiDirac(width=0.05),
                 convergence={'density': 1e-5},
                 parallel={'domain': 1},
                 mixer=MixerSum(0.05, 5, 50),
                 txt='pbe_output.txt')
e = slab.get_potential_energy()
```

<br>

### `pbe_exx.py`
```
from pathlib import Path

from ase import Atoms
from ase.build import fcc111
from gpaw import GPAW, PW, FermiDirac, MixerSum, Davidson
from gpaw.hybrids.energy import non_self_consistent_energy as nsc_energy
from gpaw.mpi import world
from gpaw.xc.rpa import RPACorrelation
from ase.parallel import paropen

resultfile = paropen('c2.pbe+exx.results.txt', 'a')

d = 2.56
a = 2**0.5 * d
slab = fcc111('Cu', a=a, size=(1, 1, 4), vacuum=10.0)
slab.pbc = True

slab += Atoms('C2',
              scaled_positions=[[0, 0, 0],
                                [1 / 3, 1 / 3, 0]],
              cell=slab.cell)


#def calculate(xc: str, d: float) -> float:
slab.positions[4:6, 2] = slab.positions[3, 2] + d
tag = f'{d:.3f}'
slab.calc = GPAW(xc='PBE',
                 mode=PW(600),
                 eigensolver=Davidson(niter=4),
                 nbands='200%',
                 kpts={'size': (12, 12, 1), 'gamma': True},
                 occupations=FermiDirac(width=0.05),
                 convergence={'density': 1e-5},
                 parallel={'domain': 1},
                 mixer=MixerSum(0.05, 5, 50),
                 txt='pbe+exx_output.txt')
e_pbe = slab.get_potential_energy()

slab.calc.write('slab.gpw', mode='all')

e_exx = nsc_energy('slab.gpw', 'EXX')

s = str(a)
s += ' '
s += str(12)
s += ' '
s += str(600)
s += ' '
s += str(e_pbe)
s += ' '
s += str(e_exx)
s += '¥n'
resultfile.write(s)
```

この際出力された`c2.pbe+exx.results.txt`の中身
```
3.620386719675124 12 600 -31.94102857829129 [  -31.94612995    19.5008031   7626.4361116  -6157.5358336
  -605.13956884  -703.89938009]¥n
```

<br>

### `rpa_init.py`
```
from pathlib import Path

from ase import Atoms
from ase.build import fcc111
from gpaw import GPAW, PW, FermiDirac, MixerSum, Davidson
from gpaw.hybrids.energy import non_self_consistent_energy as nsc_energy
from gpaw.mpi import world
from gpaw.xc.rpa import RPACorrelation

# Lattice parametes of Cu:
d = 2.56
a = 2**0.5 * d
slab = fcc111('Cu', a=a, size=(1, 1, 4), vacuum=10.0)
slab.pbc = True

# Add graphite (we adjust height later):
slab += Atoms('C2',
              scaled_positions=[[0, 0, 0],
                                [1 / 3, 1 / 3, 0]],
              cell=slab.cell)


#def calculate(xc: str, d: float) -> float:
slab.positions[4:6, 2] = slab.positions[3, 2] + d
tag = f'{d:.3f}'
slab.calc = GPAW(xc='PBE',
                 mode=PW(600),
                 eigensolver=Davidson(niter=4),
                 nbands='200%',
                 kpts={'size': (12, 12, 1), 'gamma': True},
                 occupations=FermiDirac(width=0.05),
                 convergence={'density': 1e-5},
                 parallel={'domain': 1},
                 mixer=MixerSum(0.05, 5, 50),
                 txt='rpa.pbe_output.txt')
e = slab.get_potential_energy()

slab.calc.diagonalize_full_hamiltonian()

slab.calc.write('slab.all.gpw', mode='all')
```

<br>

### `rpa.py`
```
from gpaw.xc.rpa import RPACorrelation

rpa = RPACorrelation('slab.all.gpw',
                     txt='rpa.rpa_output.txt',
                     skip_gamma=True,
                     frequency_scale=2.5)

e_rpac = rpa.calculate(ecut=[200])[0]
```
`rpa.py`の結果について、-cオプションをつけて並列化させたところ、30分で計算が終了しました。

`rpa.rpa_output.txt`の中身
```
----------------------------------------------------------
Non-self-consistent RPA correlation energy
----------------------------------------------------------
Started at:   Fri Jan 20 22:37:00 2023

Atoms                          : C2Cu4
Ground state XC functional     : PBE
Valence electrons              : 52
Number of bands                : 4864
Number of spins                : 1
Number of k-points             : 144
Number of irreducible k-points : 19
Number of q-points             : 144
Number of irreducible q-points : 19

    q: [-0.0000 -0.0000 0.0000] - weight: 0.007
    q: [0.0833 0.0833 0.0000] - weight: 0.042
    q: [0.1667 0.0833 0.0000] - weight: 0.042
    q: [0.1667 0.1667 0.0000] - weight: 0.042
    q: [0.2500 0.1667 0.0000] - weight: 0.083
    q: [0.2500 0.2500 0.0000] - weight: 0.042
    q: [0.3333 0.6667 0.0000] - weight: 0.014
    q: [0.3333 0.1667 0.0000] - weight: 0.042
    q: [0.3333 0.2500 0.0000] - weight: 0.083
    q: [0.3333 0.3333 0.0000] - weight: 0.042
    q: [-0.5833 -0.2500 0.0000] - weight: 0.083
    q: [0.4167 -0.1667 0.0000] - weight: 0.042
    q: [0.4167 0.2500 0.0000] - weight: 0.083
    q: [0.4167 0.3333 0.0000] - weight: 0.083
    q: [0.4167 0.4167 0.0000] - weight: 0.042
    q: [0.5000 0.2500 0.0000] - weight: 0.042
    q: [0.5000 0.3333 0.0000] - weight: 0.083
    q: [0.5000 0.4167 0.0000] - weight: 0.083
    q: [0.5000 0.5000 0.0000] - weight: 0.021

----------------------------------------------------------
----------------------------------------------------------

Analytical coupling constant integration

Frequencies
    Gauss-Legendre integration with 16 frequency points
    Transformed from [0,oo] to [0,1] using e^[-aw^(1/B)]
    Highest frequency point at 800.0 eV and B=2.5

Parallelization
    Total number of CPUs          : 256
    G-vector decomposition        : 1
    K-point/band decomposition    : 256

Response function bands : Equal to number of plane waves
Plane wave cutoffs (eV) : 200.000

Not calculating E_c(q) at Gamma

# 1  -  22:37:00
q = [0.083 0.083 0.000]
E_cut = 200 eV / Bands = 895:
E_c(q) = -74.136 eV

# 2  -  22:38:09
q = [0.167 0.083 0.000]
E_cut = 200 eV / Bands = 913:
E_c(q) = -69.132 eV

# 3  -  22:39:26
q = [0.167 0.167 0.000]
E_cut = 200 eV / Bands = 929:
E_c(q) = -67.524 eV

# 4  -  22:40:43
q = [0.250 0.167 0.000]
E_cut = 200 eV / Bands = 944:
E_c(q) = -64.833 eV

# 5  -  22:42:06
q = [0.250 0.250 0.000]
E_cut = 200 eV / Bands = 959:
E_c(q) = -63.593 eV

# 6  -  22:43:31
q = [0.333 0.667 0.000]
E_cut = 200 eV / Bands = 987:
E_c(q) = -58.814 eV

# 7  -  22:45:02
q = [0.333 0.167 0.000]
E_cut = 200 eV / Bands = 953:
E_c(q) = -62.245 eV

# 8  -  22:46:27
q = [0.333 0.250 0.000]
E_cut = 200 eV / Bands = 968:
E_c(q) = -61.983 eV

# 9  -  22:47:56
q = [0.333 0.333 0.000]
E_cut = 200 eV / Bands = 955:
E_c(q) = -60.996 eV

# 10  -  22:49:20
q = [-0.583 -0.250 0.000]
E_cut = 200 eV / Bands = 971:
E_c(q) = -58.881 eV

# 11  -  22:50:48
q = [0.417 -0.167 0.000]
E_cut = 200 eV / Bands = 977:
E_c(q) = -59.062 eV

# 12  -  22:52:16
q = [0.417 0.250 0.000]
E_cut = 200 eV / Bands = 968:
E_c(q) = -60.463 eV

# 13  -  22:53:44
q = [0.417 0.333 0.000]
E_cut = 200 eV / Bands = 966:
E_c(q) = -60.123 eV

# 14  -  22:55:11
q = [0.417 0.417 0.000]
E_cut = 200 eV / Bands = 965:
E_c(q) = -59.656 eV

# 15  -  22:56:36
q = [0.500 0.250 0.000]
E_cut = 200 eV / Bands = 971:
E_c(q) = -59.425 eV

# 16  -  22:58:03
q = [0.500 0.333 0.000]
E_cut = 200 eV / Bands = 970:
E_c(q) = -59.373 eV

# 17  -  22:59:33
q = [0.500 0.417 0.000]
E_cut = 200 eV / Bands = 977:
E_c(q) = -59.337 eV

# 18  -  23:01:01
q = [0.500 0.500 0.000]
E_cut = 200 eV / Bands = 964:
E_c(q) = -59.185 eV

==========================================================

Total correlation energy:
   200:   -61.4564 eV

Calculation completed at:  Fri Jan 20 23:02:25 2023

Timing:                             incl.     excl.
----------------------------------------------------------
RPA:                             1525.167     0.356   0.0% |
 chi0(q):                        1524.811     0.005   0.0% |
  Calculate CHI_0:               1502.546   150.804   9.9% |---|
   CHI_0 hermetian update:       1092.081  1092.081  71.6% |----------------------------|
   Get eigenvalues:                 0.014     0.014   0.0% |
   Get kpoints:                     0.298     0.007   0.0% |
    Group kpoints:                  0.022     0.022   0.0% |
    Initialize:                     0.269     0.244   0.0% |
     Analyze:                       0.022     0.000   0.0% |
      Group kpoints:                0.022     0.022   0.0% |
     Analyze symmetries.:           0.003     0.003   0.0% |
   Get matrix element:            251.066     3.631   0.2% |
    Get kpoint pair:               12.883     0.000   0.0% |
     fft indices:                   0.001     0.001   0.0% |
     get k-points:                 12.881     0.001   0.0% |
      Get a k-point:               12.881     0.033   0.0% |
       Load projections:            0.096     0.096   0.0% |
       load wfs:                   12.752    12.752   0.8% |
    Group kpoints:                  0.003     0.003   0.0% |
    get_pair_density:             234.550     0.010   0.0% |
     conj:                          0.053     0.053   0.0% |
     paw:                         234.486     3.142   0.2% |
      Calculate pair-densities:   231.345    28.315   1.9% ||
       fft:                       184.381   184.381  12.1% |----|
       gemm:                       18.649    18.649   1.2% |
   Initialize PAW corrections:      8.284     8.284   0.5% |
  Energy:                          22.260    22.260   1.5% ||
Read ground state:                  0.000     0.000   0.0% |
Other:                              0.302     0.302   0.0% |
----------------------------------------------------------
Total:                                     1525.469 100.0%
```

妥当な結果が得られてるのかはまだ確認できていませんが、とりあえずこの流れは実現できるはずです。