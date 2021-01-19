# Generate synthetic data for multi-spectral CT
# Author: Evelyn Cueva
# %%
from misc import TotalVariationNonNegative as TVnn
from odl.solvers import L2NormSquared
import matplotlib.pyplot as plt
import misc_dataset as miscD
import numpy as np
import misc
import scipy.io
import random
import odl
import os
# %%
# Folders for data
I0_add = 1e3
dataset = 'geometric'
folder_data = './data/{}'.format(dataset)
folder_images = './{}_images'.format(dataset)
interp = 'nearest'
dtype = 'float64'

if not os.path.isdir(folder_images):
    os.mkdir(folder_images)

if not os.path.isdir(folder_data):
    os.mkdir(folder_data)

# Energies to be observed

energies = ['E0', 'E1', 'E2']
titles = ['E_0', 'E_1', 'E_2']

# The given energy range in KeV
bins = np.arange(45, 115+1, 1)
# number of energy bins
nbins = len(bins)-1

# Load main parameters
variables = {}

with open('{}/parameters.txt'.format(folder_data)) as data_file:
    for line in data_file:
        name, value = line.split(" ")
        variables[name] = float(value)

nd = np.int(variables['ndet'])
vmax = {'E0': variables['vmaxE0'],
        'E1': variables['vmaxE1'],
        'E2': variables['vmaxE2']}
vmax_sinfo = variables["vmax_sinfo"]

if dataset == 'materials':
    sinfo_sino_max = 6
    sino_max = {'E0': 5,
                'E1': 3,
                'E2': 2}
if dataset == 'geometric':
    sinfo_sino_max = 3
    sino_max = {'E0': 2,
                'E1': 0.8,
                'E2': 1}

dom_width = variables['dom_width']
src_radius = variables["src_radius"]
det_radius = variables["det_radius"]

a = variables['scale_data']
a_str = np.str(a).replace('.', '_')
bins_list = list(bins)
view_energies = [bins_list.index(np.int(variables['E0'])),
                 bins_list.index(np.int(variables['E1'])),
                 bins_list.index(np.int(variables['E2']))]

# Model parameters for full data set
p = 720    # number of projections (angles)
n = 512   # n^2 pixels

# Undersampling problem (in small or big sizes)
sub_nangles = 90  # 120  # 60
sub_ndet = nd  # np.int(nd/2)
sub_m = 512  # np.int(n/2)  # n  # np.int(n/2)
sub_step1 = np.int(nd/sub_ndet)
sub_step2 = np.int(p/sub_nangles)

# Fan-beam acquisition geometry (2D)
src_to_rotc = 3.0                 # dist. from source to rotation center
src_to_det = 5.0                  # dist. from source to detector
det_width = 2.0                   # detector width

# %% FROM MATLAB
mat1 = scipy.io.loadmat('{}/SpectralInfo{}.mat'.format(folder_data, n))
s = mat1['s'] * I0_add  # Spectrum, initial intensity per energy
Vl = mat1['Vl']  # (70x4)
Vltmp_4 = mat1['Vltmp_v']  # (70x4) Vltmp has 4 columns, then AUt*Vltmp

if dataset == 'materials':
    Ut = mat1['Ut']  # (nxnx4) ground truth decomposed by materials
    Ut_up = mat1['Ut_up']  # (nxnx4) Phantom for each material
    Vltmp = Vltmp_4
    sinfo_sino_max = 6
if dataset == 'geometric':
    Ut = miscD.geometric_material(n, dom_width)
    Ut_up = miscD.geometric_material(n, dom_width)
    Vltmp = Vltmp_4[:, :3]
    sinfo_sino_max = 3

# the combinations of each material to create a sinogram in each energy channel

Em = np.zeros((nbins, 1))  # array for mean energy in bins
sb = np.zeros((nbins, 1))  # array for number of photons in each bin

for k in range(0, nbins):
    Ii = bins[k]  # intervals of energy
    sk = s[Ii-1]
    Em[k] = Ii * sk/np.sum(sk)
    sb[k] = np.sum(sk)

view_I0 = [s[i-1, 0] for i in view_energies]
# %% Generate sinograms for each energy (ndet*angles, 4)
distSourceOrigin = src_to_rotc
ndet = nd
nangles = p

ray_transform = misc.forward_operator(dataset, 'full')

U = odl.uniform_discr([-dom_width*0.5, -dom_width*0.5],
                      [dom_width*0.5, dom_width*0.5],
                      (n, n))
# AUt has as columns each sinogram for each material
AUt = np.zeros((nd*p, Ut.shape[2]))
for k in range(0, Ut.shape[2]):
    sinogram_k = ray_transform(U.element(Ut_up[:, :, k]))
    sino_k = sinogram_k.asarray()
    AUt[:, k] = np.reshape(sino_k, nd*p)

# %% Ground truth for each energy, based on the 4 materials and the intensities
# in each energy

# BUt has as columns each groundtruth for each material
BUt = np.zeros(((n)**2, Ut.shape[2]))
gt = np.zeros(((n)**2, nbins))

for k in range(0, Ut.shape[2]):
    gt_k = Ut_up[:, :, k]
    BUt[:, k] = np.reshape(gt_k, (n)**2)

for k in range(0, nbins):
    aux = BUt.dot(Vltmp[k].transpose())
    gt[:, k] = aux  # (n^2, 70)

name = '{}/gt_{}x{}_all_energies.npy'.format(folder_data, n, n)
np.save(name, gt)

# Choose gt in energy_views to save and plot
gt_selected = {}
cmap = 'bone'
for i, vw in enumerate(view_energies):
    energy = energies[i]
    gt_selected[energy] = np.reshape(gt[:, vw], (n, n))
    plt.clf()
    plt.imshow(gt_selected[energy], vmin=0, vmax=vmax[energy], cmap=cmap)
    plt.axis('off')
    plt.colorbar()
    name_igt = '{}_{}x{}_reference'.format(energy, n, n)
    plt.savefig('{}/{}.pdf'.format(folder_images, name_igt), format='pdf',
                dpi=1000, bbox_inches='tight', pad_inches=0.05)
    plt.clf()

name_gt = '{}/gt_{}x{}_selected_energies.npy'.format(folder_data, n, n)
np.save(name_gt, gt_selected)

# %% GENERATE DATA WITH POISSON NOISE
Y_nocrime = np.zeros((nd*p, nbins))
Y_gt = np.zeros((nd*p, nbins))

# Set rng seed
random.seed(99)
for k in range(0, nbins):
    E_k = bins[k]
    # aux is the sinogram (integral lines for each energy channel) n^2
    aux = AUt.dot(Vltmp[k].transpose())
    eta_i = s[E_k]  # I0 initial intensity
    Y_nocrime[:, k] = Y_nocrime[:, k] + np.random.poisson(eta_i*np.exp(-a*aux))
    Y_gt[:, k] = eta_i*np.exp(-aux)

plt.plot(a*aux)
Y = Y_nocrime

# log 1 = 0, since log 0 is undefined

Z = Y + 1 * [Y == 0]
Z2 = Y_gt + 1 * [Y_gt == 0]
YY = np.multiply(Z[0, :, :], 1/(sb.transpose()))
ZZ = np.multiply(Z2[0, :, :], 1/(sb.transpose()))

B = - np.log(YY) * 1/a

BB = - np.log(ZZ)
start_channel = 0
end_channel = Y.shape[1]
sino = B[:, start_channel:end_channel]  # (1024^2, 70) sinogram w poisson noise
# %%
#
# THREE SINGLE ENERGIES
#
#
# %% Generate dataset with 3 single energies
sino_single = {}
j = 0
for energy in energies:
    sino_single[energy] = np.reshape(sino[:, view_energies[j]], (p, nd))
    j += 1

np.save('{}/sinograms_{}x{}'.format(folder_data, p, nd),
        sino_single)

full_sino = sino_single

a1, a2 = full_sino['E2'].shape
a_str = np.str(a).replace('.', '_')
ref_dir = '{}/full_sino_{}x{}_a{}.npy'.format(folder_data, a1, a2, a_str)
np.save(ref_dir, full_sino)

# %% SAMPLED SINOGRAM: 90x552
sampled_sino = {}
energies = ['E0', 'E1', 'E2']
cmap = 'bone'

for energy in energies:
    B = full_sino[energy]
    sampled_sino[energy] = B[0:nangles:sub_step2, 0:ndet:sub_step1]

np.save('{}/sinograms_{}x{}.npy'.format(folder_data, sub_nangles,
        sub_ndet), sampled_sino)

# %% SINFO SINOGRAM: 90x552
sinfo_sino = np.zeros((sub_nangles, sub_ndet))

for energy in energies:
    sinfo_sino += sampled_sino[energy]

name = 'sinfo_sinogram_{}x{}'.format(sub_nangles, sub_ndet)
sinfo_dir = '{}/{}.npy'.format(folder_data, name)
np.save(sinfo_dir, sinfo_sino)

# %% SIDE INFORMATION RECONSTRUCTION: parameters and geometry

name = 'sinfo_sinogram_{}x{}'.format(sub_nangles, sub_ndet)
sinfo_dir = '{}/{}.npy'.format(folder_data, name)
sinfo_sino = np.load(sinfo_dir)
data_mode = 'single'
# %%
sub_U = odl.uniform_discr([-dom_width*0.5, -dom_width*0.5],
                          [dom_width*0.5, dom_width*0.5],
                          (sub_m, sub_m))

sub_ray_transform = misc.forward_operator(dataset, 'sample')

sino = sub_ray_transform.range.element(sinfo_sino)

# %% SIDE INFORMATION RECONSTRUCTION: using TV regularization
alphas = [1e-3, 1e-4, 5e-4]
for alpha in alphas:

    sinfo_name = 'TV_reconstruction_{}'.format(str(alpha).replace('.', '_'))
    name = 'sinfo_{}_d{}x{}_m{}_a{}'.format(sinfo_name, sub_nangles,
                                            sub_ndet, sub_m, a_str)

    sinfo_file = '{}/{}.npy'.format(folder_data, name)
    if os.path.isfile(sinfo_file):
        os.system('say "Esta configuracion ya fue ejecutada"')
        continue

    prox_options = {}
    prox_options['niter'] = 200
    prox_options['tol'] = None
    prox_options['name'] = 'FGP'
    prox_options['warmstart'] = True
    prox_options['p'] = None
    alpha = alpha
    strong_convexity = 0
    grad = None

    data_fit = 0.5 * L2NormSquared(sino.space).translated(sino)
    g = data_fit * sub_ray_transform

    Ru = TVnn(sub_U, alpha=alpha, prox_options=prox_options, grad=grad,
              strong_convexity=strong_convexity)

    function_value = g + Ru

    cb1 = odl.solvers.CallbackPrintIteration
    cb2 = odl.solvers.CallbackPrint
    cb3 = odl.solvers.CallbackPrintTiming
    # cb4 = odl.solvers.CallbackShow(step=10, vmin=0, vmax=2)

    cb = (cb1(fmt='iter:{:4d}', step=1, end=', ') &
          cb2(function_value, fmt='f(x)={0:.4g}', step=1, end=', ') &
          cb3(fmt='time: {:5.2f} s', cumulative=True, step=1)
          )

    # x = sub_U.element(np.load('{}/{}_initial.npy'.format(folder_data, name)))
    x = sub_U.one()
    # q = sub_U.zero()

    sigma = 0.4
    niter = 1000

    misc.fbs(sub_U, g, Ru, sigma, g_grad=None, x=x, niter=niter, callback=cb)
#    lc_sinfo = 0.21
#    misc.bregman_iteration(sub_U, g, Ru, sigma, lc_sinfo, g_grad=None,
#                           x=x, q=q, niter=niter, callback=cb)

    sinfo_TV = x
    np.save(sinfo_file, sinfo_TV)

    plt.close()
    plt.imshow(sinfo_TV, vmin=0, vmax=vmax_sinfo, cmap='gray')
    plt.axis('off')
    plt.colorbar()
    plt.savefig('{}/{}.pdf'.format(folder_images, name), format='pdf',
                dpi=1000, bbox_inches='tight', pad_inches=0.05)
    plt.close()
