# Read real data for multi-spectral CT
# Author: Evelyn Cueva

# %%
from misc import TotalVariationNonNegative as TVnn
from odl.solvers import L2NormSquared
import numpy as np
import misc
import h5py
import odl
import os
# %%
# Folders to save data and images
dataset = 'bird'
cmap = 'gray'

folder_data = './data/{}'.format(dataset)
folder_images = './{}_images'.format(dataset)

if not os.path.exists(folder_data):
    os.makedirs(folder_data)

if not os.path.exists(folder_images):
    os.makedirs(folder_images)

view_energies = [40, 80, 120]  # keV
energies = ['E0', 'E1', 'E2']
data_mode = 'single'

sub_nangles = 90
sub_ndet = 552
sub_m = 512
sub_step1 = np.int(552/sub_ndet)
sub_step2 = np.int(720/sub_nangles)

variables = {}

with open('{}/parameters.txt'.format(folder_data)) as data_file:
    for line in data_file:
        name, value = line.split(" ")
        variables[name] = float(value)

dom_width = variables["dom_width"]
vmax = {'E0': variables['vmaxE0'],
        'E1': variables['vmaxE1'],
        'E2': variables['vmaxE2']}
vmax_sinfo = variables["vmax_sinfo"]
src_radius = variables["src_radius"]
det_radius = variables["det_radius"]
a = variables['scale_data']
a_str = np.str(a).replace('.', '_')

# %%
interp = 'nearest'
dtype = 'float64'
structure_low = h5py.File('./data/bird/QuailChestPhantom'
                          'ALow_ct_project_2d.mat', 'r')
structure_medium = h5py.File('./data/bird/QuailChestPhantom'
                             'AMid_ct_project_2d.mat', 'r')
structure_high = h5py.File('./data/bird/QuailChestPhantom'
                           'AHigh_ct_project_2d.mat', 'r')

sinogram_low = np.array(structure_low['CtData']['sinogram'])
sinogram_mid = np.array(structure_medium['CtData']['sinogram'])
sinogram_high = np.array(structure_high['CtData']['sinogram'])

parameters = structure_low['CtData']['parameters']
distSourceDetector = parameters['distanceSourceDetector'][0][0]
distSourceOrigin = parameters['distanceSourceOrigin'][0][0]
pixelSize = parameters['pixelSize'][0][0]

# %%
ndet_full, nangles_full = sinogram_low.shape

full_data_E0 = np.array(sinogram_low)
full_data_E0[full_data_E0 < 0] = 0
full_data_E1 = np.array(sinogram_mid)
full_data_E1[full_data_E1 < 0] = 0
full_data_E2 = np.array(sinogram_high)
full_data_E2[full_data_E2 < 0] = 0

full_sino = {}
full_sino['E0'] = full_data_E0
full_sino['E1'] = full_data_E1
full_sino['E2'] = full_data_E2

# %% Geometry for REFERENCES in full size 360 x 276

U = odl.uniform_discr([-dom_width/2, -dom_width/2], [dom_width/2, dom_width/2],
                      (sub_m, sub_m))


ray_transform = misc.forward_operator(dataset, 'full')

# %% References reconstruction
ref_alphas = {'E0': 5e-3, 'E1': 2e-3, 'E2': 2e-3}

energies = ['E0', 'E1', 'E2']
method = 'TV'

for i, energy in enumerate(energies):
    alpha_TV = ref_alphas[energy]
    alpha = 0 * np.int(method == 'noprior') + alpha_TV * np.int(method == 'TV')

    isino_full = full_sino[energy]
    isino_half = isino_full[0:ndet_full:1, 0:nangles_full:1]
    a2, a1 = isino_half.shape
    name = '{}_{}x{}_reference_reconstruction_a{}'.format(energy, sub_m, sub_m,
                                                          a_str)
    if method == 'TV':
        name = name + '_alpha_' + str(alpha).replace('.', '_')
    ref_dir = '{}/{}_d{}x{}.npy'.format(folder_data, name, a1, a2)
    name = name + '_d{}x{}'.format(a1, a2)
    print(name)
    print(ref_dir)
    if os.path.isfile(ref_dir):
        os.system('say "Esta configuracion ya fue ejecutada"')
        continue

    sino = ray_transform.range.element(isino_half.transpose())
    prox_options = {}
    prox_options['niter'] = 10
    prox_options['tol'] = 1e-4
    prox_options['name'] = 'FGP'
    prox_options['warmstart'] = True
    prox_options['p'] = None
    alpha = alpha
    strong_convexity = 0
    grad = None

    data_fit = 0.5 * L2NormSquared(sino.space).translated(sino)
    g = data_fit * ray_transform

    Ru = TVnn(U, alpha=alpha, prox_options=prox_options, grad=grad,
              strong_convexity=strong_convexity)

    function_value = g + Ru

    cb1 = odl.solvers.CallbackPrintIteration
    cb2 = odl.solvers.CallbackPrint
    cb3 = odl.solvers.CallbackPrintTiming
    cb4 = odl.solvers.CallbackShow(step=50)

    cb = (cb1(fmt='iter:{:4d}', step=1, end=', ') &
          cb2(function_value, fmt='f(x)={0:.4g}', step=1, end=', ') &
          cb3(fmt='time: {:5.2f} s', cumulative=True, step=1) &
          cb4)

    x = U.one()
    sigma = 0.4
    niter = 500

    misc.fbs(U, g, Ru, sigma, g_grad=None, x=x, niter=niter, callback=cb)

    no_prior_solution = x
    no_prior_solution.show()
    fv = function_value(x)

    # Save
    solution = []
    solution.append(no_prior_solution)
    solution.append(niter)
    solution.append(fv)
    np.save(ref_dir, solution)

# %% SAMPLED SINOGRAM: 60x276
sampled_sino = {}
energies = ['E0', 'E1', 'E2']

for energy in energies:
    B = full_sino[energy]
    sampled_sino[energy] = B[0:ndet_full:sub_step1,
                             0:nangles_full:sub_step2].transpose()

np.save('{}/sinograms_{}x{}.npy'.format(folder_data, sub_nangles,
        sub_ndet), sampled_sino)

# %% SINFO SINOGRAM: 60x276
sinfo_sino = np.zeros((sub_nangles, sub_ndet))

for energy in energies:
    sinfo_sino += sampled_sino[energy]

name = 'sinfo_sinogram_{}x{}'.format(sub_nangles, sub_ndet)
sinfo_dir = '{}/{}.npy'.format(folder_data, name)
np.save(sinfo_dir, sinfo_sino)

# %% GEOMETRY for reconstructing SINFO

sub_ray_transform = misc.forward_operator(dataset, 'sample')

sino = sub_ray_transform.range.element(sinfo_sino)

# %% TV RECONSTRUCTION SINFO with TV alpha in alphas
alphas = [1e-2, 3e-2, 5e-2]
for alpha in alphas:

    sinfo_name = 'sinfo_TV_reconstruction_{}'\
                 '_d{}x{}_m{}_a{}'.format(str(alpha).replace('.', '_'),
                                          sub_nangles, sub_ndet, sub_m,
                                          a_str)

    sinfo_file = '{}/{}.npy'.format(folder_data, sinfo_name)
    if os.path.isfile(sinfo_file):
        os.system('say "Esta configuracion ya fue ejecutada"')
        continue

    prox_options = {}
    prox_options['niter'] = 10
    prox_options['tol'] = 1e-4
    prox_options['name'] = 'FGP'
    prox_options['warmstart'] = True
    prox_options['p'] = None
    alpha = alpha
    strong_convexity = 0
    grad = None

    data_fit = 0.5 * L2NormSquared(sino.space).translated(sino)
    g = data_fit * sub_ray_transform

    Ru = TVnn(U, alpha=alpha, prox_options=prox_options, grad=grad,
              strong_convexity=strong_convexity)

    function_value = g + Ru

    cb1 = odl.solvers.CallbackPrintIteration
    cb2 = odl.solvers.CallbackPrint
    cb3 = odl.solvers.CallbackPrintTiming
    # cb4 = odl.solvers.CallbackShow(step=10)

    cb = (cb1(fmt='iter:{:4d}', step=1, end=', ') &
          cb2(function_value, fmt='f(x)={0:.4g}', step=1, end=', ') &
          cb3(fmt='time: {:5.2f} s', cumulative=True, step=1))

    x = U.one()
    sigma = 0.4
    niter = 500

    misc.fbs(U, g, Ru, sigma, g_grad=None, x=x, niter=niter, callback=cb)

    sinfo_TV = x
    np.save(sinfo_file, sinfo_TV)
