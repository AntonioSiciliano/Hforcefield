import numpy as np
import time
import os
import shutil
import cellconstructor as CC

import H2model 
import H2model.Calculator

import matplotlib.pyplot as plt

import nonlinear_sscha
import nonlinear_sscha.NonLinearStructure as NLS
import nonlinear_sscha.NonLinearEnsemble as NLE
import nonlinear_sscha.Conversion as conv

import shutil

import pytest

def test_forces():
    # Get the current directory
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    # Set the force field
    ff_calculator = H2model.Calculator.ToyModelCalculator()
    ff_calculator.E = 0.0
    ff_calculator.model = 'rotating'
    re = ff_calculator.H2_re * conv.AU_TO_ANGSTROM

    # Set the initial guess for the FC constant
    Cart_dyn = CC.Phonons.Phonons('initial_H2_dyn',1)

    # The NonLinear Ensemble
    T0 = 0.
    NLensemble = NLE.NonLinearEnsemble(Cart_dyn, T0, Cart_dyn.GetSupercell())

    initial_pos = np.zeros((1,2,3))
    if ff_calculator.model == 'rotating':
        initial_pos[0,0,:] = np.array([-re /2., 0., 0.])
        initial_pos[0,1,:] = np.array([+re /2., 0., 0.])
    else:
        initial_pos[0,0,:] = np.array([+re /2., 0., 0.])
        initial_pos[0,1,:] = np.array([-re /2., 0., 0.])

    # GENERATE THE ENSEMBLE
    NLensemble.generate_nonlinear_ensemble(1, evenodd = False)
    NLensemble.xats = np.copy(initial_pos)
    NLensemble.structures[0].coords = np.copy(NLensemble.xats[0])

    NLensemble.u_disps *= 0.

    # In angstrom
#     x_range = np.linspace(0., re/5., 100) 
    x_range = np.linspace(-re/5., re/5., 200) 
    delta_x = x_range[1] - x_range[0]


    directory = 'Results'
    if os.path.isdir(directory):
        shutil.rmtree(directory)

    os.mkdir(directory)
    os.chdir(directory)

    for atom in range(2):
        for coord in range(3):
            energy = []
            force  = []

            for i in range(len(x_range)):
                NLensemble.xats = np.copy(initial_pos)
                NLensemble.xats[0,atom,coord] = np.copy(initial_pos[0,atom,coord] + x_range[i])
                NLensemble.structures[0].coords = np.copy(NLensemble.xats[0])
                r = NLensemble.xats[0,0] - NLensemble.xats[0,1]

                # Compute FORCES AND ENERGIES
                NLensemble.compute_ensemble(ff_calculator, compute_stress = False)

                energy.append(NLensemble.energies[0])
                force.append(NLensemble.forces[0,atom,coord])

            fig, ax = plt.subplots(2, 1, figsize = (10,10))

            force_num = -np.gradient(np.asarray(energy), delta_x)

            y = initial_pos[0,atom,coord] + x_range
            ax[0].plot(y, np.asarray(energy) * conv.RY_TO_mEV, 'ro', label='Energy (meV)')
            ax[1].set_ylabel('Energy (meV)')
            ax[0].legend()

            ax[1].plot(y, force_num, 'ro', label = 'Numerical')
            ax[1].plot(y, force, 'bx', label = 'Exact')
            ax[1].set_ylabel('Forces (Ry/Angstrom)')
            ax[1].set_xlabel('position at = {} coord = {} (Angstrom)'.format(atom, coord))
            ax[1].legend()
            
            if np.any(np.abs(force_num - force)>1e-1):
                raise ValueError('The force field is not working!')

            fig.tight_layout()
            plt.savefig('forces_at={}_comp={}.eps'.format(atom,coord))
    return 

if __name__ == "__main__":
    test_forces()

