import numpy as np

import ase
import ase.units
import ase.calculators.calculator as calc

import cellconstructor as CC 
import cellconstructor.Structure
import cellconstructor.Units as units


class ToyModelCalculator(calc.Calculator):

    def __init__(self, model = 'rotating',  *args, **kwargs):
        """
        Computes the potential and forces for an H atom:
        
            V(r) = H2_shift + H2_D * {1 - exp[- H2_a * (r - H2_re)]}^2 + E * x  - V_min,
        
            where r = |(x,y,z)|.
        """
        calc.Calculator.__init__(self, *args, **kwargs)
        
        # Setup what properties the calculator can load
        self.implemented_properties = ["energy", "forces"]
        
        # Chose between 'rotating' and 'vibrating'
        self.model = model
        
        # Parameters of the Morse potential in ATOMIC UNITS
        
        # The rigid energy shift in HARTREE
        self.H2_shift = -1.17225263
        
        # Depth of the potential in HARTREE
        self.H2_D     =  0.13681332
        
        # The force constant of the potential in 1/BOHR
        self.H2_a     =  1.21606669
        
        # The equilibrium bond lenght in BOHR
        self.H2_re    =  1.21606669
        
        # Crystal field in HARTREE /BOHR
        self.E = np.linspace(0., 0.001, 10)[1]
        
        # An harmonic constant in HARTREE/BOHR^2
        self.k_harm = 2. * self.H2_a **2 * self.H2_D
        
    def minimum(self):
        """
        Get the minimum of the full potential since this can be done analitycally

        Returns:
        -------
            -Vmin: double, the minimum of the potential in HARTREE
        """

        expmin = 0.5 * (1. + np.sqrt(1 + 2. * self.E * np.cos(np.pi)/(self.H2_D * self.H2_a)))

        rmin = self.H2_re - np.log(expmin) / self.H2_a

        Vmin = self.E * rmin * np.cos(np.pi) + self.H2_shift + self.H2_D * (1 - expmin)**2
    
        return Vmin
    
    def calculate(self, atoms = None,  *args, **kwargs):
        """
        COMPUTES ENERGY AND FORCES IN eV and eV/ ANGSTROM
        =================================================
        
        Returns:
        -------
            -self.results: a dict with energy and forces.
        """
        calc.Calculator.calculate(self, atoms, *args, **kwargs)
        
        # Energy and force in HARTREE and HARTREE/BOHR
        energy, force = 0., np.zeros((1,3), dtype = np.double)

        # Position in ANGSTROM converted in BOHR, np.array with shape = (1, 3)
        coords = atoms.get_positions() * units.A_TO_BOHR
        
        
        if self.model == 'rotating':
            ####################
            # ROTATIONAL MODEL #
            ####################

#             # Get the relative coordinate
#             rel_coord =  np.sqrt(coords[0,:]**2)

            # Get the radial distance
            r = np.sqrt(np.sum(coords[0,:]**2))

            # Get the energy in HARTREE subtrating the minimum of the Morse + crystal field potential
            energy = self.H2_shift + self.H2_D * (1. - np.exp(-self.H2_a * (r - self.H2_re)))**2 + self.E * coords[0,2] - self.minimum()

            # Derivative with respect the radial distance
            diff_V_r = 2. * self.H2_a * self.H2_D * (1. - np.exp(-self.H2_a * (r - self.H2_re))) * np.exp(-self.H2_a * (r - self.H2_re))

            # Get the forces for the first particle in HARTREE /BOHR
            force[0,:]  = - diff_V_r * coords /r
            force[0,2] += - self.E

        else:
            #####################
            # VIBRATIONAL MODEL #
            #####################

            energy = 0.5 * self.k_harm * coords[0,0]**2 + 0.5 * self.k_harm * coords[0,1]**2 + 0.5 * self.k_harm * (coords[0,2] + self.H2_re)**2 

            force[0,0] = - self.k_harm * coords[0,0]

            force[0,1] = - self.k_harm * coords[0,1]

            force[0,2] = - self.k_harm * (coords[0,2] + self.H2_re)

        
        # CONVERT from HARTREE, HARTREE /BOHR in -> eV, eV /ANGSTROM
        self.results = {"energy": energy * 2. * units.RY_TO_EV, "forces": force * 2. * units.RY_TO_EV /units.BOHR_TO_ANGSTROM}
    
        return self.results
