# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 17:08:32 2021

@author: jan
"""

import numpy as np
import matplotlib.pyplot as plt

# from collections import defaultdict
from scipy.optimize import linprog
from scipy.special import comb, loggamma
from pqdict import maxpq
from numba import jit


class System:
    def __init__(self, description:str="No detailed description"):
        self.description = description
        self.species = []           # list of species
        self.species_fullname = {}  # dict of full names of species, for displaying in figures
                                    #   with strucutre {species_short_name: species_long_name}
        self.reactions = []         # mapping reactions names to their ids
        # Data structure:
        # reactions = [(reactants, products, rate, additional_stochiometric),
        #              ( -----------------------||-------------------------),
        #                                      ....                          ]
        
        
    def add_species(self, *species, full_names=False):
        """
        A function to add species to the system. If only short names of species
        are used, just write them as arguments of type str. If you also want to include
        long names of the species (e.g. for figures), follow each 


        Parameters
        ----------
        *species_name : strings
            Short names of species, optionally followed by long name. See examples.
            If some other type than string is used, it will be casted into string.
        full_names : bool, optional
            Whether is each species followed by its full name.
            The default is False.

        Returns
        -------
        None.
        
        Examples
        --------
        We want to add enzyme and protein to the system. This is done by:
        
        >>> System.add("P", "Protein", "E", "Enzyme", full_names=True)
        
        or by
        
        >>> System.add("P", "E")
        """
    
        if full_names:
            if len(species)%2 != 0:
                raise ValueError("There must be even number of strings with `full_names=True`")
            for i in range(0, len(species), 2):
                S, FN = str(species[i]), str(species[i+1])
                if S in self.species:
                    print("The species '%s' already exists!"%S)
                    continue
                self.species.append(S)
                self.species_fullname[S] = FN

        
        else:
            for S in species:
                S = str(S)
                if S in self.species:
                    print("The species '%s' already exists!"%S)
                    continue
                self.species.append(S)
                self.species_fullname[S] = S
    
    
    def rm_species(self, species_name):
        try:
            del self.species[species_name]
            del self.species_fullname[species_name]
        except KeyError:
            print("No such species. Avalaible are:")
            print(", ".join(self.species.keys()))
    
    
    def add_reaction(self, reactants:tuple, products:tuple, rate:float,
                     name:str="[No description]", additional_stoichiometry:dict={}, ):
        """
        
        Function to add a reaction to the system.

        
        Parameters
        ----------
        reactants : tuple or str
            If the reaction requires only one reactant, then this is its name.
            Otherwise if tuple of reactants names.
        products : tuple or str
            If the reaction has only one product, then this is its name.
            Otherwise if tuple of products names.
        rate : float
            Rate at which the reaction occurs.
        additional_stoichiometry : dictionary, optional
            If there is dummy species that should be incorporated into system,
            write its stochiometric coefficients in form of
            "Dummy_species":coeff. The default is {}.
        name: string
            Description of the reaction. Might be handy in large systems.

        Returns
        -------
        None.
        
        Examples
        --------
        
        We want to add reaction 2X+Y -> Z with rate 2.
        This is done by
        
        >>> add_reaction(("X", "X", "Y"), ("Z",), 2)
        
        
        We want to reaction X+Y -> 2Z with rate 1, and keep track of
        how many times this reaction occurred. We will create a dummy variable
        Number_of_occurences 
        
        >>> add_reaction(("X", "X", "Y"), ("Z",), 1, {"Number_of_occurences" : 1})

        """

        
        if type(reactants) is str:
            reactants = (reactants,)
        
        if type(products) is str:
            products = (products,)
            
        for s in reactants:
            if s not in self.species:
                print("Reactant", s, "not in species list, adding now...")
                self.add_species(s)
        
        for s in products:
            if s not in self.species:
                print("Product", s, "not in species list, adding now...")
                self.add_species(s)
        
        for s in additional_stoichiometry:
            if s not in self.species:
                print("Product", s, "not in species list, adding now...")
                self.add_species(s)
        # This could be faster if we used set of species, but meh
        
        self.reactions.append((reactants, products, rate, name, additional_stoichiometry))
        # TODO: Add propensity type as reaction property

    def rm_reaction(self, index:int):
        """
        A function to remove reaction by its index.
        To see a list of reactions use print_reactions() method.
        Parameters
        ----------
        index : int
            Index of reaction to be removed. 

        Returns
        -------
        None.

        """
        
    def _reaction_repr(self)->str:
        """
        A function to generate human-readible string of reactions

        Returns
        -------
        reaction_repr : string
            Human-readible string.

        """
        
        reaction_repr = ""
        for i, r in enumerate(self.reactions):
            reaction_repr += "Reaction no. " + str(i) + " (" + r[3] + ") " + ":\n"
            reactants = defaultdict(lambda:0)
            for s in r[0]:  # iterate through reactants
                reactants[s] += 1
                
            products = defaultdict(lambda:0)
            for s in r[1]:  # iterate through products
                products[s] += 1
            
            reaction_repr += " + ".join([str(reactants[s])+s for s in reactants]) +\
                             " -> " +\
                             " + ".join([str(products[s])+s for s in products]) +\
                             " with rate %.2g\n"%r[2]
            
            if len(r[4]) > 0:
                reaction_repr += "and additional dummy stoichiometry " + \
                ", ".join([str(r[4][s])+s for s in r[4]]) + "\n"
            
            reaction_repr += "\n"
            
        return reaction_repr
    
    
    def print_reactions(self):
        """
        Prints list of reaction in human-readible form.

        Returns
        -------
        None.

        """
        print(self._reaction_repr())
    
    
    def get_species_map(self) -> dict:
        """
        Function to get mapping from species name to position.

        Returns
        -------
        dict
            Dictionary of type species_name : .

        """
        
        species_map = {}
        
        for i, s in enumerate(self.species):
            species_map[s] = i
            
        return species_map
        
    
    def get_reactant_matrix(self, species_map:dict=None) -> np.ndarray:
        R = np.zeros((len(self.species), len(self.reactions)), order="F", dtype=int)
        
        if species_map is None:
            species_map = self.get_species_map()
        
        for j, r in enumerate(self.reactions):
            for s in r[0]: # subtract reactants
                R[species_map[s], j] += 1
        
        return R
    
    def get_stoichiometry(self, species_map:dict=None) -> (np.ndarray, dict):
        """
        Function to get stoichiometry matrix – how is the number of molecules
        changed after reactions. The matrix is in column-major format.
        Rows represent species, columns represent reactions.

        Returns
        -------
        S : TYPE
            DESCRIPTION.

        """
        
        S = np.zeros((len(self.species), len(self.reactions)), order="F", dtype=int)
        if species_map is None:
            species_map = self.get_species_map()
        
        for j, r in enumerate(self.reactions):
            for s in r[0]: # subtract reactants
                S[species_map[s], j] -= 1
                
            for s in r[1]: # add products
                S[species_map[s], j] += 1
            
            for s, coef in r[4].items():
                S[species_map[s], j] += coef
        
        return S
    
    
    def get_propensities_comb(self, state, R, rates):
        """
        Function to get reaction rates 

        Parameters
        ----------
        state : np.ndarray
            1-dimensional state vector.
        R : np.ndarray
            Reactant matrix. Rows represent species, columns represent
            reactions. Order of rows should match state vector, order
            of columns should match rates vector.
        rates : np.ndarray
            1-dimenstional vector of floats of reaction rates.

        Returns
        -------
        propesnities
            1-dimensional vector of propensities of reactions.

        """
        
        return np.prod(comb(state[:, None], R), 0) * rates
    
    
    def simulate_gillespie(self, initial_state:dict, t_max:float=np.inf,
                           n_iter:int=1000000, t_list:np.ndarray=None):
        """
        

        Parameters
        ----------
        initial_state : dict
            Dictionary in form of molecule_name: copy_number.
            Only non-zero inital copy numbers are needed.
        t_max : float, optional
            Stop simulation after t_max time. Ignored if `t_list` is used.
            The default is infinity.
        n_iter : int, optional
            Emergency stop the simulation after `n_iter` steps.
            Ignored if t_list is used. The default is 1000000.
        t_list : list or np.ndarray, optional
            If state of the system should be outputted only at certain time points.
            If used, t_max and n_iter is ignored.
            Use None to output at every transition. The default is None. 

        Returns
        -------
        t_eval : np.ndarray
            Time at which the system state was evaluated.
        copy_number : np.ndarray
            Number of molecules at each time of t_eval.
            Has shape `len(t_eval) × number_of_species`
        species_list : list
            List of species in order which they are outputted.
        """

    
        # Dispatch dependent on every event time or specific event time
        if t_list is None:
            return self._gillespie(initial_state, float(t_max), n_iter)
        else:
            return self._gillespie_list(initial_state, np.array(t_list), n_iter)
    
    
    def _gillespie(self, initial_state, t_max, n_iter):
        """
        Internal functon to generate sample path by gillespie algorithm.
        All events are noted and outputed.
        """
        
        rates = np.array([r[2] for r in self.reactions])
        t = [0]
        
        species_map = self.get_species_map()
        
        S = self.get_stoichiometry(species_map)
        R = self.get_reactant_matrix(species_map)
        
        
        x = [np.zeros(len(self.species))]  # current state vector is x[-1]
        for s in initial_state:  # filling in non-zero values from arguments
            try:
                x[-1][species_map[s]] = initial_state[s]
            except KeyError:
                print("Speces", s, "is n initial_state, but is not defined in system!")
                raise KeyError()
        
        for i in range(n_iter):
            alphas = self.get_propensities_comb(x[-1], R, rates)
            alpha_sum = np.sum(alphas)
            if alpha_sum == 0:  # No reaction will happen
                t.append(t_max)
                x.append(x[-1])
                break
            reaction_id = np.random.choice(len(alphas), p=alphas/alpha_sum)
            τ = np.random.exponential(scale=1/alpha_sum)
            
            t_new = t[-1] + τ
            
            if t_new > t_max:
                t.append(t_max)
                x.append(x[-1])
                break
            else:            
                t.append(t[-1] + τ)
                x.append(x[-1] + S[:, reaction_id])
        
        return np.array(t), np.array(x, dtype=int).T, self.species.copy()

    
    def _gillespie_list(self, initial_state, t_list, n_iter):
        """
        Internal functon to generate sample path by gillespie algorithm.
        Only states at time in t_list are outputed.
        """
        rates = np.array([r[2] for r in self.reactions])
        
        t_list = np.unique(t_list) # to remove duplicities
        t_list.sort()
        t = t_list[0]  # current time
        t_max = t_list[-1]
        
        index = 1   # t_list[index] is next time when we note system state
                      
        x = np.empty((len(t_list), len(self.species)), dtype=int)
        
        species_map = self.get_species_map()
        
        S = self.get_stoichiometry(species_map)
        R = self.get_reactant_matrix(species_map)
        
        x[0] = 0 # initialize state with zeros
        for s in initial_state:  # filling in non-zero values from arguments
            try:
                x[0, species_map[s]] = initial_state[s]
            except KeyError:
                print("Speces", s, "is n initial_state, but is not defined in system!")
                raise KeyError()
        
        current_state = x[0].copy()
        
        for i in range(1, n_iter+1):
            alphas = self.get_propensities_comb(current_state, R, rates)
            alpha_sum = np.sum(alphas)
            if alpha_sum == 0:  # No reaction will happen
                x[index:, :] = current_state[None, :]
                break
            
            reaction_id = np.random.choice(len(alphas), p=alphas/alpha_sum)
            τ = np.random.exponential(scale=1/alpha_sum)
            
            t += τ
            if t > t_max:
                x[index:, :] = current_state[None, :]
                break   
            
            while t >= t_list[index]:
                x[index] = current_state
                index += 1
            
            current_state += S[:, reaction_id]
        
        return t_list, x.T, self.species.copy()
    
    
    
    def simulate_tau_leap(self, initial_state, t_max, tau=None, n_iter=1000000, t_list=None):
        """
        A function to simulate 

        Parameters
        ----------
        initial_state : TYPE
            DESCRIPTION.
        t_max : TYPE
            DESCRIPTION.
        tau : TYPE, optional
            DESCRIPTION. The default is None.
        n_iter : TYPE, optional
            DESCRIPTION. The default is 1000000.
        t_list : TYPE, optional
            DESCRIPTION. The default is None.

        Raises
        ------
        KeyError
            DESCRIPTION.

        Returns
        -------
        t : TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        """
        
        # TODO automatic tau selection
        # This will be for long...
        # There are good articles on this, e.g. Cao, Gillespie, Petzol: Efiicient step size for the tau-leaping simulation method
        # For now I have to focus on other things
        
        rates = np.array([r[2] for r in self.reactions])
        
        t = [0]  # list of time points
                      
        x = [np.zeros(len(self.species), dtype=int)]
        
        species_map = self.get_species_map()
        
        S = self.get_stoichiometry(species_map)
        R = self.get_reactant_matrix(species_map)
        
        for s in initial_state:  # filling in non-zero values from arguments
            try:
                x[0][species_map[s]] = initial_state[s]
            except KeyError:
                print("Speces", s, "is n initial_state, but is not defined in system!")
                raise KeyError()
        
        current_state = x[0].copy()
        
        for i in range(n_iter):
            alphas = self.get_propensities_comb(current_state, R, rates)
            alpha_sum = np.sum(alphas)
            if alpha_sum == 0:
                t.append(t_max)
                x.append(current_state)
                break
            
            τ = 10/np.max(alphas)
            
            if t[-1] + τ > t_max:
                τ = t_max - t[-1]
            
            # print(alphas*tau)
            while True:  # to emulate do-while loop.
                reaction_count = np.random.poisson(alphas*τ)
                species_change = S @ reaction_count
                if np.all(current_state >= -species_change):  # check for non-neagativness
                    break
                τ /= 2  # make tau smaller, so there is higher probability
                          # that next time we will success
            
            t.append(t[-1] + τ)
            current_state += species_change
            x.append(current_state.copy())
            
            if t[-1] >= t_max:
                break
        
        return t, np.array(x).T, self.species.copy()
            
            
    
    #def simulate_MLMC(self, initial_state, t_max, t_list=None):
    #   pass
    
    
    def plot_simulations(self, initial_state:dict, t_max:float,
                         n_iter:int=1000000, t_list=None, N:int=1,
                         method=None, alpha:float=0.5,
                         not_shown=set()):
        """
        A function to plot `N` sample paths in one figure. Plots one main
        figure with all species (exluding those listed in `not_shown`) and one
        figure for each of species.

        Parameters
        ----------
        initial_state : dict
            Dictionary in form of molecule_name: copy_number.
            Only non-zero inital copy numbers are needed.
        t_max : float
            Stop simulation after t_max time. Ignored if t_list is used.
        n_iter : int, optional
            Emergency stop the simulation after n_iter steps.
            Ignored if t_list is used. The default is 1000000.
        t_list : list or np.ndarray, optional
            If state of the system should be outputted only at certain time
            points. If used, t_max and n_iter is ignored.
            Use None to output at every transition. The default is None. 
        N : int, optional
            Number of simulations to run. The default is 1.
        method : callable, optional
            Method used for simulation. The default None refers
            to simulate_gillespie.
        alpha : float, optional
            Alpha channel (opacity) of the lines in the plot. The default is 0.5.
        not_shown : set or iterable
            Collection of species names that should not be included in the plot
        
        Returns
        -------
        None

        """
        
        # TODO: custom color cycle
        
        not_shown = set(not_shown)
        
        method = method or self.simulate_gillespie
        
        t, x, species_list = method(initial_state=initial_state, t_max=t_max,
                                    n_iter=n_iter, t_list=t_list)
        
        plt.figure()
        main_plot_number = plt.gcf().number
        
        colorcycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        
        species_plots_numbers = {}
        for s in species_list:
            plt.figure()
            species_plots_numbers[s] = plt.gcf().number

        # First plot (with labels)
        for j, s in enumerate(species_list):
            # Add to main plot
            if s not in not_shown:
                plt.figure(main_plot_number)
                plt.step(t, x[j], where="post", alpha=alpha,
                         label=self.species_fullname[s])
            # Add to species plot
            plt.figure(species_plots_numbers[s])
            plt.step(t, x[j], where="post", alpha=alpha,
                     c=colorcycle[j])
            plt.title(self.species_fullname[s])
            
            
        # Other plots
        for i in range(1, N):
            t, x, species_list =method(initial_state=initial_state, t_max=t_max,
                                       n_iter=n_iter, t_list=t_list)
            plt.gca().set_prop_cycle(None)
            for j, s in enumerate(species_list):
                if s not in not_shown:
                    plt.figure(main_plot_number)
                    plt.step(t, x[j], where="post", alpha=alpha,
                             c=colorcycle[j])
                plt.figure(species_plots_numbers[s])
                plt.step(t, x[j], where="post", alpha=alpha,
                         c=colorcycle[j])
                  
        plt.figure(main_plot_number)
        plt.title("All species")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        
        for i in (species_plots_numbers.values()):
            plt.figure(i)
            plt.grid()
            plt.tight_layout()
        
        plt.show()
    
    
    def get_mean_variance_MC(self, initial_state:dict, t_max:float=np.inf,
                             n_iter:int=1000000, t_list=None, N:int=5,
                             method=None, geometric:bool=False):
        """
        A function to calclulate mean and standard deviation by Monte Carlo
        method, using `Ν` independent samples. 

        Parameters
        ----------
        initial_state : dict
            Dictionary in form of molecule_name: copy_number.
            Only non-zero inital copy numbers are needed.
        t_max : float
            Stop simulation after t_max time. Ignored if t_list is used.
            THe default is infinity. 
        n_iter : int, optional
            Emergency stop the simulation after n_iter steps.
            Ignored if t_list is used. The default is 1000000.
        t_list : list or np.ndarray, optional
            If state of the system should be outputted only at certain time
            points. If used, t_max and n_iter is ignored.
            Use None to output at every transition. The default is None. 
        N : int, optional
            Number of simulations to run. The default is 5.
        method : callable, optional
            Method used for simulation. The default None refers
            to simulate_gillespie.
        geometric : bool
            Whether to compute mean and std on log scale. The default is False.
            # TODO
            
        Returns
        -------
        t_list : np.ndarray
            Vector of time points at which mean and variance was caluculated.
        mean : np.ndarray
            Matrix of means at selected time points. Rows represent species,
            columns represent time
        std : np.ndarray
            Matrix of standard deviations at selected time points.

        """
        
        if t_list is None:
            t_list = np.linspace(0, t_max, 1001)
        method = method or self._gillespie_list
        
        x = np.empty((N, len(self.species), len(t_list)), dtype=int)
        
        for i in range(N):
            _, x[i], _ = method(initial_state, t_list, n_iter)
        
        if not geometric:
            mean = np.mean(x, 0)
            std = np.std(x, 0)
        else:
            mean = np.exp(np.mean(np.log(x+1),0))-1
            std = np.exp(np.mean(np.log(x+1), 0))
        
        return t_list, mean, std, x
        
    
    def plot_mean_variance_MC(self, initial_state:dict, t_max:float=np.inf,
                              n_iter:int=1000000, t_list=None, N:int=5,
                              method=None, alpha:float=0.5,
                              not_shown=set(), geometric:bool=False):
        """
        A function to plot mean and standard deviation by Monte Carlo
        method, using `Ν` independent samples. Uses `get_mean_variance_MC`
        under the hood.

        Parameters
        ----------
        initial_state : dict
            Dictionary in form of molecule_name: copy_number.
            Only non-zero inital copy numbers are needed.
        t_max : float
            Stop simulation after t_max time. Ignored if t_list is used.
            Thxe default is infinity (`n_iter` is the only stopping criterion).
        n_iter : int, optional
            Emergency stop the simulation after n_iter steps.
            Ignored if t_list is used. The default is 1000000.
        t_list : list or np.ndarray, optional
            If state of the system should be outputted only at certain time
            points. If used, t_max and n_iter is ignored.
            Use None to output at every transition. The default is None. 
        N : int, optional
            Number of simulations to run. The default is 5.
        method : callable, optional
            Method used for simulation. The default None refers
            to simulate_gillespie.
        alpha : float, optional
            Alpha channel (opacity) of the lines in the plot. The default is 0.5.
        not_shown : set or iterable
            Collection of species names that should not be included in the 
            main plot.
        geometric : bool
            Whether to compute mean and std on log scale. The default is False.

        Returns
        -------
        t_list : np.ndarray
            Time points at which mean and variance was caluculated.
        mean : TYPE
            DESCRIPTION.
        std : TYPE
            DESCRIPTION.

        """

        
        t, mean, std, x = self.get_mean_variance_MC(initial_state, t_max,
                                                    n_iter, t_list, N, method,
                                                    geometric)
        
        species_list = self.species.copy()
        
        plt.figure()
        main_plot_number = plt.gcf().number
        
        colorcycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        
        species_plots_numbers = {}
        for s in species_list:
            plt.figure()
            species_plots_numbers[s] = plt.gcf().number

        # First line (with labels)
        for j, s in enumerate(species_list):
            # Add to main plot
            if s not in not_shown:
                plt.figure(main_plot_number)
                plt.step(t, x[0, j], where="post", alpha=alpha,
                         label=self.species_fullname[s])
            # Add to species plot
            plt.figure(species_plots_numbers[s])
            plt.step(t, x[0, j], where="post", alpha=alpha,
                     c=colorcycle[j%len(colorcycle)], label="Sample path")
            plt.title(self.species_fullname[s])
            
            
        # Other lines
        for i in range(1, N):
            plt.gca().set_prop_cycle(None)
            for j, s in enumerate(species_list):
                if s not in not_shown:
                    plt.figure(main_plot_number)
                    plt.step(t, x[i, j], where="post", alpha=alpha,
                             c=colorcycle[j%len(colorcycle)])
                plt.figure(species_plots_numbers[s])
                plt.step(t, x[i, j], where="post", alpha=alpha,
                         c=colorcycle[j%len(colorcycle)])
                  
        plt.figure(main_plot_number)
        plt.title("All species")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        
        if not geometric:
            for i, s in enumerate(species_list):
                plt.figure(species_plots_numbers[s])
                plt.fill_between(t, mean[i]-std[i], mean[i]+std[i], alpha=.2)
                plt.fill_between(t, mean[i]-2*std[i], mean[i]+2*std[i], alpha=.2)
                plt.plot(t, mean[i], "k", label="Mean")
        else:
            for i, s in enumerate(species_list):
                plt.figure(species_plots_numbers[s])
                plt.fill_between(t, mean[i]*std[i], mean[i]/std[i], alpha=.2)
                plt.fill_between(t, mean[i]*std[i]**2, mean[i]/std[i]**2, alpha=.2)
                plt.plot(t, mean[i], "k", label="Mean")
        
        for i in (species_plots_numbers.values()):
            plt.figure(i)
            plt.legend()
            plt.grid()
            plt.tight_layout()
        
        plt.show()
    
    def _split_observations(self, *data):
        # Data should be in form:
        # ((t_0, t_1, t_2, ..., t_n),
        #  {species_name_1: (x_1(t_0), x_1(t_1), ..., x_1(t_n)),
        #   species_name_2: (x_2(t_0), x_2(t_1), ..., x_2(t_n)),
        #   ... } )
        
        transitions = [] # internal data structure, containing data about transitions
        # in form (Δt, x(t_start), x(t_stop))])
        # Δt is a scalar, x(t_start) and x(t_stop) are vector of copy numbers
        # at the beginning and at the end of the time interval
        # copy numbers are in the same order as self.species.
        
        # iterate throught experiments in dataset.
        for d in data:
            self._add_trasitions(d[0], d[1], transitions)
        
        return transitions
    
    
    def _add_trasitions(self, t, copy_counts, transitions):
        x = copy_counts
        assert len(x) == len(self.species), "The number of species does not match the system!"
        assert set(x) == set(self.species), "Species in given data do not match species of the system!"
        assert len(set([len(v) for v in x.values()] + [len(t)])) == 1, "The data are not aligned!"
        
        state = np.array([x[s][0] for s in self.species])
        
        for i in range(1, len(t)):
            # for this expoeriment we are going to 
            Δt = t[i] - t[i-1]
            state_start = state.copy()
            state_end = np.array([x[s][i] for s in self.species])
            
            # transition sould be a list, which is mutable. We don't need to return the new list.
            transitions.append((Δt, state_start, state_end))
            
    
    def _get_truncated_state_space(self, Δt, state_start, state_end,
                                   rates=None, epsilon=1/100, n_max = 100000,
                                   A=None):
        rates=None
        epsilon = 1e-5
        n_max = 1000000
        #A = None
        
        # First guess on probability
        # dijkstra from the end to determine from where you cannot get into end_state
        # dijstra from the start, discarding all but feasible set
        if rates is None:
            rates = np.array([r[2] for r in self.reactions])
        else:
            rates = np.array(rates)

        dtype = state_start.dtype
        assert state_end.dtype == dtype, "Data type mismatch!"
        
        species_map = self.get_species_map()
        S = self.get_stoichiometry(species_map)
        R = self.get_reactant_matrix(species_map)

        change = state_end - state_start

        # TODO: What if we could observe only A@state, where A is a known matrix?
        # if A is None:
        #     A = np.eye(S.shape[0])
        # A = np.atleast_2d(np.array(A))
        # assert len(A.shape) == 2, "Matrix `Α` must be vector or 2-dimensional!"
        # assert A.shape[1] == S.shape[0], "Matrix A must have %d columns!"%S.shape[0]

        
        # We use Simpson 1-3-3-1 rule for average propensity
        # TODO: incorporate matrix A into this. LP might be needed for these vectors.
        P1 = self.get_propensities_comb(state_start, R, rates)
        P2 = self.get_propensities_comb((2*state_start + state_end)/3, R, rates)
        P3 = self.get_propensities_comb((state_start + 2*state_end)/3, R, rates)
        P4 = self.get_propensities_comb(state_end, R, rates)
        
        avg_a = (P1 + 3*P2 + 3*P3 + P4)/8
        avg_a0 = np.sum(avg_a)
        valid_reactions = avg_a > 0
        
        
        c = np.log(avg_a[valid_reactions]) - np.log(np.sum(avg_a))
        
        # res = linprog(-c, A_eq=A@S, b_eq=A@change, method="revised simplex") # if slow, change to interior_point
        res = linprog(-c, A_eq=S, b_eq=change, method="revised simplex") # if slow, change to interior_point
        
        if res.status != 0:
            raise ValueError("The state seems to be infeasible, or other problems in linprog phase occurred.")
        
        x = np.ceil(res.x)
        xplus1 = np.ceil(x) + 1
        
        # log of number of different paths consisting of x[i] edges S[:, i]
        # loggamma(np.sum(x)+1) - np.sum(loggamma(xplus1))

        
        
        # TODO: determine set of states from which the state_end is feasible with a reasonable probability
        # state_start should lie within the region. The treshold for probability should be approximated by
        # this expression
            
        treshold = (np.inner(np.log(avg_a[valid_reactions]), x[valid_reactions]) -\
                    np.sum(x) * np.log(avg_a0)) +\
                    loggamma(np.sum(x)+1) - np.sum(loggamma(xplus1)) +\
                    np.sum(x) * np.log(1 - np.exp(-avg_a0 * Δt)) +\
                    np.log(epsilon)        
        
        # We do two-phase Dijkstra search. In the first phase we determine the
        # set of states from which it is possible to get to state_end
        
        # Backward Dijkstra

        heap = maxpq()

        heap.additem(state_end.tobytes(), 0)
        
        probable_states_backward = set()
        
        
        for j in range(n_max):
            # most probable state so far
            try:
                statebytes, logp = heap.popitem()
            except KeyError:
                #print("Nothing left to explore")
                break
            
            if np.all(statebytes == state_end.tobytes()):
                print(np.frombuffer(statebytes, dtype))
                print(logp)
            
            if logp < treshold:
                break
            
            probable_states_backward.add(statebytes)
            
            state = np.frombuffer(statebytes, dtype)
            
            precedor_states = state[:, None] - S
            
            
            # This is not optimal, running in O(n^2) instead of O(n).
            # This will be (hopefully) fixed when 
            a = np.zeros(len(self.reactions)) # propensities
            for i in range(len(self.reactions)):
                a[i] = self.get_propensities_comb(precedor_states[:, i], R, rates)[i]
            
            a0 = np.sum(a)
            
            if a0 == 0:
                continue
            
            new_logps = logp + np.log(a) - np.log(a0) + np.log(1-np.exp(-a0*Δt))

            new_logps = np.nan_to_num(new_logps, nan=-np.inf)
            
            for i, new_logp in enumerate(new_logps):
                new_state = state - S[:, i]
                
                if new_logp <= -1.7e308: # If the most probable state is
                # infeasable, there is nothing more to do.
                    continue
                
                if new_state.tobytes() in probable_states_backward:
                    continue # This is not good. We should add the node once more
                    # so we do not "lose" probability on the run.
                    # But what else do we do in order not to search the same
                    # area of the graph multiple times?

                if new_state.tobytes() in heap: # Add probabilities
                    new_logp = self._logsum(heap[new_state.tobytes()], new_logp)
                    heap.updateitem(new_state.tobytes(), new_logp)
                    
                else:
                    heap.additem(new_state.tobytes(), new_logp)
            
            #print(len(heap))

            #print("Heap items")
            #print(*[(np.frombuffer(s, np.int32), heap[s]) for s in heap.keys()], sep="\n") # peek into heap
            #print()
            #print("Probable set items:")
            #print(*[np.frombuffer(s, np.int32) for s in probable_states_backward], sep="\n") # Peek into probable set
            #print("\n")
        
        if state_start.tobytes() not in probable_states_backward:
            raise ValueError("The trasition from " + str(state_start)+
                             " to " + str(state_end) +
                             " seems to be unfeasible. "+
                             "If this seems incorrect, try raising n_max " +
                             "parameter or lower epsilon parameter.")

        
        # Forward Dijkstra
        heap = maxpq()

        heap.additem(state_start.tobytes(), 0)
        
        probable_states = set()
        
        
        for j in range(n_max):
            # most probable state so far
            statebytes, logp = heap.popitem()
            
            if np.all(statebytes == state_end.tobytes()):
                print(np.frombuffer(statebytes, dtype))
                print(logp)
            
            if logp < treshold:
                break
            
            if statebytes not in probable_states_backward:
                continue
            
            probable_states.add(statebytes)
            
            state = np.frombuffer(statebytes, dtype)
            a = self.get_propensities_comb(state, R, rates) # propensities
            a0 = np.sum(a)
            
            if a0 == 0:
                continue
            
            new_logps = logp + np.log(a) - np.log(a0) + np.log(1-np.exp(-a0*Δt))

            new_logps = np.nan_to_num(new_logps, nan=-np.inf)
            
            for i, new_logp in enumerate(new_logps):
                new_state = state + S[:, i]
                
                if new_logp <= -1.7e308: # If the most probable state is
                # infeasable, there is nothing more to do.
                    continue
                
                if new_state.tobytes() in probable_states:
                    continue # This is not good. We should add the node once more
                    # so we do not "lose" probability on the run.
                    # But what else do we do in order not to search the same
                    # area of the graph multiple times?

                if new_state.tobytes() in heap: # Add probabilities
                    new_logp = self._logsum(heap[new_state.tobytes()], new_logp)
                    heap.updateitem(new_state.tobytes(), new_logp)
                    
                else:
                    heap.additem(new_state.tobytes(), new_logp)
                

            
            # print([*(np.frombuffer(s, np.int32), heap[s]) for s in heap.keys()], sep="\n") # peek into heap
            # print()
            # print([np.frombuffer(s, np.int32) for s in probable_states]) # Peek into probable set
            # print("\n")
        
        if state_end.tobytes() not in probable_states:
            raise ValueError("The set of probable states is too big!")

        
    # TODO: compile this function with numba
    @jit
    def _logsum(self, A, B):
        """
        If `A = ln(a)` and `B = ln(b)`, this function returns `ln(a+b)`.

        Parameters
        ----------
        A : float
        B : float

        Returns
        -------
        ln(exp(A) + exp(B)), float

        """
        A, B = max(A, B), min(A, B)
        res = A + np.log(1+np.exp(B-A))

        return np.nan_to_num(res, nan=-np.inf)
            
    
    def _calculate_trasition_likelihood(Δt, state_start, state_end):
        pass
    
    def __str__(self):
        return "An object of type System, representing a setup for chemical kinetics simulation.\n" + \
            "(" + self.description + ")" +\
            "Containing %d species: "%len(self.species) +\
            ", ".join(self.species) + " and %d reactions:\n\n"%len(self.reactions) + \
            self._reaction_repr()


if __name__ == "__main__":
    S = System("Basic SIR model")
    S.add_species("S", "Susceptible", "I", "Infectious", "R", "Recovered", full_names=True)
    #S.add_reaction(("S", "I"), ("I", "I"), 50/100, "New infection", {"I_total": 1})
    S.add_reaction(("S", "I"), ("I", "I"), 80/100, "New infection")
    S.add_reaction(("I",), ("R",), 36, "Recovery")
    # S.print_reactions()
    # print(S)
    #print(S.get_stoichiometry())
    #S.plot_simulations({"S":98, "I":2}, 1, N=50, not_shown=("I_total",))
    #S.simulate_gillespie({"S":98, "I":2}, 1)
    #t_list = t_list = [0, 0.00001, 0.1, 0.2, 0.5, 0.50000001, 1, 0.6, 0.8, 0.7, -1]
    #S.simulate_gillespie({"S":98, "I":2}, 1)
    initial_state = {"S":98, "I":2}
    t_max = 1
    t_list = np.linspace(0, t_max, 101)
    #S.plot_mean_variance_MC(initial_state, t_list=t_list, N=100, alpha=.3)
    
    #S.simulate_tau_leap(initial_state, t_max)
    #S.plot_simulations(initial_state, 1, method=S.simulate_tau_leap, N=20, alpha=.3)
    
    # ans = S.simulate_gillespie(initial_state, 1)
    # plt.figure()
    # plt.grid()
    # plt.axis("equal")
    # plt.plot(ans[1][0], ans[1][1], alpha=.3)
    # plt.scatter(ans[1][0], ans[1][1], 7, alpha=0.3)
    # plt.xlabel("Number of susceptible individuals")
    # plt.ylabel("Number of infectious individuals")
    # plt.title("Phase portrait of SIR model")
    # plt.tight_layout()
    
    
    E = System("Enzyme synthesis model")
    E.add_species("E", "Enzyme", "S", "Substrate", "C", "Complex", "P", "Protein", full_names=True)
    E.add_reaction(("E", "S"), ("C",), 0.001, "Enzyme and substrate molecule combine to form a complex")
    E.add_reaction(("C",), ("E", "S"), 0.005, "Decay of a complex")
    E.add_reaction(("C",), ("E", "P"), 0.01, "Catalytic conversion of substrate to product")
    # print(E)
    E_initial = {"E":100 ,"S":100}
    E_t_max = 500
    #E.plot_simulations(E_initial, E_t_max, N=50, alpha=0.3)
    #E.plot_mean_variance_MC(E_initial, E_t_max, N=50)
    
    
    
    SIR2 = System("SIR model with mutliple infectious phases")
    gamma = 36
    beta = 50
    N = 1000
    S_0 = 990
    
    SIR2.add_species("S", "I1", "I2", "I3", "I", "R")
    SIR2.add_reaction(("S", "I1"), ("I1", "I1"), beta/N,
                      "New infection from infected in stage 1",
                      additional_stoichiometry={"I":1})
    SIR2.add_reaction(("S", "I2"), ("I1", "I2"), beta/N,
                      "New infection from infected in stage 2",
                      additional_stoichiometry={"I":1})
    SIR2.add_reaction(("S", "I3"), ("I1", "I3"), beta/N,
                      "New infection from infected in stage 3",
                      additional_stoichiometry={"I":1})
    SIR2.add_reaction(("I1",), ("I2",), gamma*3, "Stage progression")
    SIR2.add_reaction("I2", "I3", gamma*3, "Stage progression")
    SIR2.add_reaction("I3", "R", gamma*3, "Recovery",
                      additional_stoichiometry={"I":-1})
    
    #SIR2.plot_simulations({"S":S_0, "I1":N-S_0}, 1, alpha=1)
    
    """
    self = S
    t = (0, 0.1, 0.2, 0.5)
    data = [(0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01),
            {"S": (98, 98, 98, 97, 97, 97, 96),
             "I": (2, 2, 2, 3, 3, 3, 4),
             "R": (0, 0, 0, 0, 0, 0, 0)}
           ]
    
    transitions = self._split_observations(data) 
    Δt, state_start, state_end = transitions[3]
    """    
    
    
    self = E
    t = np.arange(0, 100, 10)
    data = [t,
            {"E": (100, 45, 38, 40, 47, 49, 52, 52, 58, 64),
             "S": (100, 43, 27, 18, 18, 13, 11, 6, 7, 7),
             "C": (0, 55, 62, 60, 53, 51, 48, 48, 42, 36),
             "P": (0, 2, 11, 22, 29, 36, 41, 46, 51, 57)}
           ]
    
    transitions = self._split_observations(data) 
    Δt, state_start, state_end = transitions[2]