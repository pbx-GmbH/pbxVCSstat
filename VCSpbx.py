"""
This is the PBX library to simulate vapor compression cycles for stationary working points.
"""
from datetime import datetime

import numpy as np
import scipy.optimize
from scipy.optimize import fsolve
import CoolProp
from CoolProp.CoolProp import PropsSI as CPPSI
from tqdm import tqdm
import matplotlib.pyplot as plt
from imageio import imread
import pandas as pd


def lmtd_calc(Thi, Tho, Tci, Tco):
    """
    Calculate the mean logarithmic temperature difference. Used for calculating heat flow.

    :param Thi: Inlet temperature of hot medium
    :param Tho: Outlet temperature of hot medium
    :param Tci: Inlet temperature of cold medium
    :param Tco: Outlet temperature of cold medium
    :return: Mean logarithmic temperature difference
    """
    # calculating the logaritmic mean temperature of two fluids with defined "hot" and "cold" fluid
    dT1 = Thi - Tco
    dT2 = Tho - Tci

    # add exceptions and limit for the calculations
    if dT2 == 0:
        dT2 = 0.01
    if np.abs(dT1-dT2) < 0.00000001:
        LMTD = dT1
    else:
        LMTD = (dT1 - dT2) / np.log(dT1 / dT2)
    # if dT1 < 0:
    #     return 0.0

    # prevent NaN values:
    if np.isnan(LMTD):
        LMTD = 1e-6
    return LMTD


def dh_cond(TC, medium):
    """
    Calculate the enthalpy difference of condensation

    :param TC: Temperature of condensation
    :param medium: Medium (CoolProp string)
    :return: Enthalpy of condensation
    """
    Tcrit = CPPSI('TCRIT', medium)
    if TC > Tcrit:
        dh = -(TC - Tcrit) * 1000
    else:
        dh = CPPSI("H", "T", TC, "Q", 1, medium) - CPPSI("H", "T", TC, "Q", 0, medium)
    return dh

def calc_thirdorder_polynomial(x, y, p):
    x = x * 1.0e-5
    y = y * 1.0e-5
    xy_array = np.array([1, x, y, x*x, x*y, y*y, x*x*x, x*x*y, x*y*y, y*y*y])
    return np.dot(p, xy_array.transpose())

class System:
    """
    The system class comprises all components and connections, that are needed to calculate the model.
    It handles also the simulation itself.
    """
    def __init__(self, id: str, tolerance: float, n_max: int = 100, fun_tol: float = 0.1):
        """
        Initializes the class System

        :param id: the name of the system to later refer to it
        :param tolerance: tolerance used for the enthalpy differences at the connection points (convergence criteria)
        :param n_max: maximum number of iterations
        :param fun_tol: tolerance used for the calculation of the component models
        """
        self.id = id
        self.components = []
        self.junctions = []
        self.params = {}
        self.tolerance = tolerance

        self.n_max = n_max
        self.fun_tol = fun_tol

        self.residual_enthalpy = None
        self.residual_functions = {}

        self.residual_dict = {}

    def run(self, full_output=False):
        """
        Runs the calculation of the class System with the given parameters. This can only be done after initialization.

        :param full_output: Declare whether to generate a full output
        :return: True if simulation was successful, False if not
        """
        # first get the current enthalpy values at the junctions
        old_enthalpies = self.get_junction_enthalpies()
        counter = 0

        # calculation loop
        while True:

            # iterate through components and let them calculate their model
            for comp in self.components:
                comp.calc()

            # get the new enthalpy values at the junctions
            new_enthalpies = self.get_junction_enthalpies()

            # calculate the delta
            abs_delta = np.abs(old_enthalpies - new_enthalpies)

            # store the enthalpy residual in the residual_dict
            for i, junc in enumerate(self.junctions):
                self.residual_dict[junc.id] = abs_delta[i]

            # get the function residuals of the components
            fun_residual = np.array([])
            for comp in self.components:
                res = np.array(comp.get_function_residual())
                fun_residual = np.append(fun_residual, np.abs(res))
                for i, fr in enumerate(res):
                    key = '{}.{}'.format(comp.id, i)
                    self.residual_dict[key] = fr

            # check if delta is lower than tolerance
            if np.max(abs_delta) < self.tolerance and np.max(fun_residual) < self.fun_tol:
                break

            counter += 1

            # cancel iteration, if n_max is reached without satisfying solution
            if counter > self.n_max:
                print('Reached {} iterations without solution'.format(counter))
                for comp in self.components:
                    self.residual_functions[comp.id] = comp.get_function_residual()
                self.residual_enthalpy = abs_delta
                print('Residual enthalpies: {}'.format(abs_delta))
                print('Function residual: {}'.format(fun_residual))
                return False

            old_enthalpies = new_enthalpies.copy()

        self.residual_enthalpy = abs_delta

        for comp in self.components:
            self.residual_functions[comp.id] = comp.get_function_residual()

        # if full_output, print everything
        if full_output:
            print('---')
            print('Iteration finished after {} iterations'.format(counter))
            print('---\nResidual enthalpies difference:')
            [print('{}: {}'.format(key, self.residual_dict[key])) for key in self.residual_dict]
            print('---\nJUNCTIONS:')
            for junc in self.junctions:
                print(junc.id)
                print('Pressure: {:2f}bar'.format(junc.p/1e5))
                print('Temperature: {:2f}°C'.format(junc.T-273.15))
                print('Enthalpy: {:2f}J/kg'.format(junc.h))
                print('Massflow: {:2f}kg/s'.format(junc.mdot))
                print('---')

        return True

    def initialize(self):
        """
        Initialize the components of the system.
        Loops through all components and executes the components methods "check_junctions()" and "initialize()"

        :return:
        """
        for comp in self.components:
            comp.check_junctions()
            comp.initialize()

    def update_parameters(self):
        """
        Dummy class to update parameters.

        :return:
        """
        pass

    def add_component(self, comp):
        """
        Add a component to the system. This is done by adding the component object to the list of components.

        :param comp: component object
        :return:
        """
        self.components.append(comp)

    def add_junction(self, junc):
        """
        Add a junction to the system. This is done by adding the junction object to the list of junctions.

        :param junc:
        :return:
        """
        self.junctions.append(junc)

    def get_junction_enthalpies(self):
        """

        :return:
        """
        return np.array([junc.get_enthalpy() for junc in self.junctions])

    def plot_cycle(self, plot_dict: dict, cycle_img_path: str):
        """
        Plot the cylce on a given image. The position and values of the plottet text have to be parsed through the plot_dict.

        :param plot_dict: Dictionary that defines the position of the text fields
        :param cycle_img_path: Path to the image
        :return:
        """
        # TODO check if cycle image is a file

        # initiate the subplots
        plt.subplot(121)

        # create the picture (left) side of the plot
        cycle_image = imread(cycle_img_path)
        plt.imshow(cycle_image, interpolation='bilinear')
        plt.axis('off')

        # now run through dict and write the texts into the drawing
        # first the junctions
        for junc in plot_dict['cycle']['junctions']:
            comp = plot_dict['cycle']['junctions'][junc]
            # retrieve plot string from component
            text = comp['component'].get_plot_string()
            # plot the text
            plt.text(comp['position'][0], comp['position'][1], text, va=comp['va'], ha=comp['ha'], fontsize=8)

        # now the special text
        for special in plot_dict['cycle']['special']:
            comp = plot_dict['cycle']['special'][special]
            plt.text(comp['position'][0], comp['position'][1], comp['text'], va=comp['va'], ha=comp['ha'], fontsize=8)

        # create the top right side (Ts-diagram)
        plt.subplot(222)

    def get_export_variables(self):
        """
        Get the export variables of all components by calling the
        respective "dump_export_variables()" method of every component.

        :return: dict with all export variables
        """
        dump_dict = dict()

        # get export variables from components
        for comp in self.components:
            dump_dict.update(comp.dump_export_variables())

        # get junction variables
        for j in self.junctions:
            dump_dict.update(j.get_value_dict())
        return dump_dict

    def parameter_variation(self, parameters: iter, parameter_handles: iter, function_tolerance: float = 0.1, enthalpy_tolerance: float = 1., save_results = True):
        """
        THIS IS STILL UNDER DEVELOPMENT!
        Calculate a parameter variation field. Parameters have to be parsed with their object handle functions.

        :param parameters: Nested list of parameters
        :param parameter_handles: List of objects
        :param function_tolerance: Tolerance for solving the component models.
        :param enthalpy_tolerance: Tolerance for the enthalpy differences of junctions.
        :return:
        """
        self.results_parameter_variation = list()

        self.tolerance = enthalpy_tolerance
        self.fun_tol = function_tolerance

        # loop through the parameters list
        for params in tqdm(parameters):

            # update the parameters
            for i, p in enumerate(params):
                parameter_handles[i](p)

            res_dict = dict()

            # try to calculate the system with the new parameter setting and store the result
            try:
                converged = self.run()
                res_dict = self.get_export_variables()

                res_dict['converged'] = converged
            except:
                for i in range(11):
                    adapted_params = old_params + (np.array(params)-old_params) * (i) * 0.1
                    try:
                        for i, p in enumerate(adapted_params):
                            parameter_handles[i](p)
                        converged = self.run()
                    except:
                        res_dict['converged'] = False
                        break
                res_dict['converged'] = True

            # store parameters for next loop
            old_params = np.array(params)

            # append result into result list
            self.results_parameter_variation.append(res_dict)

        # convert the list of dictionaries to a pd.DataFrame
        self.results_parameter_variation = pd.DataFrame(self.results_parameter_variation)

        # save the result, if save_results == True
        if save_results:
            now = datetime.now()
            timestamp = now.strftime('%y%m%d-%H%M%S')
            filename = timestamp + '_parameter_variation.pkl'
            self.results_parameter_variation.to_pickle(filename)

class Component:
    """
    Base class of Components. It contains all functions, that are needed for the System Class to run but with dummy outputs.
    """
    def __init__(self, id: str, system: object):
        """
        Initialize the Component class.

        :param id: The name of the component to later refer to.
        :param system: The System object that shall contain the Component object.
        """
        self.system = system
        system.add_component(self)

        self.id = id

        self.parameters = dict()

        self.junctions = {'inlet_A': None, 'outlet_A': None}

        self.statechange = None

        self.export_variables = dict()

    def print_parameters(self):
        """
        Print the parameters of the component.

        :return:
        """
        print(self.parameters)

    def calc(self):
        """
        Dummy function for Component. This function has to be overwritten in the specific component class.

        :return:
        """
        raise ValueError('{} has no defined calc() function!'.format(self.id))

    def check_junctions(self):
        """
        Check if the component has a junction connected to every port.

        :return:
        """
        for key in self.junctions:
            if not self.junctions[key]:
                raise ValueError('{} has no junction at port {}.'.format(self.id, key))

    def get_function_residual(self):
        """
        Dummy function to return the function residual. If a component has a function residual, this function has
        to be overwritten in the component class.

        :return:
        """
        return [0.0]

    def get_Ts_data(self, npoints: int):
        """
        Dummy function for Component. This function has to be overwritten in the specific component class.

        :param npoints: Define the number of points on the Ts diagram
        :return:
        """
        pass

    def get_ph_data(self, npoints: int):
        """
        Dummy function for Component. This function has to be overwritten in the specific component class.

        :param npoints: Define the number of points on the ph diagram
        :return:
        """
        pass

    def define_export_variables(self):
        """
        Dummy function for Component. This function has to be overwritten in the specific component class.

        :return:
        """
        print('{} has no export_variables defined.'.format(self.id))
        return

    def dump_export_variables(self):
        """
        Return the values of the export variables as a dictionary. The key of the dictionary is generated the following way:
        <self.id>.<variable_key>

        :return: The dictionary with all export variables.
        """
        self.define_export_variables()
        dump_dict = {}
        for var_key in self.export_variables:
            key = '.'.join([self.id, var_key])
            dump_dict[key] = self.export_variables[var_key]
        return dump_dict

    def initialize(self):
        """
        Dummy function for Component. This function can be overwritten in the specific component class, if initialization is needed.

        :return:
        """
        pass

class Junction:
    """
    Class that defines the junctions between components. Junctions can be viewed as "high-fidelity" data storages
    and work as the communication interface between the components.
    Junction also are used to define the initial guesses.
    Junction have three defining parameters: massflow, pressure, enthalpy. All other parameters are calculated from these.
    """
    def __init__(self, id: str, system: object, medium: str, upstream_component: object, upstream_port_id: str,
                 downstream_component: object, downstream_port_id: str, mdot_init: float, p_init: float, h_init: float):
        """
        Initialize the junction.

        :param id: The name of the junction to later refer to it.
        :param system: The system that shall contain the junction.
        :param medium: The medium of the junction.
        :param upstream_component: The component object that is upstream of the junction.
        :param upstream_port_id: The port id of the upstream component. This is a string.
        :param downstream_component: The component object that is downstream of the junction.
        :param downstream_port_id: The port id of the downstream component. This is a string.
        :param mdot_init: Initial guess of massflow.
        :param p_init: Initial guess of pressure.
        :param h_init: Initial guess of enthalpy.
        """
        self.medium = medium

        self.id = id
        self.system = system
        system.add_junction(self)

        self.mdot = None
        self.p = None
        self.h = None
        self.T = None
        self.x = None

        self.set_values(mdot_init, p_init, h_init)

        if upstream_component.junctions[upstream_port_id]:
            print('{} of component {} overwritten!'.format(upstream_port_id, upstream_component.id))
        upstream_component.junctions[upstream_port_id] = self
        if downstream_component.junctions[downstream_port_id]:
            print('{} of component {} overwritten!'.format(downstream_port_id, downstream_component.id))
        downstream_component.junctions[downstream_port_id] = self


    def set_values(self, mdot: float = None, p:float = None, h: float = None):
        """
        Set the value of the junction by giving either one, two or all of the three defining parameters massflow, pressure, enthalpy.
        The function calculates the values for all other parameters and stores it in self.

        :param mdot: New value for massflow.
        :param p: New value for pressure.
        :param h: New value for enthalpy.
        :return:
        """

        if mdot:
            self.mdot = mdot
        if p:
            self.p = p
        if h:
            self.h = h

        self.T = CPPSI('T', 'P', self.p, 'H', self.h, self.medium)

        self.s = CPPSI('S', 'P', self.p, 'H', self.h, self.medium)

        # try to calculate x (only possible for refrigerants)
        try:
            self.x = self.calculate_x()
        except:
            self.x = None

    def get_pressure(self):
        """
        Return the pressure of the junction.

        :return: Pressure
        """
        return self.p

    def get_temperature(self):
        """
        Return the temperature of the junction.

        :return: Temperature
        """
        return self.T

    def get_massflow(self):
        """
        Return the mass flow of the junction.

        :return: Mass flow
        """
        return self.mdot

    def get_enthalpy(self):
        """
        Return the enthalpy of the junction.

        :return: Enthalpy
        """
        return self.h

    def get_entropy(self):
        """
        Return the entropy of the junction.

        :return: Entropy
        """
        return self.s

    def get_quality(self):
        """
        Return the vapor quality of the junction.

        :return: Quality
        """
        return self.x

    def calculate_x(self):
        """
        Calculate the quality of the refrigerant. The used definition allows values below 0 and above 1 (contrary to literature).
        The definition is:
        h-h'/(h''-h')
        The calculation is not limited to two phase region of the refrigerant, i.e. values below zero indicate subcooled liquid, values above 1 indicate superheated gas.

        :return:
        """
        # this defines h for more than just the two phase region
        h_l = CPPSI('H', 'P', self.p, 'Q', 0, self.medium)
        h_v = CPPSI('H', 'P', self.p, 'Q', 1, self.medium)
        return (self.h - h_l)/(h_v-h_l)

    def get_plot_string(self):
        """
        Return the string used to plot with the "plot_cycle()" function of class System.

        :return: text string, that contains the relevant information (T, p, h, mdot).
        """
        text = 'T: {T:.2f} °C\np: {p:.2f} bar\nh: {h:.0f} J/kg\nmdot: {mdot:.2f} g/s'.format(
            T=self.get_temperature() - 273.15,
            p=self.get_pressure() * 1e-5,
            h=self.get_enthalpy(),
            mdot=self.get_massflow() * 1e3)
        return text

    def get_value_dict(self):
        ret_dict = dict()
        ret_dict[self.id+'.p'] = self.get_pressure()
        ret_dict[self.id+'.h'] = self.get_enthalpy()
        ret_dict[self.id+'.mdot'] = self.get_massflow()
        return ret_dict


    def get_medium(self):
        """
        Returns the string for the medium.

        :return:
        """
        return self.medium

class CompressorEfficiency(Component):
    """
    Compressor model based on isentropic and volumetric efficiency. Also a parameter for electric efficiency is added.
    """
    def __init__(self, id: str, system: object, etaS: float, etaV:float, stroke: float, speed: float, etaEL:float = 1.):
        """
        Initialize the compressor object.

        :param id: name of the comporessor object
        :param system: System object, that shall contain the compressor object.
        :param etaS: Isentropic efficiency
        :param etaV: Volumetric efficiency
        :param stroke: Stroke of comporessor
        :param speed: Speed of compressor
        :param etaEL: Electric efficiency
        """
        super().__init__(id, system)
        self.etaS = etaS
        self.etaV = etaV
        self.etaEL = etaEL
        self.stroke = stroke
        self.speed = speed

        self.parameters = {'id': id, 'etaS': etaS, 'etaV': etaV, 'speed': speed, 'etaEL': etaEL}

        self.Tin = np.nan
        self.pin = np.nan
        self.pout = np.nan

        self.Pel = None
        self.P_compression = None

    def calc(self):
        """
        Calculates the compressor model. With the pressures at inlet and outlet and the temperature of the inlet,
        the model calculates the outlet enthalpy and updates the outlet junction accordingly.

        :return:
        """

        # get the inlet interface values
        self.Tin = self.junctions['inlet_A'].get_temperature()
        self.pin = self.junctions['inlet_A'].get_pressure()
        self.pout = self.junctions['outlet_A'].get_pressure()

        # compressor model based on efficiency parameters: etaS...isentropic / etaV...volumetric
        rho = CPPSI("D", "P", self.pin, "T", self.Tin, "R290")  # density of refrigerant at inlet
        mdot = self.speed / 60 * self.stroke * self.etaV * rho  # mass flow

        hin = CPPSI("H", "T", self.Tin, "P", self.pin, "R290")  # inlet enthalpy
        sin = CPPSI("S", "T", self.Tin, "P", self.pin, "R290")  # inlet entropy
        houtS = CPPSI("H", "S", sin, "P", self.pout, "R290")  # enthalpy at outlet under isentropic conditions
        self.P_compression = mdot * (houtS - hin) / self.etaS  # power input
        hout = self.P_compression / mdot + hin  # real outlet enthalpy
        Tout = CPPSI("T", "P", self.pout, "H", hout, "R290")  # outlet temperature

        self.Pel = self.P_compression/self.etaEL  # eletrical power consumption

        # update the outlet junction
        self.junctions['outlet_A'].set_values(mdot=mdot, h=hout)

    def set_speed(self, speed):
        """
        Set the speed of the compressor.

        :param speed: Speed of compressor [rpm]
        :return:
        """
        self.speed = speed

    def get_power(self):
        """
        Return the elecrtical power consumption of the compressor.

        :return: Electrical power consumption
        """
        return self.Pel

    def get_Ts_data(self, npoints: int):
        """
        Return data for Ts-diagrams.

        :param npoints: Define the number of points on the Ts diagram
        :return: Array with temperature and entropy [T, s]
        """
        T, s = np.zeros((2, npoints))
        Tin = self.junctions['inlet_A'].get_temperature()
        Tout = self.junctions['outlet_A'].get_temperature()
        sin = self.junctions['inlet_A'].get_entropy()
        T = np.linspace(Tin, Tout, npoints)
        s.fill(sin)
        return [T, s]

    def get_ph_data(self, npoints: int):
        """
        Return data for ph-diagrams.

        :param npoints: Define the number of points on the Ts diagram
        :return: Array with temperature and entropy [T, s]
        """
        pin = self.junctions['inlet_A'].get_pressure()
        pout = self.junctions['outlet_A'].get_pressure()
        hin = self.junctions['inlet_A'].get_enthalpy()
        hout = self.junctions['outlet_A'].get_enthalpy()
        p = np.linspace(pin, pout, npoints)
        h = np.linspace(hin, hout, npoints)
        return [p, h]

    def update_parameter(self, param, value):
        """
        Function to update a parameter with a string and value.

        :param param: string of the parameter to be changed.
        :param value: new value of the parameter
        :return:
        """
        if param == 'speed':
            self.set_speed(value)

        else:
            raise ValueError('Cannot set parameter {}'.format(param))

    def define_export_variables(self):
        """
        Define which variables are to be exported.

        :return:
        """
        self.export_variables = {
            'speed': self.speed,
            'Pel': self.Pel
        }
        return


class Compressor_MasterfluxAlpine(Component):
    """
    Compressor model of Masterflux Alpine compressor as provided via email to D. Radler by Edu Machado.
    """
    def __init__(self, id: str, system: object, speed: float):
        """
        Initialize the compressor object.

        :param id: name of the compressor object
        :param system: System object, that shall contain the compressor object.
        :param speed: Speed of compressor
        """
        super().__init__(id, system)
        self.speed = speed
        self.working_parameters = np.zeros(9)
        self.k = None
        self.Pel = None
        self.mdot = None


        # compressor parameters
        self.stroke = 18.243 * 2 * 1.0E-6  # in m3
        self.RPM_min = 1800
        self.RPM_max = 6500
        # model parameters
        self.params_power = np.array([[2.61351503160446E-05, 1.94406507897762E-02, -2.61535874512516E+01],
                             [2.40596664347618E-06, 2.28548550415022E-02, 2.37947319860790E+01],
                             [-2.51946735329226E-06, 1.20324211464900E-02, 1.47601128237257E+00],
                             [3.77426108757315E-07, -1.86116134522515E-02, -7.44043380970024E+00],
                             [-1.83712472528971E-07, 7.93320039283770E-03, 3.14324275029855E+00],
                             [1.30475042012445E-07, -7.12838689468400E-04, -3.41409746410319E-01],
                             [6.42191991452190E-08, 3.88134369872757E-04, 1.05210351088838E+00],
                             [-5.62121723586466E-08, 3.77645038443793E-04, -5.84602225289624E-01],
                             [1.50132257118843E-08, -2.15098941622777E-04, 1.14346993354944E-01],
                             [-2.87991861855384E-09, 2.09821522655107E-05, -4.01657493530679E-03]])

        self.params_massflow_ref = np.array([[-8.55032307735981E-08, 9.31458288703003E-04, -7.18968943227166E-01],
                                    [-2.66519043390434E-08, 4.01975426099445E-03, 1.58505108662075E-01],
                                    [2.09693655630869E-08, -1.67674555442193E-04, 1.03408907108912E-01],
                                    [-6.98430897118900E-09, 2.09188216353218E-05, -1.33471995402642E-01],
                                    [2.79929662554270E-09, -3.48031185011942E-05, 4.98428256481021E-02],
                                    [-1.19668996364721E-09, 1.04471382669788E-05, -1.29317170408179E-02],
                                    [9.19654936062527E-10, -6.35440625748187E-06, 1.09857416127085E-02],
                                    [-3.93026479716597E-10, 4.25576286265211E-06, -3.25019576352321E-03],
                                    [4.50775172771900E-11, -4.56502284807732E-07, -2.66928981585851E-05],
                                    [1.80160704721233E-11, -1.35225553553278E-07, 1.54910933414018E-04]])

        self.params_massflow_tsuction = np.array([2.04474579550577E-01, -3.36332402905712E+00, 2.46383673994489E+01,
                                         -6.31433708944579E+01])

        self.params_temp_compression = np.array([[5.27144801489871E-10, -7.49337807305126E-06, 1.25452516257171E+00],
                                        [-4.81045877105711E-10, 2.87768468434160E-06, -7.43631234653976E-03],
                                        [-7.09921866227119E-11, 8.67839961547325E-07, -4.66651696490209E-02],
                                        [1.52462983544176E-11, 6.45972863611150E-08, -3.13206792672621E-03],
                                        [4.95960479541254E-11, -4.06861878416786E-07, 2.35979602691049E-03],
                                        [1.11849648046508E-12, -1.68249538758197E-08, 2.77591029160165E-03],
                                        [5.62238500275363E-12, -8.64332447258515E-08, 5.56253904617871E-04],
                                        [-6.50054194399207E-12, 7.46878573462099E-08, -3.01167165951467E-04],
                                        [3.87185689965077E-13, -8.53951419506538E-09, 2.24475779742436E-05],
                                        [-1.15461950356974E-13, 1.48322587695103E-09, -6.08367955088717E-05]])

    def initialize(self):
        """
        Initialization for parameters, that need other objects to get information from.
        :return:
        """
        self.medium = self.junctions['inlet_A'].get_medium()

    def calc(self):
        """
        Calculates the compressor model. Efficiencies are used to calculate the states of the refrigerant at the outlet,
        as well as the electric power needed. The model is developed by Masterflux.

        :return:
        """
        # get the needed boundary parameters
        self.Tin = self.junctions['inlet_A'].get_temperature()
        self.pin = self.junctions['inlet_A'].get_pressure()
        self.pout = self.junctions['outlet_A'].get_pressure()
        self.p_ratio = self.pout / self.pin
        self.rho_in = CPPSI("D", "P", self.pin, "T", self.Tin, self.medium)
        hin = self.junctions['inlet_A'].get_enthalpy()

        # update specific heat ratio
        cp = CPPSI('CPMASS', 'P', self.pin, 'T', self.Tin, self.medium)
        cv = CPPSI('CVMASS', 'P', self.pin, 'T', self.Tin, self.medium)
        self.k = cp/cv

        # calculate the model
        self.speed_array = np.array([self.speed**2, self.speed, 1])
        self.calc_power()
        self.calc_massflow_rate()
        self.calc_hout()

        # update the outlet junction
        self.junctions['outlet_A'].set_values(mdot=self.mdot, h=self.hout)

    def calc_power(self):
        power_params_reduced = np.dot(self.params_power, self.speed_array)
        self.Pel = calc_thirdorder_polynomial(self.pin, self.pout, power_params_reduced)

    def calc_massflow_rate(self):
        pin = self.pin * 1.0e-5
        massflow_params_reduced = np.dot(self.params_massflow_ref, self.speed_array)
        massflow_ref = calc_thirdorder_polynomial(self.pin, self.pout, massflow_params_reduced)
        massflow = ((self.params_massflow_tsuction[0] * pin**3 + self.params_massflow_tsuction[1] * pin**2 + self.params_massflow_tsuction[2] * pin + self.params_massflow_tsuction[3] + 308.15)/(self.Tin))*massflow_ref
        self.mdot = massflow / 3600

    def calc_hout(self):
        td_ideal = self.Tin * self.p_ratio ** (1 - 1 / self.k)
        td_params_reduced = np.dot(self.params_temp_compression, self.speed_array)
        td_multiplier = calc_thirdorder_polynomial(self.pin, self.pout, td_params_reduced)
        Tout = td_ideal/td_multiplier
        self.hout = CPPSI('H', 'T', Tout, 'P', self.pout, self.medium)

    def set_speed(self, speed):
        """
        Update the compressor speed.

        :param speed: Speed of compressor [rpm]
        :return:
        """
        self.speed = speed

    def get_power(self):
        """
        Return the electrical power consumption of the compressor.

        :return: Electrical power consumption
        """
        return self.Pel

    def get_Ts_data(self, npoints: int):
        """
        Return data for Ts-diagrams.

        :param npoints: Define the number of points on the Ts diagram
        :return: Array with temperature and entropy [T, s]
        """
        T, s = np.zeros((2, npoints))
        Tin = self.junctions['inlet_A'].get_temperature()
        Tout = self.junctions['outlet_A'].get_temperature()
        sin = self.junctions['inlet_A'].get_entropy()
        T = np.linspace(Tin, Tout, npoints)
        s.fill(sin)
        return [T, s]

    def get_ph_data(self, npoints: int):
        """
        Return data for ph-diagrams.

        :param npoints: Define the number of points on the Ts diagram
        :return: Array with temperature and entropy [T, s]
        """
        pin = self.junctions['inlet_A'].get_pressure()
        pout = self.junctions['outlet_A'].get_pressure()
        hin = self.junctions['inlet_A'].get_enthalpy()
        hout = self.junctions['outlet_A'].get_enthalpy()
        p = np.linspace(pin, pout, npoints)
        h = np.linspace(hin, hout, npoints)
        return [p, h]

    def update_parameter(self, param, value):
        """
        Function to update a parameter with a string and value.

        :param param: string of the parameter to be changed.
        :param value: new value of the parameter
        :return:
        """
        if param == 'speed':
            self.set_speed(value)

        else:
            raise ValueError('Cannot set parameter {}'.format(param))

    def define_export_variables(self):
        """
        Define which variables are to be exported.

        :return:
        """
        self.export_variables = {
            'speed': self.speed,
            'Pel': self.Pel
        }
        return


class Condenser(Component):
    """
    Condenser model with air. This is deprecated.
    """
    def __init__(self, id: str, system: object, k: iter, area: float, subcooling: float, T_air_in: float, mdot_air_in: float):
        super().__init__(id, system)
        if len(k) != 3:
            raise ValueError('k needs to be of length 3, but len(k) = {}'.format(len(k)))
        else:
            self.k = k
        self.area = area
        self.dTSC = subcooling
        # self.parameters = {'UA': self.UA, 'subcooling': self.dTSC}

        self.T_air_in = T_air_in
        self.mdot_air = mdot_air_in

        self.TC = None
        self.TAo_desuperheat = None
        self.TAo_condenser = None
        self.TAo_subcool = None
        self.areafraction_desuperheat = None
        self.areafraction_condenser = None
        self.areafraction_subcool = None
        self.p = None

    def initialize(self):
        self.medium = self.junctions['inlet_A'].medium
        self.p = self.junctions['inlet_A'].get_pressure()
        self.TC = CPPSI('T', 'P', self.p, 'Q', 0, self.medium)
        Tmean = (self.T_air_in + self.TC)/2
        self.TAo_desuperheat = Tmean
        self.TAo_condenser = Tmean
        self.TAo_subcool = Tmean
        self.areafraction_desuperheat = 0.1
        self.areafraction_condenser = 0.8
        self.areafraction_subcool = 0.1

    def model(self, x):
        mR = self.junctions['inlet_A'].get_massflow()
        TRin = self.junctions['inlet_A'].get_temperature()
        mA = self.mdot_air
        TAi = self.T_air_in

        # Boundary for fsolve calculation to not cause the logaritmic mean temperature to generate NaN values (neg. logarithm):
        # The outlet air temperature of the superheat section must not exceed the refrigerant inlet temperature.
        if x[1] > TRin:
            x[1] = TRin - 1e-4
        if x[0] - self.dTSC < TAi:
            x[0] = self.T_air_in + self.dTSC + 1e-4

        # calculate material parameters
        cpR = np.zeros(2)
        cpR[0] = CPPSI("C", "T", x[0], "Q", 1, self.medium)
        cpR[1] = CPPSI("C", "T", x[0], "Q", 0, self.medium)
        cpA = CPPSI("C", "T", self.T_air_in, "P", 1.0e5, "AIR")

        # Calculate the mean logaritmic temperature value for all three sections of the condenser
        LMTD = np.zeros(3)
        LMTD[0] = lmtd_calc(TRin, x[0], self.T_air_in, x[1])
        LMTD[1] = lmtd_calc(x[0], x[0], self.T_air_in, x[2])
        LMTD[2] = lmtd_calc(x[0], x[0] - self.dTSC, self.T_air_in, x[3])

        # Formulation of the equation system as according to fsolve documentation ( 0 = ... ).
        # The equation set  and model definition is documented in the model description.
        dh = dh_cond(x[0], self.medium)
        f = np.zeros(7)
        f[0] = mR * cpR[0] * (TRin - x[0]) - mA * cpA * x[4] * (x[1] - TAi)
        f[1] = mR * cpR[0] * (TRin - x[0]) - self.k[0] * x[4] * self.area * LMTD[0]
        f[2] = mR * dh - self.k[1] * x[5] * self.area * LMTD[1]
        f[3] = mR * dh - mA * cpA * x[5] * (x[2] - TAi)
        f[4] = mR * cpR[1] * self.dTSC - mA * cpA * x[6] * (x[3] - TAi)
        f[5] = mR * cpR[1] * self.dTSC - self.k[2] * x[6] * LMTD[2]
        f[6] = 1 - x[4] - x[5] - x[6]

        return f

    def calc(self):

        x = np.zeros(7)
        x[0] = self.TC
        x[1] = self.TAo_desuperheat
        x[2] = self.TAo_condenser
        x[3] = self.TAo_subcool
        x[4] = self.areafraction_desuperheat
        x[5] = self.areafraction_condenser
        x[6] = self.areafraction_subcool

        x = fsolve(self.model, x0=x, xtol=self.system.fun_tol)

        self.TC = x[0]
        self.TAo_desuperheat = x[1]
        self.TAo_condenser = x[2]
        self.TAo_subcool = x[3]
        self.areafraction_desuperheat = x[4]
        self.areafraction_condenser = x[5]
        self.areafraction_subcool = x[6]

        self.p = CPPSI('P', 'T', self.TC, 'Q', 0, self.medium)

        if self.dTSC == 0:
            hout = CPPSI('H', 'P', self.p, 'Q', 0, self.medium)
        else:
            hout = CPPSI('H', 'P', self.p, 'T', self.TC-self.dTSC, self.medium)
        mdot = self.junctions['inlet_A'].get_massflow()

        self.junctions['outlet_A'].set_values(p=self.p, h=hout, mdot=mdot)
        self.junctions['inlet_A'].set_values(p=self.p)

    def set_air_parameters(self, T_air: float = None, mdot: float = None):
        if T_air:
            self.T_air_in = T_air
        if mdot:
            self.mdot_air = mdot

    def get_function_residual(self):
        x = np.zeros(7)
        x[0] = self.TC
        x[1] = self.TAo_desuperheat
        x[2] = self.TAo_condenser
        x[3] = self.TAo_subcool
        x[4] = self.areafraction_desuperheat
        x[5] = self.areafraction_condenser
        x[6] = self.areafraction_subcool

        res = self.model(x)
        Qdot = self.junctions['inlet_A'].get_massflow() * (self.junctions['outlet_A'].get_enthalpy() - self.junctions['inlet_A'].get_enthalpy())
        res[0:6] = res[0:6]/Qdot
        return res

    def set_air_temp(self, T_air:float):
        self.T_air_in = T_air

    def set_air_mdot(self, mdot_air: float):
        self.mdot_air = mdot_air

    def get_outlet_temp(self):
        return self.TAo_subcool

    def get_ph_data(self, npoints: int):
        p = np.zeros(npoints)
        p.fill(self.p)

        hin = self.junctions['inlet_A'].get_enthalpy()
        h = np.linspace()


class CondenserCounterflow(Component):
    """
    Condenser model for counter flow characteristic.
    The model is based on a three zone condeser model. (Desuperheating, Condensing, Subcooling)
    """
    def __init__(self, id: str, system: object, k: iter, area: float, subcooling: float, initial_areafractions: iter = None, upper_pressure_limit = 26e5):
        """
        Initialize the condenser model.

        :param id: Name of the condenser object.
        :param system: System object, that shall contain the condenser object.
        :param k: Array of heat transfer coefficients. [k_dsh, k_cond, k_sc]
        :param area: Heat transfer area of the condenser
        :param subcooling: Subcooling temperature difference at the outlet.
        :param initial_areafractions: Initial guess for area fraction.
        """
        super().__init__(id, system)
        if len(k) != 3:
            raise ValueError('k must be of length 3, but len(k) = {}'.format(len(k)))
        else:
            self.k_dsh = k[0]
            self.k_cond = k[1]
            self.k_sc = k[2]
        self.area = area
        self.dTSC = subcooling
        # self.parameters = {'UA': self.UA, 'subcooling': self.dTSC}
        self.upper_pressure_limit = upper_pressure_limit

        self.TC = None
        self.T_SL1 = None
        self.T_SL2 = None
        self.T_SLo = None
        self.f_dsh = None
        self.f_cond = None
        self.f_sc = None
        self.p = None

        self.junctions['inlet_B'] = None
        self.junctions['outlet_B'] = None

        if initial_areafractions:
            if len(initial_areafractions) != 3:
                raise ValueError('initial_areafractions must be of len 3')
            self.f_dsh = initial_areafractions[0]
            self.f_cond = initial_areafractions[1]
            self.f_sc = initial_areafractions[2]
        else:
            self.f_dsh = 0.1
            self.f_cond = 0.8
            self.f_sc = 0.1

    def initialize(self):
        """
        Run further initialization. These tasks cannot be done in __init__(), because there have to be junctions added to the component.

        :return:
        """

        # first get the values of the inlet junctions
        self.update_inlet_interfaces()

        # read the medium handles from inlet junctions
        self.ref = self.junctions['inlet_A'].medium
        self.ref_HEOS = CoolProp.AbstractState('HEOS', self.ref)
        self.SL = self.junctions['inlet_B'].medium

        self.p = self.junctions['inlet_A'].get_pressure()
        self.ref_HEOS.update(CoolProp.PQ_INPUTS, self.p, 0)
        self.TC = self.ref_HEOS.T()
        h_liquid = self.ref_HEOS.hmass()
        QC = self.mdot_ref * (self.h_ref_in - h_liquid)
        cpSL = CPPSI('C', 'T', self.T_SLi, 'P', 1e5, self.SL)  # heat capacity of secondary liquid
        dTSL = QC/(self.mdot_SL * cpSL)
        self.T_SL1 = self.T_SLi - self.f_dsh * dTSL
        self.T_SL2 = self.T_SL1 - self.f_cond * dTSL
        self.T_SLo = self.T_SL2 - self.f_sc * dTSL

    def define_export_variables(self):
        """
        Define which variables are to be exported.

        :return:
        """
        self.export_variables = {
            'mdot_ref': self.mdot_ref,
            'mdot_SL': self.mdot_SL,
            'p_ref': self.p,
            'T_ref_in': self.T_ref_in,
            'T_SL_in': self.T_SLi,
            'T_SL_out': self.T_SLo,
            'fA_desuperheat': self.f_dsh,
            'fA_condensing': self.f_cond,
            'fA_subcool': self.f_sc
        }
        return

    def model(self, x, recursive_call = False):
        """
        The model class, that is used to run the root determination algorithm in "calc()".\n
        The variables are:\n
        x[0] = self.p \n
        x[1] = self.T_SL1 \n
        x[2] = self.T_SL2 \n
        x[3] = self.T_SLo \n
        x[4] = self.f_dsh \n
        x[5] = self.f_cond \n
        x[6] = self.f_sc

        :param x: Array of free variables.
        :return: The result of the equation system.
        """

        # special limit to protect against negative or unrealistically low pressures
        if x[0] > self.upper_pressure_limit:
            alt_x_low = np.zeros(7)
            alt_x_low[0] = self.upper_pressure_limit - 1
            alt_x_low[1:] = x[1:]
            res_low = self.model(alt_x_low, recursive_call=True)

            alt_x_high = np.zeros(7)
            alt_x_high[0] = self.upper_pressure_limit
            alt_x_high[1:] = x[1:]
            res_high = self.model(alt_x_high, recursive_call=True)

            d_res = res_low - res_high
            return d_res*(self.upper_pressure_limit - x[0]) + res_low

        # Calculate refrigerant temperatures
        self.ref_HEOS.update(CoolProp.PQ_INPUTS, x[0], 1)
        hGas = self.ref_HEOS.hmass()

        self.ref_HEOS.update(CoolProp.PQ_INPUTS, x[0], 0)
        TC = self.ref_HEOS.T()
        hliquid = self.ref_HEOS.hmass()

        self.ref_HEOS.update(CoolProp.PT_INPUTS, x[0], TC - self.dTSC)
        h_R_out = self.ref_HEOS.hmass()

        # Calculate sec. liquid cp
        cpSL = CPPSI('C', 'T', (self.T_SLi + x[1]) / 2, 'P', 1e5, self.SL)  # heat capacity of secondary liquid

        # Calculate the mean logarithmic temperature value for all three sections of the condenser
        LMTD_dsh = lmtd_calc(self.T_ref_in, TC, x[2], x[3])
        LMTD_cond = lmtd_calc(TC, TC, x[1], x[2])
        LMTD_sc = lmtd_calc(TC, TC - self.dTSC, self.T_SLi, x[1])

        f = np.zeros(7)
        # desuperheat zone
        f[0] = self.mdot_ref * (self.h_ref_in - hGas) - self.mdot_SL * cpSL * (x[3] - x[2])
        f[1] = self.mdot_ref * (self.h_ref_in - hGas) - x[4] * self.k_dsh * self.area * LMTD_dsh

        # condensing zone
        f[2] = self.mdot_ref * (hGas - hliquid) - self.mdot_SL * cpSL * (x[2] - x[1])
        f[3] = self.mdot_ref * (hGas - hliquid) - x[5] * self.k_cond * self.area * LMTD_cond

        # subcooling zone
        f[4] = self.mdot_ref * (hliquid - h_R_out) - self.mdot_SL * cpSL * (x[1] - self.T_SLi)
        f[5] = self.mdot_ref * (hliquid - h_R_out) - x[6] * self.k_sc * self.area * LMTD_sc

        # Area conservation
        f[6] = 1 - x[4] - x[5] - x[6]

        return f

    def calc(self):
        """
        Calculates the condenser model. With the defined "model" function, it runs a scipy.optimize.root algorithm
        to determine the roots of the equation system.
        The result of the root finding is stored and the junctions are updated.

        :return:
        """
        self.update_inlet_interfaces()

        # build the x vector with all variables
        x = np.zeros(7)
        x[0] = self.p
        x[1] = self.T_SL1
        x[2] = self.T_SL2
        x[3] = self.T_SLo
        x[4] = self.f_dsh
        x[5] = self.f_cond
        x[6] = self.f_sc

        # run root finding algorithm and store the result
        sol = scipy.optimize.root(self.model, x0=x)
        x = sol.x

        # store the results in the object variables
        self.p = x[0]
        self.T_SL1 = x[1]
        self.T_SL2 = x[2]
        self.T_SLo = x[3]
        self.f_dsh = x[4]
        self.f_cond = x[5]
        self.f_sc = x[6]

        # calculate additional object variables
        self.ref_HEOS.update(CoolProp.PQ_INPUTS, self.p, 0)
        self.TC = self.ref_HEOS.T()

        if self.dTSC == 0:
            hout = CPPSI('H', 'P', self.p, 'Q', 0, self.junctions['inlet_A'].medium)
        else:
            hout = CPPSI('H', 'P', self.p, 'T', self.TC-self.dTSC, self.junctions['inlet_A'].medium)

        mdot = self.junctions['inlet_A'].get_massflow()
        hB_out = CPPSI('H', 'T', self.T_SL1, 'P', self.junctions['inlet_B'].get_pressure(), self.junctions['inlet_B'].medium)

        # update the outlet junctions
        self.junctions['outlet_A'].set_values(p=self.p, h=hout, mdot=mdot)
        self.junctions['inlet_A'].set_values(p=self.p)
        self.junctions['outlet_B'].set_values(h=hB_out)

    def update_inlet_interfaces(self):
        """
        Update the inlet interfaces by reading the parameters of the inlet junctions.

        :return:
        """
        self.mdot_ref = self.junctions['inlet_A'].get_massflow()
        self.T_ref_in = self.junctions['inlet_A'].get_temperature()
        self.h_ref_in = self.junctions['inlet_A'].get_enthalpy()
        self.mdot_SL = self.junctions['inlet_B'].get_massflow()
        self.T_SLi = self.junctions['inlet_B'].get_temperature()

    def update_parameter(self, param, value):
        """
        Function to update a parameter with a string and value.

        :param param: string of the parameter to be changed.
        :param value: new value of the parameter
        :return:
        """
        if param == 'k':
            self.set_k_value(value)

        else:
            raise ValueError('Cannot set parameter {}'.format(param))

    def set_k_value(self, k):
        """
        Set the heat transfer coefficients.

        :param k: Array of heat transfer coefficients. [k_dsh, k_cond, k_sc]
        :return:
        """
        if len(k) != 3:
            raise ValueError('k must be of length 3, but len(k) = {}'.format(len(k)))
        else:
            self.k_dsh = k[0]
            self.k_cond = k[1]
            self.k_sc = k[2]

    def get_function_residual(self):
        """
        Return the function residuals of the root finding algorithm by running the model function with the results of the root finding

        :return: Array of function residuals.
        """
        x = np.zeros(7)
        x[0] = self.p
        x[1] = self.T_SL1
        x[2] = self.T_SL2
        x[3] = self.T_SLo
        x[4] = self.f_dsh
        x[5] = self.f_cond
        x[6] = self.f_sc

        res = self.model(x)
        Qdot = self.junctions['inlet_A'].get_massflow() * (self.junctions['outlet_A'].get_enthalpy() - self.junctions['inlet_A'].get_enthalpy())
        res[0:6] = res[0:6]/Qdot
        return res


class Evaporator(Component):
    """
    Evaporator model. This is deprecated, use EvaporatorCounterflow instead.
    """
    def __init__(self, id: str, system: object, k: iter, area: float, superheat: float, boundary_switch: bool, limit_temp: bool, initial_areafractions: iter = None):
        super().__init__(id, system)
        if len(k) == 2:
            self.k = k
        else:
            raise ValueError('k has to be of length 2. len(k) = {}'.format(len(k)))

        self.area = area
        self.superheat = superheat
        self.boundary_switch = boundary_switch
        self.limit_temp = limit_temp
        self.T0 = None
        self.TSL2 = None
        self.TSLmid = None
        self.xE1 = None
        self.xE2 = None

        self.TSL_in = None

        self.junctions['inlet_B'] = None
        self.junctions['outlet_B'] = None

        if initial_areafractions:
            if len(initial_areafractions) != 2:
                raise ValueError('{} allows only initial_areafraction of size 2, but got size {}'.format(self.id, len(initial_areafractions)))
            self.xE1 = initial_areafractions[0]
            self.xE2 = initial_areafractions[1]

        else:
            self.xE1 = 0.8
            self.xE2 = 1 - self.xE1

    def initialize(self):
        self.p = self.junctions['inlet_A'].get_pressure()
        self.TSL1 = self.junctions['inlet_B'].get_temperature()
        self.T0 = CPPSI('T', 'P', self.p, 'Q', 0, self.junctions['inlet_A'].medium)
        self.TSL2 = self.TSL1 - (self.TSL1 - self.T0) * 0.8
        self.TSLmid = (self.TSL1 + self.TSL2)/2

    def define_export_variables(self):
        self.export_variables = {
            'T0': self.T0,
            'fA_evaporating': self.xE1,
            'fA_superheat': self.xE2,
            'p': self.p
        }
        return

    def model(self, x):
        TSLi = self.junctions['inlet_B'].get_temperature()
        mSL = self.junctions['inlet_B'].get_massflow()
        hRi = self.junctions['inlet_A'].get_enthalpy()
        mR = self.junctions['inlet_A'].get_massflow()
        ref = self.junctions['inlet_A'].medium
        SL = self.junctions['inlet_B'].medium

        # print('---EVAP---')
        # print(self.junctions['inlet_A'].get_temperature())

        # Boundaries for fsolve calculation to not cause the logaritmic mean temperature to generate NaN values (neg. logarithm)
        # The refrigerants oulet temperature must not be higher than the coolants inlet temperature:

        if self.boundary_switch:
            if x[0] + self.superheat > TSLi:
                x[0] = TSLi - self.superheat

            # The evaporation temperature must not be higher than the coolants outlet temperature
            if x[0] > x[1]:
                x[0] = x[1] - 1e-6
            if x[1] > x[2]:
                x[1] = x[2] + 1e-3


        # calculate material parameters
        if (x[0] < 150.) and self.limit_temp:
            cpR = CPPSI('C', 'T', 150., 'Q', 1, ref) * (x[0]/150.)  # generate a linear extrapolation for the iteration
            hRGas = CPPSI("H", "T", 150., "Q", 1, ref) * (x[0]/150.)
        else:
            cpR = CPPSI('C', 'T', x[0], 'Q', 1, ref)  # heat capacity of fully evaporated refrigerant
            hRGas = CPPSI("H", "T", x[0], "Q", 1, ref)  # enthalpy of fully evaporated refrigerant

        cpSL = CPPSI('C', 'T', (TSLi + x[1]) / 2, 'P', 1e5, SL)  # heat capacity of secondary liquid

        # Calculate the mean logarithmic temperature value for all two sections of the condenser
        LMTD = np.zeros(2)
        LMTD[0] = lmtd_calc(x[2], x[1], x[0], x[0])
        LMTD[1] = lmtd_calc(TSLi, x[2], x[0], self.superheat + x[0])

        # Formulation of the equation system as according to fsolve documentation ( 0 = ... ).
        # The equation set  and model definition is documented in the model description.
        f = np.zeros(5)

        # energy balance evaporating zone between refrigerant and sec. liquid
        f[0] = mR * (hRGas - hRi) - mSL * cpSL * (x[2] - x[1])

        # energy balance evaporating zone between refrigerant and LMTD model
        f[1] = mR * (hRGas - hRi) - self.k[0] * x[3] * self.area * LMTD[0]

        # energy balance superheating zone between refrigerant and sec. liquid
        # f[ 2 ] = mR * (hRSuperheated - hRGas) - mSL * cpSL * (TSLi - x[ 2 ])
        f[2] = mR * cpR * self.superheat - mSL * cpSL * (TSLi - x[2])

        # energy balance superheating zone between refrigerant and LMTD model
        # f[3] = mR * (hRSuperheated-hRGas) - k[1] * x[4]/100 * Atot * LMTD[1]
        f[3] = mR * cpR * self.superheat - self.k[1] * x[4] * self.area * LMTD[1]

        # area fraction balance (1 = x_evaporating + x_superheating)
        f[4] = 1 - x[3] - x[4]

        return f

    def calc(self):
        x = np.zeros(5)
        x[0] = self.T0
        x[1] = self.TSL2
        x[2] = self.TSLmid
        x[3] = self.xE1
        x[4] = self.xE2

        # x = fsolve(self.model, x0=x, xtol=self.system.fun_tol)
        sol = scipy.optimize.root(self.model, x0=x, tol=self.system.fun_tol)
        x = sol.x
        self.T0 = x[0]
        self.TSL2 = x[1]
        self.TSLmid = x[2]
        self.xE1 = x[3]
        self.xE2 = x[4]

        self.p = CPPSI('P', 'T', self.T0, 'Q', 1, self.junctions['inlet_A'].medium)
        Tout = self.T0 + self.superheat
        hout = CPPSI('H', 'T', Tout, 'P', self.p, self.junctions['inlet_A'].medium)
        self.junctions['outlet_A'].set_values(p=self.p, h=hout, mdot=self.junctions['inlet_A'].get_massflow())
        hSL2 = CPPSI('H', 'T', self.TSL2, 'P', 1e5, self.junctions['inlet_B'].medium)
        # self.junctions['inlet_A'].set_values(p=self.p)
        self.junctions['outlet_B'].set_values(h=hSL2, mdot=self.junctions['inlet_B'].get_massflow())

    def get_function_residual(self):
        x = np.zeros(5)
        x[0] = self.T0
        x[1] = self.TSL2
        x[2] = self.TSLmid
        x[3] = self.xE1
        x[4] = self.xE2

        # normalize the energy balance residuals
        res = self.model(x)
        Qdot = self.junctions['inlet_A'].get_massflow() * (self.junctions['outlet_A'].get_enthalpy() - self.junctions['inlet_A'].get_enthalpy())
        res[0:4] = res[0:4]/Qdot
        return res

    def update_parameter(self, param, value):
        if param == 'k':
            self.set_k_value(value)

        else:
            raise ValueError('Cannot set parameter {}'.format(param))

    def set_k_value(self, k):
        self.k = k


class EvaporatorCounterflow(Component):
    """
    Evaporator model for counter flow characteristic.
    The model is based on a two zone evaporator model. (Evaporating, Superheating)
    The model also "contains" an expansion organ, as it is trying to reach a superheat temperature at the outlet.
    """
    def __init__(self, id: str, system: object, k: iter, area: float, superheat: float, initial_areafractions: iter = None, lower_pressure_limit=5e4):
        """
        Initialize the evaporator model.

        :param id: Name of the object.
        :param system: System object, that shall contain the condenser object.
        :param k: Array of heat transfer coefficients [k_evap, k_sh]
        :param area: Heat transfer area of the evaporator.
        :param superheat: Superheat at the evaporator outlet
        :param initial_areafractions: Initial guess for the area fractions.
        """
        super().__init__(id, system)
        if len(k) == 2:
            self.k_ev = k[0]
            self.k_sh = k[1]
        else:
            raise ValueError('k has to be of length 2. len(k) = {}'.format(len(k)))

        self.area = area
        self.superheat = superheat
        self.lower_pressure_limit = lower_pressure_limit

        self.T0 = None
        self.TSL2 = None
        self.TSLmid = None
        self.xE1 = None
        self.xE2 = None

        self.TSL_in = None

        self.junctions['inlet_B'] = None
        self.junctions['outlet_B'] = None

        if initial_areafractions:
            if len(initial_areafractions) != 2:
                raise ValueError('{} allows only initial_areafraction of size 2, but got size {}'.format(self.id, len(initial_areafractions)))
            self.xE1 = initial_areafractions[0]
            self.xE2 = initial_areafractions[1]

        else:
            self.xE1 = 0.8
            self.xE2 = 1 - self.xE1

    def initialize(self):
        """
        Run further initialization. These tasks cannot be done in __init__(), because there have to be junctions added to the component.

        :return:
        """
        self.p = self.junctions['inlet_A'].get_pressure()
        self.TSL1 = self.junctions['inlet_B'].get_temperature()
        self.T0 = CPPSI('T', 'P', self.p, 'Q', 0, self.junctions['inlet_A'].medium)
        self.TSL2 = self.TSL1 - (self.TSL1 - self.T0) * 0.8
        self.TSLmid = (self.TSL1 + self.TSL2)/2

        self.ref = self.junctions['inlet_A'].medium
        self.ref_HEOS = CoolProp.AbstractState('HEOS', self.ref)

        self.SL = self.junctions['inlet_B'].medium

    def define_export_variables(self):
        """
        Define which variables are to be exported.

        :return:
        """
        self.export_variables = {
            'T0': self.T0,
            'fA_evaporating': self.xE1,
            'fA_superheat': self.xE2,
            'p': self.p
        }
        return

    def get_function_residual(self):
        """
        Return the function residuals of the root finding algorithm by running the model function with the results of the root finding.

        :return: Array of function residuals.
        """
        x = np.zeros(5)
        x[0] = self.p
        x[1] = self.TSL2
        x[2] = self.TSLmid
        x[3] = self.xE1
        x[4] = self.xE2

        # normalize the energy balance residuals
        res = self.model(x)
        Qdot = self.junctions['inlet_A'].get_massflow() * (self.junctions['outlet_A'].get_enthalpy() - self.junctions['inlet_A'].get_enthalpy())
        res[0:4] = res[0:4]/Qdot
        return res

    def update_parameter(self, param, value):
        """
        Function to update a parameter with a string and value.

        :param param: string of the parameter to be changed.
        :param value: new value of the parameter
        :return:
        """
        if param == 'k':
            self.set_k_value(value)

        else:
            raise ValueError('Cannot set parameter {}'.format(param))

    def set_k_value(self, k):
        """
        Set the heat transfer coefficients.

        :param k: Array of heat transfer coefficients. [k_dsh, k_cond, k_sc]
        :return:
        """
        if len(k) == 2:
            self.k_ev = k[0]
            self.k_sh = k[1]
        else:
            raise ValueError('k has to be of length 2. len(k) = {}'.format(len(k)))

    def update_inlet_interfaces(self):
        """
        Update the inlet interfaces by reading the parameters of the inlet junctions.

        :return:
        """
        self.hRi = self.junctions['inlet_A'].get_enthalpy()
        self.mR = self.junctions['inlet_A'].get_massflow()

        self.mSL = self.junctions['inlet_B'].get_massflow()
        self.T_SLi = self.junctions['inlet_B'].get_temperature()

    def model(self, x, recursive_call=False):
        """
        The model class, that is used to run the root determination algorithm in "calc()".\n
        The variables are:\n
        x[0] = self.p\n
        x[1] = self.TSL2\n
        x[2] = self.TSLmid\n
        x[3] = self.xE1\n
        x[4] = self.xE2

        :param x: Array of free variables.
        :return: The result of the equation system.
        """
        # x[0] = self.p
        # x[1] = self.TSL2
        # x[2] = self.TSLmid
        # x[3] = self.xE1
        # x[4] = self.xE2

        # special limit to protect against negative or unrealistically low pressures
        if x[0] < self.lower_pressure_limit:
            alt_x_low = np.zeros(5)
            alt_x_low[0] = self.lower_pressure_limit
            alt_x_low[1:] = x[1:]
            res_low = self.model(alt_x_low, recursive_call=True)

            alt_x_high = np.zeros(5)
            alt_x_high[0] = self.lower_pressure_limit + 1
            alt_x_high[1:] = x[1:]
            res_high = self.model(alt_x_high, recursive_call=True)

            d_res = res_low - res_high
            return d_res*(self.lower_pressure_limit - x[0]) + res_low


        cpSL = CPPSI('C', 'T', (self.T_SLi + x[1]) / 2, 'P', 1e5, self.SL)  # heat capacity of secondary liquid

        self.ref_HEOS.update(CoolProp.PQ_INPUTS, x[0], 1)  # update the CoolProp handle of refrigerant
        T0 = self.ref_HEOS.T()  # get saturation temperature
        hGas = self.ref_HEOS.hmass()  # get enthalpy of saturated gas refrigerant
        self.ref_HEOS.update(CoolProp.PT_INPUTS, x[0], T0+self.superheat)  # update the CoolProp handle of refrigerant
        hRout = self.ref_HEOS.hmass()  # get enthalpy of refrigerant outlet

        # Calculate the mean logarithmic temperature value for all two sections of the condenser
        LMTD_ev = lmtd_calc(x[2], x[1], T0, T0)
        LMTD_sh = lmtd_calc(self.T_SLi, x[2], T0, T0+self.superheat)

        f = np.zeros(5)

        f[0] = self.mR * (hGas - self.hRi) - self.mSL * cpSL * (x[2] - x[1])
        f[1] = self.mR * (hGas - self.hRi) - x[3] * self.area * self.k_ev * LMTD_ev
        f[2] = self.mR * (hRout - hGas) - self.mSL *  cpSL * (self.T_SLi - x[2])
        f[3] = self.mR * (hRout - hGas) - x[4] * self.area * self.k_sh * LMTD_sh
        f[4] = 1 - x[3] - x[4]

        return f

    def calc(self):
        """
        Calculates the evaporator model. With the defined "model" function, it runs a scipy.optimize.root algorithm
        to determine the roots of the equation system.
        The result of the root finding is stored and the junctions are updated.

        :return:
        """
        self.update_inlet_interfaces()

        # build the x vector with all variables
        x = np.zeros(5)
        x[0] = self.p
        x[1] = self.TSL2
        x[2] = self.TSLmid
        x[3] = self.xE1
        x[4] = self.xE2

        # run root finding algorithm and store the result
        sol = scipy.optimize.root(self.model, x0=x)
        x = sol.x

        # store the results in the object variables
        self.p = x[0]
        self.TSL2 = x[1]
        self.TSLmid = x[2]
        self.xE1 = x[3]
        self.xE2 = x[4]

        # calculate additional object variables
        self.T0 = CPPSI('T', 'P', self.p, 'Q', 1, self.junctions['inlet_A'].medium)
        Tout = self.T0 + self.superheat
        hout = CPPSI('H', 'T', Tout, 'P', self.p, self.junctions['inlet_A'].medium)
        hSL2 = CPPSI('H', 'T', self.TSL2, 'P', 1e5, self.junctions['inlet_B'].medium)

        # update the outlet junctions
        self.junctions['outlet_A'].set_values(p=self.p, h=hout, mdot=self.junctions['inlet_A'].get_massflow())
        self.junctions['outlet_B'].set_values(h=hSL2, mdot=self.junctions['inlet_B'].get_massflow())


class IHX(Component):
    """
    Model of the internal (suction-liquid line) heat exchanger. It is a "simple" UA heat transfer model.
    """
    def __init__(self, id: str, system: object, UA: float):
        """
        Initialize the IHX.

        :param id:  Name of the IHX object.
        :param system: System object, that shall contain the IHX object.
        :param UA: UA (i.e. kA) value for the IHX.
        """
        super().__init__(id=id, system=system)
        self.UA = UA
        self.TA_in = None
        self.TA_out = None
        self.TB_in = None
        self.TB_out = None
        self.mdot = None
        self.ref = None

        self.junctions['inlet_B'] = None
        self.junctions['outlet_B'] = None

    def initialize(self):
        """
        Run further initialization. These tasks cannot be done in __init__(), because there have to be junctions added to the component.


        :return:
        """
        self.TA_in = self.junctions['inlet_A'].get_temperature()
        self.TA_out = self.TA_in - 1.

        self.TB_in = self.junctions['inlet_B'].get_temperature()
        self.TB_out = self.TB_in + 1.

    def model(self, x):
        """
        The model class, that is used to run the root determination algorithm in "calc()".\n
        The variables are:\n
        x[0] = self.TA_out \n
        x[1] = self.TB_out

        :param x:
        :return:
        """
        # calculate heat capacities based on guessed temperature values
        cp_A = CPPSI("CPMASS", "T", self.TA_in, "P", self.pA, self.medium)
        cp_B = CPPSI("CPMASS", "T", self.TB_in, "P", self.pB, self.medium)

        # check direction of hot and cold medium flow
        if self.TA_in > self.TB_in:
            Thi = self.TA_in
            Tho = x[0]
            Tci = self.TB_in
            Tco = x[1]
        else:
            Thi = self.TB_in
            Tho = x[1]
            Tci = self.TA_in
            Tco = x[0]

        # caluclate the mean log. temperature difference
        LMTD = lmtd_calc(Thi, Tho, Tci, Tco)
        Qdot = self.UA * LMTD

        f = np.zeros(2)
        # return the zero vectors with correct sign
        if self.TA_in > self.TB_in:
            f[0] = self.mdot * cp_A * (self.TA_in - x[0]) - Qdot
            f[1] = self.mdot * cp_B * (self.TB_in - x[1]) + Qdot
        else:
            f[0] = self.mdot * cp_A * (self.TA_in - x[0]) + Qdot
            f[1] = self.mdot * cp_B * (self.TB_in - x[1]) - Qdot

        return f

    def calc(self):
        """
        Calculates the condenser model. With the defined "model" function, it runs a scipy.optimize.root algorithm
        to determine the roots of the equation system.
        The result of the root finding is stored and the junctions are updated.

        :return:
        """
        self.update_inlet_interfaces()

        x = np.zeros(2)
        x[0] = self.TA_out
        x[1] = self.TB_out

        x = fsolve(self.model, x0=x)

        self.TA_out = x[0]
        self.TB_out = x[1]

        pA_out = self.junctions['inlet_A'].get_pressure()
        pB_out = self.junctions['inlet_B'].get_pressure()
        hA_out = CPPSI('H', 'T', self.TA_out, 'P', pA_out, self.medium)
        hB_out = CPPSI('H', 'T', self.TB_out, 'P', pB_out, self.medium)
        mdot = self.junctions['inlet_A'].get_massflow()

        self.junctions['outlet_A'].set_values(p=pA_out, h=hA_out, mdot=mdot)
        self.junctions['outlet_B'].set_values(p=pB_out, h=hB_out, mdot=mdot)
        # print('---IHX---')
        # print(self.TA_out, self.TB_out)
        # print(self.junctions['outlet_A'].get_temperature(), self.junctions['outlet_B'].get_temperature())

    def get_function_residual(self):
        """
        Return the function residuals of the root finding algorithm by running the model function with the results of the root finding

        :return: Array of function residuals.
        """
        x = np.zeros(2)
        x[0] = self.TA_out
        x[1] = self.TB_out
        return self.model(x)

    def update_parameter(self, param, value):
        """
        Function to update a parameter with a string and value.

        :param param: string of the parameter to be changed.
        :param value: new value of the parameter
        :return:
        """
        if param == 'k':
            self.set_k_value(value)

        else:
            raise ValueError('Cannot set parameter {}'.format(param))

    def set_UA_value(self, UA):
        """
        Set the UA value of the IHX.

        :param UA: New UA value.
        :return:
        """
        self.UA = UA

    def define_export_variables(self):
        """
        Define which variables are to be exported.

        :return:
        """
        self.export_variables = {
            'TA_in': self.TA_in,
            'TA_out': self.TA_out,
            'TB_in': self.TB_in,
            'TB_out': self.TB_out,
            'UA': self.UA,
            'mdot': self.mdot
        }
        return

    def update_inlet_interfaces(self):
        """
        Update the inlet interfaces by reading the parameters of the inlet junctions.

        :return:
        """
        self.TA_in = self.junctions['inlet_A'].get_temperature()
        self.mdot = self.junctions['inlet_A'].get_massflow()
        self.TB_in = self.junctions['inlet_B'].get_temperature()
        self.medium = self.junctions['inlet_A'].medium

        # get the interface pressures
        self.pA = self.junctions['inlet_A'].get_pressure()
        self.pB = self.junctions['inlet_B'].get_pressure()



class Source(Component):
    """
    Source components are components, that allow the definition of boundary conditions. Source components define mass flow, pressure and enthalpy.
    """
    def __init__(self, id: str, system: object, mdot=None, p=None, h=None):
        """
        Initialize Source component.

        :param id: Name of the Source object.
        :param system: System object, that shall contain the Source object.
        :param mdot: Mass flow of the source
        :param p: Presure of the source
        :param h: Enthalpy of the source
        """
        super().__init__(id=id, system=system)
        if mdot:
            self.mdot = mdot
        if p:
            self.p = p
        if h:
            self.h = h

        self.junctions = {'outlet_A': None}

    def define_export_variables(self):
        """
        Define which variables are to be exported.

        :return:
        """
        self.export_variables = {
            'mdot': self.mdot,
            'p': self.p,
            'h': self.h
        }
        return

    def calc(self):
        """
        The "calculation" of the Source object is simply updating the outlet junction of the source with the values of the source.

        :return:
        """
        self.junctions['outlet_A'].set_values(mdot=self.mdot, p=self.p, h=self.h)

    def set_enthalpy(self, h):
        """
        Set the enthalpy of the source.

        :param h: Enthalpy
        :return:
        """
        self.h = h

    def set_mdot(self, mdot):
        """
        Set the mass flow of the source.

        :param mdot: mass flow
        :return:
        """
        self.mdot = mdot

    def set_pressure(self, p):
        """
        Set the pressure of the source.

        :param p: pressure
        :return:
        """
        self.p = p

    def update_parameter(self, param, value):
        """
        Function to update a parameter with a string and value.

        :param param: string of the parameter to be changed.
        :param value: new value of the parameter
        :return:
        """
        if param == 'h':
            self.set_enthalpy(value)
        elif param == 'mdot':
            self.set_mdot(value)
        elif param == 'p':
            self.set_pressure(value)
        else:
            raise ValueError('Cannot set parameter {}'.format(param))


class Sink(Component):
    """
    The sink component is used to terminate medium streams.
    """
    def __init__(self, id: str, system: object, mdot=None, p=None, h=None):
        """
        Initialize the source.

        :param id: Name of the sink object
        :param system: System object, that shall contain the condenser object.
        :param mdot: optional
        :param p: optional
        :param h: optional
        """
        super().__init__(id=id, system=system)
        if mdot:
            self.mdot = mdot
        if p:
            self.p = p
        if h:
            self.h = h

        self.junctions = {'inlet_A': None}

    def define_export_variables(self):
        """
        Define which variables are to be exported.

        :return:
        """
        self.export_variables = {
            'mdot': self.mdot,
            'p': self.p,
            'h': self.h
        }
        return

    def calc(self):
        """
        The "calculation" of the Sink model is simply updating the internal variables with the values of the inlet junction.

        :return:
        """
        j = self.junctions['inlet_A']
        self.mdot = j.get_massflow()
        self.p = j.get_pressure()
        self.h = j.get_enthalpy()

    def get_temperature(self):
        """
        Return the temprature of the Sink.

        :return: Temperature
        """
        T = CPPSI('T', 'P', self.p, 'H', self.h, self.junctions['inlet_A'].medium)
        return T
