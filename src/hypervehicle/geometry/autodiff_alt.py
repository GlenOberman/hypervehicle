import os
from typing import Union, Optional
import numpy as np
import functools
import pandas as pd

from art import tprint, art

HANDLED_FUNCTIONS = {}

def implements(np_function):
   "Register an __array_ufunc__ implementation for a FloatWithSens."
   def decorator(func):
       HANDLED_FUNCTIONS[np_function] = func
       return func
   return decorator

@functools.total_ordering
class FloatWithSens:
    __slots__ = ("number", "sens")
    N=0
    dtype=np.dtype('<f8')
    params=[]
    WARN_GIVEN = []

    def __init__(self, number:float, sens):
        if FloatWithSens.N==0:
            FloatWithSens.N=len(sens)
        elif len(sens)!=FloatWithSens.N:
            raise Exception("Inconsistent sensitivity component detected")
        self.number = float(number)
        self.sens = np.array(sens,dtype=FloatWithSens.dtype)

    def __add__(self,other:Union[float, int]):
        if isinstance(other,FloatWithSens):
            return FloatWithSens(self.number+other.number,self.sens+other.sens)
        else:
            return FloatWithSens(self.number+other,self.sens)
        
    def __radd__(self,other:Union[float, int]):
        return FloatWithSens(other+self.number,self.sens)
    
    def __sub__(self,other:Union[float, int]):
        if isinstance(other,FloatWithSens):
            return FloatWithSens(self.number-other.number,self.sens-other.sens)
        else:
            return FloatWithSens(self.number-other,self.sens)
        
    def __rsub__(self,other:Union[float, int]):
        return FloatWithSens(other-self.number,-1 * self.sens)
        
    def __mul__(self,other:Union[float, int]):
        if isinstance(other,FloatWithSens):
            return FloatWithSens(self.number*other.number,self.number*other.sens+self.sens*other.number)
        else:
            return FloatWithSens(self.number*other,self.sens*other)
        
    def __rmul__(self,other:Union[float,int]):
        return FloatWithSens(other*self.number,other*self.sens)
    
    def __neg__(self):
        return self * -1
        
    def __truediv__(self,other:Union[float,int]):
        if isinstance(other,FloatWithSens):
            return FloatWithSens(self.number/other.number,self.sens/other.number-self.number*other.sens/other.number**2)
        else:
            return FloatWithSens(self.number/other,self.sens/other)
        
    def __rtruediv__(self,other:Union[float,int]):
        return FloatWithSens(other/self.number,-1*other*self.sens/self.number**2)
        
    def __pow__(self,other:Union[float,int]):
        if isinstance(other,FloatWithSens):
            xtoy = self.number**other.number
            return FloatWithSens(xtoy,np.log(self.number)*xtoy*other.sens+other.number*xtoy/self.number*self.sens)
        else:
            return FloatWithSens(self.number**other,other*self.number**(other-1)*self.sens)
        
    def __rpow__(self,other:Union[float,int]):
        return FloatWithSens(other**self.number,np.log(other)*other**self.number*self.sens)
    
    def __repr__(self):
        return f"{self.number} with sensitivities {self.sens}"
    
    def __abs__(self):
        # Note: This has been implemented with sensitivities when number is zero as zero themselves, to avoid infinities
        return FloatWithSens(abs(self.number),np.sign(self.number)*self.sens)
    
    def __float__(self):
        # This strips the value of its sensitivities
        return self.number
    
    def __eq__(self,other):
        if isinstance(other,FloatWithSens):
            return self.number==other.number
        else:
            return self.number==other
        
    def __gt__(self,other):
        if isinstance(other,FloatWithSens):
            return self.number>other.number
        else:
            return self.number>other
        
    def __round__(self,ndigits):
        return FloatWithSens(round(self.number,ndigits),self.sens)
        
    def __copy__(self):
        return FloatWithSens(self.number,np.copy(self.sens))
        
    def __deepcopy__(self,memo):
        return self.__copy__()
    
    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if ufunc not in HANDLED_FUNCTIONS:
            if ufunc not in FloatWithSens.WARN_GIVEN:
                print(f"Falling back to float when calling np.{ufunc.__name__} - ufunc")
                FloatWithSens.WARN_GIVEN+=[ufunc]
            new_args=(float(arg) for arg in args)
            return ufunc(*new_args, **kwargs)
        return HANDLED_FUNCTIONS[ufunc](*args, **kwargs)
    
    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            if func not in FloatWithSens.WARN_GIVEN:
                print(f"Falling back to float when calling np.{func.__name__} - not ufunc")
                FloatWithSens.WARN_GIVEN+=[func]
            new_args=(float(arg) for arg in args)
            return func(*new_args, **kwargs)
        return HANDLED_FUNCTIONS[func](*args, **kwargs)
    
    @implements(np.isfinite)
    def isfinite(self):
        return np.isfinite(self.number) and np.all(np.isfinite(self.sens))
    
    @implements(np.isclose)
    def isclose(self,other,**kwargs):
        if isinstance(other,FloatWithSens):
            return np.isclose(float(self), other.number, **kwargs)
        else:
            return np.isclose(float(self), other, **kwargs)

    @implements(np.add)
    def add(self,other):
        if isinstance(self,FloatWithSens):
            return self+other
        elif isinstance(other,FloatWithSens):
            return other+self
        raise Exception("Unknown error when triggering np.add")
    
    @implements(np.subtract)
    def subtract(self,other):
        if isinstance(self,FloatWithSens):
            return self-other
        elif isinstance(other,FloatWithSens):
            return -other+self
        raise Exception("Unknown error when triggering np.subtract")
    
    @implements(np.multiply)
    def multiply(self,other):
        if isinstance(self,FloatWithSens):
            return self*other
        elif isinstance(other,FloatWithSens):
            return other*self
        raise Exception("Unknown error when triggering np.multiply")
    
    @implements(np.divide)
    def divide(self,other):
        if isinstance(self,FloatWithSens):
            return self/other
        elif isinstance(other,FloatWithSens):
            return self*(1/other)
        raise Exception("Unknown error when triggering np.divide")    

    @implements(np.cos)
    def cos(self):
        return FloatWithSens(np.cos(self.number),-np.sin(self.number)*self.sens)
    
    @implements(np.sin)
    def sin(self):
        return FloatWithSens(np.sin(self.number),np.cos(self.number)*self.sens)
    
    def sincos(self):
        s = np.sin(self.number)
        c = np.cos(self.number)
        return FloatWithSens(s,c*self.sens),FloatWithSens(c,-s*self.sens)
    
    @implements(np.tan)
    def tan(self):
        return FloatWithSens(np.tan(self.number),1/np.cos(self.number)**2*self.sens)
    
    @implements(np.sqrt)
    def sqrt(self):
        return self**(1/2)

    @implements(np.arctan2)
    def arctan2(y,x):
        sumsquares = float(x)**2+float(y)**2
        sens_val=np.zeros(FloatWithSens.N,dtype=FloatWithSens.dtype)
        if isinstance(y,FloatWithSens):
            sens_val+=float(x)/sumsquares*y.sens
        if isinstance(x,FloatWithSens):
            sens_val-=float(y)/sumsquares*x.sens
        return FloatWithSens(np.arctan2(float(y),float(x)),sens_val)
    
    # min and max are not implemented optimally, for sake of quick implementation
    @implements(np.max)
    def max(self,other):
        return (self + other + abs(self-other))/2

    @implements(np.min)
    def min(self,other):
        return (self + other - abs(self-other))/2
    
    @implements(np.rad2deg)
    def rad2deg(self):
        return self * (180/np.pi)

    @implements(np.isnan)
    def isnan(self):
        return np.isnan(self.number)
    
    @implements(np.interp)
    def interp(self, xp, fp, **kwargs):
        if self <= xp[0]:
            return fp[0]
        for i in range(1,len(xp)):
            if self < xp[i]:
                return fp[i-1] + (fp[i]-fp[i-1])*(self-xp[i-1])/(xp[i]-xp[i-1])
        return fp[-1]

    # To check - rem and round (see geometry.py lines 917-939)

def get_sens(val,i):
    if isinstance(val,FloatWithSens):
        return val.sens[i]
    else:
        return 0.0

class SensitivityStudyAlt:
    """
    Computes the geometric sensitivities using automatic differentiation.
    """

    def __init__(self, vehicle_constructor, verbosity: Optional[int] = 1):
        """Vehicle geometry sensitivity constructor.

        Parameters
        ----------
        vehicle_constructor : AbstractGenerator
            The Vehicle instance constructor.

        verbosity : int, optional
            The code verbosity. The default is 1.

        Returns
        -------
        VehicleSensitivity object.
        """
        self.vehicle_constructor = vehicle_constructor
        self.verbosity = verbosity

        # Parameter sensitivities
        self.parameter_sensitivities = None
        self.component_sensitivities = None
        self.component_scalar_sensitivities = None
        self.scalar_sensitivities = None
        self.property_sensitivities = None

        # Nominal vehicle instance
        self.nominal_vehicle_instance = None

        # Combined data file name
        self.combined_fn = "all_components_sensitivity_AD.csv"

    def __repr__(self):
        return "HyperVehicle sensitivity study using AD"
    
    def dvdp(
        self,
        parameter_dict: dict[str, any],
        overrides: Optional[dict[str, any]] = None,
        write_nominal_stl: Optional[bool] = True,
        nominal_stl_prefix: Optional[str] = None,
    ):
        """Computes the sensitivity of the geometry with respect to the
        parameters.

        Parameters
        ----------
        parameter_dict : dict
            A dictionary of the design parameters to perturb, and their
            nominal values.

        overrides : dict, optional
            Optional vehicle generator overrides to provide along with the
            parameter dictionary without variation. The default is None.

        perturbation : float, optional
            The design parameter perturbation amount, specified as percentage.
            The default is 20.

        vehicle_creator_method : str, optional
            The name of the method which returns a hypervehicle.Vehicle
            instance, ready for generation. The default is 'create_instance'.

        write_nominal_stl : bool, optional
            A boolean flag to write the nominal geometry STL(s) to file. The
            default is True.

        nominal_stl_prefix : str, optional
            The prefix to append when writing STL files for the nominal geometry.
            If None, no prefix will be used. The default is None.

        Returns
        -------
        sensitivities : dict
            A dictionary containing the sensitivity information for all
            components of the geometry, relative to the nominal geometry.
        """
        # Print banner
        if self.verbosity > 0:
            print_banner()
            print("Running geometric sensitivity study.")

        # TODO - return perturbed instances? After generatation to allow
        # quickly writing to STL
        from hypervehicle.generator import AbstractGenerator

        # Check overrides
        overrides = overrides if overrides else {}

        # Create Vehicle instance with nominal parameters
        if self.verbosity > 0:
            print("  Generating nominal geometry...")

        i=0
        sens_set = np.zeros(len(parameter_dict), dtype=float)
        parameter_dict_sens=parameter_dict.copy()
        for parameter, value in parameter_dict.items():
            sens_set[i]=1.0
            parameter_dict_sens[parameter] = FloatWithSens(value,sens_set.copy())
            sens_set[i]=0.0
            i+=1

        constructor_instance: AbstractGenerator = self.vehicle_constructor(
            **parameter_dict_sens, **overrides
        )
        nominal_instance = constructor_instance.create_instance()
        nominal_instance.verbosity = 0

        # Generate components
        nominal_instance.generate()
        nominal_meshes = {
            name: component.mesh
            for name, component in nominal_instance._named_components.items()
        }

        if self.verbosity > 0:
            print("    Done.")

        '''if write_nominal_stl:
            # Write nominal instance to STL files
            nominal_instance.to_stl(prefix=nominal_stl_prefix)

        # Generate meshes for each parameter
        if self.verbosity > 0:
            print("  Generating perturbed geometries...")
            print("    Parameters: ", parameter_dict.keys())

        sensitivities = {}
        analysis_sens = {}
        component_analysis_sens = {}
        property_sens = {}
        for parameter, value in parameter_dict.items():
            if self.verbosity > 0:
                print(f"    Generating for {parameter}.")

            sensitivities[parameter] = {}

            # Create copy
            adjusted_parameters = parameter_dict.copy()

            # Adjust current parameter for sensitivity analysis
            adjusted_parameters[parameter] *= 1 + perturbation / 100
            dp = adjusted_parameters[parameter] - value

            # Create Vehicle instance with perturbed parameter
            constructor_instance = self.vehicle_constructor(
                **adjusted_parameters, **overrides
            )
            parameter_instance = constructor_instance.create_instance()
            parameter_instance.verbosity = 0

            # Generate stl meshes
            parameter_instance.generate()
            parameter_meshes = {
                name: component.mesh
                for name, component in parameter_instance._named_components.items()
            }

            # Generate sensitivities for geometric analysis results
            if nominal_instance.analysis_results:
                analysis_sens[parameter] = {}
                for r, v in nominal_instance.analysis_results.items():
                    analysis_sens[parameter][r] = (
                        parameter_instance.analysis_results[r] - v
                    ) / dp

                # Repeat for components
                component_analysis_sens[parameter] = (
                    parameter_instance._volmass - nominal_instance._volmass
                ) / dp

            # Generate sensitivities for vehicle properties
            if nominal_instance.properties:
                property_sens[parameter] = {}
                for property, v in nominal_instance.properties.items():
                    property_sens[parameter][property] = (
                        parameter_instance.properties[property] - v
                    ) / dp

            # Generate sensitivities
            for component, nominal_mesh in nominal_meshes.items():
                parameter_mesh = parameter_meshes[component]
                sensitivity_df = self._compare_meshes(
                    nominal_mesh,
                    parameter_mesh,
                    dp,
                    parameter,
                )

                sensitivities[parameter][component] = sensitivity_df
        '''

        sensitivities = {}
        i=0
        for parameter in parameter_dict_sens.keys():
            sensitivities[parameter] = {}
            for component, nominal_mesh in nominal_meshes.items():
                nominal_vectors = nominal_mesh.vectors
                nominal_sensitivities = nominal_mesh.sensitivities[:,:,:,i]
                shape = nominal_vectors.shape
                flat_mesh = nominal_vectors.reshape((shape[0] * shape[2], shape[1]))
                flat_mesh_sens = nominal_sensitivities.reshape((shape[0] * shape[2], shape[1]))
                all_data = np.zeros((shape[0] * shape[2], shape[1] * 2))
                for j in range(3):
                    all_data[:, j] = flat_mesh[:,j]  # Reference locations
                    all_data[:, j+3] = flat_mesh_sens[:,j]  # Location derivatives
                
                sens_data = pd.DataFrame(data=all_data, columns=["x", "y", "z", f"dxd{parameter}", f"dyd{parameter}", f"dzd{parameter}"])
                sensitivities[parameter][component] = sens_data[~sens_data.duplicated()]
            i+=1

        if self.verbosity > 0:
            print("    Done.")

        # Return output
        self.parameter_sensitivities = sensitivities
        #self.scalar_sensitivities = analysis_sens
        #self.component_scalar_sensitivities = component_analysis_sens
        #self.property_sensitivities = property_sens
        self.component_sensitivities = self._combine(nominal_instance, sensitivities)
        self.nominal_vehicle_instance = nominal_instance

        if self.verbosity > 0:
            print("Sensitivity study complete.")

        return sensitivities
    
    def to_csv(self, outdir: Optional[str] = None):
        """Writes the sensitivity information to CSV file.

        Parameters
        ----------
        outdir : str, optional
            The output directory to write the sensitivity files to. If
            None, the current working directory will be used. The default
            is None.

        Returns
        -------
        combined_data_filepath : str
            The filepath to the combined sensitivity data.
        """
        # Check if sensitivities have been generated
        if self.component_sensitivities is None:
            raise Exception("Sensitivities have not yet been generated.")

        else:
            # Check output directory
            if outdir is None:
                outdir = os.getcwd()

            if not os.path.exists(outdir):
                # Make the directory
                os.mkdir(outdir)

            # Save sensitivity data for each component
            all_sens_data = pd.DataFrame()
            for component, df in self.component_sensitivities.items():
                df.to_csv(
                    os.path.join(outdir, f"{component}_sensitivity_alt.csv"), index=False
                )
                all_sens_data = pd.concat([all_sens_data, df])

            # Also save the combined sensitivity data
            combined_data_path = os.path.join(outdir, self.combined_fn)
            all_sens_data.to_csv(combined_data_path, index=False)

            # Also save analysis sensitivities
            if self.scalar_sensitivities:
                # Make analysis results directory
                properties_dir = os.path.join(outdir, f"scalar_sensitivities_alt")
                if not os.path.exists(properties_dir):
                    os.mkdir(properties_dir)

                # Save volume and mass
                reformatted_results = {}
                for p, s in self.component_scalar_sensitivities.items():
                    labels = []
                    values = []
                    s: pd.DataFrame
                    for component, comp_sens in s.iterrows():
                        comp_sens: pd.Series
                        for i, j in comp_sens.items():
                            labels.append(f"{component}_{i}")
                            values.append(j)

                    reformatted_results[p] = values

                # Convert to DataFrame and save
                comp_sens = pd.DataFrame(data=reformatted_results, index=labels)
                comp_sens.to_csv(
                    os.path.join(properties_dir, "volmass_sensitivity_alt.csv")
                )

                # Save others
                for param in self.scalar_sensitivities:
                    self.scalar_sensitivities[param]["cog"].tofile(
                        os.path.join(properties_dir, f"{param}_cog_sensitivity_alt.txt"),
                        sep=", ",
                    )
                    self.scalar_sensitivities[param]["moi"].tofile(
                        os.path.join(properties_dir, f"{param}_moi_sensitivity_alt.txt"),
                        sep=", ",
                    )

            # Also save user-defined property sensitivities
            if self.property_sensitivities:
                properties_dir = os.path.join(outdir, f"scalar_sensitivities_alt")
                if not os.path.exists(properties_dir):
                    os.mkdir(properties_dir)

                pd.DataFrame(self.property_sensitivities).to_csv(
                    os.path.join(properties_dir, "property_sensitivity_alt.csv")
                )

            return combined_data_path
        
    @staticmethod
    def _combine(nominal_instance, sensitivities):
        """Combines the sensitivity information for multiple parameters."""
        component_names = nominal_instance._named_components.keys()
        params = list(sensitivities.keys())

        allsens = {}
        for component in component_names:
            df = sensitivities[params[0]][component][["x", "y", "z"]]
            for param in params:
                p_s = sensitivities[param][component][
                    [f"dxd{param}", f"dyd{param}", f"dzd{param}"]
                ]
                df = pd.concat([df, p_s], axis=1)

            allsens[component] = df

        return allsens
    

def print_banner():
    """Prints the hypervehicle banner"""
    tprint("Hypervehicle", "tarty4")
    p = art("airplane2")
    print(f" {p}               {p}               {p}               {p}")