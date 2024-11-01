from abc import abstractmethod
from hypervehicle import Vehicle
import numpy as np
from hypervehicle.geometry.autodiff_alt import FloatWithSens


class AbstractGenerator:
    """Abstract Generator Interface."""

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def create_instance(self) -> Vehicle:
        pass


class Generator(AbstractGenerator):
    """Hypervehicle Parametric Generator."""

    def __init__(self, **kwargs) -> None:
        """Initialises the generator."""
        # Unpack kwargs and overwrite parameter named attributes
        for item in kwargs:
            setattr(self, item, kwargs[item])

    def dvdp(self,parameter_dict: dict[str, any]):
        i=0
        sens_set = np.zeros(len(parameter_dict), dtype=float)
        FloatWithSens.params=parameter_dict

        for item in parameter_dict:
            sens_set[i]=1.0
            setattr(self, item, FloatWithSens(parameter_dict[item],sens_set))
            sens_set[i]=0.0
            i+=1