# BSD 3-Clause License
#
# Copyright (c) 2018, Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""This module defines an optimization-based design flow inspired by Eric Chang's VLSI 2018 talk."""

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple, List, Iterable, Mapping, Union, Callable, Type

import abc
import itertools
from pathlib import Path

from scipy import optimize as sciopt
from matplotlib.backends.backend_pdf import PdfPages
import numbers
import numpy as np

from pybag.enum import LogLevel

from bag.concurrent.util import GatherHelper
from bag.io.file import read_yaml
from bag.io.sim_data import save_sim_results, load_sim_file, SweepArray
from bag.math import float_to_si_string, si_string_to_float
from bag.math.dfun import DiffFunction, ScaleAddFunction
from bag.math.interpolate import interpolate_grid, interpolate_unstructured
from bag.util.immutable import ImmutableList, Param
from bag.util.importlib import import_class
from bag.layout.template import TemplateBase
from bag.design.module import Module

from bag.simulation.cache import SimulationDB, DesignInstance
from bag.simulation.design import DesignerBase

from .fun import CustomVectorArgMapFunction, CustomVectorReduceArgMapFunction


class OptimizationError(Exception):
    """A custom exception class for optimization design script errors"""
    pass


class SweepParams:
    """A data structure containing sweep information for a multi-dimensional parametric sweep.
    This class defines some utility methods to be used by OptDesigner.

    Parameters
    ----------
    params : Dict[str, Union[Dict[str, Union[str, float]], List[Union[str, float]], np.ndarray]]
        The sweep parameters dictionary, mapping variable names to their sweep points:
        If the value is a dictionary, the value is parsed as keyworded arguments to numpy.linspace
        Otherwise, the value is directly assumed to be an array of sweep points.
        Sweep points can be provided as str (where SI string to float conversion will be applied)

    force_linear : bool
        True if sweep values should be linear. If not linear, and exception is raised. Defaults to False.
        Since most multi-variable interpolation functions require sweep points to be on a regular grid,
        this allows for linearity conditions to be verified.
    """
    def __init__(self, params: Dict[str, Union[Dict[str, Union[str, float]], List[Union[str, float]], np.ndarray]],
                 force_linear: bool = False):
        self._swp_var_list = ImmutableList(sorted(params.keys()))
        swp_params = {}
        self._is_linear = {}
        for k, v in params.items():
            if isinstance(v, dict):
                v_mod = {sub_k: _soft_cast_si_to_float(sub_v) for sub_k, sub_v in v.items()}
                swp_params[k] = np.linspace(**v_mod)
                self._is_linear[k] = True
            else:
                v_mod = np.fromiter((_soft_cast_si_to_float(sub_v) for sub_v in v), float)
                delta = np.diff(v_mod)
                self._is_linear[k] = delta.size <= 1 or np.allclose(delta[1:], delta[0], atol=1e-22)
                if force_linear and not self._is_linear[k]:
                    raise ValueError(f"List of values for variable {k} must be linear")
                swp_params[k] = v_mod
        self._swp_params = swp_params

    def __contains__(self, item: str) -> bool:
        return item in self._swp_var_list

    @property
    def is_linear(self) -> Dict[str, bool]:
        """Returns a dictionary mapping sweep variables to whether their sweep points are linear"""
        return self._is_linear

    # TODO: is this needed?
    @property
    def first_params(self) -> Dict[str, Any]:
        return {key: self._swp_params[key][0] for key in self._swp_var_list}

    @property
    def is_regular_grid(self) -> bool:
        """Returns whether the entire sweep space is on a regular grid (i.e., linear along each dimension)"""
        return all(self._is_linear.values())

    @property
    def swp_var_list(self) -> ImmutableList[str]:
        """Returns a list of sweep variable names"""
        return self._swp_var_list

    @property
    def swp_shape(self) -> List[int]:
        """Returns the sweep shape (number of points along each dimension)"""
        return [len(self._swp_params[var]) for var in self._swp_var_list]

    @property
    def swp_params(self) -> Dict[str, np.ndarray]:
        """Returns a dictionary mapping sweep variable names to the list of points"""
        return self._swp_params

    def get_swp_values(self, var: str) -> np.ndarray:
        """Returns a list of valid sweep variable values.

        Parameter
        ---------
        var : str
            the sweep variable name.

        Returns
        -------
        val_list : np.ndarray
            the sweep values of the given variable.
        """
        return self._swp_params[var]

    def swp_combo_iter(self) -> Iterable[Tuple[float, ...]]:
        """Returns an iterator of parameter combinations we sweep over.

        Returns
        -------
        combo_iter : Iterable[Tuple[float, ...]]
            an iterator of tuples of parameter values that we sweep over.
        """
        return itertools.product(*(self._swp_params[var] for var in self._swp_var_list))

    def swp_combo_iter_as_dict(self) -> Iterable[Dict[str, float]]:
        """Returns an iterator of parameter combinations we sweep over as a dictionary.

        Returns
        -------
        combo_iter : Iterable[Dict[str, float]]
            an iterator of dictionary mapping variable names to values that we sweep over.
        """
        for combo in self.swp_combo_iter():
            yield {k: v for k, v in zip(self._swp_var_list, combo)}

class DataDesigner(DesignerBase, abc.ABC):
    """A streamlined DesignerBase module created to generate data intelligently.

    1. dsn_swp_params: Compromised of List[Dict]. Each [Dict] contains:
        a) path: Yaml hierarchy [List]
        b) swp_settings: Start, stop, num. of sweep points [Dict]
        c) active: Whether sweep should be performed [Bool]
        d) interpolation: Method of interpolating to form data point. Check class fxns for options. [Str]
    """
    
    def __init__(self, root_dir: Path, sim_db: SimulationDB, dsn_specs: Mapping[str, Any]) -> None:
        self._out_dir = self.get_out_dir(sim_db, root_dir)
        self.dsn_swp = None
        super().__init__(root_dir, sim_db, dsn_specs)
        
    @classmethod
    def _parse_param_from_path (cls, param_path: Union[str, Path]) -> Param:
        """
        :param param_path: Pathlike argument
        :returns: Param class
        """
        if isinstance(param_path, (str, Path)):
            params = read_yaml(str(param_path))
        else:
            raise ValueError("Need pathlike object (string or path).")
        return Param(params)

    # Class properties
    @property
    def dut_class(self) -> Union[Type[Module], Type[TemplateBase]]:
        """Return DUT Class"""
        return self._dut_class

    # Helper methods
    @abc.abstractmethod
    def get_dut_sch_class(self) -> Type[Module]:
        """Returns the default generator class"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_dut_lay_class(self) -> Type[TemplateBase]:
        """Returns the default layout class"""
        raise NotImplementedError


    def get_dut_class_info(self, gen_specs: Param) -> Tuple[Union[Type[Module], Type[TemplateBase]], bool]:
        """Returns information about the DUT generator class.
        :param gen_specs: 
        :returns: The 


        Parameters
        ----------
        gen_specs : Param
            The generator specs.

        Returns
        -------
        dut_class : Union[Type[Module], Type[TemplateBase]]
            The DUT generator class.

        is_lay : bool
            True if the DUT generator is a layout generator, False if schematic generator.
        """
        # Get default generator classes (in case no DUT class is found)
        try:
            sch_cls = self.get_dut_sch_class().get_qualified_name()
        except NotImplementedError:
            sch_cls = None
        try:
            lay_cls = self.get_dut_lay_class().get_qualified_name()
        except NotImplementedError:
            lay_cls = None

        if 'dut_class' in gen_specs:
            dut_cls = import_class(gen_specs['dut_class'])
            if issubclass(dut_cls, Module):
                is_lay = False
            elif issubclass(dut_cls, TemplateBase):
                is_lay = True
            else:
                raise ValueError(f"Invalid generator class {dut_cls.get_qualified_name()}")
        elif 'lay_class' in gen_specs:
            dut_cls = import_class(gen_specs['lay_class'])
            if not issubclass(dut_cls, TemplateBase):
                raise ValueError(f"Invalid layout generator class {dut_cls}")
            is_lay = True
        elif 'sch_class' in gen_specs:
            dut_cls = import_class(gen_specs['sch_class'])
            if not issubclass(dut_cls, Module):
                raise ValueError(f"Incorrect schematic generator class {dut_cls}")
            is_lay = False
        elif lay_cls is not None:
            is_lay = True
            dut_cls = lay_cls
        elif sch_cls is not None:
            is_lay = False
            dut_cls = sch_cls
        else:
            raise ValueError("Either schematic or layout class must be specified")
        return dut_cls, is_lay

        
class OptDesigner(DesignerBase, abc.ABC):
    """A design script class that attempts to find a globally optimal design via a characterization database.

    The general design methodology is as follows:
    1. Over a user-defined design space (multi-dimensional sweep of design variables), an array of designs are generated
       and simulated simultaneously. Each design's measurement results are stored in an HDF5 file,
       and the measurement results from all designs are combined and stored in a single HDF5 file to represent
       a characterization database.
    2. The measured results are then modeled as continuous multivariable functions by interpolating between adjacent
       design points.
    3. These models are passed into the optimization engine, which converges to a final design based on user-defined
       target specs.

    Parameters
    ----------
    dsn_specs : Mapping[str, Any]
        The design script specifications. The following entries should be specified:

        gen_specs : Union[Mapping[str, Any], Path, str]
            The base/default generator parameters. For each set of design parameters, new generator parameters
            will be computed by overriding only the design variables.
            If a Path or str is specified, the argument will be treated as a path to a specs YAML file.

        dsn_swp_params : Dict[str, Union[Dict[str, Union[str, float]], List[Union[str, float]], np.ndarray]]
            The mapping of design sweep parameters (e.g., device sizing). Each combination is used to generate a unique
            DUT. Refer to the SweepParams constructor for more info. Defaults to {}.

        sim_cfg_swp_params : Dict[str, Union[Dict[str, Union[str, float]], List[Union[str, float]], np.ndarray]]
            The mapping of simulation configuration sweep parameters (e.g., biasing).
            Refer to the SweepParams constructor for more info. Defaults to {}.

        sim_load_swp_params : Dict[str, Union[Dict[str, Union[str, float]], List[Union[str, float]], np.ndarray]]
            The mapping of simulation load sweep parameters (e.g., capacitive loading).
            Refer to the SweepParams constructor for more info. Defaults to {}.

        dsn_fixed_params : Dict[str, Any]
            The mapping of design parameters to fixed (i.e., non-swept) values. This is useful for quickly changing
            certain design parameters by bypassing an update to gen_specs. Defaults to {}.
    """

    def __init__(self, root_dir: Path, sim_db: SimulationDB, dsn_specs: Mapping[str, Any]) -> None:
        self._out_dir = self.get_out_dir(sim_db, root_dir)
        self.dsn_fixed_params = None
        self.dsn_swp = None
        self.sim_cfg_swp = None
        self.sim_load_swp = None
        self._dut_class = None
        self._is_lay = None
        self.base_gen_specs = None
        self._sim_swp_order = None
        self._swp_order = None
        self._sim_swp_params = None
        self._swp_params = None
        self._swp_shape = None
        self._em_sim = None
        super().__init__(root_dir, sim_db, dsn_specs)

    @classmethod
    def _parse_params(cls, params: Union[Mapping[str, Any], str, Path]) -> Param:
        """Returns the parsed parameter file if a Pathlike argument is specified, otherwise passthrough is performed.

        Parameters
        ----------
        params : Union[Mapping[str, Any], str, Path]
            The parameters to parse. If a string or a Path, then it is assumed to be the path to a yaml file and
            its contents are returned.

        Returns
        -------
        new_params : Param
            the parsed parameters cast to an immutable dictionary.
        """
        if isinstance(params, (str, Path)):
            params = read_yaml(str(params))
        return Param(params)

    def commit(self):
        super().commit()

        self.dsn_fixed_params = self._dsn_specs.get('dsn_fixed_params', {})

        self.dsn_swp = SweepParams(self._dsn_specs.get('dsn_swp_params', {}))
        self.sim_cfg_swp = SweepParams(self._dsn_specs.get('sim_cfg_swp_params', {}))
        self.sim_load_swp = SweepParams(self._dsn_specs.get('sim_load_swp_params', {}))

        base_gen_specs = self._parse_params(self._dsn_specs['gen_specs'])

        self._dut_class, self._is_lay = self.get_dut_class_info(base_gen_specs)

        em_params = self.dsn_specs.get('em_params', {})
        self._em_sim = True if em_params else False

        self.base_gen_specs = base_gen_specs['params']

        sim_swp_var_list = self.sim_cfg_swp_vars.to_list() + self.sim_load_swp_vars.to_list()
        self._sim_swp_order = ['corner'] + sim_swp_var_list
        self._swp_order = self.dsn_swp_vars.to_list() + self._sim_swp_order

        self._sim_swp_params = dict(
            corner=np.array(self.env_list),
            **self.sim_cfg_swp.swp_params,
            **self.sim_load_swp.swp_params
        )

        self._swp_params = dict(
            **self.dsn_swp.swp_params,
            **self._sim_swp_params
        )

        self._swp_shape = tuple(len(self._swp_params[var]) for var in self._swp_order)

    @property
    def is_lay(self) -> bool:
        """Return whether the specified DUT generator is a layout generator (True) or schematic generator (False)"""
        return self._is_lay

    @property
    def dut_class(self) -> Union[Type[Module], Type[TemplateBase]]:
        """Return the DUT generator class"""
        return self._dut_class

    @property
    def dsn_swp_vars(self) -> ImmutableList[str]:
        """Return the list of design sweep variable names"""
        return self.dsn_swp.swp_var_list

    @property
    def sim_cfg_swp_vars(self) -> ImmutableList[str]:
        """Return the list of simulation configuration sweep variable names"""
        return self.sim_cfg_swp.swp_var_list

    @property
    def sim_load_swp_vars(self) -> ImmutableList[str]:
        """Return the list of simulation loading sweep variable names"""
        return self.sim_load_swp.swp_var_list

    @property
    def env_list(self) -> List[str]:
        """Return the list of corners"""
        return self._dsn_specs['env_list']

    @property
    def dsn_basename(self) -> str:
        """Return the design basename"""
        return self._dsn_specs['dsn_basename']

    def get_dut_class_info(self, gen_specs: Param) -> Tuple[Union[Type[Module], Type[TemplateBase]], bool]:
        """Returns information about the DUT generator class.

        Parameters
        ----------
        gen_specs : Param
            The generator specs.

        Returns
        -------
        dut_class : Union[Type[Module], Type[TemplateBase]]
            The DUT generator class.

        is_lay : bool
            True if the DUT generator is a layout generator, False if schematic generator.
        """
        # Get default generator classes (in case no DUT class is found)
        try:
            sch_cls = self.get_dut_sch_class().get_qualified_name()
        except NotImplementedError:
            sch_cls = None
        try:
            lay_cls = self.get_dut_lay_class().get_qualified_name()
        except NotImplementedError:
            lay_cls = None

        if 'dut_class' in gen_specs:
            dut_cls = import_class(gen_specs['dut_class'])
            if issubclass(dut_cls, Module):
                is_lay = False
            elif issubclass(dut_cls, TemplateBase):
                is_lay = True
            else:
                raise ValueError(f"Invalid generator class {dut_cls.get_qualified_name()}")
        elif 'lay_class' in gen_specs:
            dut_cls = import_class(gen_specs['lay_class'])
            if not issubclass(dut_cls, TemplateBase):
                raise ValueError(f"Invalid layout generator class {dut_cls}")
            is_lay = True
        elif 'sch_class' in gen_specs:
            dut_cls = import_class(gen_specs['sch_class'])
            if not issubclass(dut_cls, Module):
                raise ValueError(f"Incorrect schematic generator class {dut_cls}")
            is_lay = False
        elif lay_cls is not None:
            is_lay = True
            dut_cls = lay_cls
        elif sch_cls is not None:
            is_lay = False
            dut_cls = sch_cls
        else:
            raise ValueError("Either schematic or layout class must be specified")
        return dut_cls, is_lay

    @staticmethod
    def get_out_dir(sim_db: SimulationDB, sim_dir: Union[Path, str]) -> Path:
        """Returns the root output directory for permanent data storage.

        Parameters
        ----------
        sim_db : SimulationDB
            The simulation database.

        sim_dir : Union[Path, str]
            The simulation directory.

        Returns
        -------
        out_dir : Path
            The output directory.
        """
        if isinstance(sim_dir, str):
            sim_dir = Path(sim_dir)
        if sim_dir.is_absolute():
            try:
                out_dir = sim_dir.relative_to(sim_db._sim._dir_path)
            except ValueError:
                return sim_dir
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                return out_dir
        else:
            return sim_dir

    def get_data_dir(self, dsn_name: str, meas_type: str = '') -> Path:
        """Returns the data directory path for the given measurement."""
        args = (dsn_name, meas_type) if meas_type else (dsn_name, )
        return self._out_dir.joinpath(*args)

    def get_meas_dir(self, dsn_name: str, meas_type: str = '') -> Path:
        """Returns the measurement directory path for the given measurement."""
        args = (dsn_name, meas_type) if meas_type else (dsn_name, )
        return self._work_dir.joinpath(*args)

    @property
    def swp_order(self) -> List[str]:
        """Returns an ordered list of all sweep variables."""
        return self._swp_order

    @property
    def swp_params(self) -> Dict[str, np.ndarray]:
        """Returns a dictionary mapping of sweep variable names to values."""
        return self._swp_params

    @property
    def swp_shape(self) -> tuple:
        """Returns the number of sweep points along each dimension."""
        return self._swp_shape

    @property
    def sim_swp_order(self) -> List[str]:
        """Returns an ordered list of simulation sweep variables."""
        return self._sim_swp_order

    @property
    def sim_swp_params(self) -> Dict[str, np.ndarray]:
        """Returns a dictionary mapping of simulation sweep variable names to values."""
        return self._sim_swp_params

    def get_dut_sch_class(self) -> Type[Module]:
        """Returns the default schematic generator class."""
        raise NotImplementedError

    def get_dut_lay_class(self) -> Type[TemplateBase]:
        """Returns the default layout generator class."""
        raise NotImplementedError

    @classmethod
    def get_dut_gen_specs(cls, is_lay: bool, base_gen_specs: Param, dsn_params: Mapping[str, Any]) \
            -> Union[Param, Dict[str, Any]]:
        """Returns the updated generator specs with some design variables.

        Parameters
        ----------
        is_lay : bool
            True if DUT is layout, False if schematic.

        base_gen_specs : Param
            The base/default generator specs.

        dsn_params : Mapping[str, Any]
            The design variables.

        Returns
        -------
        gen_specs : Union[Param, Dict[str, Any]]
            The updated generator specs.
        """
        return base_gen_specs

    @classmethod
    def get_em_dut_gen_specs(cls, base_gen_specs: Param, gen_params: Mapping[str, Any]
                             ) -> Union[Param, Dict[str, Any]]:
        """Returns the updated generator specs with some design variables.

        Parameters
        ----------
        base_gen_specs : Param
            The base/default generator specs.

        gen_params : Mapping[str, Any]
            The design variables.

        Returns
        -------
        gen_specs : Union[Param, Dict[str, Any]]
            The updated generator specs.
        """
        return base_gen_specs

    def process_meas_results(self, res: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Processes and returns measurement results.
        If any particular post-processing needs to be done, this method should be overriden by subclasses.

        Parameters
        ----------
        res : Dict[str, Any]
            Measurement results.

        params : Dict[str, Any]
            Design parameters.

        Returns
        -------
        new_res : Dict[str, Any]
            The updated measurement results.
        """
        return res

    @abc.abstractmethod
    async def verify_design(self, dut: DesignInstance, dsn_params: Dict[str, Any],
                            sim_swp_params: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Simulates and verifies design. This method is to be implemented by subclasses.

        Parameters
        ----------
        dut : DesignInstance
            The DUT.

        dsn_params : Dict[str, Any]
            Design parameters.

        sim_swp_params : Dict[str, np.ndarray]
            Simulation sweep parameters.

        Returns
        -------
        res : Dict[str, Any]
            The measurement results.
        """
        raise NotImplementedError

    def get_design_name(self, combo: Mapping[str, Any]) -> str:
        """Constructs the design name based on the specified combination of design parameters."""
        name = self.dsn_basename

        for var in self.dsn_swp_vars:
            if var not in combo:
                continue
            val = combo[var]
            if isinstance(val, str) or isinstance(val, int):
                name += f'_{var}_{val}'
            elif np.isscalar(val):
                name += f'_{var}_{float_to_si_string(val)}'
            else:
                raise ValueError('Unsupported parameter type: %s' % (type(val)))

        return name.replace('.', 'p')

    def get_results_fname(self, params: Dict[str, Any]) -> Path:
        """Returns the path to the design's measured results.

        Parameters
        ----------
        params : Dict[str, Any]
            The design parameters.

        Returns
        -------
        fpath : Path
            The measurement results path for the specified design parameters.
        """
        return self._out_dir / self.get_design_name(params) / 'results.hdf5'

    def load_results(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Loads and returns previously saved measurement results.

        Parameters
        ----------
        params : Dict[str, Any]
            The design parameters.

        Returns
        -------
        res : Dict[str, Any]
            The saved measurement results.
        """
        return load_sim_file(str(self.get_results_fname(params)))

    def save_results(self, res: Dict[str, Any], params: Dict[str, Any]):
        """Saves the measurement results.

        Parameters
        ----------
        res : Dict[str, Any]
            The measurement results.

        params : Dict[str, Any]
            The design parameters.
        """
        res = self.process_meas_results(res, params)
        save_sim_results(res, str(self.get_results_fname(params)))

    def check_results_exists(self, params: Dict[str, Any]) -> bool:
        """Checks if previous measurement results exist.

        Parameters
        ----------
        params : Dict[str, Any]
            The design parameters.

        Returns
        -------
        exists : bool
            True if previous results exist, False if not.
        """
        return self.get_results_fname(params).exists()

    async def characterize_designs(self):
        """Generates and characterizes all designs."""
        self.log('Characterizing designs...')
        db_data = dict(sweep_params={}, **self.swp_params)

        gatherer = GatherHelper()

        sl_list = []

        for dsn_params in self.dsn_swp.swp_combo_iter_as_dict():
            gatherer.append(self.characterize_single_design(dsn_params))

            # Get multi-dimensional index corresponding to design sweep point
            sl = [slice(None)] * len(self.swp_order)
            for dsn_var, dsn_val in dsn_params.items():
                sl[self.swp_order.index(dsn_var)] = np.where(self.dsn_swp.swp_params[dsn_var] == dsn_val)
            sl_list.append(tuple(sl))

        res_list = await gatherer.gather_err()

        for sl, res_item in zip(sl_list, res_list):
            for out_var in res_item['sweep_params']:
                if out_var not in db_data:
                    db_data[out_var] = np.full(self.swp_shape, np.nan, res_item[out_var].dtype)
                    db_data['sweep_params'][out_var] = self.swp_order
                db_data[out_var][sl] = res_item[out_var]

        # Save simulation results
        save_sim_results(db_data, str(self._out_dir / 'db.hdf5'))
        self.log('Characterization complete!')

    async def characterize_single_design(self, dsn_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generates and characterizes a single design.

        Parameters
        ----------
        dsn_params : Dict[str, Any]
            The design parameters.

        Returns
        -------
        res : Dict[str, Any]
            The measurement results.
        """
        res_fpath = self.get_results_fname(dsn_params)
        run_meas = self.check_run_meas(res_fpath, self.get_meas_var_list())

        if not run_meas:
            prev_res = load_sim_file(str(res_fpath))
            self.reorder_data_swp(prev_res, self.sim_swp_order)
            self.log(f'Reusing previous results for {dsn_params}', LogLevel.DEBUG)
            res = prev_res
        else:
            dsn_name = self.get_design_name(dsn_params)
            dsn_params = await self.pre_setup(dsn_params)
            dut_gen_params = self.get_dut_gen_specs(self._is_lay, self.base_gen_specs,
                                                    {**self.dsn_fixed_params, **dsn_params})
            ext_dut_name = f'{dsn_name}_ext' if self._em_sim else dsn_name
            dut = await self.async_new_dut(ext_dut_name, self.dut_class, dut_gen_params,
                                           export_lay=self._sim_db.extract and self._sim_db.gen_sch_dut)
            if self._em_sim:
                em_dut_gen_params = self.get_em_dut_gen_specs(self.base_gen_specs,
                                                              {**self.dsn_fixed_params, **dsn_params})
                em_dut, gds_file, gds_cached = await self.async_new_em_dut(f'{dsn_name}_em', self.dut_class,
                                                                           em_dut_gen_params, export_lay=True)
                sp_file = await self.async_gen_nport(em_dut, gds_file, gds_cached, self._dsn_specs['em_params'],
                                                     gds_file.parent)
                dsn_params['sp_file'] = sp_file

            res = await self.verify_design(dut, dsn_params, self.sim_swp_params)

        self.save_results(res, dsn_params)

        return res

    async def pre_setup(self, dsn_params: Dict[str, Any]) -> Dict[str, Any]:
        """Processes and returns design parameters.
        If any particular pre-processing needs to be done, this method should be overridden by subclasses.

        Parameters
        ----------
        dsn_params : Dict[str, Any]
            Design parameters.

        Returns
        -------
        new_params : Dict[str, Any]
            The updated design parameters.
        """
        return dsn_params

    def check_run_meas(self, res_fpath: Path, var_list: Optional[List[str]] = None) -> bool:
        """Checks to see if a design should be re-simulated.

        Parameters
        ----------
        res_fpath : Path
            File path to (previous) measurement results.

        var_list : Optional[List[str]]
            The list of measurement variables to check. Defaults to None.

        Returns
        -------
        run_meas : bool
            True to run measurement, False if not.
        """
        if self._sim_db._force_sim:  # force_sim is enabled, so always rerun
            return True
        if not res_fpath.exists():  # cannot find previous results
            return True

        # check for error when loading previous results
        try:
            prev_res = load_sim_file(str(res_fpath))
        except OSError:
            return True
        prev_sim_swp_order_list = list(prev_res['sweep_params'].values())
        # check if the sweep variable list is the same per measurement
        # TODO: if not, should this quietly rerun?
        if not all((sorted(_order) == sorted(prev_sim_swp_order_list[0]) for _order in prev_sim_swp_order_list)):
            raise ValueError("All measurements must have the same sweep variables")

        prev_sim_swp_order = prev_sim_swp_order_list[0]
        same_swp_var = sorted(prev_sim_swp_order) == sorted(self.sim_swp_order)
        if not same_swp_var:
            return True

        for var in self.sim_swp_order:  # Check if each variable has the same set of values
            if var not in prev_res:
                return True
            if self.sim_swp_params[var].shape != prev_res[var].shape:
                return True
            if var == 'corner':
                if not np.all(self.sim_swp_params[var] == prev_res[var]):
                    return True
            elif not np.allclose(self.sim_swp_params[var], prev_res[var]):
                return True

        # Check if all measurement variables are in previous results
        for var in var_list or []:
            if var not in prev_res:
                return True
            if var not in prev_res['sweep_params']:
                return True

        return False

    @classmethod
    def get_meas_var_list(cls) -> List[str]:
        """Return the expected measurement variables. Used for caching."""
        return []

    @classmethod
    def reorder_data_swp(cls, data: Dict[str, Any], swp_order: List[str]):
        """Reorders the simulation data to have the same order of sweep variables."""
        for out_var, in_var_order in data['sweep_params'].items():
            if in_var_order != swp_order:
                ax_dest = [swp_order.index(var) for var in in_var_order]
                # noinspection PyTypeChecker
                data[out_var] = SweepArray(np.moveaxis(data[out_var], list(range(len(ax_dest))), ax_dest), swp_order)

    def make_models(self, db_path: Optional[Path] = None) -> Tuple[Dict[str, List[DiffFunction]], List[str]]:
        """Computes models of the characterized database by interpolating simulation data.

        Parameters
        ----------
        db_path : Optional[Path]
            File path to the database. By default, set to db.hdf5 in the output directory.

        Returns
        -------
        fn_table : Dict[str, List[DiffFunction]]
            A dictionary mapping measured values to a list of functions (1 per corner).

        swp_names : List[str]
            The ordered list of sweep variables found in the database.
        """
        self.log('Generating database models...')
        interp_method = 'spline'  # TODO: make this a function argument?
        if db_path is None:
            db_path = self._out_dir / 'db.hdf5'
        db_data = load_sim_file(str(db_path))
        # Get measurement and design variable names
        out_names = list(db_data['sweep_params'].keys())
        swp_names = [elem for elem in db_data['sweep_params'][out_names[0]]]
        # remove corner from interpolation, since each corner will have its own interpolation function
        interp_names = [elem for elem in swp_names if elem != 'corner']

        scale_list = []
        points = []
        for name in interp_names:
            cur_xvec = db_data[name]
            scale_list.append((cur_xvec[0], cur_xvec[1] - cur_xvec[0]))
            points.append(cur_xvec)

        fn_table = {}
        corner_idx = swp_names.index('corner')
        sl: List[Union[slice, int]] = [slice(None) for _ in range(len(swp_names))]

        is_regular = True
        for idx, swp_var in enumerate(swp_names):
            if idx == corner_idx:
                continue
            if idx < corner_idx:
                is_linear = self.dsn_swp.is_linear[swp_var]
            elif swp_var in self.sim_cfg_swp:
                is_linear = self.sim_cfg_swp.is_linear[swp_var]
            else:
                is_linear = self.sim_load_swp.is_linear[swp_var]
            if not is_linear:
                is_regular = False
                break
        for out_var in db_data['sweep_params']:
            out_arr = db_data[out_var]
            fun_list = []
            for idx in range(len(self.env_list)):
                sl[corner_idx] = idx
                if is_regular:
                    fun_list.append(interpolate_grid(scale_list, out_arr[tuple(sl)], method=interp_method,
                                                     extrapolate=True, num_extrapolate=1))
                else:
                    fun_list.append(interpolate_unstructured(points, out_arr[tuple(sl)], method=interp_method,
                                                             extrapolate=True))
            fn_table[out_var] = fun_list
        self.log('Model generation complete!')
        return fn_table, swp_names

    def optimize(self, opt_var: str, fn_table: Dict[str, List[DiffFunction]],
                 swp_order: List[str], maximize: bool = False,
                 var_constraints: Optional[Dict[str, Union[float, Tuple[Optional[float], Optional[float]]]]] = None,
                 spec_constraints: Optional[Dict[str, Union[float, Tuple[Optional[float], Optional[float]]]]] = None,
                 reduce_fn: Callable = np.mean, custom_constraints_fn: Optional[Callable] = None,
                 rng: Optional[np.random.Generator] = None, rng_seed: Optional[int] = None,
                 num_success: int = 64, max_ratio_fail: float = 0.75, plot_conv: bool = False) \
            -> Tuple[Dict[str, List[Union[float, np.ndarray]]], float, Dict[str, np.ndarray]]:
        """Runs the optimization engine.

        The function to optimize and constraints are computed and passed into scipy.optimize.minimize.
        This solver will find a locally optimal design. To increase the chances of hitting the global optimal design,
        the solver is re-run multiple times with randomly generated initial conditions and the best design of
        all optimization results is chosen.

        Parameters
        ----------
        opt_var : str
            The measurement variable to optimize.

        fn_table : Dict[str, List[DiffFunction]]
            A mapping of measurement variables to function models.

        swp_order : List[str]
            The ordered list of sweep variables.

        maximize : bool
            True if opt_var should be maximized, False if minimized.

        var_constraints : Optional[Dict[str, Union[float, Tuple[Optional[float], Optional[float]]]]]
            A mapping of sweep variables to constraints.
            If a (single) float is passed in, the variable is constrained to exactly that value.
            If a tuple of 2 optional floats is passed in, the variable is constrained to (lower bound, upper bound).

        spec_constraints : Optional[Dict[str, Union[float, Tuple[Optional[float], Optional[float]]]]]
            A mapping of spec/measurement variables to constraints.
            If a (single) float is passed in, the variable is constrained to exactly that value.
            If a tuple of 2 optional floats is passed in, the variable is constrained to (lower bound, upper bound).

        reduce_fn : Callable
            The reduction function to apply to derive the optimization function.
            Database interpolation models are vector functions by nature (by corner). Since the optimizer aims to
            minimize a scalar, the vector must be reduced to a scalar.

        custom_constraints_fn : Optional[Callable]
            A function that can be called to generate additional optimization constraints as a function of fn_table.

        rng : Optional[np.random.Generator]
            The random number generator to use. Defaults to numpy.random.default_rng.

        rng_seed : Optional[int]
            The random number generator seed to use. Only used if the default RNG is used.

        num_success : int
            The number of successful optimization runs.

        max_ratio_fail : float
            Maximum ratio of optimization runs that can fail.

        plot_conv : bool
            True to plot the convergence trend.

        Returns
        -------
        opt_x_fmt : Dict[str, List[Union[float, np.ndarray]]]
            The dictionary mapping design parameters to optimal values.

        opt_y : float
            The optimized spec value.

        spec_vals : Dict[str, np.ndarray]
            The dictionary mapping spec/measured variables to their values at the optimal design point.
        """
        self.log('Running optimization...')
        var_constraints = var_constraints or {}
        spec_constraints = spec_constraints or {}
        opt_fns: List[DiffFunction] = fn_table[opt_var]

        # Remove corner
        swp_order = [var for var in swp_order if var != 'corner']

        # Process variable constraints
        fixed_vals = {}
        input_bounds = {}
        for var, bounds in var_constraints.items():
            if var not in self.dsn_swp_vars and var not in self.sim_cfg_swp_vars and var not in self.sim_load_swp_vars:
                raise KeyError(f"Variable {var} is invalid")
            if isinstance(bounds, tuple):
                input_bounds[swp_order.index(var)] = bounds
            elif isinstance(bounds, (int, float, np.ndarray)):
                if isinstance(bounds, np.ndarray) and var not in self.sim_cfg_swp_vars:
                    raise ValueError("Cannot set array bound")
                fixed_vals[swp_order.index(var)] = bounds
            else:
                raise ValueError(f"Unrecognized bound of type ({type(bounds)}) for variable ({var})")
        map_idx_list = [swp_order.index(var) for var in self.sim_cfg_swp_vars if var not in var_constraints]
        opt_fn = CustomVectorReduceArgMapFunction(opt_fns, map_idx_list, fixed_vals, input_bounds, reduce_fn=reduce_fn)
        if maximize:
            min_fn = ScaleAddFunction(opt_fn, 0.0, -1.0)
        else:
            min_fn = opt_fn

        if rng is None:
            rng = np.random.default_rng(rng_seed)

        const_fn_table = {}
        for k, fn_list in fn_table.items():
            const_fn_table[k] = CustomVectorArgMapFunction(fn_list, map_idx_list, fixed_vals)

        # Process specification constraints
        constraints = []
        for spec, bounds in spec_constraints.items():
            if spec not in self.get_meas_var_list():
                raise KeyError(f"Variable {spec} is invalid")
            if isinstance(bounds, tuple):
                bnd_l, bnd_h = bounds
                if bnd_l is not None and bnd_h is not None:
                    assert bnd_l < bnd_h, f'Lower bound ({bnd_l}) should be less than upper bound ({bnd_h})'
                if bnd_l is not None:
                    constraints.extend(const_fn_table[spec] >= bnd_l)
                if bnd_h is not None:
                    constraints.extend(const_fn_table[spec] <= bnd_h)
            elif isinstance(bounds, numbers.Number):
                constraints.extend(const_fn_table[spec] == bounds)
            else:
                raise ValueError(f"Unrecognized bound of type ({type(bounds)}) for spec ({spec})")

        bounds = opt_fn.input_ranges_norm
        if custom_constraints_fn is not None:
            constraints.extend(custom_constraints_fn(const_fn_table))

        opt_x, opt_y = None, np.inf
        fail_cnt = 0
        succ_cnt = 0
        max_fail = round(num_success / (1 - max_ratio_fail) * max_ratio_fail)
        # noinspection PyTypeChecker
        current_opt_arr = np.full(num_success, np.nan, dtype=float)
        while True:
            x0 = np.array(opt_fn.random_input(rng))
            # noinspection PyTypeChecker
            rv = sciopt.minimize(min_fn, x0, bounds=bounds, constraints=constraints, options=dict(disp=False))
            self.log(f'succ: {rv.success}, x0: {x0}, opt x: {rv.x}, opt y: {-rv.fun if maximize else rv.fun}',
                     LogLevel.DEBUG)
            if not rv.success:
                fail_cnt += 1
                if fail_cnt > max_fail:
                    raise OptimizationError("Too many failures")
                continue
            if rv.fun < opt_y:
                opt_x, opt_y = rv.x, rv.fun
            current_opt_arr[succ_cnt] = opt_y
            succ_cnt += 1
            if succ_cnt == num_success:
                break

        # Normalized inputs are used for the optimizer to enable proper convergence. Denormalize before returning
        opt_x_denormed = opt_fn.denorm_input(opt_x)
        if maximize:
            opt_y = -opt_y

        spec_vals = {k: fn(opt_x) for k, fn in const_fn_table.items()}

        if plot_conv:
            import matplotlib.pyplot as plt
            with PdfPages(self._out_dir / 'convergence.pdf') as pdf:
                plt.plot(current_opt_arr)
                pdf.savefig()

        opt_x_unmapped = opt_fn.unmap_input(opt_x_denormed)
        opt_x_fmt = {var: val for var, val in zip(swp_order, opt_x_unmapped)}
        self.log('Optimization complete!')
        return opt_x_fmt, opt_y, spec_vals


def _soft_cast_si_to_float(si: Union[float, str]) -> float:
    if isinstance(si, str):
        return si_string_to_float(si)
    return si
