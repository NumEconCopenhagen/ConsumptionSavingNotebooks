Contains Jupyter Notebooks showcasing the [ConSav package](https://github.com/NumEconCopenhagen/ConsumptionSaving).

# Getting Started

The main tool in the [ConSav package](https://github.com/NumEconCopenhagen/ConsumptionSaving) is the **ModelClass** class with predefined methods for e.g. saving and loading. The main selling point is that it provides an easy interface to calling Python functions jit compilled with [Numba](http://numba.pydata.org/), and C++ functions. Each concrete model inherits these methods and then adds methods for e.g. solving and simulating.

The main folders are:

* **01. BufferStockModel/** Example with canonical buffer-stock model.
* **02. DurableConsumptionModel/** Example with the solution method proposed in [A Guide to Solve Non-Convex Consumption-Saving Models](https://doi.org/10.1007/s10614-020-10045-x), [Druedahl](https://sites.google.com/view/jeppe-druedahl/), 2021, *Computational Economics*.
* **03. G2EGM/** Python version of the G2EGM algorithm from [A General Endogenous Grid Method for Multi-Dimensional Models with Non-Convexities and Constraints](https://doi.org/10.1016/j.jedc.2016.11.005), [Druedahl](https://sites.google.com/view/jeppe-druedahl/) and [Jørgensen](http://www.tjeconomics.com/), 2017, *Journal of Economic Dynamics and Control*, 74 ([MATLAB version](https://github.com/JeppeDruedahl/G2EGM)).
* **04. Tools/** Showcases the various tools.

The repository **[EconModelNotebooks](https://github.com/NumEconCopenhagen/EconModelNotebooks)** contains a number of examples on using the underlying model class.

**To get started:**

1. Install the EconModel package: ``pip install EconModel``
2. Install the ConSav package: ``pip install ConSav``
3. Clone or download this repository
4. Open your notebook of choice

If you are **new to Python** then try out this online course, [Introduction to programming and numerical analysis](https://sites.google.com/view/numeconcph-introprog/home).

We recommend running the notebooks in [JupyerLab](https://jupyterlab.readthedocs.io/en/stable/). A set of guides on how to install Python and JupyterLab is available [here](https://sites.google.com/view/numeconcph-introprog/guides).