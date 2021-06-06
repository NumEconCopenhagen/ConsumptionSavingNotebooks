Contains Jupyter Notebooks showcasing the [ConSav package](https://github.com/NumEconCopenhagen/ConsumptionSaving).

# Getting Started

The main tool in the [ConSav package](https://github.com/NumEconCopenhagen/ConsumptionSaving) is the **ModelClass** class with predefined methods for e.g. saving and loading. The main selling point is that it provides an easy interface to calling Python functions jit compilled with [Numba](http://numba.pydata.org/), and C++ functions. Each concrete model inherits these methods and then adds methods for e.g. solving and simulating. To get started there are three options:

1. The **00. DynamicProgramming/** folder contains a simple introduction to both dynamic programming and the **ConSav** package. In particular, the [Your First Consumption-Saving Model notebook](https://github.com/NumEconCopenhagen/ConsumptionSavingNotebooks/blob/master/00.%20DynamicProgramming/02.%20Your%20First%20Consumption-Saving%20Model.ipynb) showcases the fundamentals of the **ConSav** package.
2. The simplest full example is the canonical buffer-stock consumption model, see the [BufferStockModel notebook](https://github.com/NumEconCopenhagen/ConsumptionSavingNotebooks/blob/master/01.%20BufferStockModel/01.%20BufferStockModel.ipynb).
3. The [DurableConsumptioModel notebook](https://github.com/NumEconCopenhagen/ConsumptionSavingNotebooks/blob/master/02.%20DurableConsumptionModel/01.%20Example.ipynb) contains more advanced examples. Specifically, it implements the solution methods proposed in [A Guide On Solving Non-Convex Consumption-Saving Models](https://doi.org/10.1007/s10614-020-10045-x).

If you are **new to Python** then try out this online course, [Introduction to programming and numerical analysis](https://numeconcopenhagen.netlify.com/).

**To get started:**

1. Install the EconModel package: ``pip install EconModel``
2. Install the ConSav package: ``pip install ConSav``
3. Clone or download this repository
4. Open your notebook of choice

We recommend running the notebooks in [JupyerLab](https://jupyterlab.readthedocs.io/en/stable/). A set of guides on how to install Python and JupyterLab is available [here](https://numeconcopenhagen.netlify.com/guides/).

# Overview
The main folder are:

* **00. DynamicProgramming/** Tutorial on dynamic programming and the ConSav package.
* **01. BufferStockModel/** Example with canonical buffer-stock model.
* **02. DurableConsumptionModel/** Example with the solution method proposed in [A Guide to Solve Non-Convex Consumption-Saving Models](https://doi.org/10.1007/s10614-020-10045-x), [Druedahl](https://sites.google.com/view/jeppe-druedahl/), 2021, *Computational Economics*.
* **03. G2EGM/** Python version of the G2EGM algorithm from [A General Endogenous Grid Method for Multi-Dimensional Models with Non-Convexities and Constraints](https://doi.org/10.1016/j.jedc.2016.11.005), [Druedahl](https://sites.google.com/view/jeppe-druedahl/) and [Jørgensen](http://www.tjeconomics.com/), 2017, *Journal of Economic Dynamics and Control*, 74 ([MATLAB version](https://github.com/JeppeDruedahl/G2EGM)).
* **04. Tools/** Showcases the various tools.

The repository **[EconModelNotebooks](https://github.com/NumEconCopenhagen/EconModelNotebooks)** contains a number of examples on using the underlying model class.