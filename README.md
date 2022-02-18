Welcome to the CONTOMO project
===============================

This is a scientific code originally developed to adress 
high speed - sparse data reconstruction in tomography.

The package contains support for reconstruction as well as generation of
4d phantoms for numerical experiments.

The central idea is to view the tomography problem as an 
inital value andvection PDE to be propagated in time. Solutions
are obtianed by discretizing the density field in space in finite volume
fashion and for each sought time recovering the driving velocity field through
the projected advection equations using iterative methods.

Installation
---------------------------

---
**NOTE**

`contomo` is dependent on the [astra-toolbox](https://www.astra-toolbox.com/) which needs a [nvidia gpu with cuda](https://en.wikipedia.org/wiki/CUDA) to execute ray tracing.

--- 

To install `contomo` it is recomended to use [Anaconda](https://www.anaconda.com/). Start in a new environment

    conda create -n contomo
    conda activate contomo

Once in the new environment you will need to install the [astra-toolbox](https://www.astra-toolbox.com/)

    conda install -c astra-toolbox astra-toolbox

After this is is doen, we go ahead and install some additional dependecies

    conda install -c astra-toolbox astra-toolbox
    
Finally to install `contomo` clone the repo to your local machine

    git clone https://github.com/AxelHenningsson/contomo.git
 
and next run a local installation in the repo folder as
 
    cd contomo
    pip install -e .
 
 you can try and run some things from the sandbox folder named test_... to see that things look ok.
 
 Good luck!
 
