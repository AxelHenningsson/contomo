Welcome to the CONTOMO project
===============================

This is a scientific code originally developed to adress 
high speed - sparse data reconstruction in tomography as published `here`_:

publication citation, yerar, journal, etc.

.. _here: https://domain.invalid/

The central idea is to view the tomography problem as an 
inital value andvection PDE to be propagated in time. Solutions
are obtianed by discretizing the density field in space in finite volume
fashion and for each sought time recovering the driving velocity field through
the projected advection equations using standard iteratove methods.

The package contains support for reconstruction as well as generation of
4d phantoms for numerical experiments.

To learn more about applications, underlying theory and limitations check out 
the original publicationalong with the coding example