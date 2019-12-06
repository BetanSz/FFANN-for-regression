# FFANN-for-regression
Feed forward (fully conected) Artificial Neural Networks for hyper-parameter study in
few-group homogenized nuclear cross section of a typicial PWR fuel assembly.

For Python2.7
This script allows to model multi-layered fully conected Artificial Neural Networks (ANN) as
presented in the Article:
[1] "Few-group cross sections library compression by artificial neural networks." Proceding of the
 conference: Physics of reactor conference,Cambridge, United Kingdom, 2020

This is a work in progress and so many Limitations/TODO:

Presentation
* add example of dummy X,Y data generation
* improve runtime print format if verboes=True
* Improve docstring in general and put in markdown format
* Check that serialized results are actually independent of GPU usage (they are prolly not)
* Too long and complex core functions (train..)
* Add bash files of all intresting runs

Limitations/TODO of the Code:
* Handle Test batch better
* Functions must share support, i.e. same X tensor for all |Y| functions
* Statistical data is obtained during training insed of saving model hooks
* In regards to GPU memory only training batch is available solely for improved convergence. An OOM
 error is not currently handled
* Data is presumed to be in numpy or tensor format. No data check no unit test.
* Examples of multi-output not tested though already possible

As we nuclears say N-joy,
Esteban Szames
