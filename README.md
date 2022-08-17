# Prediction and Analysis of the Pacific Juvenile Loggerhead Sea Turtle's Migrations
Author: Elisabeth Kneip\
Supervisors: Philippe Gaspar, Emmanuel Hanert and Sébastien Jodogne

Python code implemented in a master thesis in order to obtain the diploma of Computer Science and Engineering at UCLouvain. This work is based on a dataset processed by [Mercator Océan](https://www.mercator-ocean.eu/) and originating [NOAA PIFSC](https://www.fisheries.noaa.gov/about/pacific-islands-fisheries-science-center). It is therefore not published in this repository.

### Replication of the Results Presented in the Writings
<sup> Tracks need to be added to the folder.</sup>\
The file [*settings*](settings.py) allows to choose a model to train as well as a target value. Then, [*main*](main.py) offers the different experiments presented in the master thesis. 

Chapter 5 used the *all_var* function with target value `vs` and model `dnn`, `lstm` or `condRNN`. 

Section 6.1.1 and 6.3 used the *all_var* function with target model `svm-savgol` and respectively target value `lat` or `lon`.

Section 6.1.2 used the same function, set the target value to `lat` and changed the models to `dnn`, `lstm` or `rf`. 

Sections 6.1.3 and 6.3.1 used the functions *exp_1* and *exp_2* with respectively target value `lat` or `lon`, and model `svm-savgol`. 

Sections 6.2 and 6.4 used the the functions *plot_scores_delta* and *plot_var_delta* with model `svm` and respectively target value `delta_lat` or `delta_lon`. 


