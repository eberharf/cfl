import pytest 

import cfl 
# This module includes unit tests to check that 
# each public module in the cfl package is able to be imported 
# using the statement 'import cfl' 

# As new modules are added, this file should be periodically 
# updated to include any new additions


#########CLUSTER METHODS 

def test_bad_import_kmeans(): 
    with pytest.raises(Exception): #this should not work 
        cluster_methods.kmeans.KMeans 

#test that kmeans is imported successfully 
def test_import_kmeans(): 
    assert cfl.cluster_methods.kmeans.KMeans #check if class exists 
         
# check that evaluate_clustering.py was imported 
def test_import_evaluate_clustering(): 
    assert cfl.cluster_methods.evaluate_clustering #check if file was imported 

#########DENSITY ESTIMATION METHODS 
#test imports of all concrete CDE methods 

def test_import_cde(): 
    assert cfl.density_estimation_methods.condExpVB.CondExpVB  #check if class exists 

def test_bad_import_cde(): 
    with pytest.raises(Exception): 
        density_estimation_methods.condExpVB.CondExpVB

def test_import_chalupka(): 
    assert cfl.density_estimation_methods.condExpKC.CondExpKC  #check if class exists 

def test_import_CNN(): 
    assert cfl.density_estimation_methods.condExpCNN.CondExpCNN  #check if class exists 

############CORE CFL OBJECTS 
#test import of concrete cfl method (two-step cfl)
def test_import_two_step(): 
    assert cfl.core_cfl_objects.two_step_cfl.Two_Step_CFL_Core #check if class exists

####### CFL WRAPPER OBJECT (CFL.py)
def test_import_CFL(): 
    assert cfl.cfl.CFL #check if class exists 

def test_bad_import_cfl(): 
    with pytest.raises(Exception): 
        cfl.CFL 

########VISUALIZATION.py
def test_import_visualize(): 
    assert cfl.visualization #check that file imports successfully

