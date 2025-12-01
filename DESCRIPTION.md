** Write a DESCRIPTION.md file that describes what
cluster_maker does and explains the functionality of its main parts. **

We are going to create a guide that will explain every script hosted
on *cluster_maker*.

# algorithms.py

Is the script were the k-means algorithm is developed.
We can see how random points are selected as centroids,
and the algorithm starts to assign each point a centroid
and recalculating centroids, until it converges.

# data_analyser.py

It provides several tools that perform data analysis on DataFrames.

# data_exporter.py

It provides several functions that export data coming from DataFrames
to CSVs files.

# dataframe_builder.py

It creates syntetic data from given information about clusters centers, in order to
perform the kmeans algorithm with it.

# evaluation.py

It analyse the algorithm and gives a value for the inertia and elbow curve
which will help us to choose a correct value of k, in other words,
the number of clusters.

# interface.py

This is the main script of the package, it is the one that unifies all the other functions.
It executes all others functions, it creates the data, runs the kmeans algorithm and 
its evaluation, and finally plot and export the results.

# plotting_clustered.py

It has two main scripts which plot the kmeans result and the elbow evaluation.

# preprocessing.py

It selects and normalise columns from a given data for them being used by dataframe_builder.