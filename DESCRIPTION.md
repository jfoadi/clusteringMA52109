The module cluster_maker is a small educational Python package for simulating clustered datasets and running clustering analyses with a simple, user-friendly interface.
It is a Python module aimed at generating synthetic datasets and applying common clustering algorithms.

The synthetic dataset is generated using the module dataframe_builder.py, which contains functions define_dataframe_structure and simulate_data, which define a pandas DataFrame choosing the centres of each cluster, and simulate clustered data around these given cluster centres respectively. This is then exported to a CSV file and saved in the 'data' folder, using functions defined in data_exporter.py.

Other functions include the ability to preprocess the generated data in the preprocessing.py module, which standardises features to have zero mean and unit variance. We can also fit K-means models and label data, as described in the algorithms.py module.

Finally, we are able to plot the clustered data using the functions in plotting_clustered.py, and generate performance metrics, such as the inertia and the silhouette score (as defined in evaluation.py), as well as diagnostic plots, such as elbow plots and 2D cluster plots.

This module is intended for data scientists, programmers, or students needing repeatable, controlled datasets for testing clustering algorithms. It solves the problem of quickly demonstrating clustering workflows without relying on external, static datasets.