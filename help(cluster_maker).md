Help on package cluster_maker:

NAME
    cluster_maker - cluster_maker

DESCRIPTION
    An educational Python package for generating synthetic clustered data,
    running clustering algorithms, evaluating results, and producing
    user-friendly plots. Designed for practicals and exams where students
    work with an incomplete or faulty version of the package and must fix it.

    Allowed libraries:
    - Python standard library
    - numpy
    - pandas
    - matplotlib
    - scipy
    - scikit-learn

PACKAGE CONTENTS
    algorithms
    data_analyser
    data_exporter
    dataframe_builder
    evaluation
    interface
    plotting_clustered
    preprocessing
    stability

FUNCTIONS
    assign_clusters(X: 'np.ndarray', centroids: 'np.ndarray') -> 'np.ndarray'
        Assign each sample to the nearest centroid (Euclidean distance).

    calculate_correlation(data: 'pd.DataFrame') -> 'pd.DataFrame'
        Compute the correlation matrix for numeric columns in the DataFrame.

        Parameters
        ----------
        data : pandas.DataFrame

        Returns
        -------
        corr : pandas.DataFrame
            Correlation matrix.

    calculate_descriptive_statistics(data: 'pd.DataFrame') -> 'pd.DataFrame'
        Compute descriptive statistics for each numeric column in the DataFrame.

        Parameters
        ----------
        data : pandas.DataFrame

        Returns
        -------
        corr : pandas.DataFrame
            Correlation matrix.

    calculate_descriptive_statistics(data: 'pd.DataFrame') -> 'pd.DataFrame'
        Compute descriptive statistics for each numeric column in the DataFrame.

        Parameters
        ----------
        data : pandas.DataFrame

        Returns
        -------
        stats : pandas.DataFrame
            Result of `data.describe()` including count, mean, std, etc.

    cluster_stability_score(X: 'np.ndarray', k: 'int', n_runs: 'int' = 20, noise_scale: 'float' = 0.05, random_state: 'int | None' = None) -> 'float'
        Compute clustering stability while correcting for label switching.

        Cluster labels are aligned across runs by sorting cluster centroids,
        ensuring that label 0 always refers to the same cluster structure,
        label 1 to the next cluster, etc.

        This produces a meaningful stability score in [0,1].

    compute_inertia(X: 'np.ndarray', labels: 'np.ndarray', centroids: 'np.ndarray') -> 'float'
        Compute the within-cluster sum of squared distances (inertia).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        labels : ndarray of shape (n_samples,)
        centroids : ndarray of shape (k, n_features)

        Returns
        -------
        inertia : float

    define_dataframe_structure(column_specs: 'List[Dict[str, Any]]') -> 'pd.DataFrame'
        Define a seed DataFrame describing cluster centres.

        Parameters
        ----------
        column_specs : list of dict
            Each dict must contain:
            - 'name': str    – the column name
            - 'reps': list   – list of centre values, one per cluster

            Example:
            [
                {"name": "x", "reps": [0.0, 5.0, -5.0]},
                {"name": "y", "reps": [0.0, 5.0, -5.0]},
            ]

        Returns
        -------
        seed_df : pandas.DataFrame
            DataFrame with one row per cluster and one column per feature.

    elbow_curve(X: 'np.ndarray', k_values: 'List[int]', random_state: 'Optional[int]' = None, use_sklearn: 'bool' = True) -> 'Dict[int, float]'
        Compute inertia values for multiple K values (elbow method).

        Parameters
        ----------
        X : ndarray
        k_values : list of int
        random_state : int or None
        use_sklearn : bool, default True
            If True, use scikit-learn KMeans; otherwise use manual kmeans.

        Returns
        -------
        inertia_dict : dict
            Mapping from k to inertia.

    export_formatted(data: 'pd.DataFrame', file: 'Union[str, TextIO]', include_index: 'bool' = False) -> 'None'
        Export a DataFrame as a formatted text table.

        Parameters
        ----------
        data : pandas.DataFrame
        file : str or file-like
            Filename or open file handle.
        include_index : bool, default False

    export_summary(summary_df: 'pd.DataFrame', csv_path: 'str', txt_path: 'str') -> 'None'
        Export a summary DataFrame (created by summarise_numeric_columns)
        to both a CSV file and a neatly formatted text file.

        Parameters
        ----------
        summary_df : pandas.DataFrame
            Summary statistics DataFrame from data_analyser.summarise_numeric_columns.
        csv_path : str
            Output CSV file path.
        txt_path : str
            Output plain-text file path.

        Notes
        -----
        - Includes friendly, clear error messages.

    export_to_csv(data: 'pd.DataFrame', filename: 'str', delimiter: 'str' = ',', include_index: 'bool' = False) -> 'None'
        Export a DataFrame to CSV.

        Parameters
        ----------
        data : pandas.DataFrame
        filename : str
            Output filename.
        delimiter : str, default ","
        include_index : bool, default False

    init_centroids(X: 'np.ndarray', k: 'int', random_state: 'Optional[int]' = None) -> 'np.ndarray'
        Initialise centroids by randomly sampling points from X without replacement.

    kmeans(X: 'np.ndarray', k: 'int', max_iter: 'int' = 300, tol: 'float' = 0.0001, random_state: 'Optional[int]' = None) -> 'Tuple[np.ndarray, np.ndarray]'
        Simple manual K-means implementation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        k : int
            Number of clusters.
        max_iter : int, default 300
            Maximum number of iterations.
        tol : float, default 1e-4
            Convergence tolerance on centroid movement.
        random_state : int or None

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        centroids : ndarray of shape (k, n_features)

    plot_clusters_2d(X: 'np.ndarray', labels: 'np.ndarray', centroids: 'Optional[np.ndarray]' = None, title: 'Optional[str]' = None) -> 'Tuple[plt.Figure, plt.Axes]'
        Plot clustered data in 2D using the first two features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        labels : ndarray of shape (n_samples,)
        centroids : ndarray of shape (k, n_features) or None
        title : str or None

        Returns
        -------
        fig, ax : matplotlib Figure and Axes

    plot_elbow(k_values: 'List[int]', inertias: 'List[float]', title: 'str' = 'Elbow Curve') -> 'Tuple[plt.Figure, plt.Axes]'
        Plot inertia vs k (elbow method).

        
        Parameters
        ----------
        k_values : list of int
        inertias : list of float
        title : str, default "Elbow Curve"

        Returns
        -------
        fig, ax : matplotlib Figure and Axes

    run_clustering(input_path: 'str', feature_cols: 'List[str]', algorithm: 'str' = 'kmeans', k: 'int' = 3, standardise: 'bool' = True, output_path: 'Optional[str]' = None, random_state: 
'Optional[int]' = None, compute_elbow: 'bool' = False, elbow_k_values: 'Optional[List[int]]' = None, compute_stability: 'bool' = False) -> 'Dict[str, Any]'
        High-level function to run the full clustering workflow.

        Steps:
        1. Load data from CSV
        2. Select feature columns
        3. Optionally standardise features
        4. Run the chosen clustering algorithm
        5. Compute evaluation metrics
        6. Generate plots
        7. Optionally write labelled data to CSV

        Parameters
        ----------
        input_path : str
            Path to the input CSV file.
        feature_cols : list of str
            Names of feature columns to use.
        algorithm : {"kmeans", "sklearn_kmeans"}, default "kmeans"
        k : int, default 3
            Number of clusters.
        standardise : bool, default True
        output_path : str or None, default None
            If provided, the input data with cluster labels will be saved to this CSV.
        random_state : int or None, default None
        compute_elbow : bool, default False
            If True, compute inertia for multiple k values.
        elbow_k_values : list of int or None, default None
            k-values for elbow curve. If None and compute_elbow is True, defaults
            to range 1..(k+5).

        Returns
        -------
        result : dict
            Dictionary containing:
            - "data": DataFrame with added "cluster" column
            - "labels": ndarray of cluster labels
            - "centroids": ndarray of cluster centroids
            - "metrics": dict with "inertia" and optional "silhouette"
            - "fig_cluster": Figure for the cluster plot
            - "fig_elbow": Figure for the elbow plot or None
            - "elbow_inertias": dict mapping k -> inertia (if computed)

    select_features(data: 'pd.DataFrame', feature_cols: 'List[str]') -> 'pd.DataFrame'
        Select a subset of columns to use as features, ensuring they are numeric.

        Parameters
        ----------
        data : pandas.DataFrame
        feature_cols : list of str
            Column names to select.

        Returns
        -------
        X_df : pandas.DataFrame
            DataFrame containing only the selected feature columns.

        Raises
        ------
        KeyError
            If any requested column is missing.
        TypeError
            If any selected column is non-numeric.

    silhouette_score_sklearn(X: 'np.ndarray', labels: 'np.ndarray') -> 'float'
        Compute the silhouette score using scikit-learn.

        Returns
        -------
        score : float

    simulate_data(seed_df: 'pd.DataFrame', n_points: 'int' = 100, cluster_std: 'float' = 1.0, random_state: 'int | None' = None) -> 'pd.DataFrame'
        Simulate clustered data around the given cluster centres.

        Parameters
        ----------
        seed_df : pandas.DataFrame
            Rows represent cluster centres, columns represent features.
        n_points : int, default 100
            Total number of data points to simulate.
        cluster_std : float, default 1.0
            Standard deviation of Gaussian noise added around centres.
        random_state : int or None, default None
            Random seed for reproducibility.

        Returns
        -------
        data : pandas.DataFrame
            Simulated data with all original feature columns plus a 'true_cluster'
            column indicating the generating cluster.

    sklearn_kmeans(X: 'np.ndarray', k: 'int', random_state: 'Optional[int]' = None) -> 'Tuple[np.ndarray, np.ndarray]'
        Thin wrapper around scikit-learn's KMeans.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        centroids : ndarray of shape (k, n_features)

    standardise_features(X: 'np.ndarray') -> 'np.ndarray'
        Standardise features to zero mean and unit variance.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_scaled : ndarray of shape (n_samples, n_features)

    summarise_numeric_columns(df: 'pd.DataFrame') -> 'pd.DataFrame'
        Create a summary table for all numeric columns in the DataFrame.

        For each numeric column, compute:
        - mean
        - standard deviation
        - minimum
        - maximum
        - number of missing values

        Non-numeric columns are ignored but clearly reported in a warning.

        Returns
        -------
        summary_df : pandas.DataFrame
            A DataFrame where each row corresponds to a numeric column and
            each column contains one of the summary statistics.

        Notes
        -----
        - Robust to non-numeric columns.
        - Produces human-readable statistics.

    update_centroids(X: 'np.ndarray', labels: 'np.ndarray', k: 'int', random_state: 'Optional[int]' = None) -> 'np.ndarray'
        Update centroids by taking the mean of points in each cluster.
        If a cluster becomes empty, re-initialise its centroid randomly from X.

DATA
    __all__ = ['define_dataframe_structure', 'simulate_data', 'calculate_d...

FILE
    /Users/Yasvanti.Akilakulasingam/Documents/bath/MSc/Programming for data science/clusteringMA52109/cluster_maker/__init__.py