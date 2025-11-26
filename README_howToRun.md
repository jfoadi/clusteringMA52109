# cluster_maker – How to Run the Package and Demos

This project contains the `cluster_maker` Python package used throughout the
MA52109 mock exam. Several demo scripts and tests depend on Python being able to
locate the `cluster_maker` package correctly.

To avoid `ModuleNotFoundError: No module named 'cluster_maker'`, it is important
to run all scripts **from the project root directory**, not from inside the
`demo/` or `cluster_maker/` folders.

---

## 🚀 Running Scripts Correctly

### ✔️ 1. Navigate to the project root

Before running any demo or test, make sure your terminal is inside:

clusteringMA52109/


You should see directories such as:

cluster_maker/
demo/
tests/
demo_output/
pyproject.toml


If you run scripts from inside the `demo/` or `cluster_maker/` folders, Python
will *not* be able to import the package.

---

## ✔️ 2. Running demo scripts

You can run any script inside `demo/` using either of these two recommended
methods:

### **Method A — Run as a module (recommended)**
python -m demo.analyse_from_csv path/to/file.csv


### **Method B — Run the script file directly**
python demo/analyse_from_csv.py path/to/file.csv


Both methods work **as long as you run the command from the project root**.

Running the script directly *from inside the demo folder* will cause Python to
lose access to the `cluster_maker` package.

---

## ✔️ 3. Running tests

All tests should be run from the project root using:

python -m unittest discover tests


or to run a specific file:

python -m unittest tests.test_dataframe_builder


This ensures that the package is importable during the test run.

---

## ❗ Why running from the root is required

Python locates modules based on the directory structure.  
When executed correctly, this structure is visible:

clusteringMA52109/
cluster_maker/
demo/
tests/


If you run a script from inside a subfolder, Python changes `sys.path` and can
no longer find the package, resulting in import errors.

Using the `-m` flag always treats the directory structure as a proper Python
package, ensuring imports work correctly.

---

## 👍 Summary

- Always run commands from the **project root**.
- Use `python -m demo.scriptname ...` for module-safe execution.
- Direct script execution works only when launched from the root.
- All tests must be run from the root for imports to work.
- Following this avoids `ModuleNotFoundError` and ensures consistent behaviour.

---

If you encounter any import errors, verify that you are **in the project root**
and use the `-m` option to run scripts as modules.
