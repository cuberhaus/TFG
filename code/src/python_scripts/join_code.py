import os

import nbformat

"""
This script is used to consolidate all the code from the notebooks and python files into a single file.
"""


def extract_code_from_notebook(notebook_path):
    """
    Extract the code from a Jupyter Notebook.
    :param notebook_path:
    :return:
    """
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    code_cells = [cell.source for cell in nb.cells if cell.cell_type == 'code']
    return '\n\n'.join(code_cells)


def consolidate_code(notebook_paths, py_files, output_file, base_path):
    """
    Consolidate all the code from the notebooks and python files into a single file.
    :param notebook_paths:
    :param py_files:
    :param output_file:
    :param base_path:
    :return:
    """
    with open(output_file, 'w') as outfile:
        # Process Python files
        for file in py_files:
            full_path = os.path.join(base_path, file)
            print(f"Processing {full_path}")
            with open(full_path, 'r') as infile:
                outfile.write(f"# Code from {full_path}\n")
                outfile.write(infile.read())
                outfile.write("\n\n")

        # Process Jupyter Notebooks
        for notebook_path in notebook_paths:
            full_path = os.path.join(base_path, notebook_path)
            print(f"Processing {full_path}")
            notebook_code = extract_code_from_notebook(full_path)
            outfile.write(f"# Code from {full_path}\n")
            outfile.write(notebook_code)
            outfile.write("\n\n")
    print(f"Consolidated code written to {output_file}")


# Get the absolute path of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the paths
select = 1
if select == 0:
    notebook_path = [
        '../j_notebooks/main.ipynb'
    ]
    py_files = [
        '../clases/custom_dataset.py',
        '../clases/data_utils.py',
        '../clases/model_utils.py',
        '../test_model.py',
        '../train_and_save_model.py',
    ]
elif select == 1:
    notebook_path = []
    py_files = [
        '../clases/custom_dataset.py',
        '../clases/data_utils.py',
        '../clases/model_utils.py',
        '../shell_scripts/train_models.sh',
        '../test_model.py',
        '../train_and_save_model.py',
        '../train_models.py',
    ]
elif select == 2:
    notebook_path = []
    py_files = [
        '../clases/custom_dataset.py',
        '../clases/data_utils.py',
        '../clases/model_utils.py',
        '../python_scripts/copy_files_to_cyclegan_structure.py'
        '../python_scripts/create_masks.py',
        '../test_model.py',
        '../train_and_save_model.py',
    ]
else:
    notebook_path = []
    py_files = []

output_file = os.path.join(SCRIPT_DIR, '../../tmp/consolidated_code.py')

# Run the consolidation
consolidate_code(notebook_path, py_files, output_file, SCRIPT_DIR)
