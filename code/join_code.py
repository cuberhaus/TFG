import nbformat


def extract_code_from_notebook(notebook_path):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    code_cells = [cell.source for cell in nb.cells if cell.cell_type == 'code']
    return '\n\n'.join(code_cells)


def consolidate_code(notebook_paths, py_files, output_file):
    with open(output_file, 'w') as outfile:
        # Process Python files
        for file in py_files:
            with open(file, 'r') as infile:
                outfile.write(f"# Code from {file}\n")
                outfile.write(infile.read())
                outfile.write("\n\n")

        # Process Jupyter Notebooks
        for notebook_path in notebook_paths:
            notebook_code = extract_code_from_notebook(notebook_path)
            outfile.write(f"# Code from {notebook_path}\n")
            outfile.write(notebook_code)
            outfile.write("\n\n")


# Define the paths
select = 1
if select == 0:
    notebook_path = ['main.ipynb']
    py_files = ['custom_dataset.py', 'model_utils.py', 'data_utils.py']
elif select == 1:
    notebook_path = []
    py_files = ['train_and_save_model.py', 'custom_dataset.py', 'model_utils.py', 'data_utils.py']
elif select == 3:
    notebook_path = ['main_old.ipynb', 'cycleGAN.ipynb']
    py_files = ['custom_dataset.py', 'model_utils.py', 'data_utils.py', 'create_masks.py',
                'copy_files_to_cyclegan_structure.py']


output_file = 'tmp/consolidated_code.py'

# Run the consolidation
consolidate_code(notebook_path, py_files, output_file)
