import matplotlib.pyplot as plt
import os
import glob


def plot_losses(file_path, output_folder='tmp/losses'):
    # Read loss values from file
    with open(file_path, 'r') as file:
        losses = [float(line.strip()) for line in file.readlines()]

    # Create a plot
    plt.figure()
    plt.plot(losses)
    plt.title(f"Losses for {os.path.basename(file_path)}")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

    # Prepare output file path
    plot_file_name = os.path.basename(file_path).replace('.txt', '.png')
    output_file_path = os.path.join(output_folder, plot_file_name)

    print(f"Saving plot to {plot_file_name}")
    plt.savefig(plot_file_name)
    plt.close()


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    # Directory containing the loss files
    directory_losses = os.path.join(SCRIPT_DIR, '..', 'tmp/losses')
    directory_plots = os.path.join(SCRIPT_DIR, '..', 'tmp/plots_losses')

    # Iterate over all loss files in the directory
    for file_path in glob.glob(os.path.join(directory_losses, "*_losses.txt")):
        plot_losses(file_path, directory_plots)


if __name__ == "__main__":
    main()
