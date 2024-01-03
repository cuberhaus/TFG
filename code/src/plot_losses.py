import matplotlib.pyplot as plt
import os
import glob

"""
Plots all losses from the losses directory and outputs them to a new plots directory
"""


def plot_losses(file_path, batch_plots, epoch_plots):
    with open(file_path, 'r') as file:
        losses = [float(line.strip()) for line in file.readlines()]

    plt.figure()
    plt.plot(losses)
    plt.title(f"Losses for {os.path.basename(file_path)}")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

    if "batch_losses" in file_path:
        output_folder = batch_plots
    elif "epoch_losses" in file_path:
        output_folder = epoch_plots
    else:
        print(f"Unknown file type for {file_path}, skipping...")
        return

    plot_file_name = os.path.basename(file_path).replace('.txt', '.png')
    output_file_path = os.path.join(output_folder, plot_file_name)

    print(f"Saving plot to {output_file_path}")
    plt.savefig(output_file_path)
    plt.close()


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    directory_losses = os.path.join(SCRIPT_DIR, '..', 'tmp/losses')
    directory_plots = os.path.join(SCRIPT_DIR, '..', 'tmp/plots_losses/')
    print(directory_plots)
    batch_plots = os.path.join(directory_plots, 'batch_plots')
    epoch_plots = os.path.join(directory_plots, 'epoch_plots')
    print(batch_plots)

    if not os.path.exists(directory_plots):
        os.makedirs(directory_plots)
    if not os.path.exists(batch_plots):
        os.makedirs(batch_plots)
    if not os.path.exists(epoch_plots):
        os.makedirs(epoch_plots)

    for file_path in glob.glob(os.path.join(directory_losses, "*_losses.txt")):
        plot_losses(file_path, batch_plots, epoch_plots)


if __name__ == "__main__":
    main()
