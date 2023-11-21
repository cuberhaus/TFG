def train_dummy_model(num_epochs):
    for epoch in range(num_epochs):
        print(f"Training... Epoch {epoch + 1}/{num_epochs}")
        # Simulate some training process here
    # Return the last epoch number
    return epoch + 1

# Example usage
num_epochs = 5
last_epoch = train_dummy_model(num_epochs)
print(f"Training completed. Last epoch processed was Epoch {last_epoch}.")
