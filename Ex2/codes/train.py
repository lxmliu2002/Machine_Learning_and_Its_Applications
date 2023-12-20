import data_process
import lenet5
from layers.loss import SoftMax
import tqdm
import numpy as np
from layers.optimizer import Adam

batch_size = 256
epochs = 10
lr = 1e-3

data = data_process.set_data()
Lenet5 = lenet5.LeNet5()
optimizer = Adam(Lenet5.get_params(), lr)

loss_list = []
acc_list = []

def train(loss_list, acc_list):
    best_acc = 0
    best_weight = None

    for epoch in range(epochs):
        # Training
        total_batches = int(data['train_image'].shape[0] / batch_size)
        pbar = tqdm.tqdm(range(total_batches), ncols = 150)
        for batch_idx in pbar:
            batch_X, batch_y = data_process.get_batch(data["train_image"], data["train_label"], batch_size)
            pred = Lenet5.forward(batch_X)
            loss = SoftMax.loss(pred, batch_y)
            grad = SoftMax.gradient(pred, batch_y)
            acc = SoftMax.accuracy(pred, batch_y)

            loss_list.append(loss)
            acc_list.append(acc)

            Lenet5.backward(grad)
            optimizer.step()

            pbar.set_description(f"Epoch: {epoch+1}/{epochs}")
            pbar.set_postfix(loss=loss, acc=acc)

        # Validation
        val_train, val_label = data["val_train"], data["val_label"]
        val_train_pred = Lenet5.forward(val_train)
        val_train_pred = np.argmax(val_train_pred, axis = 1)
        val_acc = np.mean(val_train_pred == val_label.reshape(1, val_label.shape[0]))

        if val_acc > best_acc:
            best_acc = val_acc
            best_weight = Lenet5.get_params()

        pbar.set_postfix(val_acc = val_acc)

    return best_weight

def test(model, best_weight, data):
    test_image, test_label = data["test_image"], data["test_label"]

    # Set the model parameters to the best weight
    model.set_params(best_weight)

    # Set the model to evaluation mode
    model.eval()

    # Forward pass
    y_pred = model.forward(test_image)
    y_pred = np.argmax(y_pred, axis=1)

    # Calculate accuracy
    acc = np.mean(y_pred == test_label.reshape(1, test_label.shape[0]))

    # Print more detailed test results
    print(f"Test Accuracy: {acc}")