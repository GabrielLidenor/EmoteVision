import pytest
import tensorflow
import numpy as np
from src.models import FacialRecognitionModel

@pytest.fixture
def model_config():
    """Standard model parameters"""
    return {
           "num_classes": 10,
           "input_shape": (150, 150, 3),
           "batch_size": 4,
           }

@pytest.fixture
def model(model_config):
    """Model fixture with compilation"""
    model = FacialRecognitionModel(
            num_classes = model_config["num_classes"],
            input_shape = model_config["input_shape"]
            )

    model.compile(
            optimizer = "adam",
            loss = "categorical_crossentropy",
            metrics = ["accuracy"],
            )

    return model

def test_shape_geometry(model, model_config):
    """Validates tensor transformations match expected (batch_size, num_classes) dimensions."""
    batch_size = model_config["batch_size"]
    h, w, c = model_config["input_shape"]

    dummy_input = tensorflow.random.uniform((batch_size, h, w, c))
    output = model(dummy_input)

    expected_shape = (batch_size, model_config["num_classes"])

    assert (
            output.shape == expected_shape
            ), f"Expected shape {expected_shape}, got {output.shape}"

def test_probability_contracts(model, model_config):
    """Asserts output tensor contain no NaN/Inf values and sum to 1.0 across predictions."""
    batch_size = model_config["batch_size"]
    h, w, c = model_config["input_shape"]

    dummy_input = tensorflow.random.uniform((batch_size, h, w, c))
    outputs = model(dummy_input).numpy()

    assert not np.isnan(outputs).any(), "Output tensor contains NaN values!"
    assert not np.isinf(outputs).any(), "Output tensor contains Inf values!"

    assert (outputs >= 0.0).all(), "Softmax output contains negative values!"

    row_sums = np.sum(outputs, axis = 1)
    np.testing.assert_allclose(
            row_sums,
            np.ones(batch_size),
            rtol = 1e-5,
            err_msg = "Outputs do not sum to 1.0 per sample"
            )

def test_gradient_flow_single_sample_overfit(model_config):
    """Executes a single-sample overfitting routine to verify backpropagation and weight updates."""
    model = FacialRecognitionModel(
            num_classes = model_config["num_classes"],
            input_shape = model_config["input_shape"],
            )

    model.compile(
            optimizer = tensorflow.keras.optimizers.Adam(learning_rate = 0.01),
            loss = "categorical_crossentropy",
            )

    h, w, c = model_config["input_shape"]
    x_single = tensorflow.random.uniform((1, h, w, c))
    y_single = tensorflow.one_hot([3], depth = model_config["num_classes"])

    initial_loss = model.train_on_batch(x_single, y_single)

    # several steps training on the same sample
    for _ in range(15):
        current_loss = model.train_on_batch(x_single, y_single)

    assert (
            current_loss < initial_loss
           ), f"Model failed to overfit single sample. Initial loss: {initial_loss}, final loss: {current_loss}"

def test_frozen_layers_gradients(model_config):
    """Ensures our frozen base layers receive no gradient updates."""
    model = FacialRecognitionModel(
            num_classes = model_config["num_classes"],
            input_shape = model_config["input_shape"],
            )

    h, w, c = model_config["input_shape"]
    x = tensorflow.random.uniform((1, h, w, c))
    y = tensorflow.one_hot([0], depth = model_config["num_classes"])

    loss_fn = tensorflow.keras.losses.CategoricalCrossentropy()

    with tensorflow.GradientTape() as tape:
        preds = model(x, training = True)
        loss = loss_fn(y, preds)

    # Calculate gradients accross all model variables
    trainable_vars = model.trainable_variables
    base_vars = model.base_model.variables

    # Verify base model vairables are not in trainable_variables
    base_var_ids = {id(v) for v in base_vars}
    trainable_vars_ids = {id(v) for v in trainable_vars}

    assert base_var_ids.isdisjoint(
            trainable_vars_ids
            ), "Base model variables found in trainable_variables!"


def test_integration_train_on_batch(model, model_config):
    """Performs trial train_on_batch steps to check compilation and loss/optimizer compatibility."""
    batch_size = model_config["batch_size"]
    h, w, c = model_config["input_shape"]

    x_batch = tensorflow.random.uniform((batch_size, h, w, c))
    y_batch = tensorflow.one_hot([0, 1, 2, 3], depth = model_config["num_classes"])

    # Perform a single training step
    results = model.train_on_batch(x_batch, y_batch)

    assert results is not None, "train_on_batch returned None"

    if isinstance(results, list):
        loss_val = results[0]
    else:
        loss_val = results

    assert not np.isnan(loss_val), "Loss evaluated to NaN during training step"
