"""
RoadGuard AI - Custom Dual-Output CNN Architecture
Shared feature extractor with two specialized heads:
1. Traffic Sign Classification (43 classes)
2. Occlusion Detection (2 classes: 0=Clear, 1=Occluded)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_roadguard_model(
    input_shape: tuple = (64, 64, 3),
    num_sign_classes: int = 43,
    num_occlusion_classes: int = 2,
    dropout_rate: float = 0.20,
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """
    Builds and compiles the dual-output RoadGuard CNN model.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    
    # ------------------ Shared Feature Extractor ------------------ #
    
    # Conv Block 1: 32 filters, 3x3 kernel
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu", name="relu1_1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu", name="relu1_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    
    # Conv Block 2: 64 filters, 3x3 kernel
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu", name="relu2_1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu", name="relu2_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    
    # Conv Block 3: 128 filters, 3x3 kernel
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="relu3_1")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.Activation("relu", name="relu3_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    
    # Global spatial pooling + dense feature embedding
    gap = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    flat = layers.Flatten(name="flatten_features")(x)
    combined = layers.Concatenate(name="concat_features")([gap, flat])
    
    shared_dense = layers.Dense(256, activation="relu", name="shared_representation")(combined)
    shared_dense = layers.BatchNormalization(name="bn_shared")(shared_dense)
    shared_dense = layers.Dropout(dropout_rate, name="drop_shared")(shared_dense)
    
    # ------------------ Head 1: Traffic Sign Classification ------------------ #
    sign_dense = layers.Dense(128, activation="relu", name="sign_dense")(shared_dense)
    sign_dense = layers.BatchNormalization(name="bn_sign")(sign_dense)
    sign_dense = layers.Dropout(dropout_rate, name="drop_sign")(sign_dense)
    sign_output = layers.Dense(
        num_sign_classes,
        activation="softmax",
        name="sign_output"
    )(sign_dense)
    
    # ------------------ Head 2: Occlusion Detection ------------------ #
    occ_dense = layers.Dense(64, activation="relu", name="occ_dense")(shared_dense)
    occ_dense = layers.BatchNormalization(name="bn_occ")(occ_dense)
    occ_dense = layers.Dropout(dropout_rate, name="drop_occ")(occ_dense)
    occ_output = layers.Dense(
        num_occlusion_classes,
        activation="softmax",
        name="occlusion_output"
    )(occ_dense)
    
    # Assembly & Compile
    model = models.Model(
        inputs=inputs,
        outputs=[sign_output, occ_output],
        name="RoadGuard_DualOutput_CNN"
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss={
            "sign_output": "sparse_categorical_crossentropy",
            "occlusion_output": "sparse_categorical_crossentropy"
        },
        loss_weights={
            "sign_output": 1.0,
            "occlusion_output": 1.0
        },
        metrics={
            "sign_output": ["accuracy"],
            "occlusion_output": ["accuracy"]
        }
    )
    return model
