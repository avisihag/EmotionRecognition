from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau


early_stopping = {

    "1": EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.0005,
        patience=11,
        verbose=1,
        restore_best_weights=True,
    ),

    "2": EarlyStopping(
        monitor='accuracy',
        min_delta=0.0005,
        patience=15,
        verbose=1,
        restore_best_weights=True,
    ),

    "3": EarlyStopping(
        monitor='accuracy',
        min_delta=0.0006,
        patience=12,
        verbose=1,
        restore_best_weights=True,
    ),

    "4": EarlyStopping(
        monitor='accuracy',
        min_delta=0.0005,
        patience=9,
        verbose=1,
        restore_best_weights=True,
    ),

    "5": EarlyStopping(
        monitor='accuracy',
        min_delta=0.0005,
        patience=7,
        verbose=1,
        restore_best_weights=True,
    ),

}

lr_schedulers = {

    "1": ReduceLROnPlateau(
        monitor='val_accuracy',
        min_delta=0.0001,
        factor=0.1,
        patience=4,
        min_lr=1e-7,
        verbose=1,
    ),

    "2": ReduceLROnPlateau(
        monitor='val_accuracy',
        min_delta=0.0001,
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1,
    ),

    "3": ReduceLROnPlateau(
        monitor='val_accuracy',
        min_delta=0.0001,
        factor=0.2,
        patience=4,
        min_lr=1e-7,
        verbose=1,
    ),

    "4": ReduceLROnPlateau(
        monitor='val_accuracy',
        min_delta=0.0001,
        factor=0.35,
        patience=4,
        min_lr=1e-7,
        verbose=1,
    ),

}
