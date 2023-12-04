from collections import defaultdict

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_ranking as tfr


def get_dataset(fold, split, batch_size=32):
    """Return an MSLR_WEB10K split as a tf.data.Dataset.

    The returned dataset contains:
    Batches of 32 queries each.
      - For each query:
          - A padded (to max nr of docs/query in the batch) list of feature vectors.
          - A list of labels for each document. -1 for padding documents.
    """
    ds = tfds.load(f"mslr_web/10k_fold{fold}:1.2.0", split=split)

    # Add a mask tensor.
    ds = ds.map(
        lambda feature_map: {
            "_mask": tf.ones_like(feature_map["label"], dtype=tf.bool),
            "float_features": feature_map["float_features"],
            "label": feature_map["label"],
        }
    )

    # Shuffle data and create padded batches for queries with different lengths.
    # Padded items will have False on `_mask`.
    ds = ds.shuffle(buffer_size=ds.cardinality()).padded_batch(batch_size=batch_size)

    # Create (features, labels) tuples from data and set -1 label for masked items.
    ds = ds.map(
        lambda feature_map: (
            feature_map,
            tf.where(feature_map["_mask"], feature_map.pop("label"), -1.0),
        )
    )

    return ds


def get_model(shape: tuple) -> tf.keras.Model:
    inputs = {
        "float_features": tf.keras.Input(
            shape=(None, 136), dtype=tf.float32, name="float_features"
        ),
    }
    x = tf.keras.layers.BatchNormalization()(inputs["float_features"])

    # for layer in layers:
    #     x = layer(x)

    for layer_width in shape:
        x = tf.keras.layers.Dense(units=layer_width)(x)
        x = tf.keras.layers.Activation("relu")(x)

    scores = tf.squeeze(tf.keras.layers.Dense(units=1)(x), axis=-1)

    # Create, compile and train the model.
    model = tf.keras.Model(inputs=inputs, outputs=scores)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tfr.keras.losses.ListMLELoss(),
        metrics=[
            tfr.keras.metrics.get("ndcg", topn=1, name="NDCG@1"),
            tfr.keras.metrics.get("ndcg", topn=5, name="NDCG@5"),
            tfr.keras.metrics.get("ndcg", topn=10, name="NDCG@10"),
        ],
    )

    return model


# >>> model = get_model([
# ...     tf.keras.layers.Reshape((-1,136,1)),
# ...     tf.keras.layers.Conv1D(32, 5, 5, activation="relu"),
# ...     tf.keras.layers.Reshape((-1,864)),
# ...     tf.keras.layers.Dense(64,activation="relu"),
# ... ])

# for shape in [[32, 16], [64, 32], [128, 64, 32]]:
#     print(f"\n\n\nTraining model with shape {shape}.\n")
#     model = get_model(shape)
#     ds = get_dataset(1, "train")
#     ds_vali = get_dataset(1, "vali")
#     model.fit(
#         ds,
#         validation_data=ds_vali,
#         epochs=50,
#     )

SHAPES = [
    (8,),
    (12,),
    (16,),
    (24,),
    (32,),
    (48,),
    (64,),
    (96,),
    (128,),
    (16, 8),
    (32, 16),
    (64, 32),
    (96, 48),
    (128, 64),
    (32, 16, 8),
    (64, 32, 16),
    (128, 64, 32),
    (256, 128, 64),
]

histories = defaultdict(list)
scores = defaultdict(list)

for shape in SHAPES:
    print(f"\n\n\nModel with shape {shape}.\n")
    for fold in range(1, 6):
        model = get_model(shape)
        ds = get_dataset(fold, "train")
        ds_vali = get_dataset(fold, "vali")
        history = model.fit(ds, validation_data=ds_vali, epochs=50, verbose=2)
        histories[str(shape)].append(history.history)
        scores[str(shape)].append(model.evaluate(ds_vali))


import json

print("\n\n\n")
print(json.dumps(dict(histories=dict(histories), scores=dict(scores))))
