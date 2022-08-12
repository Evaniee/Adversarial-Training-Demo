import neural_structured_learning as nsl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from datetime import datetime

#Constants
DATASET = 'mnist'               # Dataset to load from tensorflow_datasets
INPUT_SHAPE = [28,28,1]         # Shape of Input Images
NUM_CLASSES = 10                # Number of possible Labels
BATCH_SIZE = 32                 # Training Batch Size
IMAGE_INPUT_NAME = 'image'
LABEL_INPUT_NAME = 'label'
EPOCHS = 100                    # How many epochs to train for
ADV_MULTIPLIER = 0.2
ADV_STEP_SIZE = 0.2
ADV_GRADIENT_NORM = 'infinity'


def get_datasets():
    print('Getting datasets')
    train_ds = tfds.load(DATASET, split='train[:80%]')
    train_ds = train_ds.map(normalize).shuffle(10000).batch(BATCH_SIZE).map(convert_to_tuples)
    val_ds = tfds.load(DATASET, split='train[80%:]').map(normalize).shuffle(10000).batch(BATCH_SIZE).map(convert_to_tuples)
    test_ds = tfds.load(DATASET, split='test').map(normalize).shuffle(10000).batch(BATCH_SIZE).map(convert_to_tuples)
    return train_ds, val_ds, test_ds

def normalize(features):
  features[IMAGE_INPUT_NAME] = tf.cast(
      features[IMAGE_INPUT_NAME], dtype=tf.float32) / 255.0
  return features

def convert_to_tuples(features):
  return features[IMAGE_INPUT_NAME], features[LABEL_INPUT_NAME]


def convert_to_dictionaries(image, label):
  return {IMAGE_INPUT_NAME: image, LABEL_INPUT_NAME: label}

def create_model():
    print('Creating model')
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=INPUT_SHAPE, dtype=tf.float32, name=IMAGE_INPUT_NAME))
    model.add(tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(NUM_CLASSES))
    model.summary()
    return model;

def train_model(model, train_data, val_data, cp_callbacks):
    print('Training model')
    history = model.fit(train_data,validation_data=val_data,epochs=EPOCHS,verbose=2,callbacks=[cp_callbacks])
    return history

def load_model(model):
    path = str(input("Path to file saved: "))
    return model.load_weights(path)

def plot_training(history, adv_history):
    epoch_range = range(EPOCHS)

    plt.figure(figsize=(8,8))
    plt.subplot(2,3,1)
    plt.plot(epoch_range, history.history['sparse_categorical_accuracy'], label='Base Training Accuracy')
    plt.plot(epoch_range, history.history['val_sparse_categorical_accuracy'], label='Base Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Base Accuracy')

    plt.subplot(2,3,2)
    plt.plot(epoch_range, adv_history.history['sparse_categorical_accuracy'], label='Advsarial Training Accuracy')
    plt.plot(epoch_range, adv_history.history['val_sparse_categorical_accuracy'], label='Adversarial Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Adversarial Accuracy')

    plt.subplot(2,3,3)
    plt.plot(epoch_range, history.history['sparse_categorical_accuracy'], label='Base Training Accuracy')
    plt.plot(epoch_range, history.history['val_sparse_categorical_accuracy'], label='Base Validation Accuracy')
    plt.plot(epoch_range, adv_history.history['sparse_categorical_accuracy'], label='Adversarial Training Accuracy')
    plt.plot(epoch_range, adv_history.history['val_sparse_categorical_accuracy'], label='Adversarial Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Comparison Accuracy')
    
    plt.subplot(2,3,4)
    plt.plot(epoch_range, history.history['loss'], label='Base Training Loss')
    plt.plot(epoch_range, history.history['val_loss'], label='Base Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Base Loss')

    plt.subplot(2,3,5)
    plt.plot(epoch_range, adv_history.history['loss'], label='Adversarial Training Loss')
    plt.plot(epoch_range, adv_history.history['val_loss'], label='Adversarial Validation Loss')
    plt.plot(epoch_range, adv_history.history['scaled_adversarial_loss'], label='Scaled Adversarial Training Loss')
    plt.plot(epoch_range, adv_history.history['val_scaled_adversarial_loss'], label='Scaled Adversarial Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Adversairal Loss')

    plt.subplot(2,3,6)
    plt.plot(epoch_range, history.history['loss'], label='Base Training Loss')
    plt.plot(epoch_range, history.history['val_loss'], label='Base Validation Loss')
    plt.plot(epoch_range, adv_history.history['loss'], label='Adversarial Training Loss')
    plt.plot(epoch_range, adv_history.history['val_loss'], label='Adversarial Validation Loss')
    plt.plot(epoch_range, adv_history.history['scaled_adversarial_loss'], label='Scaled Adversarial Training Loss')
    plt.plot(epoch_range, adv_history.history['val_scaled_adversarial_loss'], label='Scaled Adversarial Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Comparison Loss')
    plt.show()

# Show Batch as Example
def batch_display(batch_index):
    batch_images = perturbed_images[batch_index]
    batch_labels = labels[batch_index]
    batch_preds = predictions[batch_index]
    n_col = 4
    n_row = (BATCH_SIZE + n_col - 1) // n_col

    print('accuracy in batch %d:' % batch_index)
    for name, pred in batch_preds.items():
      print('\t%s model: %d / %d (%f%%)' % (name, np.sum(batch_labels == pred), BATCH_SIZE, ((np.sum(batch_labels == pred) / BATCH_SIZE) * 100)))

    plt.figure()
    for i, (image, label) in enumerate(zip(batch_images, batch_labels)):
      base_pred = batch_preds['base'][i]
      adv_pred = batch_preds['adv-regularized'][i]
      plt.subplot(4, 8, i+1)
      plt.title(' A: %d, B: %d, T: %d' % (adv_pred, base_pred, label))
      plt.imshow(tf.keras.utils.array_to_img(image), cmap='gray')
      plt.axis('off')
    plt.show()

# Retrive Datasets
train_ds, val_ds, test_ds = get_datasets()
adv_train_ds = train_ds.map(convert_to_dictionaries)
adv_val_ds = val_ds.map(convert_to_dictionaries)
adv_test_ds = test_ds.map(convert_to_dictionaries)

# Adversarial Configuration
adv_config = nsl.configs.make_adv_reg_config(
        multiplier=ADV_MULTIPLIER,
        adv_step_size=ADV_STEP_SIZE,
        adv_grad_norm=ADV_GRADIENT_NORM)


# Train & Validate

# Basic Model
base_model = create_model()
base_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['sparse_categorical_accuracy'])


# Adv Model
adv_model = nsl.keras.AdversarialRegularization(
    create_model(),
    label_keys=[LABEL_INPUT_NAME],
    adv_config=adv_config)
adv_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['sparse_categorical_accuracy'])

# Non-Attack Evaluation

# Savepaths
currentTime = datetime.now().strftime('%Y%m%d%H%M%S')

base_savepath = 'saved_models/' + DATASET + '/' + str(EPOCHS) + '/base/' + currentTime + '/cp-{epoch:04d}.ckpt'
base_cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=base_savepath, 
    verbose=2, 
    save_weights_only=True,
    save_freq='epoch')

adv_savepath = 'saved_models/' + DATASET + '/' + str(EPOCHS) + '/adv/' + currentTime + '/cp-{epoch:04d}.ckpt'
adv_cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=adv_savepath, 
    verbose=2, 
    save_weights_only=True,
    save_freq='epoch')

# Base
base_history = train_model(base_model, train_ds, val_ds, base_cp_callback)
results = base_model.evaluate(test_ds, verbose=2)
#base_model.save_weights(base_savepath)
print("Base model saved to", adv_savepath)
base_results = dict(zip(base_model.metrics_names, results))

# Adversarial
adv_history = train_model(adv_model, adv_train_ds, adv_val_ds, adv_cp_callback)
results = adv_model.evaluate(adv_test_ds, verbose=2)
#adv_model.save_weights(adv_savepath)
print("Adversarial model saved to", adv_savepath)
adv_results = dict(zip(adv_model.metrics_names, results))

#base_model = load_model(base_model)

#Generate attack test data
models_to_attack = { 'base': base_model,
                   'adv-regularized': adv_model.base_model }
attack_metrics = { metric_name: tf.keras.metrics.SparseCategoricalAccuracy()
            for metric_name in models_to_attack.keys()}
# Model used to perturb data
perturb_model = nsl.keras.AdversarialRegularization(base_model, label_keys=[LABEL_INPUT_NAME], adv_config = adv_config)
perturb_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['sparse_categorical_accuracy'])
perturbed_images, labels, predictions = [], [], []

                                                    
# Test all attack batches
for batch in adv_test_ds:
    batch_perturbed = perturb_model.perturb_on_batch(batch)
    batch_perturbed[IMAGE_INPUT_NAME] = tf.clip_by_value(batch_perturbed[IMAGE_INPUT_NAME], 0.0, 1.0)
    true_label = batch_perturbed.pop(LABEL_INPUT_NAME)
    perturbed_images.append(batch_perturbed[IMAGE_INPUT_NAME].numpy())
    labels.append(true_label.numpy())
    predictions.append({})
    # Evaluate all models on this batch
    for metric_name, model in models_to_attack.items():
        predicted_label = model(batch_perturbed)
        attack_metrics[metric_name](true_label, predicted_label)
        predictions[-1][metric_name] = tf.argmax(predicted_label, axis=-1).numpy()

                                                    
# Show Results

# Clean Test Statistics
print('\nbase model on Clean Data\n\taccuracy:', base_results['sparse_categorical_accuracy'],'\n\tloss:',base_results['loss'])
print('\nadversarial model on Clean Data\n\taccuracy:', adv_results['sparse_categorical_accuracy'],'\n\tloss:',adv_results['loss'])

# Attack Test Accuracy
for name, metric in attack_metrics.items():
  print('\n%s model accuracy against adversarial examples: %f' % (name, metric.result().numpy()))

# Plot Graphs
plot_training(base_history,adv_history)

print('Tested against', len(adv_test_ds) / BATCH_SIZE, ' batches of size ', BATCH_SIZE, ' (Final Batch may be smaller)')
while True :
    batch_index = int(input('\nBatch to view: '))
    if batch_index >= 0 and batch_index < len(adv_test_ds) / BATCH_SIZE - 1:
        batch_display(batch_index)
    else:
        break;
    


                                                
