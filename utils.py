import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
from IPython import display

def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def plot_waveform(waveform, ax):
  if len(waveform.shape) > 1:
    waveform = np.squeeze(waveform, axis=-1)
  # Plot the waveform.
  ax.plot(np.arange(waveform.shape[0]), waveform)
  # Label the axes.
  ax.set_title('Waveform')


def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)
  ax.set_title('Spectrogram')
  
def plot_data(waveform, spectrogram, src, instr):
  fig, axes = plt.subplots(2, figsize=(10, 8))
  plot_waveform(waveform, axes[0])
  plot_spectrogram(spectrogram, axes[1])
  plt.suptitle(src + " " + instr)
  plt.show()
  
def save_as_json(data, path):
  with open(path, 'w') as f:
    json.dump(data, f, ensure_ascii=False)
    
def plot_history(history):
  fig, axs = plt.subplots(1, 2, figsize=(12, 4))
  axs[0].plot(history['accuracy'])
  axs[0].plot(history['val_accuracy'])
  axs[0].set_title('Accuracy')
  axs[0].set_ylabel('Accuracy')
  axs[0].set_xlabel('Epoch')
  axs[0].legend(['Train', 'Test'], loc='upper left')
  axs[1].plot(history['loss'])
  axs[1].plot(history['val_loss'])
  axs[1].set_title('Loss')
  axs[1].set_ylabel('Loss')
  axs[1].set_xlabel('Epoch')
  axs[1].legend(['Train', 'Test'], loc='upper left')
  plt.show()
    
def predict(model, audio, classes):
  display.display(display.Audio(audio.numpy(), rate=16000))
  input_data = audio
  if len(model.get_config()["layers"][0]["config"]["batch_input_shape"]) == 4:
    input_data = get_spectrogram(audio)
  input_data = tf.expand_dims(input_data, axis=0)
  prediction = tf.nn.softmax(model.predict(input_data)[0])
  print("Predicted class: " + classes[np.argmax(prediction)])
  fig = plt.figure(figsize=(10, 5))
  plt.bar(classes, prediction)
  plt.title('Prediction')
  plt.xlabel('Class')
  plt.ylabel('Confidence')
  plt.tight_layout()
  plt.show()
  return classes[np.argmax(prediction)]
  
  
