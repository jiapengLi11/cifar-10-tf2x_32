# CIFAR-10 TensorFlow Models (32x32)

This is an early TensorFlow image classification project for CIFAR-10. It contains two training pipelines:

- a custom CNN in `models/Net.py`
- a VGG19-style network in `models/VGG.py`

## Project Structure

- `train_net.py`: train the custom CNN
- `train_vgg19.py`: train the VGG19-style model
- `test_net.py`: run inference with the custom CNN
- `test_vgg19.py`: run inference with the VGG19-style model
- `images/`: sample images for quick testing
- `figures/`: saved training curves from earlier runs
- `models/`: model definitions

## Setup

```bash
pip install -r requirements.txt
```

## Train

Train the custom CNN:

```bash
python train_net.py
```

Train the VGG19-style model:

```bash
python train_vgg19.py
```

Training weights will be saved to:

- `log-self/self.h5`
- `log-vgg19/vgg19.h5`

## Predict

Run the custom CNN on sample images:

```bash
python test_net.py --weights log-self/self.h5 --image-dir images
```

Run the VGG19-style model on sample images:

```bash
python test_vgg19.py --weights log-vgg19/vgg19.h5 --image-dir images
```

Add `--show` if you want to display the images with OpenCV.

## Notes

- The repository does not include trained `.h5` weights.
- CIFAR-10 is loaded automatically through `tensorflow.keras.datasets.cifar10`.
- `figures/` contains exported training curves from earlier experiments, while large TensorBoard event files are excluded.
