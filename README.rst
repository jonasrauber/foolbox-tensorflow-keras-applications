=============================================================
TensorFlow Keras Applications (pretrained models) for Foolbox
=============================================================

This repository provides `Keras Applications <https://keras.io/applications/>`_ (pretrained ImageNet models) in a `Foolbox Native <https://github.com/bethgelab/foolbox>`_ compatible format.

Example
-------

.. code-block:: python

   import foolbox as fbn

   url = "https://github.com/jonasrauber/foolbox-tensorflow-keras-applications"
   fmodel = fbn.zoo.get_model(url, name="MobileNetV2", overwrite=True)

   images, labels = fbn.samples(fmodel, dataset='imagenet', batchsize=16)
   print(fbn.accuracy(fmodel, images, labels))
   # -> 0.9375

   # you can now attack fmodel using Foolbox attacks
   # ...
