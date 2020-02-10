=============================================================
TensorFlow Keras Applications (pretrained models) for Foolbox
=============================================================

This repository Keras Applications (pretrained models) in a `Foolbox Native <https://github.com/bethgelab/foolbox>`_ compatible format.

Example
-------

.. code-block:: python

   import foolbox as fbn

    url = "https://github.com/jonasrauber/foolbox-tensorflow-keras-applications"
   fmodel = fbn.zoo.get_model(url, name="MobileNetV2")

   images, labels = foolbox.utils.samples(fmodel, dataset='imagenet', batchsize=16)
   print(fbn.accuracy(fmodel, images, labels))

   # you can now attack fmodel using Foolbox attacks
   # ...
