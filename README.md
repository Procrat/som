Self-organizing maps
====================

This package aims to provide a small framework for using and extending
self-organizing maps (SOMs).

Examples
--------


Quick usage guide
-----------------
Run or take a look at `examples.py` for a quick glance on how it is meant to be
used.

Structure
---------
At the moment, this package mainly consists of three files:
- `som`: The common structure of SOMs.
- `som.basic_som`: An implementation of the standard SOM.
- `som.asom`: An implementation of an extension, namely an assymetric SOM,
  which no longer uses a grid, but a plane where the nodes can position freely.

Extending the base implementation
---------------------------------
If you want to implement a SOM extension using this framework, you basicly have
two options.
1. If you just want to adjust a small part of the regular SOM, say the
   neighborhood function, you might want to build a new class which extends the
   `BasicSOM` class and overwrites certain functions.
2. If you want to try something completely different, e.g. using a different
   topology (as the case with assymetric SOMs), you might want to build a new
   class which extends the `SOM` class which asks for an initialized codebook
   in it's `__init__` method. This codebook should be subclass of the
   `Topology` class. This might at first seem a bit confusing but it should be
   quite clear if you take a quick look at the three files in the package.

Context of this project / Licensing
----------------------------
I made this framework as a side effect of the project of a graduate course
`Intelligent Systems` at Ghent University. As I have no immediate plans to
further develop this package, I made it publicly available under the MIT
license.
If you happen to find it useful, feel free to let me know at stijnseghers at
gmail.com.
