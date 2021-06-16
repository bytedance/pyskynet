# PySkynet

PySkynet is a library for using [skynet](https://github.com/cloudwu/skynet) in python. Including a lua library *numsky* for dealing with numpy.ndarray object.

### update submodule

~~~~sh
$ git submodule update --init --recursive
~~~~

### build

compile & install

~~~~sh
$ python setup.py install
~~~~

or compile inplace

~~~~sh
$ python setup.py build_ext -i
~~~~

### run example

~~~~sh
$ cd examples
$ python example_script.py
~~~~

or

~~~~sh
$ cd examples
$ pyskynet example.lua
~~~~

