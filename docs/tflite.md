
# lua-tflite

wrap tensorflow-lite in lua

### usage

```
local ns = require "numsky" -- tflite dependent numsky
local tflite = require "tflite"
```

### compile

TODO

### tflite class

* tflite.interpreter
	* tflite.interpreter(model [, options]) # (tflite.model | string [, tflite.options| integer]) -> tflite.interpreter # create a *tflite.interpreter* object, model is a string or *tflite.model*, options is thread_num or *tflite.options* object, default is 1
	* interpreter.input\_tensors # input tensor table，map from index or name to interpreter's input tensor, index start from 1
	* interpreter.output\_tensors # output tensor table, map from index or name to interpreter's output tensor，index start from 1
	* interpreter:allocate\_tensors() # allocate tensors
	* interpreter:invoke()

* tflite.tensor
	* tensor:set(arr) # (numsky.ndarray) # copy from *numsky.ndarray*
	* tensor:get() # () -> numsky.ndarray # copy to *numsky.ndarray* and return
	* tensor.name
	* tensor.shape
	* tensor.ndim
	* tensor.dtype
	* tensor.data # lightuserdata，point to tensor's data

* tflite.model
	* tflite.model() # create a *tflite.model*

* tflite.options
	* tflite.options() # create a *tflite.options*
	* options.set\_num\_threads # set thread_num
