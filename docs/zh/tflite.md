
# lua-tflite

## 实现了一些对tensorflow-lite的封装，用来提供给lua层调用的API

### 使用

```
local ns = require "numsky" -- tflite模块依赖numsky
local tflite = require "tflite"
```

### 编译方式

TODO

### tflite的类封装

* tflite.interpreter
	* tflite.interpreter(model [, options]) # (tflite.model | string [, tflite.options| integer]) -> tflite.interpreter #创建一个tflite的interpreter对象。model为模型的二进制string或tflite.model对象。options为线程数或tflite.options对象，默认为1。
	* interpreter.input\_tensors # input tensor table，可通过index或者name查找interpreter的input tensor，index从1开始。
	* interpreter.output\_tensors # output tensor table, 可通过index或者name查找interpreter的output tensor，index从1开始。
	* interpreter:allocate\_tensors() # 分配tensors
	* interpreter:invoke()

* tflite.tensor
	* tensor:set(arr) # (numsky.ndarray) # 将numsky.ndarray对象arr拷贝给tensor 对象
	* tensor:get() # () -> numsky.ndarray # 将tensor对象拷贝为numsky.ndarray对象并返回
	* tensor.name
	* tensor.shape
	* tensor.ndim
	* tensor.dtype
	* tensor.data # 一个lightuserdata，指向tensor数据的指针

* tflite.model
	* tflite.model() # 创建一个tflite.model

* tflite.options
	* tflite.options() # 创建一个tflite.options
	* options.set\_num\_threads # 设置线程数
