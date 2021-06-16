
# numsky

## 一个lua版本的numpy (TODO)

* lua的table下标从1开始，numsky的数组索引也从1开始

### 数据类型

* numsky.bool
	* 在lua中，false和nil为false，其它值（包括0）为true
* numsky.int8
* numsky.int16
* numsky.int32
* numsky.int64
* numsky.uint8
* numsky.uint16
* numsky.uint32
* numsky.float32
* numsky.float64

### 构造

* numsky.zeros(shape, dtype)
* numsky.empty(shape, dtype)
* numsky.ones(shape, dtype)
* numsky.array(table, dtype)
* numsky.arange(start,stop,step,dtype)
* numsky.linspace(start,stop,step,dtype)

| **numpy**                                                   | **numsky**                                                 |
|:-----------------------------------------------------------:|:----------------------------------------------------------:|
| ```arr = np.array([1,2,3,4,5,6], dtype=np.int8)```          | ```local arr = ns.array({1,2,3,4,5,6}, ns.int8)```         |
| ```arr = np.arange(1,101, dtype=np.float32) # 左开右闭```   | ```local arr = np.arange(1,100, ns.float32) -- 左闭右闭``` |
| ```arr = np.linspace(1,20,20,False) #                 ```   | ```local arr = np.linspace(1, 20, 20, false) --        ``` |

### 切片、取值、赋值

* numsky.slice(start, stop, step)

| **numpy**                                             | **numsky**                                               |
|:-----------------------------------------------------:|:--------------------------------------------------------:|
| ```slice(0,4,1) # 范围区间左开右闭，index从0开始```   | ```ns.slice(1,4,1) -- 范围区间左闭右闭，index从1开始```  |
| ```arr[0,-2,1:4,-1:-4:-1]```                          | ```arr[{1,-2,ns.slice(2,4),ns.slice(-1,-4,-1)}] ```      |
| ```arr[np.array([0,1,2,3])]```                        | ```arr[ns.array({1,2,3,4})] ```                          |

### 运算

* 一元运算
	* numsky.unm -- 取负
	* numsky.bnot -- 按位取反
	* numsky.abs
	* numsky.ceil
	* numsky.floor
	* numsky.acos
	* numsky.asin
	* numsky.atan
	* numsky.cos
	* numsky.sin
	* numsky.tan
	* numsky.deg
	* numsky.rad
	* numsky.log
	* numsky.exp
	* numsky.sqrt

* 二元运算
	* numsky.add
	* numsky.sub
	* numsky.mul
	* numsky.div
	* numsky.idiv
	* numsky.mod
	* numsky.pow
	* numsky.band
	* numsky.bor
	* numsky.bxor
	* numsky.shl
	* numsky.shr
	* numsky.eq
	* numsky.lt
	* numsky.le
	* numsky.ne
	* numsky.gt
	* numsky.ge
	* numsky.fmax
	* numsky.fmin
	* numsky.atan2

* reduce运算
	* numsky.sum
	* numsky.prod
	* numsky.any
	* numsky.all
	* numsky.max
	* numsky.min

| **numpy**                       | **numsky**                            |
|:-------------------------------:|:-------------------------------------:|
| ```arr1 + arr2```               | ```arr1 + arr2```                     |
| ```arr1 & arr2```               | ```arr1 & arr2```                     |
| ```arr1 > arr2```               | ```ns.gt(arr1, arr2) -- TODO ```      |

### 属性

| **numpy**                     | **numsky**                          |
|:-----------------------------:|:-----------------------------------:|
| ```arr.ndim```                | ```arr.ndim```                      |
| ```arr.shape```               | ```arr.shape```                     |
| ```arr.strides ```            | ```arr.strides```                   |
| ```arr.dtype```               | ```arr.dtype```                     |

### ndarray对象方法

* arr.reshape
* arr.flatten
* arr.astype
* arr.copy

### TODO

* numsky.where, numsky.argmax, numsky.argmin, numsky.std, numsky.var, numsky.mean,
