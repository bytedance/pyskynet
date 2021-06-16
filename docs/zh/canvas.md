
# numsky.canvas

## 简介
numsky.canvas是基于xml+lua实现的一套用于特征预处理的DSL(Domain Specific Language)


### 示例

```lua
local ns = require "numsky"
local canv = ns.canvas([[<?lua
	local x,y = ...  -- reset阶段传入静态参数
?>
<var local="d">... </var> <!-- render阶段传入动态参数 -->
<arr shape="x,y">
	<arr for="i=1,x">
		<float32 for="j=1,y">d*i+j</float32>
	</arr>
</arr>
]])

canv:reset(3,4)
local arr = canv:render(10)
print(arr)
```

### 详细示例

* [example_canv.py](../examples/example_canv.py)

### 基本设计

* lua对象、多维数组、图形(TODO)
* 静态属性、动态属性(TODO)

### 元素

* lua对象

| **标签**           | **说明**        | **示例**     |
|:------------------:|:---------------:|:---------------------------:|
| ```any```          |```构造任意lua对象```| ```<any>"sfdsfs"</any>```         |
| ```table```        |```构造一个lua table```| ```<table><any name="key">"value"</any></table>```         |

* 多维数组

| **标签**                      | **说明**                 | **示例**     |
|:-----------------------------:|:------------------------:|:---------------------------:|
| ```arr```                     |```构造一个数组```        | ```<arr><float32>1,2,3</float32></arr>```         |
| ```arr1d, arr2d,...```        |```同上，但直接给定维度```| ```<arr1d><float32>1,2,3</float32></arr1d>```     |
| ```int8,uint8,float32,...```  |```构造若干个标量数据```  | ```<int8>1,2,3</int8>```         |

* 图形

| **标签**           | **说明**         | **示例**     |
|:------------------:|:----------------:|:---------------------------:|
| ```camera```          |```构造一个相机对象```| ```<camera><rect>1,2,3</rect></camera>```         |
| ```mesh```         |```绘制一个mesh图形```| ```<mesh vertices="{{1,0,0},{0,1,0},{1,1,0}}" indices="{{1,2,3}}">1,2,3</mesh>```         |
| ```point,line,rect,polygon,circle,sector```         |```绘制一个图形```| ```<rect>1,2,3</rect>``` |

* 其他

| **标签**                      | **说明**                 | **示例**     |
|:-----------------------------:|:------------------------:|:---------------------------:|
| ```block```                   |```表示一个块```        | ```<arr><block><float32>1,2,3</float32></block></arr>```         |
| ```var```        |```定义一个lua变量```| ```<var local="sth">1,2,3</var>```     |
| ```proc```  |```执行一段lua语句```  | ```<proc>print(321)</proc>```         |
