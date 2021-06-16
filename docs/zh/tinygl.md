# TinyGL

## 简介
numsky.canvas中支持以基于xml+lua的方式调用TinyGL绘制图形。

### 基于xml+lua的示例
```
<camera ortho="-10,10,-10,10" shape="20,20,3">
	<rect size="1,1" pos="1,1" rot="80" scale="1,1" layer="1">
        255,255,255 -- 填充channel
	</rect>
</camera>
```
* 代码参见[example_tinygl.py](../examples/example_tinygl.py)

### xml元素的属性

* camera

| **名称**           | **说明**         |
|:------------------:|:----------------:|
| ```静态属性:``` |
| ```ortho```         |```正交投影参数，left,right,bottom,top[,near,far]```|
| ```perspective```   |```透视投影参数，fovy,aspect,near,far```|
| ```shape```         |```高、宽、通道数```|
| ```动态属性:``` |
| ```pos```         |```位移，(x,y), (x,y,z)```|
| ```rot```         |```旋转，角度制，(rotZ,) (rotX,rotY,rotZ) 类似unity```|

* mesh, 自定义图形

| **名称**           | **说明**         |
|:------------------:|:----------------:|
| ```静态属性:``` |
| ```fill```         |```填充类型，可以为point,line,triangle```|
| ```vertices```     |```顶点数组，一个n*3的float类型的numsky数组```|
| ```indices```     |```面数组，一个n*3的int类型的numsky数组，下标从1开始```|
| ```动态属性:``` |
| ```pos```         |``````|
| ```rot```         |``````|
| ```scale```         |``````|
| ```layer```         |``````|

* point,line,rect,circle,polygon,sector，内置图形

| **名称**           | **说明**         |
|:------------------:|:----------------:|
| ```静态属性:``` |
| ```fill```         |```填充类型，可以为point，line，triangle```|
| ```size```         |```若干参数，具体由类型决定```|
| ```pivot```         |```控制旋转和缩放的中心点```|
| ```动态属性:``` |
| ```pos```         |``````|
| ```rot```         |``````|
| ```scale```         |```缩放 (x,y), (x,y,z)```|
| ```layer```         |```绘制的channel层，从1开始```|

* size 属性

| **标签**           | **size属性**     | **默认中心点**
|:------------------:|:-----------------:|:------------------:|
| ```point```         |```无``` | ```点的位置``` |
| ```line```         |```(xsize=1)``` | ```线段中点``` |
| ```rect```         |```(xsize=1,ysize=1)``` | ``` 矩形中点 ``` |
| ```polygon```         |```(radius=0.5,edge_num=5)``` | ``` 多边形中点 ``` |
| ```circle```         |```(radius=0.5)``` | ```圆心``` |
| ```sector```         |```(radius=0.5,degree=45)``` | ```扇形圆心``` |

### lua api

* TODO
