# TinyGL

* modify from [TinyGL](https://bellard.org/TinyGL/)
* supporting draw graphic object in numsky.canvas

### Example
```
<Camera Ortho="-10,10,-10,10" shape="20,20,3">
	<Rect Size="1,1" pos="1,1" rot="80" scale="1,1" layer="1">
        255,255,255 -- fill channel
	</Rect>
</Camera>
```
* [example_tinygl.py](../examples/example_tinygl.py)

### xml tag's attribute

* camera

| **attribute**           | **description**         |
|:------------------:|:----------------:|
| ```Ortho```         |```orthogonal projection's parameter，left,right,bottom,top[,near,far]```|
| ```Perspective```   |```perspective projection's parameter，fovy,aspect,near,far```|
| ```Shape```         |```height、width、channel```|
| ```pos```         |```position，(x,y), (x,y,z)```|
| ```rot```         |```rotation，degree angle，(rotZ,) (rotX,rotY,rotZ) like unity```|

* mesh

| **attribute**           | **description**         |
|:------------------:|:----------------:|
| ```x-type```         |```fill type, point/line/triangle```|
| ```Vertices```     |```vertex array, numsky.ndarray object. shape is (n,3), dtype is float```|
| ```Indices```     |```face array, index for vertex array index start from 1. numsky.ndarray object, shape is (n,3), dtype is int```|
| ```pos```         |``````|
| ```rot```         |``````|
| ```scale```         |``````|
| ```layer```         |``````|

* point,line,rect,circle,polygon,sector

| **attribute**           | **description**         |
|:------------------:|:----------------:|
| ```x-type```         |```fill type，point/line/triangle```|
| ```Size```         |```some parameter for builtin mesh```|
| ```Pivot```         |```center for rotation and scale```|
| ```pos```         |``````|
| ```rot```         |``````|
| ```scale```         |```scale (x,y), (x,y,z)```|
| ```layer```         |```channel draw layer, start from 1```|

* size attribute

| **tag**           | **size attribute**     |
|:------------------:|:-----------------:|
| ```point```         |```invalid``` |
| ```line```         |```(xsize=1)``` |
| ```rect```         |```(xsize=1,ysize=1)``` |
| ```polygon```         |```(radius=0.5,edge_num=5)``` |
| ```circle```         |```(radius=0.5)``` |
| ```sector```         |```(radius=0.5,degree=45)``` |

### lua api

* TODO
