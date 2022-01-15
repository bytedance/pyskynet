
## numsky.canvas

Implementing a DSL(Domain Specific Language) with xml+lua for feature processing.


### Example

```lua
local ns = require "numsky"
local canv = ns.canvas([[<?reset
	local x,y = ...  -- pass static arguments in reset
?>
<var x-local="d">... </var> <!-- pass dynamic arguments in render -->
<Arr Shape="x,y">
	<Arr x-for="i=1,x">
		<float32 x-for="j=1,y">d*i+j</float32>
	</Arr>
</Arr>
]])

canv:reset(3,4)
local arr = canv:render(10)
print(arr)
```

### More Examples

* [example_canv.py](../examples/example_canv.py)

### Element

* lua object

| **tag**           | **description**        | **example**     |
|:------------------:|:---------------:|:---------------------------:|
| ```Any```          |```make any lua object```| ```<Any>"sfdsfs"</Any>```         |
| ```Table```        |```make a lua table```| ```<Table><Any x-name="key">"value"</Any></Table>```         |

* multi-dimensions array

| **tag**                      | **description**                 | **example**     |
|:-----------------------------:|:------------------------:|:---------------------------:|
| ```Arr```                     |```make an array```        | ```<Arr><float32>1,2,3</float32></Arr>```         |
| ```arr1d, arr2d,...```        |```make an array, but nd is determinated```| ```<Arr1d><float32>1,2,3</float32></Arr1d>```     |
| ```int8,uint8,float32,...```  |```make a scalar```  | ```<int8>1,2,3</int8>```         |

* graphic

| **tag**           | **description**         | **example**     |
|:------------------:|:----------------:|:---------------------------:|
| ```Camera```          |```make a camera```| ```<Camera><Rect>1,2,3</Rect></Camera>```         |
| ```Mesh```         |```draw a mesh in camera```| ```<Mesh Vertices="{{1,0,0},{0,1,0},{1,1,0}}" Indices="{{1,2,3}}">1,2,3</mesh>```         |
| ```Point,Line,Rect,Polygon,Circle,Sector```         |```draw a builtin mesh in camera```| ```<Rect>1,2,3</Rect>``` |

* other

| **tag**                      | **description**                 | **example**     |
|:-----------------------------:|:------------------------:|:---------------------------:|
| ```block```                   |```make a block, contain a lua scope```        | ```<Arr><block><float32>1,2,3</float32></block></Arr>```         |
| ```var```        |```define lua variable```| ```<var x-local="sth">1,2,3</var>```     |
| ```proc```  |```execute lua statement```  | ```<proc>print(321)</proc>```         |
