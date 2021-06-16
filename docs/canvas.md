
## numsky.canvas

Implementing a DSL(Domain Specific Language) with xml+lua for feature processing.


### Example

```lua
local ns = require "numsky"
local canv = ns.canvas([[<?lua
	local x,y = ...  -- pass static arguments in reset
?>
<var local="d">... </var> <!-- pass dynamic arguments in render -->
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

### More Examples

* [example_canv.py](../examples/example_canv.py)

### Element

* lua object

| **tag**           | **description**        | **example**     |
|:------------------:|:---------------:|:---------------------------:|
| ```any```          |```make any lua object```| ```<any>"sfdsfs"</any>```         |
| ```table```        |```make a lua table```| ```<table><any name="key">"value"</any></table>```         |

* multi-dimensions array

| **tag**                      | **description**                 | **example**     |
|:-----------------------------:|:------------------------:|:---------------------------:|
| ```arr```                     |```make an array```        | ```<arr><float32>1,2,3</float32></arr>```         |
| ```arr1d, arr2d,...```        |```make an array, but nd is determinated```| ```<arr1d><float32>1,2,3</float32></arr1d>```     |
| ```int8,uint8,float32,...```  |```make a scalar```  | ```<int8>1,2,3</int8>```         |

* graphic

| **tag**           | **description**         | **example**     |
|:------------------:|:----------------:|:---------------------------:|
| ```camera```          |```make a camera```| ```<camera><rect>1,2,3</rect></camera>```         |
| ```mesh```         |```draw a mesh in camera```| ```<mesh vertices="{{1,0,0},{0,1,0},{1,1,0}}" indices="{{1,2,3}}">1,2,3</mesh>```         |
| ```point,line,rect,polygon,circle,sector```         |```draw a builtin mesh in camera```| ```<rect>1,2,3</rect>``` |

* other

| **tag**                      | **description**                 | **example**     |
|:-----------------------------:|:------------------------:|:---------------------------:|
| ```block```                   |```make a block, contain a lua scope```        | ```<arr><block><float32>1,2,3</float32></block></arr>```         |
| ```var```        |```define lua variable```| ```<var local="sth">1,2,3</var>```     |
| ```proc```  |```execute lua statement```  | ```<proc>print(321)</proc>```         |
