
import sys
sys.path.append("../")

import pyskynet

pyskynet.start()


canvas = pyskynet.canvas("""
<?reset

local Left, Right, Up, Bottom = -10,10,-10,10
local MapH, MapW, MapC = ...

?>

<var x-local="camera_direction">...</var>

<Camera Ortho="Left, Right, Up, Bottom" Shape="MapH, MapW, MapC" rot="camera_direction">
    <Rect Size="3,3" x-for="i=1,3" rot="120*i" pos="3,3" scale="i,4-i" layer="i">
        255
    </Rect>
    <Sector Size="5,45" x-for="i=1,3" rot="120*i" pos="-3,-3" layer="i">
        255
    </Sector>
    <block x-for="x=-5,5">
        <Circle x-if="x%2==1" pos="x,0" scale="3,3" layer="1">
            255,255
        </Circle>
    </block>
    <Mesh Vertices="{{1,0,0}, {0,1,0}, {1,1,0}, {0,0,0}}" x-type="point" pos="3,-3">
        255,255
    </Mesh>
</Camera>
""", "temp.xml")

canvas.reset(100,100,3)

arr, = canvas.render(90)

print(arr)
import cv2
cv2.imwrite("tinygl.bmp", arr)

