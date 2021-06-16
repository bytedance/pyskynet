
import sys
sys.path.append("../")

import pyskynet

pyskynet.start()


canvas = pyskynet.canvas("""
<?lua

local left, right, up, bottom = -10,10,-10,10
local MapH, MapW, MapC = ...

?>

<var local="camera_direction">...</var>

<camera ortho="left, right, up, bottom" shape="MapH, MapW, MapC" rot="camera_direction">
    <rect size="3,3" for="i=1,3" rot="120*i" pos="3,3" scale="i,4-i" layer="i">
        255
    </rect>
    <sector size="5,45" for="i=1,3" rot="120*i" pos="-3,-3" layer="i">
        255
    </sector>
    <block for="x=-5,5">
        <circle if="x%2==1" pos="x,0" scale="3,3" layer="1">
            255,255
        </circle>
    </block>
    <mesh vertices="{{1,0,0}, {0,1,0}, {1,1,0}, {0,0,0}}" fill="point" pos="3,-3">
        255,255
    </mesh>
</camera>
""", "temp.xml")

canvas.reset(100,100,3)

arr, = canvas.render(90)

import cv2
cv2.imwrite("tinygl.bmp", arr)

