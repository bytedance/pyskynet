import sys
sys.path.append("../")

import pyskynet
import pyskynet.foreign as foreign

import numpy as np

pyskynet.start()

canvas = pyskynet.canvas("""<?lua
local LEN = 100  -- define static variable
local s1, s2 = ...      -- pass ... from canv:reset(...)

?>

<var x-local="n">1</var> <!-- define dynamic variable  -->
<var x-local="v1,v2,v3">...</var> <!-- pass ... from canv:render(...) -->
<var x-function="func(a,b)"> <!-- define function -->
    return math.max(a,b)
</var>

<!-- make an array -->
<arr>
    <int32> 321 </int32>
    <int32> 13,4,4,23</int32> <!-- multi int32 value, fill into arr-->
    <float32> LEN,n,v1,v2,v3 </float32> <!-- use static & dynamic dynamic variable -->
    <float32><?lua  -- more complex code
        if func(v1, v2) then
            return v1
        else
            return v2
        end
    ?></float32>
</arr>

<!-- make a 2d array -->
<arr x-type="int32"> <!-- put dtype， default float32 -->
    <arr>
        <int32>1,2,3</int32>
    </arr>
    <arr x-for="i=1,10">  <!-- expand with for loop -->
        <int32>i,i,i</int32>
    </arr>
    <arr x-for="k,v in pairs({5,3,4})" x-sort="v"> <!-- sort by number -->
        <int32>k,v,k+v</int32>
    </arr>

    <!-- exception when not match -->
    <!--arr>
        <int32>1,2,3,4,4,5,</int32>
    </arr-->
</arr>

<!-- make a table -->
<table>
    <int32>321</int32>
    <arr name="fds"> <!-- name as table's key-->
        <int32>32131</int32>
    </arr>
    <table> <!--table in table-->
        <int32 name="num">321</int32>
    </table>
</table>

""", "hello.xml")


canvas.reset(3,4)
a,b,c = canvas.render(1,2,3)

print(a)
print(b)
print(c)
