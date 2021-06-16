
import sys
sys.path.append("../")


import cv2
import pyskynet
import pyskynet.foreign as foreign

pyskynet.start()

service = pyskynet.scriptservice("""
        local pyskynet = require "pyskynet"
        local foreign = require "pyskynet.foreign"
        local ns = require "numsky"
        local tinygl = require "numsky.tinygl"
        local rapidjson = require "rapidjson"


        local box_v = ns.array({{-4.9,-4.9,-3}, {-4.9,-4.9,-3}, {-4.9,-4.9,-3}}, ns.float32)
        local box_i = ns.array({{1,2,3}}, ns.int32)
        local mesh1 = tinygl.mesh(box_v, box_i)

        local box_v = ns.array({{4.9,4.9,-3}, {4.9,4.9,-3}, {4.9,4.9,-3}}, ns.float32)
        local box_i = ns.array({{1,2,3}}, ns.int32)
        local mesh2 = tinygl.mesh(box_v, box_i)

        local camera = tinygl.camera(101,101,3)
        camera:rotation(0,0,0)
        camera:position(0,0,0)
        camera:ortho(-10,10,-10,10)
        local rect1 = tinygl.rect(-0.5,-0.5,2,1)
        rect1:rotation(0,0,0)
        rect1:scale(5,5,0)
        local rect2 = tinygl.rect(-0.5,-0.5,1,1)
        rect2:rotation(0,0,0)
        rect2:scale(5,5,0)
        rect2:setColor(255,255,255)
        local circle = tinygl.circle(0,0,5)
        circle:setColor(0,255,255,255)
        circle:setFillType(2)
        circle:rotation(0,0,90)
        circle:scale(1,0.5,1)
        circle:position(0,0,0)
        local tri = tinygl.circle(0,0,5,3)
        tri:setColor(255,255,255)
        local sector = tinygl.sector(0,0,5,15)
        sector:setColor(255,255,0)
        local line = tinygl.line(0,0,5,5)
        line:setColor(255,255,0)
        local point = tinygl.point(-3,-3)
        point:setColor(255,255,0)
        foreign.dispatch("test", function()
            camera:draw(rect1)
            camera:draw(rect2)
            camera:draw(sector)
            camera:draw(line)
            camera:draw(point)
            camera:draw(mesh1)
            camera:draw(mesh2)
            local pixel_arr, depth_arr = camera:array()
            return pixel_arr:copy(), depth_arr:copy()
        end)
        pyskynet.start(function()
        end)
""")

arr1, arr2 = foreign.call(service, "test")

import numpy as np



cv2.imwrite("tinygl1.bmp", arr1);
cv2.imwrite("tinygl2.bmp", arr2);

#arr1 = arr1.reshape((arr1.shape[0], arr1.shape[1]))
#print(np.where(arr1==255))
#print(arr1)
#print(np.sum(arr1))
print(arr1.shape)
