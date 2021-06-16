
import sys
sys.path.append("../")

import pyskynet
from pyskynet.skynet_py_main import old_setenv, old_getenv, py_setenv, py_getenv


def test_old_env():
    print(old_getenv(b"rewrw"))
    print(old_setenv(b"rewrw", b"rewrewrewrewrwe"))
    print(old_getenv(b"rewrw"))
    pyskynet.start()
    print(old_getenv(b"rewrw"))
    print(old_setenv(b"rewrw", b"rewrewrewrewrwe"))
    print(old_getenv(b"rewrw"))
    print(old_setenv(b"rewrw", b"kljldsfsdfs"))
    pyskynet.scriptservice("""
        local skynet = require "skynet"

        skynet.start(function()
            print(skynet.getenv("rewrw"))
            skynet.setenv("rewrw", "sfsfds")
        end)
    """)

def test_raw_py_env():
    pyskynet.start()
    print(py_getenv(b"rewrw"))
    print(py_setenv(b"rewrw", b"rewrewrewrewrwe"))
    print(py_getenv(b"rewrw"))
    pyskynet.scriptservice("""
        local pyskynet = require "pyskynet"
        local pyskynet_modify = require "pyskynet.modify"

        pyskynet.start(function()
            print("nil next:", pyskynet_modify.nextenv(nil) )
            print("rewrw next:", pyskynet_modify.nextenv("rewrw") )
        end)
    """)


def test_ENV():
    pyskynet.start()
    pyskynet.ENV["rewkl"] = 32113
    print(pyskynet.ENV["rewkl"])
    for k in pyskynet.ENV:
        print(pyskynet.ENV[k])
    pyskynet.scriptservice("""
        local pyskynet = require "pyskynet"
        local pyskynet_modify = require "pyskynet.modify"

        pyskynet.start(function()
            for k,v in pairs(pyskynet.ENV) do
                print(k,v)
            end
        end)
    """)


test_old_env()
#test_raw_py_env()
#test_ENV()

