
import sys
sys.path.append("../")

import pyskynet
from pyskynet import setenv, getenv


def test_old_env():
    print(getenv(b"rewrw"))
    setenv(b"rewrw", b"rewrewrewrewrwe")
    setenv(b"rew", {"rew":321})
    pyskynet.start()
    pyskynet.scriptservice("""
        local pyskynet = require "pyskynet"

        pyskynet.start(function()
            print(pyskynet.getenv("rewrw"))
            print(pyskynet.getenv("rew").rew)
        end)
    """)




test_old_env()

