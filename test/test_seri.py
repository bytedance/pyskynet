
import sys
sys.path.append("../")

import pyskynet
import pyskynet.foreign as foreign
import pyskynet.skynet_py_foreign_seri as foreign_seri

import numpy as np

pyskynet.start()

echo_service = None

@foreign.dispatch("echo")
def _echo(*args):
    return args

def turnon_lua():
    global echo_service
    echo_service = pyskynet.scriptservice("""
            local pyskynet = require "pyskynet"
            local skynet = require "skynet"
            local foreign = require "pyskynet.foreign"
            local foreign_seri = require "pyskynet.foreign_seri"
            pyskynet.start(function()
                local function trash_ret(msg, ...)
                    foreign_seri.trash(msg)
                    return ...
                end
                foreign.dispatch("echo", function(...)
                    return ...
                end)
                --[[foreign.dispatch("echo_ref", function(...)
                    local msg, sz = foreign_seri.refpack(...)
                    return trash_ret(msg, foreign_seri.refunpack(msg, sz))
                end)
                foreign.dispatch("echo_ref_hook", function(...)
                    local msg, sz = foreign_seri.refpack(...)
                    local hook = foreign_seri.packhook(msg)
                    if hook then
                        return trash_ret(hook, foreign_seri.refunpack(hook, sz))
                    else
                        return trash_ret(msg, foreign_seri.refunpack(msg, sz))
                    end
                end)]]
                foreign.dispatch("echo_remote", function(...)
                    local msg, sz = foreign_seri.remotepack(...)
                    return trash_ret(msg, foreign_seri.remoteunpack(msg, sz))
                end)
                foreign.dispatch("echo_remote_hook", function(...)
                    local msg, sz = foreign_seri.remotepack(...)
                    local hook = foreign_seri.__packhook(msg)
                    if hook then
                        return trash_ret(hook, foreign_seri.remoteunpack(hook, sz))
                    else
                        return trash_ret(msg, foreign_seri.remoteunpack(msg, sz))
                    end
                end)
            end)
    """)


def equal(left, right):
    if type(left) != type(right):
        return (left == right) == True
    elif type(left) == dict:
        if len(left) != len(right):
            return False
        for k in left:
            if not (k in right):
                return False
            if not equal(left[k], right[k]):
                return False
        return True
    elif type(left) == np.ndarray:
        return left.shape == right.shape and (left == right).all()
    else:
        return left == right

def tostring(arg):
    if type(arg) == dict:
        re = {}
        for k,v in arg.items():
            re[k] = tostring(v)
        return re
    elif type(arg) == list:
        re = []
        for k in arg:
            re.append(tostring(k))
        return re
    elif type(arg) == np.ndarray:
        return "arr("+str(arg.shape)+","+str(arg.dtype)+")"
    else:
        return arg


def check_case(*args):
    def check_equal(name, left, right):
        for i in range(len(left)):
            assert equal(left[i], right[i]), "check error :%s:%s,%s,%s,%s"%(name,
                    left[i],  type(left[i]),
                    right[i], type(right[i]))
    for name, pack, unpack in [
            ("foreign", foreign_seri.__refpack, foreign_seri.__refunpack),
            ("foreign_remote", foreign_seri.remotepack, foreign_seri.remoteunpack)]:
        # pack to capsule
        msg_ptr, msg_size = pack(*args)
        result = unpack(msg_ptr, msg_size)
        foreign.trash(msg_ptr)
        check_equal(name, result, args)
        # pack hook to capsule
        msg_ptr, msg_size = pack(*args)
        hook_ptr = foreign_seri.__packhook(msg_ptr)
        if hook_ptr:
            result = unpack(hook_ptr, msg_size)
            foreign.trash(hook_ptr)
            check_equal(name, result, args)
        else:
            foreign_seri.__unref(msg_ptr)
            foreign.trash(msg_ptr)
        # pack to bytes
        #msg_ptr, msg_size = pack(*args)
        #msg_bytes = foreign_seri.tobytes(msg_ptr, msg_size)
        #result = unpack(msg_bytes)
        #check_equal(name, result, args)
    if echo_service:
        for name in ["echo", "echo_remote", "echo_remote_hook"]:
            result = foreign.call(echo_service, name, *args)
            check_equal(name, result, args)
    result = foreign.call(".python", "echo", *args)
    check_equal(name, result, args)
    print("check ok:", *[tostring(arg) for arg in args])

def test_number():
    t = (0, 0x10, 0x100, 0x1000, 0x10000, 0x100000, -4)
    check_case(*t)
    t = (1, 0x20, 0x200, 0x2000, 0x20000, 0x200000, -8)
    check_case(*t)
    t = (1.0, -20.0, 0x200, 0x2000, 0x20000, 0x200000, -8)
    check_case(*t)

def test_dict():
    t = (1,{1:123,b"bfds":321,b"rewrw":{b"rewrw":b"rewrewrwrw"}}, 32132131321312)
    check_case(*t)
    t = ({1:123,b"bfds":321,b"rewrw":{b"rewrw":b"rewrewrwrw"}},)
    check_case(*t)

def test_list():
    t = ([1,2,3,4], [i for i in range(100)], [{321:b"rewrw"}, {b"fdsfds":234234}])
    check_case(*t)


def test_arr():
    t1 = np.array([0,1,2,3], dtype=np.bool)
    t2 = np.array([0,1,2,3], dtype=np.int8)
    t3 = np.array([0,1,2,3], dtype=np.uint8)
    t4 = np.array([0,1,2,3], dtype=np.int16)
    check_case(t1,t2,t3,t4)
    t1 = np.array([0,1,2,3], dtype=np.uint16)
    t2 = np.array([0,1,2,3], dtype=np.int32)
    t3 = np.array([0,1,2,3], dtype=np.uint32)
    t4 = np.array([0,1,2,3], dtype=np.int64)
    check_case(t1,t2,t3,t4)
    t1 = np.array([[1,2,3,4], [4,5,6,7]])
    t2 = np.array([[1,2,3,4], [4,5,6,7], [4,5,6,7]])
    t3 = np.array([[[1,2,3],[3,4,5]], [[3,4,5],[5,6,7]]])
    check_case(t1,t2,t3)
    t1 = np.arange(1000)
    t2 = np.arange(1000000)
    t3 = t2.reshape(1000,1000)
    t4 = t2.reshape(100,100,100)
    t5 = np.empty((2, 10000000))
    #t5 = np.empty((2, 100000000))
    #t6 = t5.reshape(100000,10000)
    check_case(t1,t2,t3,t4,t5)

turnon_lua()
for i in range(10000000000):
    test_number()
    test_dict()
    test_arr()
    test_list()
