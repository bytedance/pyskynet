
import sys
sys.path.append("../")

import pyskynet.foreign as foreign
import pyskynet.skynet_py_foreign_seri as foreign_seri

import numpy as np

def equal(left, right):
    if type(left) != type(right):
        return (left == right) == True
    elif type(left)==memoryview:
        return left.tobytes() == right.tobytes()
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


def check_case(*args):
    for name, pack, unpack in [
            ("foreign", foreign_seri.pack, foreign_seri.unpack),
            ("foreign_remote", foreign_seri.remotepack, foreign_seri.remoteunpack)]:
        # pack to capsule
        msg_ptr, msg_size = pack(*args)
        result = unpack(msg_ptr, msg_size)
        for i in range(len(args)):
            assert equal(args[i], result[i]), "check error :%s,%s,%s,%s,%s"%(name,
                    args[i],  type(args[i]),
                    result[i], type(result[i]))
        # pack to bytes
        msg_ptr, msg_size = pack(*args)
        msg_bytes = foreign_seri.tobytes(msg_ptr, msg_size)
        result = unpack(msg_bytes)
        for i in range(len(args)):
            assert equal(args[i], result[i]), "check error :%s,%s,%s,%s,%s"%(name,
                    args[i],  type(args[i]),
                    result[i], type(result[i]))
    print("check ok:", *args)

def test_number():
    t = (0, 0x10, 0x100, 0x1000, 0x10000, 0x100000, -4)
    check_case(*t)
    t = (1, 0x20, 0x200, 0x2000, 0x20000, 0x200000, -8)
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

test_number()
test_dict()
test_arr()
test_list()
