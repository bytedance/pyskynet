
import time
import sys
sys.path.append("../../")

import pyskynet
import pyskynet.foreign as foreign

import numpy as np

lua_service = None

def start():
    global lua_service
    pyskynet.start()
    lua_service = pyskynet.scriptservice("""
            local pyskynet = require "pyskynet"
            local foreign = require "pyskynet.foreign"
            local ns = require "numsky"
            local function replace_slice(t)
                if type(t) == "table" then
                    if t.step then
                        return ns.slice(t.start, t.stop, t.step)
                    else
                        return ns.slice(t.start, t.stop)
                    end
                else
                    return t
                end
            end
            foreign.dispatch("oper2", function(arr1, op, arr2)
                if op == "+" then
                    return arr1 + arr2
                elseif op == "-" then
                    return arr1 - arr2
                elseif op == "*" then
                    return arr1 * arr2
                elseif op == "/" then
                    return arr1 / arr2
                elseif op == "//" then
                    return arr1 // arr2
                elseif op == "^" then
                    return arr1 ^ arr2
                elseif op == "%" then
                    return arr1 % arr2
                end
                if op == "|" then
                    return arr1 | arr2
                elseif op == "&" then
                    return arr1 & arr2
                elseif op == "~" then
                    return arr1 ~ arr2
                elseif op == ">>" then
                    return arr1 >> arr2
                elseif op == "<<" then
                    return arr1 << arr2
                end
            end)
            foreign.dispatch("index_one", function(arr, i)
                i = replace_slice(i)
                return arr[i]
            end)
            foreign.dispatch("newindex_one", function(arr, i, value)
                i = replace_slice(i)
                arr[i] = value
                return arr
            end)
            local function replace_slice_table(l)
                for i=1, #l do
                    if type(l[i]) == "table" then
                        l[i] = replace_slice(l[i])
                    end
                end
                return l
            end
            foreign.dispatch("index_table", function(arr, l)
                l = replace_slice_table(l)
                return arr[l]
            end)
            foreign.dispatch("newindex_table", function(arr, l, value)
                l = replace_slice_table(l)
                arr[l] = value
                return arr
            end)
            pyskynet.start(function()
            end)
    """)


def index_tolua(i):
    if type(i) == int:
        if i < 0:
            return i
        else:
            return i + 1
    elif type(i) == slice:
        if i.start == None:
            start = None
        elif i.start >= 0:
            start = i.start + 1
        else:
            start = i.start
        stop = i.stop
        return {"start":start, "stop":stop, "step":i.step}
    elif type(i) == np.ndarray:
        if i.dtype.kind == 'i':
            k = i.copy()
            k[k>=0] += 1
            return k
        elif i.dtype.kind == 'u':
            return i + 1
        else:
            return i
    elif type(i) == tuple:
        i = list(i)
        for n, k in enumerate(i):
            i[n] = index_tolua(k)
        return i
    else:
        raise Exception("unvalid indexing key")

class TestObject(object):
    def __init__(self, arr=None):
        self.arr = arr.copy()

    def oper(self, op, right):
        arr = self.arr.copy()
        if type(right) == np.ndarray:
            case = f"({arr.shape},{arr.dtype}) {op} ({right.shape},{right.dtype})"
        else:
            case = f"({arr.shape},{arr.dtype}) {op} {right}"
        py_re_arr = None
        if op == "+":
            py_re_arr = arr + right
        elif op == "-":
            py_re_arr = arr - right
        elif op == "*":
            py_re_arr = arr * right
        elif op == "/":
            py_re_arr = arr / right
        elif op == "//":
            py_re_arr = arr // right
        elif op == "^":
            py_re_arr = arr ** right
        elif op == "%":
            py_re_arr = arr % right
        else:
            pass
        if op == "|":
            py_re_arr = arr | right
        elif op == "&":
            py_re_arr = arr & right
        elif op == "~":
            py_re_arr = arr ^ right
        elif op == ">>":
            py_re_arr = arr >> right
        elif op == "<<":
            py_re_arr = arr << right
        else:
            pass
        try:
            lua_re_arr, = foreign.call(lua_service, "oper2", arr, op, right)
            #if lua_re_arr.dtype != py_re_arr.dtype:
                #print(f"check fail dtype not match lua:dtype:{lua_re_arr.dtype} py:dtype:{py_re_arr.dtype}", case, )
            if lua_re_arr.shape != py_re_arr.shape:
                print(f"check fail dtype not match lua:shape:{lua_re_arr.shape} py:shape:{py_re_arr.shape}", case)
            elif not np.all(lua_re_arr == py_re_arr):
                print("check fail", case)
                print(arr, right)
                print("lua:")
                print(lua_re_arr)
                print("py:")
                print(py_re_arr)
            else:
                print("check ok", case)
        except Exception as e:
            print("throw except", case, e)
        return py_re_arr

    def roper(self, op, left):
        arr = self.arr.copy()
        if type(left) == np.ndarray:
            case = f"({left.shape},{left.dtype}) {op} ({arr.shape},{arr.dtype})"
        else:
            case = f"{left} {op} ({arr.shape},{arr.dtype})"
        py_re_arr = None
        if op == "+":
            py_re_arr = left + arr
        elif op == "-":
            py_re_arr = left - arr
        elif op == "*":
            py_re_arr = left * arr
        elif op == "/":
            py_re_arr = left / arr
        elif op == "//":
            py_re_arr = left // arr
        elif op == "^":
            py_re_arr = left ** arr
        elif op == "%":
            py_re_arr = left % arr
        else:
            pass
        if op == "|":
            py_re_arr = left | arr
        elif op == "&":
            py_re_arr = left & arr
        elif op == "~":
            py_re_arr = left ^ arr
        elif op == ">>":
            py_re_arr = left >> arr
        elif op == "<<":
            py_re_arr = left << arr
        else:
            pass
        try:
            lua_re_arr, = foreign.call(lua_service, "oper2", left, op, arr)
            #if lua_re_arr.dtype != py_re_arr.dtype:
                #print(f"check fail dtype not match lua:dtype:{lua_re_arr.dtype} py:dtype:{py_re_arr.dtype}", case)
            if lua_re_arr.shape != py_re_arr.shape:
                print(f"check fail dtype not match lua:shape:{lua_re_arr.shape} py:shape:{py_re_arr.shape}", case)
            elif not np.all(lua_re_arr == py_re_arr):
                print("check fail", case)
                print(left, arr)
                print("lua:")
                print(lua_re_arr)
                print("py:")
                print(py_re_arr)
            else:
                print("check ok", case)
        except Exception as e:
            print("throw except", case, e)
        return py_re_arr

    # test op right
    def __add__(self, right):
        return self.oper("+", right)

    def __sub__(self, right):
        return self.oper("-", right)

    def __mul__(self, right):
        return self.oper("*", right)

    def __truediv__(self, right):
        return self.oper("/", right)

    def __floordiv__(self, right):
        return self.oper("//", right)

    def __pow__(self, right):
        return self.oper("^", right)

    def __mod__(self, right):
        return self.oper("%", right)

    def __and__(self, right):
        return self.oper("&", right)

    def __or__(self, right):
        return self.oper("|", right)

    def __xor__(self, right):
        return self.oper("~", right)

    def __lshift__(self, right):
        return self.oper("<<", right)

    def __rshift__(self, right):
        return self.oper(">>", right)

    # left op test
    def __radd__(self, right):
        return self.roper("+", right)

    def __rsub__(self, right):
        return self.roper("-", right)

    def __rmul__(self, right):
        return self.roper("*", right)

    def __rtruediv__(self, right):
        return self.roper("/", right)

    def __rfloordiv__(self, right):
        return self.roper("//", right)

    def __rpow__(self, right):
        return self.roper("^", right)

    def __rmod__(self, right):
        return self.roper("%", right)

    def __rand__(self, right):
        return self.roper("&", right)

    def __ror__(self, right):
        return self.roper("|", right)

    def __rxor__(self, right):
        return self.roper("~", right)

    def __rlshift__(self, right):
        return self.roper("<<", right)

    def __rrshift__(self, right):
        return self.roper(">>", right)

    def __getitem__(self, k):
        arr = self.arr.copy()
        func = None
        if type(k) != tuple:
            func = "index_one"
        else:
            func = "index_table"
        case = f"arr:shape={self.arr.shape} dtype={self.arr.dtype}. key:{k}"
        try:
            re_arr, = foreign.call(lua_service, func, arr, index_tolua(k))
            if np.all(arr[k] == re_arr):
                print("check ok", case)
            else:
                print("check fail", case)
        except Exception as e:
            print("throw except", case, e)
        return arr[k]

    def __setitem__(self, k, v):
        arr = self.arr.copy()
        func = None
        if type(k) != tuple:
            func = "newindex_one"
        else:
            func = "newindex_table"
        case = f"arr:shape={self.arr.shape} dtype={self.arr.dtype}. key:{k}, value:{v.shape} {v.dtype}"
        try:
            re_arr, = foreign.call(lua_service, func, arr.copy(), index_tolua(k), v)
            arr[k] = v
            if np.all(arr == re_arr):
                print("check ok", case)
            else:
                print("check fail", case)
        except Exception as e:
            print("throw except", case, e)

