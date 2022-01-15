
from setuptools import Extension, setup
import os
import sys
import re

def is_suffix(fname, suffix):
    return fname.rfind(suffix) == len(fname) - len(suffix)

def list_path(path, suffix, exclude_files=[]):
    if path[-1] != "/":
        path += "/"
    re = []
    for f in os.listdir(path):
        if is_suffix(f, suffix) and (f not in exclude_files):
            re.append(path+f)
    return re

# SKYNET_3RD = "./skynet/3rd"

# JEMALLOC_STATICLIB = SKYNET_3RD + "/jemalloc/lib/libjemalloc_pic.a"
# JEMALLOC_INC = SKYNET_3RD + "/jemalloc/include/jemalloc"


LUA_PATH = "./skynet/3rd/lua"
SKYNET_SRC_PATH = "./skynet/skynet-src"

LIBRARIES = ["pthread", "m", "readline"]
MACROS = [("NOUSE_JEMALLOC", None), ("BUILD_FOR_PYSKYNET", None), ("__STDC_NO_ATOMICS__", None)]

TFLITE_LIB = None

if sys.platform == "linux":
    MACROS += [("LUA_USE_LINUX", None)]
    LIBRARIES += ["dl"]
    TFLITE_LIB = "3rd/nn_libs/tflite/lib/manylinux/libtensorflow-lite.a"
elif sys.platform == "darwin":
    MACROS += [("LUA_USE_MACOSX", None)]
    TFLITE_LIB = "3rd/nn_libs/tflite/lib/macosx/libtensorflow-lite.a"
else:
    TFLITE_LIB = "3rd/nn_libs/tflite/lib/win/tensorflow-lite.lib"
    raise Exception("no build config for platform %s" % sys.platform)

INCLUDE_DIRS = [SKYNET_SRC_PATH, LUA_PATH, "./src", "./src/c_src", "./skynet/lualib-src", "3rd/numsky/src"]


def create_skynet_extensions():
    SKYNET_CSERVICES = ["snlua", "logger", "gate", "harbor"]
    ext_cservices = []
    for cservice in SKYNET_CSERVICES:
        ext = Extension('skynet.cservice.'+cservice,
            include_dirs=INCLUDE_DIRS,
            sources=['skynet/service-src/service_'+cservice+'.c'],
            define_macros=MACROS,
            extra_objects=[])
        ext_cservices.append(ext)
    clib_skynet_src = ["lua-skynet.c",
                "lua-seri.c",
                "lua-socket.c",
                "lua-mongo.c",
                "lua-netpack.c",
                "lua-memory.c",
                "lua-multicast.c",
                "lua-cluster.c",
                "lua-crypt.c",
                "lsha1.c",
                "lua-sharedata.c",
                "lua-stm.c",
                "lua-debugchannel.c",
                "lua-datasheet.c",
                "lua-sharetable.c"]
    ext_skynet = Extension('skynet.luaclib.skynet',
        include_dirs=INCLUDE_DIRS,
        define_macros=MACROS,
        sources=["skynet/lualib-src/" + s for s in clib_skynet_src],
        extra_objects=[])
    ext_lpeg = Extension('skynet.luaclib.lpeg',
        include_dirs=[LUA_PATH, "skynet/3rd/lpeg"],
        sources=list_path("skynet/3rd/lpeg", ".c"),
        define_macros=MACROS,
        extra_objects=[])
    ext_md5 = Extension('skynet.luaclib.md5',
        include_dirs=[LUA_PATH, "skynet/3rd/lua-md5"],
        sources=list_path("skynet/3rd/lua-md5", ".c"),
        define_macros=MACROS,
        extra_objects=[])
    ext_bson = Extension('skynet.luaclib.bson',
        include_dirs=[SKYNET_SRC_PATH, LUA_PATH, "skynet/lualib-src/lua-bson"],
        sources=["skynet/lualib-src/lua-bson.c"],
        define_macros=MACROS,
        extra_objects=[])
    return ext_cservices + [ext_skynet, ext_lpeg, ext_md5, ext_bson]

def create_lua_extensions():
    lua_service_pyholder = Extension('pyskynet.service.pyholder',
        sources=['src/c_src/service_pyholder.c'],
        include_dirs=INCLUDE_DIRS,
        define_macros=MACROS,
        libraries=LIBRARIES)
    lua_foreign_seri = Extension('pyskynet.lualib.pyskynet.foreign_seri',
        sources=['3rd/numsky/src/lua-foreign_seri.c'],
        include_dirs=INCLUDE_DIRS,
        define_macros=MACROS,
        libraries=LIBRARIES)
    #lua_seri_v2 = Extension('pyskynet.lualib.pyskynet.seri_v2',
    #    sources=list_path('3rd/numsky/src/foreign_seri/', '.cpp'),
    #    include_dirs=INCLUDE_DIRS,
    #    define_macros=MACROS,
    #    extra_compile_args=['-std=c++11'],
    #    libraries=LIBRARIES)
    lua_modify = Extension('pyskynet.lualib.pyskynet.modify',
        sources=['src/c_src/lua-modify.c'],
        include_dirs=INCLUDE_DIRS,
        define_macros=MACROS,
        libraries=LIBRARIES)
    lua_numsky = Extension('pyskynet.lualib.numsky',
        sources=list_path("3rd/numsky/src/numsky/ndarray", ".cpp") +
                list_path("3rd/numsky/src/numsky/ufunc", ".cpp") +
                list_path("3rd/numsky/src/numsky/canvas", ".cpp") +
                list_path("3rd/numsky/src/numsky/tinygl", ".cpp") +
                list_path("3rd/TinyGL/tinygl", ".cpp") +
                list_path("3rd/numsky/src/numsky", ".cpp"),
        include_dirs=INCLUDE_DIRS + ["3rd/rapidxml", "3rd/TinyGL"],
        define_macros=MACROS,
        extra_compile_args=['-std=c++11'],
        libraries=LIBRARIES)
    return [lua_service_pyholder, lua_foreign_seri, lua_modify, lua_numsky]

def create_3rd_extensions():
    lua_pb = Extension('pyskynet.lualib.pb',
        sources=["3rd/lua-protobuf/pb.c"],
        include_dirs=["3rd/lua-protobuf", LUA_PATH])
    lua_rapidjson = Extension('pyskynet.lualib.rapidjson',
        sources=list_path("3rd/lua-rapidjson/src", ".cpp"),
        extra_compile_args=["-std=c++11"],
        include_dirs=["3rd/lua-rapidjson/src", "3rd/lua-rapidjson/rapidjson/include", LUA_PATH, "3rd/"])
    if sys.platform == "linux":
        lua_unqlite = Extension('pyskynet.lualib.unqlite',
            sources=["src/c_src/lua-unqlite.cpp", "3rd/unqlite/unqlite.c"],
            include_dirs=["3rd/unqlite/"]+INCLUDE_DIRS)
        return [lua_pb, lua_rapidjson, lua_unqlite]
    else:
        # lua_unqlite can't be compiled success on MacOS ...
        return [lua_pb, lua_rapidjson]

