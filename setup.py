
from distutils.command.build_ext import build_ext
from setuptools import Extension, setup

from Cython.Build import cythonize
import numpy as np
import os
import sys
import re


def list_path(path, suffix, exclude_files=[]):
    if path[-1] != "/":
        path += "/"
    re = []
    for f in os.listdir(path):
        if (f.rfind(suffix) == len(f) - len(suffix)) and (f not in exclude_files):
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

INCLUDE_DIRS = [SKYNET_SRC_PATH, LUA_PATH, np.get_include(), "./src", "./src/c_src", "./skynet/lualib-src"]


def build_skynet():
    MYCFLAGS = []
    for k, v in MACROS:
        MYCFLAGS.append("-D"+k)
    MYCFLAGS = " ".join(MYCFLAGS)
    if sys.platform == "linux":
        os.system("cd skynet;MYCFLAGS='%s' make linux"%MYCFLAGS)
    elif sys.platform == "darwin":
        os.system("cd skynet;MYCFLAGS='%s' make macosx"%MYCFLAGS)

def create_cython_extensions():
    ext_main = Extension('pyskynet.skynet_py_main',
        include_dirs=INCLUDE_DIRS,
        sources=['src/cy_src/skynet_py_main.pyx'] +
                list_path(SKYNET_SRC_PATH, ".c", ["skynet_main.c", "skynet_start.c", "skynet_env.c"]) +
                list_path("src/skynet_modify", ".c") +
                list_path("src/skynet_foreign", ".c", ["test.c"]) +
                list_path(LUA_PATH, ".c", ["lua.c", "luac.c"]),
        define_macros=MACROS,
        libraries=LIBRARIES,
        extra_objects=[])

    ext_seri = Extension('pyskynet.skynet_py_foreign_seri',
        include_dirs=INCLUDE_DIRS,
        sources=['src/cy_src/skynet_py_foreign_seri.pyx'],
        define_macros=MACROS,
        libraries=LIBRARIES)

    ext_mq = Extension('pyskynet.skynet_py_mq',
        include_dirs=INCLUDE_DIRS,
        sources=['src/cy_src/skynet_py_mq.pyx'],
        define_macros=MACROS,
        libraries=LIBRARIES)

    return [ext_main, ext_mq, ext_seri]


def create_lua_extensions():
    lua_service_pyholder = Extension('pyskynet.service.pyholder',
        sources=['src/c_src/service_pyholder.c'],
        include_dirs=INCLUDE_DIRS,
        define_macros=MACROS,
        libraries=LIBRARIES)
    lua_foreign_seri = Extension('pyskynet.lualib.pyskynet.foreign_seri',
        sources=['src/c_src/lua-foreign_seri.c'],
        include_dirs=INCLUDE_DIRS,
        define_macros=MACROS,
        libraries=LIBRARIES)
    lua_modify = Extension('pyskynet.lualib.pyskynet.modify',
        sources=['src/c_src/lua-modify.c'],
        include_dirs=INCLUDE_DIRS,
        define_macros=MACROS,
        libraries=LIBRARIES)
    lua_numsky = Extension('pyskynet.lualib.numsky',
        sources=list_path("src/c_src/numsky", ".cpp") +
                list_path("src/c_src/numsky/ndarray", ".cpp") +
                list_path("src/c_src/numsky/ufunc", ".cpp") +
                list_path("src/c_src/numsky/canvas", ".cpp") +
                list_path("src/c_src/numsky/tinygl", ".cpp") +
                list_path("3rd/TinyGL/tinygl", ".cpp"),
        include_dirs=INCLUDE_DIRS + ["3rd/rapidxml", "3rd/TinyGL"],
        define_macros=MACROS,
        extra_compile_args=['-std=c++11'],
        libraries=LIBRARIES)
    lua_drlua = Extension('pyskynet.lualib.drlua',
        sources=["src/drlua/drlua.cpp"],
        include_dirs=INCLUDE_DIRS,
        define_macros=MACROS,
        extra_compile_args=['-std=c++11'],
        libraries=LIBRARIES)
    return [lua_service_pyholder, lua_foreign_seri, lua_modify, lua_numsky, lua_drlua]


def create_3rd_extensions():
    lua_pb = Extension('pyskynet.lualib.pb',
        sources=["3rd/lua-protobuf/pb.c"],
        include_dirs=["3rd/lua-protobuf", LUA_PATH])
    lua_rapidjson = Extension('pyskynet.lualib.rapidjson',
        sources=list_path("3rd/lua-rapidjson/src", ".cpp"),
        extra_compile_args=["-std=c++11"],
        include_dirs=["3rd/lua-rapidjson/src", "3rd/lua-rapidjson/rapidjson/include", LUA_PATH, "3rd/"])
    return [lua_pb, lua_rapidjson]


def create_tflite_extensions():
    lua_tflite = Extension('pyskynet.lualib.tflite',
        sources=["src/c_src/lua-tflite.cpp", "src/c_src/numsky/lua-numsky.cpp"],
        include_dirs=INCLUDE_DIRS + ["3rd/nn_libs/tflite/tflite_cinclude"],
        define_macros=MACROS,
        extra_compile_args=['-std=c++11'],
        libraries=LIBRARIES,
        extra_objects=[TFLITE_LIB])
    return [lua_tflite]


cython_extensions = cythonize(create_cython_extensions())
lua_extensions = create_lua_extensions()


class build_ext_rename(build_ext):
    def get_ext_filename(self, ext_name):
        ext_name_last = ext_name.split(".")[-1]
        is_cy_ext = False
        for cy_ext in cython_extensions:
            if ext_name_last == cy_ext.name.split(".")[-1]:
                is_cy_ext = True
        is_lua_ext = not is_cy_ext
        # for lua_ext in lua_extensions:
        #    if ext_name_last == lua_ext.name.split(".")[-1]:
        #        is_lua_ext = True
        # if is_cy_ext and is_lua_ext:
        #    raise Exception("ext_name duplicate error: %s" % ext_name)
        # if not is_cy_ext and not is_lua_ext:
        #    raise Exception("ext_name not found error: %s" % ext_name)
        if is_cy_ext:
            return super().get_ext_filename(ext_name)
        if is_lua_ext:
            ext_path = ext_name.split('.')
            return os.path.join(*ext_path) + ".so"


class build_ext_purec(build_ext):
    def get_ext_filename(self, ext_name):
        ext_path = ext_name.split('.')
        return os.path.join(*ext_path) + ".so"


def get_version():
    with open("pyskynet/__init__.py") as fo:
        data = fo.read()
        result = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', data)
        return result.group(1)


def main():
    build_skynet()
    setup(
            name="pyskynet",
            version=get_version(),
            author="cz",
            author_email="chenze.3057@bytedance.com",
            description="PySkynet is a library for using skynet in python.",
            ext_modules=cython_extensions + lua_extensions + create_3rd_extensions(),
            cmdclass={"build_ext": build_ext_rename},
            packages=["pyskynet", "skynet"],
            package_data={
                "pyskynet": ["service/*",
                            "lualib/*",
                            "lualib/*/*"],
                "skynet": ["service/*",
                          "cservice/*",
                          "luaclib/*",
                          "lualib/*",
                          "lualib/*/*",
                          "lualib/*/*/*"],
            },
            zip_safe=False,
            entry_points={
                "console_scripts": [
                    "pyskynet=pyskynet.boot:main",
                ]
            },
            install_requires=[
                "cffi ~= 1.14.2",
                "gevent >= 20.6.0",
                # "Cython ~= 0.29.21",
                # "numpy >= 1.19.0",
            ],
            python_requires='>=3.6',
        )


main()
