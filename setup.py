
from distutils.command.build_ext import build_ext
from distutils.command.build import build
from setuptools import Extension, setup

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

INCLUDE_DIRS = [SKYNET_SRC_PATH, LUA_PATH, "./src", "./src/c_src", "./skynet/lualib-src"]


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
    ext_ltls = Extension('skynet.luaclib.ltls',
        include_dirs=[SKYNET_SRC_PATH, LUA_PATH, "skynet/lualib-src/ltls"],
        sources=["skynet/lualib-src/ltls.c"],
        libraries=["ssl"],
        define_macros=MACROS,
        extra_objects=[])
    return ext_cservices + [ext_skynet, ext_lpeg, ext_md5, ext_bson, ext_ltls]


def create_cython_extensions():
    ext_main = Extension('pyskynet.skynet_py_main',
        include_dirs=INCLUDE_DIRS,
        sources=['src/cy_src/skynet_py_main.pyx'] +
                list_path(SKYNET_SRC_PATH, ".c", ["skynet_main.c", "skynet_start.c", "skynet_env.c"]) +
                list_path("src/skynet_modify", ".c") +
                list_path("src/skynet_foreign", ".c", ["test.c"]) +
                list_path(LUA_PATH, ".c", ["lua.c", "luac.c"]),
        depends=['src/cy_src/skynet_py.pxd'],
        define_macros=MACROS,
        libraries=LIBRARIES,
        extra_objects=[])

    ext_seri = Extension('pyskynet.skynet_py_foreign_seri',
        include_dirs=INCLUDE_DIRS,
        sources=['src/cy_src/skynet_py_foreign_seri.pyx'],
        depends=['src/cy_src/skynet_py.pxd'],
        define_macros=MACROS,
        libraries=LIBRARIES)

    ext_mq = Extension('pyskynet.skynet_py_mq',
        include_dirs=INCLUDE_DIRS,
        sources=['src/cy_src/skynet_py_mq.pyx'],
        depends=['src/cy_src/skynet_py.pxd'],
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
    return [lua_service_pyholder, lua_foreign_seri, lua_modify, lua_numsky]


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


class build_with_numpy_cython(build):
    def finalize_options(self):
        super().finalize_options()
        import numpy
        for extension in self.distribution.ext_modules:
            np_inc = numpy.get_include()
            if not (np_inc in extension.include_dirs):
                extension.include_dirs.append(np_inc)
        from Cython.Build import cythonize
        self.distribution.ext_modules = cythonize(self.distribution.ext_modules, language_level=3)


class build_ext_rename(build_ext):
    def get_ext_filename(self, ext_name):
        ext_name_last = ext_name.split(".")[-1]
        # cython library start with skynet_py
        if ext_name_last.find("skynet_py_") == 0:
            # for cython library
            return super().get_ext_filename(ext_name)
        else:
            # for lua library
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
    setup(
            name="pyskynet",
            version=get_version(),
            author="cz",
            author_email="chenze.3057@bytedance.com",
            license='MIT',
            description="PySkynet is a library for using skynet in python.",
            ext_modules=create_skynet_extensions() + create_cython_extensions() + create_lua_extensions() + create_3rd_extensions(),
            cmdclass={"build_ext": build_ext_rename, "build": build_with_numpy_cython},
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
                "numpy",
            ],
            url='https://github.com/bytedance/pyskynet',
            setup_requires=["cython", "numpy"],
            python_requires='>=3.5',
        )


main()
