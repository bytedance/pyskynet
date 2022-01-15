
from distutils.command.build_ext import build_ext
from distutils.command.build import build
import setuptools.command.install
import setuptools.command.develop
from setup_ext import *

def create_skynet_extensions_ltls(ssl=False):
    tls_library_dirs=[]
    tls_include_dirs=[SKYNET_SRC_PATH, LUA_PATH, "skynet/lualib-src/ltls"]
    if ssl and type(ssl) == str:
        search_path = [ssl, "/usr", "/usr/local", "/usr/local/opt"]
    else:
        search_path = ["/usr", "/usr/local", "/usr/local/opt"]
    ssl_found = False
    for path in search_path:
        if os.path.isfile(path+"/include/openssl/ssl.h"):
            tls_library_dirs.append(path+"/lib")
            tls_include_dirs.append(path+"/include")
            ssl_found = True
            break
    if not ssl_found:
        if ssl:
            raise Exception("'openssl/ssl.h' not found")
        else:
            print("'openssl/ssl.h' not found, skynet/luaclib/ltls.so not installed")
        return []
    else:
        ext_ltls = Extension('skynet.luaclib.ltls',
            include_dirs=tls_include_dirs,
            library_dirs=tls_library_dirs,
            sources=["skynet/lualib-src/ltls.c"],
            libraries=["ssl"],
            define_macros=MACROS,
            extra_objects=[])
        return [ext_ltls]

def create_cython_extensions():
    ext_main = Extension('pyskynet.skynet_py_main',
        include_dirs=INCLUDE_DIRS,
        sources=['src/cy_src/skynet_py_main.pyx'] +
                list_path(SKYNET_SRC_PATH, ".c", ["skynet_main.c", "skynet_start.c", "skynet_env.c"]) +
                list_path("src/skynet_modify", ".c") +
                list_path("3rd/numsky/src/skynet_foreign/", ".c") +
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

def create_tflite_extensions():
    lua_tflite = Extension('pyskynet.lualib.tflite',
        sources=["src/c_src/lua-tflite.cpp", "src/c_src/numsky/lua-numsky.cpp"],
        include_dirs=INCLUDE_DIRS + ["3rd/nn_libs/tflite/tflite_cinclude"],
        define_macros=MACROS,
        extra_compile_args=['-std=c++11'],
        libraries=LIBRARIES,
        extra_objects=[TFLITE_LIB])
    return [lua_tflite]

install_opts = {
        "ssl":False,
        "tflite":False,
        }

class build_with_numpy_cython(build):
    def finalize_options(self):
        super().finalize_options()
        self.distribution.ext_modules=create_skynet_extensions() + create_cython_extensions() + create_lua_extensions() + create_3rd_extensions() + create_skynet_extensions_ltls(install_opts["ssl"])
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


#class build_ext_purec(build_ext):
#    def get_ext_filename(self, ext_name):
#        ext_path = ext_name.split('.')
#        return os.path.join(*ext_path) + ".so"


class CommandMixin(object):
    user_options = [
        ('ssl=', None, 'build with ssl'),
        ('tflite', None, 'build with tflite'),
    ]

    def initialize_options(self):
        super().initialize_options()
        # Initialize options
        self.ssl = False
        self.tflite = False

    def finalize_options(self):
        super().finalize_options()

    def run(self):
        # Use options
        install_opts["ssl"] = self.ssl
        install_opts["tflite"] = self.tflite
        super().run()


class InstallCommand(CommandMixin, setuptools.command.install.install):
    user_options = getattr(setuptools.command.install.install, 'user_options', []) + CommandMixin.user_options

class DevelopCommand(CommandMixin, setuptools.command.develop.develop):
    user_options = getattr(setuptools.command.develop.develop, 'user_options', []) + CommandMixin.user_options


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
            ext_modules=[], # setted in build_with_numpy_cython
            cmdclass={"build_ext": build_ext_rename, "build": build_with_numpy_cython, "install":InstallCommand, "develop":DevelopCommand},
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
