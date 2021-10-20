
from setup_ext import *
import shutil

class DumpSource(object):
    def __init__(self, copy_path="./webgl", plugin_path="./plugin"):
        self.plugin_path = plugin_path
        self.copy_path = copy_path
        self.name2ext = self._create_name2ext()
        self.include_dir_set = ['3rd/lua-rapidjson/rapidjson/include', '3rd/lua-rapidjson/src', '3rd/rapidxml',
                'skynet/3rd/lpeg', '3rd/lua-protobuf', '3rd/numsky/src', '3rd/TinyGL', '3rd/lua-5.4.3/src']
        self.copy_file_set = set()
        self.header_set = set()
        self.source_set = set()

    def run(self):
        #for name, ext in self.name2ext.items():
            #for h in ext.include_dirs:
                #self.include_dir_set.add(h)
        for include_dir in self.include_dir_set:
            self.copy_header(include_dir)

        for name, ext in self.name2ext.items():
            self.copy_src(ext)

        for header in self.header_set:
            self.fix_header(header)

        shutil.copy("3rd/numsky/src/lua-serialize.c", self.copy_path+"/lua-serialize.c")
        self.source_set.add("lua-serialize.c")

        for source in self.source_set:
            self.fix_header(source)

        for name, ext in self.name2ext.items():
            self.create_cpp(name, ext)


    def fix_header(self, header):
        l = []
        prefix = "../"*header.count("/")
        with open(self.copy_path+"/"+header, "r") as fi:
            while True:
                line = fi.readline()
                if not line:
                    break
                match_list = re.findall('^#include\s+[<"]([^<"]+)[>"]', line)
                if len(match_list) == 1 and (match_list[0] in self.header_set):
                    newline = '#include "%s"\n'%(prefix+match_list[0])
                    print(header, newline)
                    l.append(newline)
                else:
                    l.append(line)
        with open(self.copy_path+"/"+header, "w") as fo:
            for line in l:
                fo.write(line)

    def copy_file(self, src, dst):
        dst = os.path.normpath(dst)
        assert not (dst in self.copy_file_set), dst + " has been copied"
        self.copy_file_set.add(dst)
        shutil.copy(src, dst)

    def copy_header(self, include_dir="", rel_dir="."):
        copy_dir = self.copy_path+"/"+rel_dir
        if not os.path.isdir(copy_dir):
            os.mkdir(copy_dir)
        for f in os.listdir(include_dir+"/"+rel_dir):
            if f[0] == ".":
                continue
            fullpath = include_dir+"/"+rel_dir+"/"+f
            if os.path.isdir(fullpath):
                self.copy_header(include_dir, rel_dir+"/"+f)
            elif os.path.isfile(fullpath) and (is_suffix(f, ".h") or is_suffix(f, ".hpp")):
                self.copy_file(fullpath, copy_dir+"/"+f)
                self.header_set.add(os.path.normpath(rel_dir+"/"+f))

    def copy_src(self, ext):
        for src in ext.sources:
            self.copy_file(src, self.copy_path+"/"+os.path.basename(src))
            self.source_set.add(os.path.normpath(os.path.basename(src)))

    def create_cpp(self, name, ext):
        cpp_format = '#include "%s/%s"'
        c_format = 'extern "C" {\n#include "%s/%s"\n}\n'
        with open(self.plugin_path+"/"+name+"_webgl.cpp", "w") as fo:
            for src in ext.sources:
                if is_suffix(src, ".c"):
                    str_format = c_format
                else:
                    str_format = cpp_format
                fname = str_format%("../"+self.copy_path, os.path.basename(src))
                fname = os.path.normpath(fname)
                fo.write(fname)
                fo.write("\n")

    def _create_name2ext(self):
        all_exts = create_lua_extensions() + create_3rd_extensions() + create_skynet_extensions()
        all_names = ["pb", "rapidjson", "numsky", "lpeg", "foreign_seri"]
        #all_names = ["pb"]
        lua_ext = Extension('xlua',
            include_dirs=[],
            sources=list_path("3rd/lua-5.4.3/src", ".c", ["luac.c"])+
                    list_path("3rd/numsky/src/skynet_foreign/", ".c"))
        name2ext = {"xlua":lua_ext}
        for ext in all_exts:
            for name in all_names:
                if len(re.findall(name+"$", ext.name)) > 0:
                    name2ext[name] = ext
        return name2ext

    def dump_source_one(name, ext):
        print(ext.sources)

if __name__=="__main__":
    dump = DumpSource()
    dump.run()
