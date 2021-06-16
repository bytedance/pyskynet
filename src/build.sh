rm -r build_linux
mkdir build_linux
cd build_linux
cmake ..
cd ..
cmake --build build_linux
#cp build_linux/libdrlua.so unity/libdrlua.so
