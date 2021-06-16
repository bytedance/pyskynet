rm -r build_win
mkdir build_win
cd build_win
cmake.exe ..
cd ..
cmake.exe --build build_win --config Release
