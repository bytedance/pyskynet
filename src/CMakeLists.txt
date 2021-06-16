
cmake_minimum_required(VERSION 2.8)

project(drlua)

add_definitions( -DNDEBUG )
add_definitions( -DBUILD_FOR_DRLUA )

SET(CMAKE_BUILD_TYPE "Release")

if (WIN32)
	add_definitions( -DLUA_BUILD_AS_DLL)
	add_definitions( -DTFL_COMPILE_LIBRARY )
	add_compile_options( /O2 )
else ()
	add_definitions( -DLUA_USE_POSIX)
	add_compile_options( -fPIC )
	add_compile_options( -O3 )
endif ()

# 1. lua code
set(LUA_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../3rd/lua-5.3)
file(GLOB LUA_SRC ${LUA_PATH}/*.c)
list(REMOVE_ITEM LUA_SRC ${LUA_PATH}/lua.c ${LUA_PATH}/luac.c)

# 2. foreign src
file(GLOB FOREIGN_SRC
	skynet_foreign/*.c)

# 3. numsky src
set(TINYGL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../3rd/TinyGL)
file(GLOB NUMSKY_SRC
	c_src/numsky/*.cpp
	c_src/numsky/ndarray/*.cpp
	c_src/numsky/ufunc/*.cpp
	c_src/numsky/canvas/*.cpp
	c_src/numsky/tinygl/*.cpp
	${TINYGL_PATH}/tinygl/*.cpp
	)

# 4. tflite src
set(TFLITE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../3rd/nn_libs/tflite)
set(TFLITE_SRC c_src/lua-tflite.cpp)

# 5. drlua src
set(DRLUA_SRC drlua/drlua.cpp)

# 6. rapid json src TODO
set(RAPIDJSON_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../3rd/lua-rapidjson)
file(GLOB RAPIDJSON_SRC
	${RAPIDJSON_PATH}/src/*.cpp
	)

# 7. pb src TODO
set(PB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../3rd/lua-protobuf)
set(PB_SRC ${PB_PATH}/pb.c)



include_directories(
	./c_src
	./drlua
	./
	../3rd
	../3rd/rapidxml
	${LUA_PATH}
	${TINYGL_PATH}/
	${TFLITE_PATH}/tflite_cinclude
	${RAPIDJSON_PATH}/src
	${RAPIDJSON_PATH}/rapidjson/include
	${PB_PATH}
)
#add_executable( hello ${Dosth_sources})
#add_library( numsky MODULE
#	${NUMSKY_SRC}
#	${FOREIGN_SRC}
#	)


add_library( drlua SHARED
	${NUMSKY_SRC}
	${FOREIGN_SRC}
	${DRLUA_SRC}
	${LUA_SRC}
	${TFLITE_SRC}
	${RAPIDJSON_SRC}
	${PB_SRC}
	)

if(WIN32)
	target_link_libraries( drlua ${TFLITE_PATH}/lib/win/tensorflow-lite.lib)
else ()
	target_link_libraries( drlua ${TFLITE_PATH}/lib/manylinux/libtensorflow-lite.a)
	target_link_libraries( drlua m pthread)
endif ()
