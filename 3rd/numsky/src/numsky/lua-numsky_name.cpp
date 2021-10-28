
#include "numsky/lua-numsky.h"
#include "numsky/ndarray/lua-numsky_ndarray.h"
#include "numsky/ufunc/lua-numsky_ufunc.h"
#include "numsky/tinygl/lua-numsky_tinygl.h"
#include "numsky/canvas/AstNode.h"

namespace luabinding {
    template <> const char* Class_<numsky_dtype>::metaname= "numsky.dtype";
    template <> const char* Class_<numsky_ndarray>::metaname= NS_ARR_METANAME;
    template <> const char* Class_<numsky_nditer>::metaname= "numsky.nditer";
    template <> const char* Class_<numsky_slice>::metaname= "numsky.slice";
    template <> const char* Class_<numsky_ufunc>::metaname= "numsky.ufunc";

    template <> const char* Class_<numsky_canvas>::metaname= "numsky.canvas";

    template <> const char* Class_<tinygl::Mesh>::metaname= "tinygl.mesh";
	template <> const char* Class_<tinygl::Camera>::metaname= "tinygl.camera";
}

namespace numsky {
	template <> const char* MeshEnumVariable<MESH_POINT>::mesh_name = "Point";
	template <> const char* MeshEnumVariable<MESH_LINE>::mesh_name = "Line";
	template <> const char* MeshEnumVariable<MESH_RECT>::mesh_name = "Rect";
	template <> const char* MeshEnumVariable<MESH_POLYGON>::mesh_name = "Polygon";
	template <> const char* MeshEnumVariable<MESH_CIRCLE>::mesh_name = "Circle";
	template <> const char* MeshEnumVariable<MESH_SECTOR>::mesh_name = "Sector";
}
