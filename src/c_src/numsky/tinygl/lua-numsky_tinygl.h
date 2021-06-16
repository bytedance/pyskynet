#pragma once

#include <vector>
#include <functional>
#include <memory>
#include <map>

#include "tinygl/math.h"
#include "tinygl/Mesh.h"
#include "tinygl/Camera.h"
#include "numsky/lua-numsky.h"


namespace numsky {
	enum MESH_BUILTIN_ENUM {
		MESH_POINT = 0,
		MESH_LINE = 1,
		MESH_RECT = 2,
		MESH_POLYGON = 3,
		MESH_CIRCLE = 4,
		MESH_SECTOR = 5,
	};

    template <MESH_BUILTIN_ENUM N> class MeshEnumVariable {
	public:
		static const char* mesh_name;
	};

	// if no indices then indices_stacki = 0
	std::unique_ptr<tinygl::Mesh> tinygl_mesh_new(ThrowableContext *ctx, int vertices_stacki, int indices_stacki);
	// with out check ysize, xsize, channel
	inline std::unique_ptr<tinygl::Camera> tinygl_camera_newunsafe(const tinygl::ScreenShape & shape) {
		int ysize = shape.ysize;
		int xsize = shape.xsize;
		int channel = shape.pixelsize;
		skynet_foreign *pixel_base = skynet_foreign_newbytes(ysize*xsize*channel);
		skynet_foreign *depth_base = skynet_foreign_newbytes(ysize*xsize*sizeof(uint16_t));
		auto decref = [](void *data) { skynet_foreign_decref(reinterpret_cast<skynet_foreign*>(data));};
		tinygl::Camera *mCamera = new tinygl::Camera(ysize, xsize, channel,
				depth_base, decref, depth_base->data,
				pixel_base, decref, pixel_base->data);
		return std::unique_ptr<tinygl::Camera>(mCamera);
	}

	// create pixel arr
	template <bool InLua> inline std::unique_ptr<numsky_ndarray, void(*)(numsky_ndarray*)> ltinygl_camera_pixel_array(lua_State*L, tinygl::Camera *camera) {
		tinygl::ScreenShape shape = camera->getScreen()->getShape();
		skynet_foreign *pixel_base = reinterpret_cast<skynet_foreign*>(camera->get_pbuf_base());
		auto pixel_arr = numsky::ndarray_new_preinit<InLua>(L, 3, 'B');
		pixel_arr->dimensions[0] = shape.d[0];
		pixel_arr->dimensions[1] = shape.d[1];
		pixel_arr->dimensions[2] = shape.d[2];
		numsky_ndarray_autostridecount(pixel_arr.get());
		skynet_foreign_incref(pixel_base);
		numsky_ndarray_refdata(pixel_arr.get(), pixel_base, pixel_base->data);
		return pixel_arr;
	}

	// create depth arr
	template <bool InLua> inline std::unique_ptr<numsky_ndarray, void(*)(numsky_ndarray*)> ltinygl_camera_depth_array(lua_State*L, tinygl::Camera *camera) {
		tinygl::ScreenShape shape = camera->getScreen()->getShape();
		skynet_foreign *depth_base = reinterpret_cast<skynet_foreign*>(camera->get_zbuf_base());
		auto depth_arr = numsky::ndarray_new_preinit<true>(L, 2, 'H');
		depth_arr->dimensions[0] = shape.d[0];
		depth_arr->dimensions[1] = shape.d[1];
		numsky_ndarray_autostridecount(depth_arr.get());
		skynet_foreign_incref(depth_base);
		numsky_ndarray_refdata(depth_arr.get(), depth_base, depth_base->data);
		return depth_arr;
	}


	template <MESH_BUILTIN_ENUM mesh_type> std::unique_ptr<tinygl::Mesh> tinygl_mesh_builtin(lua_State *L, int stacki);
	template <MESH_BUILTIN_ENUM mesh_type> inline int ltinygl_mesh_builtin(lua_State *L) {
		auto ptr = numsky::tinygl_mesh_builtin<mesh_type>(L, 1);
		luabinding::ClassUtil<tinygl::Mesh>::newwrap(L, ptr.release());
		return 1;
	}
} // namespace numsky
