#include "numsky/lua-numsky.h"
#include "numsky/ndarray/lua-numsky_ndarray.h"
#include "numsky/tinygl/lua-numsky_tinygl.h"

namespace numsky {
	template <> const char* MeshEnumVariable<MESH_POINT>::mesh_name = "point";
	template <> const char* MeshEnumVariable<MESH_LINE>::mesh_name = "line";
	template <> const char* MeshEnumVariable<MESH_RECT>::mesh_name = "rect";
	template <> const char* MeshEnumVariable<MESH_POLYGON>::mesh_name = "polygon";
	template <> const char* MeshEnumVariable<MESH_CIRCLE>::mesh_name = "circle";
	template <> const char* MeshEnumVariable<MESH_SECTOR>::mesh_name = "sector";

	std::unique_ptr<tinygl::Mesh> tinygl_mesh_new(ThrowableContext *ctx, int vertices_stacki, int indices_stacki) {
		lua_State* L = ctx->L;
		std::vector<int> axis;
		axis.push_back(0);
		// item about vertices;
		auto vertices_arr_ptr = check_temp_ndarray(ctx, vertices_stacki, 'f');
		auto vertices_arr = vertices_arr_ptr.get();
		if(vertices_arr->dtype->kind != 'f') {
			ctx->throw_func("vertices array's dtype must be float32 or float64 ");
		}
		if(vertices_arr->nd != 2) {
			ctx->throw_func("vertices array's nd must be 2");
		}
		if(vertices_arr->dimensions[1] != 3) {
			ctx->throw_func("vertices array's 2nd dim must be 3");
		}
		double (*dataptr2float64)(char*) = lnumsky_template_fp(L, vertices_arr->dtype->typechar, numsky::dataptr_to_float64);
		int vlen = vertices_arr->dimensions[0];
		auto vget = [&](tinygl::V3f& v, int i){
			char *dataptr = vertices_arr->dataptr + vertices_arr->strides[0] * i;
			for(int j=0;j<3;j++) {
				v.v[j] = dataptr2float64(dataptr + vertices_arr->strides[1] * j);
			}
		};
		if(indices_stacki!=0) {
			// item about indices;
			auto indices_arr_ptr = check_temp_ndarray(ctx, indices_stacki, 'i');
			auto indices_arr = indices_arr_ptr.get();
			if(indices_arr->dtype->kind != 'u' && indices_arr->dtype->kind != 'i'){
			   	ctx->throw_func("indices array's dtype must be integer or unsigned integer");
			}
			if(indices_arr->nd != 2) {
				ctx->throw_func("indices array's nd must be 2");
			}
			if(indices_arr->dimensions[1] != 3) {
				ctx->throw_func("indices array's 2nd dim must be 3");
			}
			int64_t (*dataptr2int64)(char*) = lnumsky_template_fp(L, indices_arr->dtype->typechar, numsky::dataptr_to_int64);
			// build mesh
			int ilen = indices_arr->dimensions[0];
			std::unique_ptr<tinygl::Mesh> mesh(new tinygl::Mesh(vlen, vget, ilen, [&](tinygl::V3i& v, int i) {
					char *dataptr = indices_arr->dataptr + indices_arr->strides[0] * i;
					for(int j=0;j<3;j++) {
						int index = dataptr2int64(dataptr + indices_arr->strides[1] * j) - 1;
						if(index < 0 || index >= vlen) {
							ctx->throw_func("mesh indices range error");
						}
						v.v[j] = index;
					}
			}));
			return mesh;
		} else {
			// build mesh
			std::unique_ptr<tinygl::Mesh> mesh(new tinygl::Mesh(
						vlen, vget, 0, [](tinygl::V3i& v, int i) {}
					));
			mesh->shader.fill_type = tinygl::FILL_POINT;
			return mesh;
		}
	}

	template <> std::unique_ptr<tinygl::Mesh> tinygl_mesh_builtin<MESH_RECT>(lua_State *L, int stacki) {
		float xmin = luaL_checknumber(L, stacki);
		float ymin = luaL_checknumber(L, stacki + 1);
		float xsize = luaL_checknumber(L, stacki + 2);
		float ysize = luaL_checknumber(L, stacki + 3);
		return tinygl::Mesh::create_rect(xmin, ymin, xsize, ysize);
	}

	template <> std::unique_ptr<tinygl::Mesh> tinygl_mesh_builtin<MESH_CIRCLE>(lua_State *L, int stacki) {
		float x = luaL_checknumber(L, stacki);
		float y = luaL_checknumber(L, stacki + 1);
		float radius = luaL_checknumber(L, stacki + 2);
		return tinygl::Mesh::create_circle(x, y, radius);
	}

	template <> std::unique_ptr<tinygl::Mesh> tinygl_mesh_builtin<MESH_POLYGON>(lua_State *L, int stacki) {
		float x = luaL_checknumber(L, stacki);
		float y = luaL_checknumber(L, stacki + 1);
		float radius = luaL_checknumber(L, stacki + 2);
		unsigned char edge_num = luaL_checkinteger(L, stacki + 3);
		return tinygl::Mesh::create_polygon(x, y, radius, edge_num);
	}

	template <> std::unique_ptr<tinygl::Mesh> tinygl_mesh_builtin<MESH_SECTOR>(lua_State *L, int stacki) {
		float x = luaL_checknumber(L, stacki);
		float y = luaL_checknumber(L, stacki + 1);
		float radius = luaL_checknumber(L, stacki + 2);
		float degree = luaL_checknumber(L, stacki + 3);
		return tinygl::Mesh::create_sector(x, y, radius, degree);
	}

	template <> std::unique_ptr<tinygl::Mesh> tinygl_mesh_builtin<MESH_LINE>(lua_State *L, int stacki) {
		float from_x = luaL_checknumber(L, stacki);
		float from_y = luaL_checknumber(L, stacki + 1);
		float to_x = luaL_checknumber(L, stacki + 2);
		float to_y = luaL_checknumber(L, stacki + 3);
		return tinygl::Mesh::create_line(from_x, from_y, to_x, to_y);
	}

	template <> std::unique_ptr<tinygl::Mesh> tinygl_mesh_builtin<MESH_POINT>(lua_State *L, int stacki) {
		float x = luaL_checknumber(L, stacki);
		float y = luaL_checknumber(L, stacki + 1);
		return tinygl::Mesh::create_point(x, y);
	}
} // namespace numsky
