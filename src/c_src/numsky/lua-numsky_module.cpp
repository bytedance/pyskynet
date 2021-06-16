#define LUA_LIB

#include <time.h>

#include "numsky/lua-numsky.h"
#include "numsky/ndarray/lua-numsky_ndarray.h"
#include "numsky/ufunc/lua-numsky_ufunc.h"
#include "numsky/tinygl/lua-numsky_tinygl.h"
#include "numsky/canvas/AstNode.h"
#include "numsky/lua-numsky_module.h"

template <numsky::MESH_BUILTIN_ENUM N> void binding_mesh_ctor(luabinding::Module_ & m) {
	m.setFunction(numsky::MeshEnumVariable<N>::mesh_name, numsky::ltinygl_mesh_builtin<N>);
	binding_mesh_ctor<(numsky::MESH_BUILTIN_ENUM)(N-1)>(m);
}

template <> void binding_mesh_ctor<numsky::MESH_POINT>(luabinding::Module_ & m) {
	m.setFunction(numsky::MeshEnumVariable<numsky::MESH_POINT>::mesh_name, numsky::ltinygl_mesh_builtin<numsky::MESH_POINT>);
}


extern "C" {
	static void bindClass(lua_State*L) {
		luabinding::ClassUtil<numsky_dtype>::bind(L);
		luabinding::ClassUtil<numsky_ndarray>::bind(L);
		luabinding::ClassUtil<numsky_canvas>::bind(L);
		luabinding::ClassUtil<tinygl::Mesh>::bind(L);
		luabinding::ClassUtil<tinygl::Camera>::bind(L);
	}

	LUAMOD_API int luaopen_numsky_tinygl(lua_State *L) {

		bindClass(L);

		luabinding::Module_ m(L);
		m.start();

		// tinygl mesh
		m.setFunction("mesh", luabinding::Class_<tinygl::Mesh>::ctor);

		// tinygl camera
		m.setFunction("camera", luabinding::Class_<tinygl::Camera>::ctor);

		// builtin mesh
		binding_mesh_ctor<numsky::MESH_SECTOR>(m);

		m.setField("FILL_POINT", [](lua_State*L) {
			lua_pushinteger(L, tinygl::FILL_POINT);
		});

		m.setField("FILL_LINE", [](lua_State*L) {
			lua_pushinteger(L, tinygl::FILL_LINE);
		});

		m.setField("FILL_TRIANGLE", [](lua_State*L) {
			lua_pushinteger(L, tinygl::FILL_TRIANGLE);
		});

		m.finish();

		return 1;
	}

	LUAMOD_API int luaopen_numsky(lua_State *L) {

		bindClass(L);

		luabinding::Module_ m(L);
		m.start();

		// dtype
		luaL_getmetatable(m.L, luabinding::Class_<numsky_dtype>::metaname);
		int dtype_meta_stacki = lua_gettop(L);
		for(size_t i=0;i<sizeof(NS_DTYPE_CHARS);i++){
			numsky_dtype *dtype = numsky_get_dtype_by_char(NS_DTYPE_CHARS[i]);
			m.setField(dtype->name, [&](lua_State*L) {
				lua_geti(m.L, dtype_meta_stacki, NS_DTYPE_CHARS[i]);
			});
		}
		lua_pop(L, 1);

		// ufunc
		luabinding::ClassUtil<numsky_ufunc>::bind(L);
		luaL_getmetatable(m.L, luabinding::Class_<numsky_ufunc>::metaname);
		int ufunc_meta_stacki = lua_gettop(L);
		for(int i=numsky::UFUNC_add;i<=numsky::UFUNC_atan2;i++){
			lua_geti(m.L, ufunc_meta_stacki, i);
			numsky_ufunc *ufunc=luabinding::ClassUtil<numsky_ufunc>::check(L, -1);
			m.setField(ufunc->name, [&](lua_State*L) {
				lua_geti(m.L, ufunc_meta_stacki, i);
			});
			lua_pop(L, 1);
		}
		for(int i=numsky::UFUNC_unm;i<=numsky::UFUNC_sqrt;i++){
			lua_geti(m.L, ufunc_meta_stacki, i);
			numsky_ufunc *ufunc=luabinding::ClassUtil<numsky_ufunc>::check(L, -1);
			m.setField(ufunc->name, [&](lua_State*L) {
				lua_geti(m.L, ufunc_meta_stacki, i);
			});
			lua_pop(L, 1);
		}
		lua_pop(L, 1);
		m.setFunction("sum", numsky::methods_sum);
		m.setFunction("prod", numsky::methods_prod);
		m.setFunction("any", numsky::methods_any);
		m.setFunction("all", numsky::methods_all);
		m.setFunction("max", numsky::methods_max);
		m.setFunction("min", numsky::methods_min);


		// slice
		luabinding::ClassUtil<numsky_slice>::bind(L);
		m.setFunction("slice", luabinding::Class_<numsky_slice>::ctor);
		m.setFunction("s", luabinding::Class_<numsky_slice>::ctor);


		// nditer
		luabinding::ClassUtil<numsky_nditer>::bind(L);
		m.setFunction("nditer", luabinding::Class_<numsky_nditer>::ctor);


		// tuple
		lnumsky_tuple_bind_lib(m);


		// ndarray
		m.setFunction("array", luabinding::Class_<numsky_ndarray>::ctor);

		m.setFunction("empty", numsky::ctor_empty);
		m.setFunction("zeros", numsky::ctor_zeros);
		m.setFunction("ones", numsky::ctor_ones);

		m.setFunction("arange", numsky::ctor_arange);
		m.setFunction("linspace", numsky::ctor_linspace);


		// canvas
		m.setFunction("canvas", luabinding::Class_<numsky_canvas>::ctor);

		m.finish();

		return 1;
	}

}
