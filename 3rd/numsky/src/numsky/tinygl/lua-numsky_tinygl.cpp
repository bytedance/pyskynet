#include "numsky/lua-numsky.h"
#include "numsky/ndarray/lua-numsky_ndarray.h"
#include "numsky/tinygl/lua-numsky_tinygl.h"

static int lmesh_setColor(lua_State *L) {
	auto mesh = luabinding::ClassUtil<tinygl::Mesh>::check(L, 1);
	int pixelsize = lua_gettop(L) - 1;
	std::unique_ptr<unsigned char[]> color(new unsigned char[pixelsize]);
	for(int i=0;i<pixelsize;i++) {
		color[i] = luaL_checkinteger(L, i + 2);
	}
	mesh->shader.setColor(pixelsize, color.get());
	return 0;
}

static int lmesh_setLayer(lua_State *L) {
	auto mesh = luabinding::ClassUtil<tinygl::Mesh>::check(L, 1);
	int layer = luaL_checkinteger(L, 2);
	mesh->shader.layer = layer - 1;
	return 0;
}

static int lmesh_setFillType(lua_State *L) {
	auto mesh = luabinding::ClassUtil<tinygl::Mesh>::check(L, 1);
	int fill_type = luaL_checkinteger(L, 2);
	mesh->shader.fill_type = static_cast<tinygl::FILL_TYPE>(fill_type);
	return 0;
}

static int lmesh_rotation(lua_State *L) {
	auto mesh = luabinding::ClassUtil<tinygl::Mesh>::check(L, 1);
	mesh->setRotation(
			luaL_checknumber(L, 2),
			luaL_checknumber(L, 3),
			luaL_checknumber(L, 4));
	return 0;
}

static int lmesh_position(lua_State *L) {
	auto mesh = luabinding::ClassUtil<tinygl::Mesh>::check(L, 1);
	mesh->setPosition(
			luaL_checknumber(L, 2),
			luaL_checknumber(L, 3),
			luaL_checknumber(L, 4));
	return 0;
}

static int lmesh_scale(lua_State *L) {
	auto mesh = luabinding::ClassUtil<tinygl::Mesh>::check(L, 1);
	mesh->setScale(
			luaL_checknumber(L, 2),
			luaL_checknumber(L, 3),
			luaL_checknumber(L, 4));
	return 0;
}

static int lmesh_del(lua_State *L) {
	auto mesh = luabinding::ClassUtil<tinygl::Mesh>::check(L, 1);
	delete mesh;
	return 0;
}

static int lcamera_ortho(lua_State*L){
	tinygl::Camera * came = luabinding::ClassUtil<tinygl::Camera>::check(L, 1);
    came->ortho(
            luaL_checknumber(L, 2),
            luaL_checknumber(L, 3),
            luaL_checknumber(L, 4),
            luaL_checknumber(L, 5));
    return 0;
}

static int lcamera_perspective(lua_State*L){
	tinygl::Camera * came = luabinding::ClassUtil<tinygl::Camera>::check(L, 1);
    came->perspective(
            luaL_checknumber(L, 2),
            luaL_checknumber(L, 3),
            luaL_checknumber(L, 4),
            luaL_checknumber(L, 5));
	return 0;
}

static int lcamera_rotation(lua_State*L){
	tinygl::Camera * mCame = luabinding::ClassUtil<tinygl::Camera>::check(L, 1);
	mCame->setRotation(
			luaL_checknumber(L, 2),
			luaL_checknumber(L, 3),
			luaL_checknumber(L, 4));
	return 0;
}

static int lcamera_position(lua_State*L){
	tinygl::Camera * mCame = luabinding::ClassUtil<tinygl::Camera>::check(L, 1);
	mCame->setPosition(
			luaL_checknumber(L, 2),
			luaL_checknumber(L, 3),
			luaL_checknumber(L, 4));
	return 0;
}

static int lcamera_draw(lua_State*L) {
	tinygl::Camera *mCame = luabinding::ClassUtil<tinygl::Camera>::check(L, 1);
	tinygl::Mesh *mesh = luabinding::ClassUtil<tinygl::Mesh>::check(L, 2);
	mCame->draw(mesh);
	return 0;
}

static int lcamera_array(lua_State*L) {
	tinygl::Camera *camera = luabinding::ClassUtil<tinygl::Camera>::check(L, 1);
	lua_getuservalue(L, 1);
	if(lua_istable(L, -1)) {
		lua_geti(L, -1, 1);
		lua_geti(L, -2, 2);
		return 2;
	} else {
		lua_createtable(L, 2, 0);
		int usertable_stacki = lua_gettop(L);
		lua_pushvalue(L, usertable_stacki);
		lua_setuservalue(L, 1);

		numsky::ltinygl_camera_pixel_array<true>(L, camera);
		numsky::ltinygl_camera_depth_array<true>(L, camera);

		// set user value
		lua_pushvalue(L, -2);
		lua_seti(L, usertable_stacki, 1);
		lua_pushvalue(L, -1);
		lua_seti(L, usertable_stacki, 2);
		return 2;
	}
}

static int lcamera_del(lua_State*L) {
	tinygl::Camera *mCame = luabinding::ClassUtil<tinygl::Camera>::check(L, 1);
	delete mCame;
	return 0;
}

namespace luabinding {
    template <> void Class_<tinygl::Mesh>::clazz(Class_<tinygl::Mesh> & c1) {
		c1.setMetaFunction("__gc", lmesh_del)
			.setMetaDefaultIndex()
			.setFieldFunction("scale", lmesh_scale)
			.setFieldFunction("position", lmesh_position)
			.setFieldFunction("rotation", lmesh_rotation)
			.setFieldFunction("setFillType", lmesh_setFillType)
			.setFieldFunction("setLayer", lmesh_setLayer)
			.setFieldFunction("setColor", lmesh_setColor);
	}

    template <> int Class_<tinygl::Mesh>::ctor(lua_State*L) {
		std::unique_ptr<tinygl::Mesh> mesh;
		numsky::ThrowableContext ctx(L);
		if(lua_gettop(L) == 1) {
			mesh = numsky::tinygl_mesh_new(&ctx, 1, 0);
		} else if (lua_gettop(L) == 2){
			mesh = numsky::tinygl_mesh_new(&ctx, 1, 2);
		} else {
			luaL_error(L, "tinygl.mesh can only take 1 or 2 arguments");
		}
		unsigned char color[3] = {255,0,255};
		mesh->shader.setColor(3, color);

		luabinding::ClassUtil<tinygl::Mesh>::newwrap(L, mesh.release());
		return 1;
	}
} // namespace luabinding

namespace luabinding {
    template <> void Class_<tinygl::Camera>::clazz(Class_<tinygl::Camera>& c2) {
		c2.setMetaFunction("__gc", lcamera_del)
			.setMetaDefaultIndex()
			.setFieldFunction("rotation", lcamera_rotation)
			.setFieldFunction("position", lcamera_position)
			.setFieldFunction("draw", lcamera_draw)
			.setFieldFunction("array", lcamera_array)
			.setFieldFunction("ortho", lcamera_ortho)
			.setFieldFunction("perspective", lcamera_perspective);
	}

    template <> int Class_<tinygl::Camera>::ctor(lua_State*L) {
		int ysize = luaL_checkinteger(L, 1);
		int xsize = luaL_checkinteger(L, 2);
		int channel = luaL_checkinteger(L, 3);
		if(xsize <=0 || ysize <=0 || channel <=0) {
			luaL_error(L, "camera's screen's size & channel must > 0");
		}
		std::unique_ptr<tinygl::Camera> came_ptr = numsky::tinygl_camera_newunsafe(tinygl::ScreenShape(ysize, xsize, channel));
		luabinding::ClassUtil<tinygl::Camera>::newwrap(L, came_ptr.release());
		return 1;
	}
} // namespace luabinding

