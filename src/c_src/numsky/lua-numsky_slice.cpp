#include "numsky/lua-numsky.h"

/****************
 * meta methods *
 ****************/

static int
lnumsky_slice__tostring(lua_State *L){
	struct numsky_slice* slice = luabinding::ClassUtil<numsky_slice>::check(L, 1);
	char buf[80];
	if(slice->start == 0 && slice->stop == 0) {
		snprintf(buf, sizeof(buf), "slice(nil,nil,%d)", slice->step);
	} else if (slice->start == 0){
		snprintf(buf, sizeof(buf), "slice(nil,%d,%d)", slice->stop, slice->step);
	} else if (slice->stop == 0) {
		snprintf(buf, sizeof(buf), "slice(%d,nil,%d)", slice->start, slice->step);
	} else {
		snprintf(buf, sizeof(buf), "slice(%d,%d,%d)", slice->start, slice->stop, slice->step);
	}
	lua_pushstring(L, buf);
	return 1;
}


/**************
 * properties *
 **************/

static void lnumsky_slice_start_setter(lua_State *L, struct numsky_slice* slice, int idx){
	int start;
	if(lua_isnil(L, idx)) {
		start = 0;
	} else {
		start = luaL_checkinteger(L, idx);
		if(start == 0) {
			luaL_error(L, "slice.start can't be 0");
			return ;
		}
	}
	slice->start = start;
}

static void lnumsky_slice_start_getter(lua_State *L, struct numsky_slice* slice){
	if(slice->start != 0) {
		lua_pushinteger(L, slice->start);
	} else {
		lua_pushnil(L);
	}
}

static void lnumsky_slice_stop_setter(lua_State *L, struct numsky_slice* slice, int idx){
	int stop;
	if(lua_isnil(L, idx)) {
		stop = 0;
	} else {
		stop = luaL_checkinteger(L, idx);
		if(stop == 0) {
			luaL_error(L, "slice.stop can't be 0");
			return ;
		}
	}
	slice->stop = stop;
}

static void lnumsky_slice_stop_getter(lua_State *L, struct numsky_slice* slice){
	if(slice->stop != 0) {
		lua_pushinteger(L, slice->stop);
	} else {
		lua_pushnil(L);
	}
}

static void lnumsky_slice_step_setter(lua_State *L, struct numsky_slice* slice, int idx){
	slice->step = luaL_checkinteger(L, idx);
	if(slice->step == 0) {
		luaL_error(L, "slice.step can't be 0");
	}
}

static void lnumsky_slice_step_getter(lua_State *L, struct numsky_slice* slice){
	lua_pushinteger(L, slice->step);
}


/***********
 * binding *
 ***********/

namespace luabinding {

    template <> void Class_<numsky_slice>::clazz(Class_<numsky_slice>&c) {
		c.setFieldProperty("start", lnumsky_slice_start_getter, lnumsky_slice_start_setter)
			.setFieldProperty("stop", lnumsky_slice_stop_getter, lnumsky_slice_stop_setter)
			.setFieldProperty("step", lnumsky_slice_step_getter, lnumsky_slice_step_setter)
			.setMetaDefaultIndex()
			.setMetaDefaultNewIndex()
			.setMetaFunction("__tostring", lnumsky_slice__tostring);
	}

    template <> int Class_<numsky_slice>::ctor(lua_State*L) {
		if(lua_isnone(L, 1)) {
			struct numsky_slice* slice = luabinding::ClassUtil<numsky_slice>::newalloc(L);
			slice->start = 0;
			slice->stop = 0;
			slice->step = 1;
			return 1;
		} else {
			int step = 0;
			if(lua_isnone(L, 3)) {
				step = 1;
			} else {
				step = luaL_checkinteger(L, 3);
				if(step == 0){
					return luaL_error(L, "slice.step can't be 0");
				}
			}
			struct numsky_slice* slice = luabinding::ClassUtil<numsky_slice>::newalloc(L);
			slice->step = step;
			lnumsky_slice_start_setter(L, slice, 1);
			lnumsky_slice_stop_setter(L, slice, 2);
			return 1;
		}
	}
} // namespace luabinding

