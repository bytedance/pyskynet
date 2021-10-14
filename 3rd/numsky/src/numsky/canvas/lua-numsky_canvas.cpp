#include "numsky/lua-numsky.h"

#include "numsky/ndarray/lua-numsky_ndarray.h"
#include "numsky/canvas/AstNode.h"
#include "numsky/canvas/ValNode.h"

#include "rapidxml.hpp"
#include <map>

const char unknown_name[] = "unknown xml->lua";

static int canvas_reset(lua_State* L) {
	numsky_canvas *canv = luabinding::ClassUtil<numsky_canvas>::check(L, 1);
	int arg_top = lua_gettop(L);
	lua_getuservalue(L, 1);
	lua_getfield(L, -1, "_reset");
	// push args
	for(int i=2;i<=arg_top;i++) {
		lua_pushvalue(L, i);
	}
	lua_call(L, arg_top - 1, 1);
	int ft_stacki = lua_gettop(L);
	lua_pushvalue(L, -1);
	lua_setfield(L, arg_top + 1, "_ftable");
	numsky::canvas::PostParseContext post_ctx(L, ft_stacki);
	canv->post_parse(&post_ctx);
	return 0;
}

static int canvas_render(lua_State* L) {
	numsky_canvas *canv = luabinding::ClassUtil<numsky_canvas>::check(L, 1);
	int narg = lua_gettop(L) - 1;
	lua_getuservalue(L, 1);
	lua_getfield(L, -1, "_ftable");
	numsky::canvas::EvalContext eval_ctx(L, canv, narg, lua_gettop(L));
	auto val_node = canv->eval(&eval_ctx);
	val_node->pre_eval(&eval_ctx, NULL);
	val_node->ret_eval(&eval_ctx, 0);
	delete val_node;
	return eval_ctx.get_nret();
}

static int canvas_destroy(lua_State* L){
	numsky_canvas *canv = luabinding::ClassUtil<numsky_canvas>::check(L, 1);
	delete canv;
	return 0;
}

static void canvas_lua_script_getter(lua_State* L, numsky_canvas *canv) {
	lua_pushlstring(L, canv->get_lua_script().c_str(), canv->get_lua_script().size());
}

static void canvas_xml_script_getter(lua_State* L, numsky_canvas *canv) {
	lua_pushlstring(L, canv->get_xml_script().c_str(), canv->get_xml_script().size());
}

static int canvas_dump_xml(lua_State* L){
	numsky_canvas *canv = luabinding::ClassUtil<numsky_canvas>::check(L, 1);
	std::string s = canv->dump_xml(0);
	lua_pushlstring(L, s.c_str(), s.size());
	return 1;
}

namespace luabinding {
    template <> const char* Class_<numsky_canvas>::metaname= "numsky.canvas";

    template <> void Class_<numsky_canvas>::clazz(Class_<numsky_canvas> & c) {
		c.setFieldFunction("reset", canvas_reset)
		   .setFieldFunction("render", canvas_render)
		   .setFieldFunction("dump_xml", canvas_dump_xml)
		   .setFieldProperty("lua_script", canvas_lua_script_getter, NULL)
		   .setFieldProperty("xml_script", canvas_xml_script_getter, NULL)
		   .setMetaFunction("__gc", canvas_destroy)
		   .setMetaDefaultIndex();
	}

	template <> int Class_<numsky_canvas>::ctor(lua_State* L) {
		size_t text_len;
		const char* text = luaL_checklstring(L, 1, &text_len);
		const char * chunkname;
		if(lua_isnone(L, 2)) {
			chunkname = unknown_name;
		} else {
			chunkname = luaL_checkstring(L, 2);
		}
		numsky_canvas *canv = new numsky_canvas();
		luabinding::ClassUtil<numsky_canvas>::newwrap(L, canv);
		int reti = lua_gettop(L);
		lua_newtable(L);
		lua_pushvalue(L, -1);
		lua_setuservalue(L, -3);
		canv->parse(L, text, text_len);
		int err = luaL_loadbufferx(L, canv->get_lua_script().c_str(), canv->get_lua_script().size(), chunkname, "t");
		if(err != LUA_OK) {
			return luaL_error(L, "%s", lua_tostring(L, -1));
		}
		lua_setfield(L, -2, "_reset");
		lua_pushvalue(L, reti);
		return 1;
	}
}
