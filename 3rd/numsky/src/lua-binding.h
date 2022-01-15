
#pragma once

extern "C" {

#include "lua.h"
#include "lauxlib.h"
#include <stdlib.h>

}
#include <vector>
#include <tuple>
#include <type_traits>
#include <memory>
#include <sstream>
#include <string>
#include <functional>

#define UPVALUE_META_TABLE 1
#define UPVALUE_FIELD_TABLE 2
#define UPVALUE_CFUNCTION 3

#define PROPERTY_META_NAME "luabinding.property"

namespace luaUtils {

struct FormatTool {
	static void write(std::ostringstream & stream, lua_State *L, const char* s) {
		while(s[0]) {
			if(s[0] == '%') {
				if(s[1] != '%') {
					luaL_error(L, "format error, args and %% not match");
					return ;
				}
				s++;
			}
			stream << s[0];
			s++;
		}
	}
	template <typename T, typename...Args> static void write(std::ostringstream & stream, lua_State* L, const char*s, T first, Args ...args){
		while(s[0]) {
			if(s[0] == '%') {
				if(s[1] != '%') {
					switch(s[1]) {
						case 'd':
						case 's':
						case 'f':
							stream << first;
							break;
						default:
							luaL_error(L, "format error, only support %%d, %%s, %%f");
							break;
					}
					write(stream, L, s+2, args...);
					return ;
				}
				s++;
			}
			stream << s[0];
			s++;
		}
	}
};

template <typename...Args> static inline void lassert(bool value, lua_State*L, const char* format, Args...args) {
	if(value) {
		return ;
	} else {
		std::ostringstream stream;
		FormatTool::write(stream, L, format, args...);
		luaL_error(L, "%s", stream.str().c_str());
		return ;
	}
}

template <typename...Args> static inline int lerror(lua_State*L, const char* format, Args...args) {
	std::ostringstream stream;
	FormatTool::write(stream, L, format, args...);
	luaL_error(L, "%s", stream.str().c_str());
	return 0;
}

template <typename...Args> static inline std::string lformat(lua_State*L, const char* format, Args...args) {
	std::ostringstream stream;
	FormatTool::write(stream, L, format, args...);
	return stream.str();
}

} // namespace luaUtils

namespace luabinding {

template <typename T> class Class_;

template <typename T> struct ClassTypeVariable {
	using ispointer = std::true_type;
	//using TPadding = int;
};

template <typename T> struct Property_ {
	const char* name;
	void (*getter)(lua_State*L, T *userdata);
	void (*setter)(lua_State*L, T *userdata, int idx);
};

class Module_ {
	private:
		int libstacki;
	public:
		lua_State *L;
		explicit Module_(lua_State *cur_L) : L(cur_L){
			luaL_checkversion(L);
		}

		void start() {
			luaL_newmetatable(L, PROPERTY_META_NAME);
			struct luaL_Reg l_methods[] = {
				{ NULL,  NULL },
			};
			luaL_newlibtable(L, l_methods);
			libstacki = lua_gettop(L);
		}

		void finish() {
			lua_pushvalue(L, libstacki);
		}

		void setFunction(const char* name, lua_CFunction func) {
			lua_pushcfunction(L, func);
			lua_setfield(L, libstacki, name);
		}
		void setField(const char* name, const std::function<void(lua_State*L)> & push_one) {
			int top = lua_gettop(L);
			push_one(L);
			if(lua_gettop(L) - top != 1) {
				luaL_error(L, "must push one value when set field");
			}
			lua_setfield(L, libstacki, name);
		}
};


template <typename T, typename IsPointer> struct ClassUtilBase;
template <typename T> struct ClassUtilBase<T, std::true_type> {
   static inline T* check(lua_State* L, int idx) {
	  return *reinterpret_cast<T**>(luaL_checkudata(L, idx, Class_<T>::metaname));
   }
   static inline T* test(lua_State* L, int idx) {
	  auto ptr = luaL_testudata(L, idx, Class_<T>::metaname);
	  if(ptr == NULL) {
		 return NULL;
	  } else {
		 return *reinterpret_cast<T**>(ptr);
	  }
   }
   static inline void newwrap(lua_State* L, T* ptr) {
	  *(reinterpret_cast<T**>(lua_newuserdata(L, sizeof(T*)))) = ptr;
	  luaL_getmetatable(L, Class_<T>::metaname);
	  lua_setmetatable(L, -2);
   }
   static int default__gc(lua_State* L) {
	  T *obj = luabinding::ClassUtilBase<T, std::true_type>::check(L, 1);
	  auto fdestroy = reinterpret_cast<void (*)(T*)>(lua_touserdata(L, lua_upvalueindex(UPVALUE_CFUNCTION)));
	  fdestroy(obj);
	  return 0;
   }
};

template <typename T> struct ClassUtilBase<T, std::false_type> {
   static inline T* check(lua_State* L, int idx) {
	  return reinterpret_cast<T*>(luaL_checkudata(L, idx, Class_<T>::metaname));
   }
   static inline T* test(lua_State* L, int idx) {
	  return reinterpret_cast<T*>(luaL_testudata(L, idx, Class_<T>::metaname));
   }
   static void cDeleter(T* t) {
	   free(t);
   }
   static inline std::unique_ptr<T, void (*)(T*)> unialloc(int padding) {
	   std::unique_ptr<T, void (*)(T*)> ptr(reinterpret_cast<T*>(malloc(sizeof(T) + sizeof(typename ClassTypeVariable<T>::TPadding)*padding)), cDeleter);
	   return ptr;
   }
   static inline T* newalloc(lua_State* L) {
	  auto data = reinterpret_cast<T*>(lua_newuserdata(L, sizeof(T)));
	  luaL_getmetatable(L, Class_<T>::metaname);
	  lua_setmetatable(L, -2);
	  return data;
   }
   static inline T* newalloc(lua_State* L, int padding) {
	  auto data = reinterpret_cast<T*>(lua_newuserdata(L, sizeof(T) + sizeof(typename ClassTypeVariable<T>::TPadding)*padding));
	  luaL_getmetatable(L, Class_<T>::metaname);
	  lua_setmetatable(L, -2);
	  return data;
   }
};

template <typename T> struct ClassUtil : public ClassUtilBase<T, typename ClassTypeVariable<T>::ispointer> {
   static inline T* check(lua_State* L, int idx) {
	  return ClassUtilBase<T, typename ClassTypeVariable<T>::ispointer>::check(L, idx);
   }
   static inline T* test(lua_State* L, int idx) {
	  return ClassUtilBase<T, typename ClassTypeVariable<T>::ispointer>::test(L, idx);
   }
   static inline void bind(lua_State* L) {
	   luaL_getmetatable(L, luabinding::Class_<T>::metaname);
	   if(!lua_istable(L, -1)) {
		   Class_<T> c(L);
		   c.start();
		   Class_<T>::clazz(c);
		   c.finish();
		   lua_pop(L, 1);
	   } else {
		   lua_pop(L, 1);
	   }
   }
   static int default__index(lua_State* L) {
	  T *obj = luabinding::ClassUtil<T>::check(L, 1);
	  luaL_checktype(L, 2, LUA_TSTRING);
	  ClassUtil<T>::upget_function_or_property(L, obj);
	  return 1;
   }
   static int default__newindex(lua_State* L) {
	   T *obj = luabinding::ClassUtil<T>::check(L, 1);
	   luaL_checktype(L, 2, LUA_TSTRING);
	   lua_pushvalue(L, 2);
	   lua_rawget(L, lua_upvalueindex(UPVALUE_FIELD_TABLE));
	   int value_type = lua_type(L, -1);
	   if(value_type == LUA_TUSERDATA) {
		  auto property = reinterpret_cast<Property_<T>*>(lua_touserdata(L, -1));
		  if(property->setter != NULL) {
			  property->setter(L, obj, 3);
		  } else {
			  luaL_error(L, "userdata '%s' 's property '%s' can't be set", luabinding::Class_<T>::metaname, lua_tostring(L, 2));
		  }
	   } else {
		  luaL_error(L, "userdata '%s' has no property '%s' ", luabinding::Class_<T>::metaname, lua_tostring(L, 2));
	   }
	   return 0;
   }
   static void upget_function_or_property(lua_State *L, T* obj) {
	  lua_pushvalue(L, -1);
	  lua_rawget(L, lua_upvalueindex(UPVALUE_FIELD_TABLE));
	  int value_type = lua_type(L, -1);
	  if(value_type == LUA_TFUNCTION) {
		 return ;
	  } else if(value_type == LUA_TUSERDATA) {
		 auto property = reinterpret_cast<Property_<T>*>(lua_touserdata(L, -1));
		 if(property->getter != NULL) {
			property->getter(L, obj);
		 } else {
			luaL_error(L, "getter is NULL");
		 }
	  } else {
		 luaL_error(L, "userdata '%s' index unexcepted field '%s'", Class_<T>::metaname, lua_tostring(L, 2));
	  }
   }
};

template <typename T> class Class_ {
	public:
		static const char* metaname;
		static int ctor(lua_State *L);
		static void clazz(Class_<T> & c);
	private:
		int bottom;
		int timeta; // stack index for metatable
		int tifield; // stack index for field table
	public:
		lua_State *L;
		Class_<T> (lua_State* l) {
			L = l;
			int cur_type = luaL_getmetatable(L, Class_<T>::metaname);
			if(cur_type == LUA_TNIL) {
				luaL_newmetatable(L, Class_<T>::metaname);
				lua_newtable(L); // upvalue(UPVALUE_FIELD_TABLE), as an uptable for field
				lua_setfield(L, -2, "fieldtable");
				lua_pop(L, 1);
			}
		}
		Class_<T> & start(){
			bottom = lua_gettop(L);
			luaL_getmetatable(L, Class_<T>::metaname);
			lua_getfield(L, -1, "fieldtable");
			// TODO(cz) check valid?
			timeta = bottom + 1;
			tifield = bottom + 2;
			return *this;
		}
		Class_<T> & setFieldFunction(const char* name, lua_CFunction func){
			lua_pushvalue(L, timeta);
			lua_pushvalue(L, tifield);
			lua_pushcclosure(L, func, 2);
			lua_setfield(L, tifield, name);
			return *this;
		}
		Class_<T> & setFieldProperty(const char* name, void (*getter)(lua_State*, T*), void (*setter)(lua_State*, T*, int)){
			auto property = reinterpret_cast<Property_<T>*>(lua_newuserdata(L, sizeof(Property_<T>)));
			property->name = name;
			property->getter = getter;
			property->setter = setter;
			luaL_getmetatable(L, PROPERTY_META_NAME);
			lua_setmetatable(L, -2);
			lua_setfield(L, tifield, name);
			return *this;
		}
		Class_<T> & setMetaFunction(const char* name, lua_CFunction func) {
			lua_pushvalue(L, timeta);
			lua_pushvalue(L, tifield);
			lua_pushcclosure(L, func, 2);
			lua_setfield(L, timeta, name);
			return *this;
		}
		Class_<T> & setMetaDefaultGC(void (*fdestroy)(T*)){
			lua_pushvalue(L, timeta);
			lua_pushvalue(L, tifield);
			lua_pushlightuserdata(L, reinterpret_cast<void*>(fdestroy));
			lua_pushcclosure(L, ClassUtil<T>::default__gc, 3);
			lua_setfield(L, timeta, "__gc");
			return *this;
		}
		Class_<T> & setMetaDefaultIndex(){
			lua_pushvalue(L, timeta);
			lua_pushvalue(L, tifield);
			lua_pushcclosure(L, ClassUtil<T>::default__index, 2);
			lua_setfield(L, timeta, "__index");
			return *this;
		}
		Class_<T> & setMetaDefaultNewIndex(){
			lua_pushvalue(L, timeta);
			lua_pushvalue(L, tifield);
			lua_pushcclosure(L, ClassUtil<T>::default__newindex, 2);
			lua_setfield(L, timeta, "__newindex");
			return *this;
		}
		void finish(){
			lua_settop(L, bottom);
		}
};


} // namespace luabinding
