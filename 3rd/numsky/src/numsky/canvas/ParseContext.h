
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <sstream>
#include <memory>
#include "rapidxml.hpp"

#include "numsky/lua-numsky.h"
#include "numsky/ndarray/lua-numsky_ndarray.h"
#include "numsky/tinygl/lua-numsky_tinygl.h"

extern "C" {
#include "lua.h"
#include "lauxlib.h"
}

#define NS_CANVAS_NAME_FUNCS "____"

namespace numsky {
	namespace canvas {
		class StreamWrapper {
			public:
				std::ostringstream stream;
				int line;
				StreamWrapper() : line(1) {}
				StreamWrapper & operator<<(int & i) {
					stream<<i;
					return *this;
				}
				StreamWrapper & operator<<(const char * s) {
					stream<<s;
					return *this;
				}
				StreamWrapper & operator<<(std::string & s) {
					for(auto iter=s.begin();iter!=s.end();iter++) {
						if(*iter=='\n') {
							line ++;
						}
					}
					stream<<s;
					return *this;
				}
				StreamWrapper & operator<<(std::string && s) {
					for(auto iter=s.begin();iter!=s.end();iter++) {
						if(*iter=='\n') {
							line ++;
						}
					}
					stream<<s;
					return *this;
				}
				void fixline(int data_line) {
					while(line < data_line) {
						line ++;
						stream<<"\n";
					}
				}
		};
		class ParseContext {
			private:
				std::string NAME_FUNCS;
			public:
				lua_State* L;
			private:
				StreamWrapper streamw;
				const char* text;
				int fi_counter;
				int si_counter;
				std::map<int, int> pos2line; // xml string's position to line number
				int calc_line(const char* ptr);
			public:
				inline int xnode_calc_line(rapidxml::xml_node<> *xnode) {
					if(xnode->name_size()!=0) {
						return calc_line(xnode->name());
					} else {
						return 1;
					}
				}
				ParseContext(lua_State*L, std::string &xml_script);
				void raise(const char *where, const char* what);
				void raise(const char *where, const char* what, const std::string &after);
				inline void put_do(rapidxml::xml_node<> *xnode) {
					if(xnode->name_size()!=0) {
						streamw.fixline(calc_line(xnode->name()));
					}
					streamw<<" do ";
				}
				inline void put_end(rapidxml::xml_node<> *xnode) {
					streamw<<" end ";
				}
				inline void put_global(const char *data, int data_len) {
					streamw.fixline(calc_line(data));
					streamw<<" "<<std::string(data, data_len)<<" ";
				}
				inline int put_static_explist(rapidxml::xml_attribute<> *xattr) {
					streamw.fixline(calc_line(xattr->name()));
					int si = -- si_counter;
					streamw<<" "<<NAME_FUNCS<<"["<<si<<"]=function() return "<<std::string(xattr->value(), xattr->value_size())<<" end ";
					return si;
				}
				template <bool USE_PI> inline int put_explist(const char *data, int data_len) {
					streamw.fixline(calc_line(data));
					int fi = ++ fi_counter;
					if(USE_PI) {
						streamw<<" "<<NAME_FUNCS<<"["<<fi<<"]=function() "<<std::string(data, data_len)<<" end ";
					} else {
						streamw<<" "<<NAME_FUNCS<<"["<<fi<<"]=function() return "<<std::string(data, data_len)<<" end ";
					}
					return fi;
				}
				inline int put_if(rapidxml::xml_attribute<> *xif, std::string & value) {
					streamw.fixline(calc_line(xif->value()));
					int fi_if = ++ fi_counter;
					streamw<<NAME_FUNCS<<"["<<fi_if<<"]=function() return "<<value<<" end ";
					return fi_if;
				}
				inline std::tuple<int, int> put_forin(rapidxml::xml_attribute<> *xfor, std::string & value) {
					streamw.fixline(calc_line(xfor->value()));
					size_t in_index = value.find(" in ");
					if(in_index == std::string::npos || in_index == 0 || in_index + 4 >= value.size()) {
						raise(xfor->name(), "for in syntax error", value);
					}
					int fi_forvar = ++ fi_counter;
					int fi_forgen = ++ fi_counter;
					std::string var = value.substr(0, in_index);
					std::string gen = value.substr(in_index+4);
					streamw<<" local "<<var<<" ";
					streamw<<NAME_FUNCS<<"["<<fi_forvar<<"]=function(...) "<<var<<"=... end ";
					streamw<<NAME_FUNCS<<"["<<fi_forgen<<"]=function() return "<<gen<<" end ";
					return std::make_tuple(fi_forvar, fi_forgen);
				}
				inline std::tuple<int, int> put_foreq(rapidxml::xml_attribute<> *xfor, std::string & value) {
					streamw.fixline(calc_line(xfor->value()));
					size_t eq_index = value.find("=");
					if(eq_index == std::string::npos || eq_index == 0 || eq_index + 1 >= value.size()) {
						raise(xfor->name(), "for eq syntax error", value);
					}
					int fi_forvar = ++ fi_counter;
					int fi_forseq = ++ fi_counter;
					// TODO(cz) check var is valid identify
					std::string var = value.substr(0, eq_index);
					std::string seq = value.substr(eq_index+1);
					streamw<<" local "<<var<<" ";
					streamw<<NAME_FUNCS<<"["<<fi_forvar<<"]=function(...) "<<var<<"=... end ";
					streamw<<NAME_FUNCS<<"["<<fi_forseq<<"]=function() return "<<seq<<" end ";
					return std::make_tuple(fi_forvar, fi_forseq);
				}
				inline int put_proc(const char *data, int data_len) {
					streamw.fixline(calc_line(data));
					std::string script(data, data_len);
					int fi_proc = ++ fi_counter;
					streamw<<NAME_FUNCS<<"["<<fi_proc<<"]=function() "<<script<<" end ";
					return fi_proc;
				}
				template <bool USE_PI> inline int put_varlocal(rapidxml::xml_attribute<> *xlocal, const char *data, int data_len) {
					streamw.fixline(calc_line(xlocal->value()));
					std::string var(xlocal->value(), xlocal->value_size());
					std::string script(data, data_len);
					int fi_assign = ++ fi_counter;
					streamw<<" local "<<var<<" ";
					streamw<<NAME_FUNCS<<"["<<fi_assign<<"]=function(...) "<<var<<"=";
					if(!USE_PI){
						streamw<<script<<" end ";
					} else {
						streamw<<"(function(...) "<<script<<" end)(...) end ";
					}
					return fi_assign;
				}
				inline void put_varfunction(rapidxml::xml_attribute<> *xfunction, const char *data, int data_len) {
					streamw.fixline(calc_line(xfunction->value()));
					std::string head(xfunction->value(), xfunction->value_size());
					std::string script(data, data_len);
					streamw<<" local function "<<head<<" "<<script<<" end ";
				}
				inline std::string finish(){
					streamw<<" return "<<NAME_FUNCS<<" ";
					return streamw.stream.str();
				}
				inline numsky_dtype* try_parse_dtype(const char *data) {
					std::string tag = data;
					for(size_t i=0;i<sizeof(NS_DTYPE_CHARS);i++) {
						numsky_dtype *try_dtype = numsky_get_dtype_by_char(NS_DTYPE_CHARS[i]);
						if(tag == try_dtype->name) {
							return try_dtype;
						}
					}
					return NULL;
				}
				inline int parse_int(const char *data) {
					if(!lua_stringtonumber(L, data)) {
						raise(data, "parse int failed");
					}
					int succ = 0;
					int re = lua_tointegerx(L, -1, &succ);
					if(!succ) {
						raise(data, "parse int failed");
					}
					lua_pop(L, 1);
					return re;
				}
		};

		class PostParseContext : public numsky::ThrowableContext {
			public:
				int ft_stacki;
				int cur_line;
				PostParseContext(lua_State*l, int fti) : ThrowableContext(l), ft_stacki(fti), cur_line(0) {}
				void throw_func(const std::string & s) final;
				inline void set_cur_line(int line) {
					cur_line = line;
				}
				inline void raise(int line, const char* what) {
					luaL_error(L, "xml:PostParseError:line:%d, %s", line, what);
				}
				inline void raise(int line, const char* what, std::string &after) {
					luaL_error(L, "xml:PostParseError:line:%d, %s: %s", line, what, after.c_str());
				}
				template <uint8_t MIN_LEN> inline int check_length(int line, int stacki) {
					int type = lua_type(L, stacki);
					int re = 0;
					if(type==LUA_TNUMBER) {
						int isnum = 0;
						re = lua_tointegerx(L, stacki, &isnum);
						if(!isnum) {
							raise(line, "check integer error, numbertointeger failed");
						}
					} else {
						if(lua_toboolean(L, stacki)){
							raise(line, "check integer error, not number && not false value");
						}
						re = 0;
					}
					if(re < MIN_LEN) {
						raise(line, "check integer error");
					}
					return re;
				}
				inline std::unique_ptr<double[]> eval_numbers(int si, int & nresults) {
                    int bottom = lua_gettop(L);
					lua_geti(L, ft_stacki, si);
					lua_call(L, 0, LUA_MULTRET);
					nresults = lua_gettop(L) - bottom;
					std::unique_ptr<double[]> re(new double[nresults]);
					for(int i=-nresults;i<=-1;i++) {
						int isnum = 0;
						re[i+nresults] = lua_tonumberx(L, i, &isnum);
						if(!isnum) {
							raise(cur_line, "number expected");
						}
					}
                    lua_settop(L, bottom);
					return re;
				}
				inline std::unique_ptr<int64_t[]> eval_integers(int si, int & nresults) {
                    int bottom = lua_gettop(L);
					lua_geti(L, ft_stacki, si);
					lua_call(L, 0, LUA_MULTRET);
					nresults = lua_gettop(L) - bottom;
					std::unique_ptr<int64_t[]> re(new int64_t[nresults]);
					for(int i=-nresults;i<=-1;i++) {
						int isint = 0;
						re[i+nresults] = lua_tointegerx(L, i, &isint);
						if(!isint) {
							raise(cur_line, "integer expected");
						}
					}
                    lua_settop(L, bottom);
					return re;
				}
				template <typename TFunc> inline void eval(int si, TFunc callback) {
                    int bottom = lua_gettop(L);
					lua_geti(L, ft_stacki, si);
					lua_call(L, 0, LUA_MULTRET);
					int nresults = lua_gettop(L) - bottom;
                    callback(nresults);
                    lua_settop(L, bottom);
				}
				inline std::unique_ptr<tinygl::Mesh> eval_mesh(int si_vertices, int si_indices) {
                    int bottom = lua_gettop(L);
					if(si_vertices == 0) {
						raise(cur_line, "mesh must set vertices");
					}
					lua_geti(L, ft_stacki, si_vertices);
					lua_call(L, 0, 1);
					std::unique_ptr<tinygl::Mesh> mesh;
					if(si_indices != 0) {
						lua_geti(L, ft_stacki, si_indices);
						lua_call(L, 0, 1);
						mesh = numsky::tinygl_mesh_new(this, -2, -1);
					} else {
						mesh = numsky::tinygl_mesh_new(this, -1, 0);
					}
                    lua_settop(L, bottom);
					return mesh;
				}
		};

	}
}
