
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <sstream>
#include "numsky/canvas/AstNode.h"

extern "C" {
#include "lua.h"
#include "lauxlib.h"
}

namespace numsky {
	namespace canvas {
		class EvalContext {
			public:
				lua_State*L;
			private:
				IAstNode *cursor;
				int nargs;
				int nret;
				int ft_stacki;
			public:
				EvalContext(lua_State*l, IAstNode *an, int n, int fti) : L(l), cursor(an), nargs(n), nret(0), ft_stacki(fti) {}
				inline void set_cursor(IAstNode *an) {
					cursor = an;
				}
				inline void ret_top(BaseAstNode *an, int tablei) {
					if(tablei > 0) {
						if(an->xname != NULL) {
							lua_setfield(L, tablei, an->xname);
						} else {
							lua_seti(L, tablei, luaL_len(L, tablei) + 1);
						}
					} else {
						nret ++;
					}
				}
				inline void assert_length(int set_len, int real_len) {
					if(set_len != real_len) {
						std::string detail = "expect:"+std::to_string(set_len)+" but get:"+std::to_string(real_len);
						raise("len error,", detail);
					}
				}
				inline void raise(const char* what) {
					luaL_error(L, "xml:EvalError:line:%d, %s", cursor->line, what);
				}
				inline void raise(const char* what, std::string after) {
					luaL_error(L, "xml:EvalError:line:%d, %s: %s", cursor->line, what, after.c_str());
				}
				inline std::unique_ptr<uint8_t[]> eval_bytes(int fi, int & nresults) {
                    int bottom = lua_gettop(L);
					lua_geti(L, ft_stacki, fi);
					lua_call(L, 0, LUA_MULTRET);
					nresults = lua_gettop(L) - bottom;
					std::unique_ptr<uint8_t[]> re(new uint8_t[nresults]);
					for(int i=-nresults;i<=-1;i++) {
						int isint = 0;
						re[i+nresults] = lua_tointegerx(L, i, &isint);
						if(!isint) {
							raise("int expected");
						}
					}
                    lua_settop(L, bottom);
					return re;
				}
				inline std::unique_ptr<double[]> eval_numbers(int fi, int & nresults) {
                    int bottom = lua_gettop(L);
					lua_geti(L, ft_stacki, fi);
					lua_call(L, 0, LUA_MULTRET);
					nresults = lua_gettop(L) - bottom;
					std::unique_ptr<double[]> re(new double[nresults]);
					for(int i=-nresults;i<=-1;i++) {
						int isnum = 0;
						re[i+nresults] = lua_tonumberx(L, i, &isnum);
						if(!isnum) {
							raise("number expected");
						}
					}
                    lua_settop(L, bottom);
					return re;
				}
				template <typename TFunc> inline void scalar(int fi, TFunc callback) {
                    int bottom = lua_gettop(L);
					lua_geti(L, ft_stacki, fi);
					lua_call(L, 0, LUA_MULTRET);
					int nresults = lua_gettop(L) - bottom;
                    callback(nresults);
                    lua_settop(L, bottom);
				}
				inline std::tuple<int, int> eval_any(int fi) {
                    int bottom = lua_gettop(L);
					lua_geti(L, ft_stacki, fi);
					lua_call(L, 0, LUA_MULTRET);
					return std::make_tuple<int, int>(bottom + 1, lua_gettop(L) - bottom);
				}
				inline int64_t integer(int fi) {
                    int bottom = lua_gettop(L);
					lua_geti(L, ft_stacki, fi);
					lua_call(L, 0, 1);
					if(lua_type(L, -1) != LUA_TNUMBER) {
						raise("integer expected");
					}
					int isint;
					int num = lua_tointegerx(L, -1, &isint);
					if(!isint) {
						raise("integer expected");
					}
                    lua_settop(L, bottom);
					return num;
				}
				inline double number(int fi) {
                    int bottom = lua_gettop(L);
					lua_geti(L, ft_stacki, fi);
					lua_call(L, 0, 1);
					if(lua_type(L, -1) != LUA_TNUMBER) {
						raise("sort field must be number");
					}
					double num = lua_tonumber(L, -1);
                    lua_settop(L, bottom);
					return num;
				}
				inline bool boolean(int fi) {
                    int bottom = lua_gettop(L);
					lua_geti(L, ft_stacki, fi);
					lua_call(L, 0, 1);
					bool re = lua_toboolean(L, -1);
                    lua_settop(L, bottom);
					return re;
				}
				inline void process(int fi_proc) {
					lua_geti(L, ft_stacki, fi_proc);
					lua_call(L, 0, 0);
				}
				inline void assign(int fi_assign) {
					lua_geti(L, ft_stacki, fi_assign);
					for(int i=0;i<nargs;i++) {
						lua_pushvalue(L, i+2);
					}
					lua_call(L, nargs, 0);
				}
                template <typename TFunc> inline void forin(int fi_forvar, int fi_forgen, TFunc iteratee){
                    int bottom1 = lua_gettop(L);
					lua_geti(L, ft_stacki, fi_forgen);
					lua_call(L, 0, 3);
                    int bottom2 = bottom1 + 3;
                    while (true) {
                        lua_geti(L, ft_stacki, fi_forvar);
                        lua_pushvalue(L, bottom1+1); // next
                        lua_pushvalue(L, bottom1+2); // t
                        lua_pushvalue(L, bottom1+3); // key
                        int err = lua_pcall(L, 2, LUA_MULTRET, 0);
						if(err != LUA_OK) {
							luaL_error(L, "xml:EvalError:line:%d: %s", cursor->line, lua_tostring(L, -1));
						}
                        if(lua_isnoneornil(L, bottom2 + 2)) {
                            break;
                        } else {
                            lua_copy(L, bottom2 + 2, bottom2);
                            lua_call(L, lua_gettop(L) - bottom2 - 1, 0);
                            iteratee();
                            lua_settop(L, bottom2);
                        }
                    }
                    lua_settop(L, bottom1);
                }
                template <typename TFunc> inline void foreq(int fi_forvar, int fi_forseq, TFunc iteratee){
                    int bottom1 = lua_gettop(L);
					lua_geti(L, ft_stacki, fi_forseq);
					lua_call(L, 0, 3);
					int type_step = lua_type(L, -1);
					if(lua_type(L, -3) != LUA_TNUMBER || lua_type(L, -2) != LUA_TNUMBER) {
						raise("foreq's start & stop must be number");
						return ;
					} else if(type_step != LUA_TNIL && type_step != LUA_TNUMBER) {
						raise("foreq's step must be nil or number");
						return ;
					}
					if(lua_isinteger(L, -3) && lua_isinteger(L, -2) && (lua_isinteger(L, -1) || type_step == LUA_TNIL)) {
						int64_t start = lua_tointeger(L, -3);
						int64_t stop = lua_tointeger(L, -2);
						int64_t step = 1;
						if(type_step != LUA_TNIL) {
							step = lua_tointeger(L, -1);
						}
						if(step > 0) {
							for(int64_t i=start;i<=stop;i+=step) {
								lua_geti(L, ft_stacki, fi_forvar);
								lua_pushinteger(L, i);
								lua_call(L, 1, 0);
								iteratee();
							}
						} else if(step < 0) {
							for(int64_t i=start;i>=stop;i-=step) {
								lua_geti(L, ft_stacki, fi_forvar);
								lua_pushinteger(L, i);
								lua_call(L, 1, 0);
								iteratee();
							}
						} else {
							raise("step can't be 0");
						}
					} else {
						double start = lua_tonumber(L, -3);
						double stop = lua_tonumber(L, -2);
						double step = 1.0;
						if(type_step != LUA_TNIL) {
							step = lua_tonumber(L, -1);
						}
						if(step > 0) {
							for(double i=start;i<=stop;i+=step) {
								lua_geti(L, ft_stacki, fi_forvar);
								lua_pushnumber(L, i);
								lua_call(L, 1, 0);
								iteratee();
							}
						} else if(step < 0) {
							for(double i=start;i>=stop;i-=step) {
								lua_geti(L, ft_stacki, fi_forvar);
								lua_pushnumber(L, i);
								lua_call(L, 1, 0);
								iteratee();
							}
						} else {
							raise("step can't be 0");
						}
					}
                    lua_settop(L, bottom1);
                }
				inline int get_nret() {
					return nret;
				}
		};
	}
}
