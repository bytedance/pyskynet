
#include <sstream>
#include <type_traits>
#include <algorithm>
#include "numsky/canvas/AstNode.h"
#include "numsky/canvas/ValNode.h"

// node implement
namespace numsky {
	namespace canvas {
		ExpandControl *ConAstNode::get_expand_control() {
			return NULL;
		}
		void ConAstNode::post_parse(PostParseContext *ctx) {}
		TypeGuard * ConAstNode::get_type_guard() {
			return NULL;
		}
	} // namespace canvas

	// VarAstNode
	namespace canvas {
		IValNode* VarAstNode::eval(EvalContext *ctx) {
			if(xlocal != NULL) {
				ctx->assign(fi_assign);
			}
			return NULL;
		}
		void VarAstNode::xparse_attr_xlocal(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			if(xlocal != NULL || xfunction != NULL) {
				ctx->raise(xattr->name(), "var has put local or function");
			} else {
				xlocal = xattr;
			}
		}
		void VarAstNode::xparse_attr_xfunction(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			if(xlocal != NULL || xfunction != NULL) {
				ctx->raise(xattr->name(), "var has put local or function");
			} else {
				xfunction = xattr;
			}
		}
		void VarAstNode::xparse_data(ParseContext *ctx, const char* data, int data_len, bool isScope) {
			if(setted) {
				ctx->raise(data, "var can't has multi ");
			} else {
				setted = true;
				if(xlocal != NULL) {
					if(isScope) {
						fi_assign = ctx->put_varlocal<true>(xlocal, data, data_len);
					} else {
						fi_assign = ctx->put_varlocal<false>(xlocal, data, data_len);
					}
				} else if(xfunction != NULL) {
					ctx->put_varfunction(xfunction, data, data_len);
				} else {
					ctx->raise(data, "var must set function or local attr");
				}
			}
		}
		void VarAstNode::xparse_finish(ParseContext *ctx, rapidxml::xml_node<> *xnode) {
			if(!setted) {
				ctx->raise(xnode->name(), "invalid var element (must has (function or local attribute) and data )");
			}
		}
	} // namespace canvas
	// ProcAstNode
	namespace canvas {
		IValNode* ProcAstNode::eval(EvalContext *ctx) {
			if(fi_proc>0){
				ctx->process(fi_proc);
			}
			return NULL;
		}
		void ProcAstNode::xparse_data(ParseContext *ctx, const char* data, int data_len, bool isScope) {
			if(isScope) {
				ctx->raise(data, "don't use scope in proc ");
			}
			if(fi_proc > 0) {
				ctx->raise(data, "proc data has been setted");
			} else {
				fi_proc = ctx->put_proc(data, data_len);
			}
		}
	} // namespace canvas

	// ScalarAstNode
	namespace canvas {
		IValNode* ScalarAstNode::eval(EvalContext *ctx) {
			return new ScalarValNode(this);
		}
		TypeGuard * ScalarAstNode::get_type_guard() {
			return &type_guard;
		}
		void ScalarAstNode::xparse_attr_Shape(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			ctx->raise(xattr->name(), "scalar can't has shape");
		}
		void ScalarAstNode::xparse_data(ParseContext *ctx, const char*data, int data_len, bool isScope) {
			if(fi_data!=0) {
				ctx->raise(data, "scalar's data has been setted");
			} else {
				if(isScope) {
					fi_data = ctx->put_explist<true>(data, data_len);
				} else {
					fi_data = ctx->put_explist<false>(data, data_len);
				}
			}
		}
		template <typename Tdst, typename Tsrc> static char* T_cpy_func(lua_State*L, char *left, char *right, int count) {
			Tsrc *src = reinterpret_cast<Tsrc*>(right);
			Tdst *dst = reinterpret_cast<Tdst*>(left);
			return reinterpret_cast<char*>(std::copy(src, src+count, dst));
		}
		static char* Error_cpy_func(lua_State*L, char *left, char *right, int count) {
			luaL_error(L, "fatal error, unexception case happen");
			return NULL;
		}

		// if success, return 0, else return index for first error
		template <typename T> static int T_lcpy_func(lua_State*L, char *dataptr, int count) {
			for(int i=count;i>0;i--) {
				int t = lua_type(L, -i);
				if(t == LUA_TNUMBER && !std::is_same<T, bool>::value) {
					numsky::dataptr_cast<T>(dataptr) = lua_tonumber(L, -i);
				} else if(t == LUA_TBOOLEAN && std::is_same<T, bool>::value) {
					numsky::dataptr_cast<T>(dataptr) = lua_toboolean(L, -i);
				} else {
					// return index for error
					return count-i+1;
				}
				dataptr += sizeof(T);
			}
			return 0;
		}

		void ScalarAstNode::xparse_finish(ParseContext *ctx, rapidxml::xml_node<> *xnode) {
			if (dtype == NULL) {
				ctx->raise(xnode->name(), "scalar's dtype must set");
				return ;
			}
			if(arr_parent == nullptr) {
				cpy_func = Error_cpy_func;
			} else if (arr_parent->dtype == NULL){
				ctx->raise(xnode->name(), "scalar's parent's dtype not setted");
				cpy_func = Error_cpy_func;
			} else {
				cpy_func = lnumsky_template_fp2(ctx->L, arr_parent->dtype->typechar, dtype->typechar, T_cpy_func);
			}
			lcpy_func = lnumsky_template_fp(ctx->L, dtype->typechar, T_lcpy_func);
			if(fi_data==0) {
				ctx->raise(xnode->name(), "scalar must set data");
			}
		}
		void ScalarAstNode::post_parse(PostParseContext *ctx) {
			type_guard.eval(ctx, line, 0);
		}
	} // namespace canvas

	// LuaAstNode
	namespace canvas {
		void AnyAstNode::xparse_data(ParseContext *ctx, const char* data, int data_len, bool isScope) {
			if(fi_data!=0) {
				ctx->raise(data, "<any> node's data has been setted");
			} else {
				if(isScope) {
					fi_data = ctx->put_explist<true>(data, data_len);
				} else {
					fi_data = ctx->put_explist<false>(data, data_len);
				}
			}
		}
		IValNode* AnyAstNode::eval(EvalContext *ctx) {
			return new AnyValNode(this);
		}
	} // namespace canvas
} // namespace numsky
