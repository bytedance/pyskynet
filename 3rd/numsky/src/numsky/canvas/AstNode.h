
#pragma once

#include "numsky/canvas/IAstNode.h"
#include "numsky/canvas/CameraAstNode.h"
#include "numsky/canvas/ArrayAstNode.h"
#include "numsky/canvas/LuaAstNode.h"

namespace numsky {
	namespace canvas {
		class ConAstNode : public IAstNode {
		public:
			ExpandControl* get_expand_control() final;
			void post_parse(PostParseContext *ctx) final;
			TypeGuard * get_type_guard() final;
		};

		class VarAstNode : public ConAstNode {
			rapidxml::xml_attribute<> *xlocal;
			rapidxml::xml_attribute<> *xfunction;
			int fi_assign;
			bool setted;
		protected:
			void xparse_attr_local(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) final;
			void xparse_attr_function(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) final;
			void xparse_data(ParseContext *ctx, const char* data, int data_len, bool isPI) final;
			void xparse_finish(ParseContext *ctx, rapidxml::xml_node<> *xnode) final;
		public:
			VarAstNode(): xlocal(NULL), xfunction(NULL), fi_assign(0), setted(false) {}
			IValNode* eval(EvalContext *ctx) final;
		};

		class ProcAstNode : public ConAstNode {
			int fi_proc;
		protected:
			void xparse_data(ParseContext *ctx, const char* data, int data_len, bool isPI) final;
		public:
			ProcAstNode(): fi_proc(0) {}
			IValNode* eval(EvalContext *ctx) final;
		};

		class ScalarAstNode : public BaseAstNode {
		private:
			numsky_dtype *dtype;
			AbstractArrayAstNode *arr_parent;
			int fi_data; // lua function index for data
			char* (*cpy_func)(lua_State*L, char *left, char *right, int count);
			int (*lcpy_func)(lua_State*L, char *dataptr, int count);
		protected:
			void xparse_attr_shape(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) final;
			void xparse_data(ParseContext *ctx, const char* data, int data_len, bool isPI) final;
			void xparse_finish(ParseContext *ctx, rapidxml::xml_node<> *xnode) final;

		public:
			TypeGuard type_guard;
			friend class ScalarValNode;
			ScalarAstNode(AbstractArrayAstNode *parent, numsky_dtype* dtype_from_tag): BaseAstNode(), arr_parent(parent), fi_data(0), type_guard(&ctrl) {
				dtype = dtype_from_tag;
			}
			TypeGuard * get_type_guard() final;
			IValNode* eval(EvalContext *ctx) final;
			void post_parse(PostParseContext *ctx) final;
		};

		//
		class AnyAstNode: public BaseAstNode {
		public:
			int fi_data; // lua function index for data
		protected:
			void xparse_data(ParseContext *ctx, const char* data, int data_len, bool isPI) final;
		public:
			friend class AnyValNode;
			AnyAstNode(): BaseAstNode(), fi_data(0) {}
			IValNode* eval(EvalContext *ctx) final;
		};

	} // namespace canvas
} // namespace numsky

