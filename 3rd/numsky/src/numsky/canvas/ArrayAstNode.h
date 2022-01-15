
#pragma once

#include <memory>
#include "numsky/canvas/IAstNode.h"

namespace numsky {
	namespace canvas {
		// AbstractArray: List, Array
		class AbstractArrayAstNode : public BaseAstNode {
		protected:
			void xparse_attr_ndim(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) final;
			void xparse_finish(ParseContext *ctx, rapidxml::xml_node<> *xnode) final;
			bool xparse_has_scope() final;
		public:
			TypeGuard type_guard;
			int ndim;
			numsky_dtype *dtype;
			explicit AbstractArrayAstNode(int v_ndim) : BaseAstNode(), type_guard(&ctrl), ndim(v_ndim), dtype(NULL) {}
			BaseAstNode *xparse_child_arr(ParseContext *ctx, rapidxml::xml_node<> *xnode, int child_ndim) final;
			BaseAstNode *xparse_child_block(ParseContext *ctx, rapidxml::xml_node<> *xnode) final;
			BaseAstNode *xparse_child_scalar(ParseContext *ctx, rapidxml::xml_node<> *xnode, numsky_dtype *scalar_dtype) final;
			TypeGuard * get_type_guard() final;
			void post_parse(PostParseContext *ctx) override;
		};

		class ArrayAstNode : public AbstractArrayAstNode {
		public:
			std::unique_ptr<npy_intp[]> shape;
		protected:
			void xparse_attr_xtype(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) final;
		public:
			explicit ArrayAstNode(int v_ndim) : AbstractArrayAstNode(v_ndim), shape(new npy_intp[CANVAS_MAX_DIM]) {
				for(int i=0;i<CANVAS_MAX_DIM;i++) {
					shape[i] = 0;
				}
				dtype = numsky_get_dtype_by_char('f');
				type_guard.point_shape(shape.get());
			}
			IValNode* eval(EvalContext *ctx) final;
			void post_parse(PostParseContext *ctx) final;
		};

		class ListAstNode : public AbstractArrayAstNode {
		private:
			AbstractArrayAstNode *arr_parent;
		protected:
			void xparse_attr_xtype(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) final;
		public:
			ListAstNode(int v_ndim, AbstractArrayAstNode *parent) : AbstractArrayAstNode(v_ndim), arr_parent(parent) {
				dtype = parent->dtype;
				type_guard.point_shape(arr_parent->type_guard.shape + 1);
			}
			IValNode* eval(EvalContext *ctx) final;
			void post_parse(PostParseContext *ctx) final;
		};

		class ArrBlockAstNode : public AbstractBlockAstNode {
		private:
			AbstractArrayAstNode* arr_parent;
		public:
			TypeGuard type_guard;
			explicit ArrBlockAstNode(AbstractArrayAstNode *parent): AbstractBlockAstNode(), arr_parent(parent), type_guard(&ctrl) {
				type_guard.point_shape(parent->type_guard.shape + 1);
			}
			void post_parse(PostParseContext *ctx) final;
			TypeGuard * get_type_guard() final;
			BaseAstNode *get_parent() final;
			IValNode* eval(EvalContext *ctx) final;
		};


	} // namespace canvas
} // namespace numsky
