
#include <sstream>
#include <type_traits>
#include <string>
#include "numsky/canvas/AstNode.h"
#include "numsky/canvas/ValNode.h"
#include "numsky/canvas/ArrayAstNode.h"

// node implement
namespace numsky {
	// AbstractArrayAstNode
	namespace canvas {
		void AbstractArrayAstNode::xparse_attr_ndim(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			int attr_ndim = ctx->parse_int(xattr->value());
			if(ndim == 0){
				ndim = attr_ndim;
			} else if(ndim != attr_ndim) {
				ctx->raise(xattr->name(), "ndim attr conflict");
			}
			if(ndim <= 0) {
				ctx->raise(xattr->name(), "arr's ndim can't be <= 0 ");
			}
		}

		BaseAstNode *AbstractArrayAstNode::xparse_child_arr(ParseContext *ctx, rapidxml::xml_node<> *xnode, int child_ndim) {
			ListAstNode *child;
			if(ndim > 0) {
				if(child_ndim > 0) {
				   	if(ndim != child_ndim + 1) {
						ctx->raise(xnode->name(), "ndim not match..");
					}
					child = new ListAstNode(child_ndim, this);
				} else {
					child = new ListAstNode(ndim - 1, this);
				}
			} else {
				if(child_ndim > 0) {
					ndim = child_ndim + 1;
					child = new ListAstNode(child_ndim, this);
				} else {
					child = new ListAstNode(0, this);
				}
			}
			child->parse_xml(ctx, xnode);
			if(ndim == 0) {
				ndim = child->ndim + 1;
			} else if(ndim != child->ndim + 1) {
				delete child;
				ctx->raise(xnode->name(), "ndim not match...");
				return NULL;
			}
			if(ndim > CANVAS_MAX_DIM) {
				ctx->raise(xnode->name(), "dim too large");
			}
			return child;
		}

		BaseAstNode *AbstractArrayAstNode::xparse_child_block(ParseContext *ctx, rapidxml::xml_node<> *xnode) {
			auto child = new ArrBlockAstNode(this);
			child->parse_xml(ctx, xnode);
			return child;
		}
		BaseAstNode *AbstractArrayAstNode::xparse_child_scalar(ParseContext *ctx, rapidxml::xml_node<> *xnode, numsky_dtype* scalar_dtype) {
			if(ndim == 0) {
				ndim = 1;
			} else if(ndim != 1) {
				ctx->raise(xnode->name(), "ndim not match");
				return NULL;
			}
			auto child = new ScalarAstNode(this, scalar_dtype);
			child->parse_xml(ctx, xnode);
			return child;
		}
		void AbstractArrayAstNode::xparse_finish(ParseContext *ctx, rapidxml::xml_node<> *xnode) {
			if(ndim == 0) {
				ctx->raise(xnode->name(), "arr's ndim can't be undeterminated");
			}
		}
		bool AbstractArrayAstNode::xparse_has_scope() {
			return true;
		}

		TypeGuard* AbstractArrayAstNode::get_type_guard() {
			return &type_guard;
		}

		void AbstractArrayAstNode::post_parse(PostParseContext* ctx){
			type_guard.eval(ctx, line, ndim);
			if(type_guard.len > 0) {
				if(type_guard.shape[0] == 0) {
					type_guard.shape[0] = type_guard.len;
				} else if(type_guard.shape[0]!=type_guard.len) {
					ctx->raise(line, "dim 0 not match with len");
				}
			}
			int len_count = 0;
			bool determine = true;
			for(auto iter=children.begin();iter!=children.end();iter++) {
				(*iter)->post_parse(ctx);
				TypeGuard *child_guard = (*iter)->get_type_guard();
				if(child_guard==NULL) {
					continue;
				} else {
					int cur_len_count = child_guard->len_count(ctx, line);
					if(cur_len_count == 0) {
						determine = false;
						continue;
					} else {
						len_count += cur_len_count;
					}
				}
			}
			if(determine) {
				if(type_guard.shape[0] == 0) {
					type_guard.shape[0] = len_count;
				} else if(type_guard.shape[0]!=len_count) {
					ctx->raise(line, "dim 0 not match after child count");
				}
				if(type_guard.len == 0) {
					type_guard.len = len_count;
				} else if(type_guard.len!=len_count){
					ctx->raise(line, "len not match after child count");
				}
			}
		}
	} // namespace canvas

	// ArrayAstNode & ListAstNode
	namespace canvas {
		IValNode* ListAstNode::eval(EvalContext *ctx) {
			return new ListValNode(this);
		}

		void ListAstNode::xparse_attr_dtype(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			ctx->raise(xattr->name(), "list arr don't need dtype, TODO deal this error with other logic");
		}

		void ListAstNode::post_parse(PostParseContext* ctx){
			AbstractArrayAstNode::post_parse(ctx);
		}

		IValNode* ArrayAstNode::eval(EvalContext *ctx) {
			return new ArrayValNode(this);
		}

		void ArrayAstNode::xparse_attr_dtype(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			std::string value(xattr->value(), xattr->value_size());
			numsky_dtype *arr_dtype = ctx->try_parse_dtype(xattr->value());
			if(arr_dtype==NULL) {
				ctx->raise(xattr->name(), "dtype unknown");
			} else {
				this->dtype = arr_dtype;
			}
		}

		void ArrayAstNode::post_parse(PostParseContext* ctx){
			for(int i=0;i<CANVAS_MAX_DIM;i++) {
				shape[i] = 0;
			}
			AbstractArrayAstNode::post_parse(ctx);
		}

	} // namespace canvas

	// BlockAstNode
	namespace canvas {
		void ArrBlockAstNode::post_parse(PostParseContext *ctx) {
			type_guard.eval(ctx, line, arr_parent->ndim - 1);
			int len_count = 0;
			bool determine = true;
			for(auto iter=children.begin();iter!=children.end();iter++) {
				(*iter)->post_parse(ctx);
				TypeGuard *child_guard = (*iter)->get_type_guard();
				if(child_guard==NULL) {
					continue;
				} else {
					int cur_len_count = child_guard->len_count(ctx, line);
					if(cur_len_count == 0) {
						determine = false;
						continue;
					} else {
						len_count += cur_len_count;
					}
				}
			}
			if(determine) {
				if(type_guard.len == 0) {
					type_guard.len = len_count;
				} else if(type_guard.len!=len_count){
					ctx->raise(line, "len not match after child count");
				}
			}
		}
		TypeGuard* ArrBlockAstNode::get_type_guard() {
			return &type_guard;
		}
		BaseAstNode* ArrBlockAstNode::get_parent() {
			return arr_parent;
		}
		IValNode* ArrBlockAstNode::eval(EvalContext *ctx) {
			return new ArrBlockValNode(this);
		}
	} // namespace canvas
} // namespace numsky
