
#include "numsky/canvas/AstNode.h"
#include "numsky/canvas/ValNode.h"
#include <sstream>
#include <type_traits>

namespace numsky {
	namespace canvas {
		BaseAstNode* AbstractLuaAstNode::xparse_child_table(ParseContext *ctx, rapidxml::xml_node<> *xnode) {
			BaseAstNode *child = new TableAstNode();
			child->parse_xml(ctx, xnode);
			return child;
		}
		BaseAstNode* AbstractLuaAstNode::xparse_child_any(ParseContext *ctx, rapidxml::xml_node<> *xnode) {
			BaseAstNode *child = new AnyAstNode();
			child->parse_xml(ctx, xnode);
			return child;
		}
		BaseAstNode* AbstractLuaAstNode::xparse_child_block(ParseContext *ctx, rapidxml::xml_node<> *xnode) {
			BaseAstNode *child = new LuaBlockAstNode(this);
			child->parse_xml(ctx, xnode);
			return child;
		}
		BaseAstNode* AbstractLuaAstNode::xparse_child_arr(ParseContext *ctx, rapidxml::xml_node<> *xnode, int child_ndim) {
			BaseAstNode *child = new ArrayAstNode(child_ndim);
			child->parse_xml(ctx, xnode);
			return child;
		}
		BaseAstNode* AbstractLuaAstNode::xparse_child_scalar(ParseContext *ctx, rapidxml::xml_node<> *xnode, numsky_dtype *scalar_dtype) {
			BaseAstNode *child = new ScalarAstNode(nullptr, scalar_dtype);
			child->parse_xml(ctx, xnode);
			return child;
		}
		BaseAstNode* AbstractLuaAstNode::xparse_child_camera(ParseContext *ctx, rapidxml::xml_node<> *xnode) {
			BaseAstNode *child = new CameraAstNode();
			child->parse_xml(ctx, xnode);
			return child;
		}
		BaseAstNode* AbstractLuaAstNode::xparse_child_mesh(ParseContext *ctx, rapidxml::xml_node<> *xnode, int mesh_enum) {
			ctx->raise(xnode->name(), "TODO");
			return NULL;
		}
		bool AbstractLuaAstNode::xparse_has_scope() {
			return true;
		}
	}
	// TableAstNode
	namespace canvas {
		IValNode* TableAstNode::eval(EvalContext *ctx) {
			return new TableValNode(this);
		}
	}
	// LuaBlockAstNode
	namespace canvas {
		IValNode* LuaBlockAstNode::eval(EvalContext *ctx) {
			return new LuaBlockValNode(this);
		}
		BaseAstNode* LuaBlockAstNode::get_parent() {
			return lua_parent;
		}
	}
}

void numsky_canvas::xparse_pi_reset(numsky::canvas::ParseContext *ctx, const char*data, int data_len) {
	ctx->put_global(data, data_len);
}

numsky::canvas::IValNode* numsky_canvas::eval(numsky::canvas::EvalContext *ctx) {
	return new numsky::canvas::ReturnValNode(this);
}


void numsky_canvas::parse(lua_State *L, const char* text, size_t text_len) {
	xml_script.append(text, text_len);
	numsky::canvas::ParseContext ctx(L, xml_script);
	try {
		xml_doc.parse<rapidxml::parse_no_entity_translation|rapidxml::parse_validate_closing_tags|rapidxml::parse_pi_nodes>(const_cast<char*>(xml_script.c_str()));
	} catch (rapidxml::parse_error err) {
		ctx.raise(err.where<char>(), err.what());
	}
	parse_xml(&ctx, &xml_doc);
	lua_script = ctx.finish();
}
