
#pragma once

#include "numsky/canvas/IAstNode.h"

namespace numsky {
	namespace canvas {

		// AbstractLua: TableLua, numsky_canvas
		class AbstractLuaAstNode : public BaseAstNode {
		protected:
			BaseAstNode *xparse_child_any(ParseContext *ctx, rapidxml::xml_node<> *xnode) final;
			BaseAstNode *xparse_child_block(ParseContext *ctx, rapidxml::xml_node<> *xnode) final;
			BaseAstNode *xparse_child_table(ParseContext *ctx, rapidxml::xml_node<> *xnode) final;
			BaseAstNode *xparse_child_arr(ParseContext *ctx, rapidxml::xml_node<> *xnode, int child_ndim) final;
			BaseAstNode *xparse_child_scalar(ParseContext *ctx, rapidxml::xml_node<> *xnode, numsky_dtype *scalar_dtype) final;
			BaseAstNode *xparse_child_camera(ParseContext *ctx, rapidxml::xml_node<> *xnode) final;
			BaseAstNode* xparse_child_mesh(ParseContext *ctx, rapidxml::xml_node<> *xnode, int mesh_enum) final;
			bool xparse_has_scope() final;
		public:
			AbstractLuaAstNode(): BaseAstNode() {}
		};

		class TableAstNode : public AbstractLuaAstNode {
		public:
			TableAstNode(): AbstractLuaAstNode() {}
			IValNode* eval(EvalContext *ctx) final;
		};

		class LuaBlockAstNode : public AbstractBlockAstNode {
			AbstractLuaAstNode* lua_parent;
		public:
			LuaBlockAstNode(AbstractLuaAstNode *parent): AbstractBlockAstNode(), lua_parent(parent) {}
			BaseAstNode *get_parent() final;
			IValNode* eval(EvalContext *ctx) final;
		};
	}
}

class numsky_canvas : public numsky::canvas::AbstractLuaAstNode {
private:
	std::string xml_script; // source xml script
	std::string lua_script; // middle lua script for create upvalues & functions
	rapidxml::xml_document<> xml_doc; // xml document

protected:
	void xparse_pi_reset(numsky::canvas::ParseContext *ctx, const char*data, int data_len) final;

public:
	friend class numsky::canvas::ReturnValNode;
	numsky_canvas() : numsky::canvas::AbstractLuaAstNode() {}

	// from ast node
	numsky::canvas::IValNode* eval(numsky::canvas::EvalContext *ctx) final;
	// self
	void parse(lua_State*L, const char* text, size_t text_len);

	inline const std::string & get_xml_script() {
		return xml_script;
	}
	inline const std::string & get_lua_script() {
		return lua_script;
	}
};

