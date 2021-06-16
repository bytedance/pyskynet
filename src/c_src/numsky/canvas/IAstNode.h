
#pragma once
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <sstream>

#include "numsky/lua-numsky.h"
#include "numsky/ndarray/lua-numsky_ndarray.h"
#include "numsky/canvas/ParseContext.h"
#include "numsky/canvas/TypeGuard.h"
#include "numsky/canvas/ExpandControl.h"
#include "rapidxml.hpp"

#define CANVAS_MAX_DIM 16

namespace numsky {
	namespace canvas {
		class IValNode;
		class EvalContext;

		class IAstNode {
		public:
			int line;
			IAstNode() : line(0) {}
			friend class AttrParse;
		protected:
			// attrs
			virtual void xparse_attr_name(ParseContext *ctx, rapidxml::xml_attribute<> *xattr);

			// control
			virtual void xparse_attr_for(ParseContext *ctx, rapidxml::xml_attribute<> *xattr);
			virtual void xparse_attr_if(ParseContext *ctx, rapidxml::xml_attribute<> *xattr);
			virtual void xparse_attr_sort(ParseContext *ctx, rapidxml::xml_attribute<> *xattr);

			// type
			virtual void xparse_attr_dtype(ParseContext *ctx, rapidxml::xml_attribute<> *xattr);
			virtual void xparse_attr_ndim(ParseContext *ctx, rapidxml::xml_attribute<> *xattr);
			virtual void xparse_attr_len(ParseContext *ctx, rapidxml::xml_attribute<> *xattr);
			virtual void xparse_attr_count(ParseContext *ctx, rapidxml::xml_attribute<> *xattr);
			virtual void xparse_attr_shape(ParseContext *ctx, rapidxml::xml_attribute<> *xattr);

			// var
			virtual void xparse_attr_local(ParseContext *ctx, rapidxml::xml_attribute<> *xattr);
			virtual void xparse_attr_function(ParseContext *ctx, rapidxml::xml_attribute<> *xattr);

			// graphic
			virtual void xparse_attr_rot(ParseContext *ctx, rapidxml::xml_attribute<> *xattr);
			virtual void xparse_attr_pos(ParseContext *ctx, rapidxml::xml_attribute<> *xattr);
			// graphic camera
			virtual void xparse_attr_ortho(ParseContext *ctx, rapidxml::xml_attribute<> *xattr);
			virtual void xparse_attr_perspective(ParseContext *ctx, rapidxml::xml_attribute<> *xattr);
			// graphic mesh
			virtual void xparse_attr_scale(ParseContext *ctx, rapidxml::xml_attribute<> *xattr);
			virtual void xparse_attr_layer(ParseContext *ctx, rapidxml::xml_attribute<> *xattr);
			virtual void xparse_attr_fill(ParseContext *ctx, rapidxml::xml_attribute<> *xattr);
			virtual void xparse_attr_pivot(ParseContext *ctx, rapidxml::xml_attribute<> *xattr);
			virtual void xparse_attr_size(ParseContext *ctx, rapidxml::xml_attribute<> *xattr);
			virtual void xparse_attr_vertices(ParseContext *ctx, rapidxml::xml_attribute<> *xattr);
			virtual void xparse_attr_indices(ParseContext *ctx, rapidxml::xml_attribute<> *xattr);

			// datas
			virtual void xparse_data(ParseContext *ctx, const char *data, int data_len, bool isPI);
			virtual void xparse_children(ParseContext *ctx, rapidxml::xml_node<> *xnode);
			virtual void xparse_finish(ParseContext *ctx, rapidxml::xml_node<> *xnode);

			virtual bool xparse_has_scope();

		public:
			void parse_xml(ParseContext *ctx, rapidxml::xml_node<> *xnode);

			// functions for post parse
			virtual TypeGuard * get_type_guard() = 0;
			virtual void post_parse(PostParseContext *ctx) = 0;

			// functions for eval
			virtual IValNode* eval(EvalContext *ctx) = 0;
			virtual ExpandControl* get_expand_control() = 0;

			virtual ~IAstNode() {}
		};

		class BaseAstNode : public IAstNode {
		public:
			char* xname;
			ExpandControl ctrl;
			std::vector<IAstNode*> children;
			friend class TagParse;

		protected:
			// attrs
			void xparse_attr_name(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) final;
			void xparse_attr_if(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) final;
			void xparse_attr_sort(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) final;
			void xparse_attr_for(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) final;

			void xparse_attr_len(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) override;
			void xparse_attr_shape(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) override;
			void xparse_attr_count(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) override;

			// datas
			void xparse_children(ParseContext *ctx, rapidxml::xml_node<> *xnode) final;

		public:
			// child
			virtual BaseAstNode* xparse_child_any(ParseContext *ctx, rapidxml::xml_node<> *xnode) ;
			virtual BaseAstNode* xparse_child_block(ParseContext *ctx, rapidxml::xml_node<> *xnode) ;
			virtual BaseAstNode* xparse_child_table(ParseContext *ctx, rapidxml::xml_node<> *xnode) ;
			virtual BaseAstNode* xparse_child_arr(ParseContext *ctx, rapidxml::xml_node<> *xnode, int child_ndim) ;
			virtual BaseAstNode* xparse_child_scalar(ParseContext *ctx, rapidxml::xml_node<> *xnode, numsky_dtype *scalar_dtype) ;
			virtual BaseAstNode* xparse_child_camera(ParseContext *ctx, rapidxml::xml_node<> *xnode) ;
			virtual BaseAstNode* xparse_child_mesh(ParseContext *ctx, rapidxml::xml_node<> *xnode, int mesh_enum);

		protected:
			IAstNode* xparse_child_var(ParseContext *ctx, rapidxml::xml_node<> *xnode);
			IAstNode* xparse_child_proc(ParseContext *ctx, rapidxml::xml_node<> *xnode);

		public:
			template <typename TAstNode> friend class ChildableValNode;
			BaseAstNode() : xname(NULL) {}
			std::string dump_xml(int depth);

			// functions for post parse
			TypeGuard * get_type_guard() override;
			void post_parse(PostParseContext *ctx) override;

			ExpandControl* get_expand_control() override;

			virtual ~BaseAstNode();
		};

		// AbstractBlock: LuaBlock, ArrBlock
		class AbstractBlockAstNode : public BaseAstNode {
		public:
			BaseAstNode *xparse_child_block(ParseContext *ctx, rapidxml::xml_node<> *xnode) final;
			BaseAstNode *xparse_child_table(ParseContext *ctx, rapidxml::xml_node<> *xnode) final;
			BaseAstNode *xparse_child_arr(ParseContext *ctx, rapidxml::xml_node<> *xnode, int child_ndim) final;
			BaseAstNode *xparse_child_scalar(ParseContext *ctx, rapidxml::xml_node<> *xnode, numsky_dtype *scalar_dtype) final;
			BaseAstNode* xparse_child_mesh(ParseContext *ctx, rapidxml::xml_node<> *xnode, int mesh_enum) final;
			virtual BaseAstNode *get_parent() = 0;
			bool xparse_has_scope() final;
			AbstractBlockAstNode(): BaseAstNode() {}
		};

		class ReturnValNode;
	}
}
