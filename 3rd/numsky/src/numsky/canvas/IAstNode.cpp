
#include "numsky/canvas/AstNode.h"
#include "numsky/canvas/ValNode.h"
#include <sstream>
#include <type_traits>
#include <map>

// node implement
namespace numsky {
	// IAstNode
	namespace canvas {
		class AttrParse {
			std::map<std::string, void(*)(IAstNode*, ParseContext*, rapidxml::xml_attribute<>*)> nameToFunc;
		public:
			void parse(IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
				auto iter = nameToFunc.find(attr->name());
				if(iter==nameToFunc.end()) {
					ctx->raise(attr->name(), "invalid attr", attr->name());
				} else {
					iter->second(node, ctx, attr);
				}
			}
			AttrParse() {
				nameToFunc["x-name"] = [](IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
					node->xparse_attr_xname(ctx, attr);
				};
				nameToFunc["x-for"] = [](IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
					node->xparse_attr_xfor(ctx, attr);
				};
				nameToFunc["x-if"] = [](IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
					node->xparse_attr_xif(ctx, attr);
				};
				nameToFunc["x-sort"] = [](IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
					node->xparse_attr_xsort(ctx, attr);
				};
				nameToFunc["x-type"] = [](IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
					node->xparse_attr_xtype(ctx, attr);
				};
				nameToFunc["ndim"] = [](IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
					node->xparse_attr_ndim(ctx, attr);
				};
				nameToFunc["len"] = [](IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
					node->xparse_attr_len(ctx, attr);
				};
				nameToFunc["count"] = [](IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
					node->xparse_attr_count(ctx, attr);
				};
				nameToFunc["Shape"] = [](IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
					node->xparse_attr_Shape(ctx, attr);
				};
				nameToFunc["x-local"] = [](IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
					node->xparse_attr_xlocal(ctx, attr);
				};
				nameToFunc["x-function"] = [](IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
					node->xparse_attr_xfunction(ctx, attr);
				};
				// camera
				nameToFunc["rot"] = [](IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
					node->xparse_attr_rot(ctx, attr);
				};
				nameToFunc["pos"] = [](IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
					node->xparse_attr_pos(ctx, attr);
				};
				nameToFunc["scale"] = [](IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
					node->xparse_attr_scale(ctx, attr);
				};
				nameToFunc["Ortho"] = [](IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
					node->xparse_attr_Ortho(ctx, attr);
				};
				nameToFunc["Perspective"] = [](IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
					node->xparse_attr_Perspective(ctx, attr);
				};
				// mesh
				nameToFunc["layer"] = [](IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
					node->xparse_attr_layer(ctx, attr);
				};
				nameToFunc["Pivot"] = [](IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
					node->xparse_attr_Pivot(ctx, attr);
				};
				nameToFunc["Size"] = [](IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
					node->xparse_attr_Size(ctx, attr);
				};
				nameToFunc["Vertices"] = [](IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
					node->xparse_attr_Vertices(ctx, attr);
				};
				nameToFunc["Indices"] = [](IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
					node->xparse_attr_Indices(ctx, attr);
				};
			}
		};
		static AttrParse attr_parse;

		void IAstNode::xparse_attr_xname(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			ctx->raise(xattr->name(), "attr name not implement in this tag");
		}
		void IAstNode::xparse_attr_xfor(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			ctx->raise(xattr->name(), "attr x-for not implement in this tag");
		}
		void IAstNode::xparse_attr_xif(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			ctx->raise(xattr->name(), "attr x-if not implement in this tag");
		}
		void IAstNode::xparse_attr_xsort(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			ctx->raise(xattr->name(), "attr x-sort not implement in this tag");
		}
		void IAstNode::xparse_attr_xtype(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			ctx->raise(xattr->name(), "attr x-type not implement in this tag");
		}
		void IAstNode::xparse_attr_ndim(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			ctx->raise(xattr->name(), "attr ndim not implement in this tag");
		}
		void IAstNode::xparse_attr_len(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			ctx->raise(xattr->name(), "attr len not implement in this tag");
		}
		void IAstNode::xparse_attr_count(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			ctx->raise(xattr->name(), "attr count not implement in this tag");
		}
		void IAstNode::xparse_attr_Shape(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			ctx->raise(xattr->name(), "attr Shape not implement in this tag");
		}
		void IAstNode::xparse_attr_xlocal(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			ctx->raise(xattr->name(), "attr x-local not implement in this tag");
		}
		void IAstNode::xparse_attr_xfunction(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			ctx->raise(xattr->name(), "attr x-function not implement in this tag");
		}
		void IAstNode::xparse_attr_rot(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			ctx->raise(xattr->name(), "attr rot not implement in this tag");
		}
		void IAstNode::xparse_attr_pos(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			ctx->raise(xattr->name(), "attr pos not implement in this tag");
		}
		void IAstNode::xparse_attr_scale(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			ctx->raise(xattr->name(), "attr scale not implement in this tag");
		}
		void IAstNode::xparse_attr_layer(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			ctx->raise(xattr->name(), "attr layer not implement in this tag");
		}
		void IAstNode::xparse_attr_Ortho(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			ctx->raise(xattr->name(), "attr Ortho not implement in this tag");
		}
		void IAstNode::xparse_attr_Perspective(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			ctx->raise(xattr->name(), "attr Perspective not implement in this tag");
		}
		void IAstNode::xparse_attr_Pivot(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			ctx->raise(xattr->name(), "attr Pivot not implement in this tag");
		}
		void IAstNode::xparse_attr_Size(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			ctx->raise(xattr->name(), "attr Size not implement in this tag");
		}
		void IAstNode::xparse_attr_Vertices(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			ctx->raise(xattr->name(), "attr Vertices not implement in this tag");
		}
		void IAstNode::xparse_attr_Indices(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			ctx->raise(xattr->name(), "attr Indices not implement in this tag");
		}

		// datas
		void IAstNode::xparse_data(ParseContext *ctx, const char *data, int data_len, bool isScope) {
			ctx->raise(data, "xml data not implement in this tag");
		}
		void IAstNode::xparse_pi_reset(ParseContext *ctx, const char *data, int data_len) {
			ctx->raise(data, "xml pi_reset not implement in this tag");
		}

		void IAstNode::xparse_children(ParseContext *ctx, rapidxml::xml_node<> *xnode) {
			for(rapidxml::xml_node<> *xchild = xnode->first_node();xchild;xchild=xchild->next_sibling()){
				if(xchild->type() == rapidxml::node_element) {
					ctx->raise(xchild->name(), "this tag's child node not implement");
				}
			}
		}

		void IAstNode::xparse_finish(ParseContext *ctx, rapidxml::xml_node<> *xnode) {}

		bool IAstNode::xparse_has_scope() {
			return false;
		}

		void IAstNode::parse_xml(ParseContext *ctx, rapidxml::xml_node<> *xnode) {
			line = ctx->xnode_calc_line(xnode);
			if(xparse_has_scope()) {
				ctx->put_do(xnode);
			}
			for(auto attr=xnode->first_attribute();attr;attr=attr->next_attribute()) {
				attr_parse.parse(this, ctx, attr);
			}
			bool parsed_data = false;
			for(rapidxml::xml_node<> *xchild = xnode->first_node();xchild;xchild=xchild->next_sibling()){
				if(xchild->type()==rapidxml::node_data) {
					if(parsed_data) {
						ctx->raise(xchild->value(), "node can only have one data element");
					} else {
						xparse_data(ctx, xchild->value(), xchild->value_size(), false);
						parsed_data = true;
					}
				} else if(xchild->type()==rapidxml::node_pi){
					std::string pi(xchild->name(), xchild->name_size());
					if(parsed_data) {
						ctx->raise(xchild->value(), "node can only have one data element");
					} else {
						if(pi=="lua") {
							xparse_data(ctx, xchild->value(), xchild->value_size(), true);
						} else if(pi=="reset"){
							xparse_pi_reset(ctx, xchild->value(), xchild->value_size());
						} else {
							ctx->raise(xchild->name(), "PI target must be lua or reset");
						}
						parsed_data = true;
					}
				}
			}
			xparse_children(ctx, xnode);
			xparse_finish(ctx, xnode);
			if(xparse_has_scope()) {
				ctx->put_end(xnode);
			}
		}
	}
	// BaseAstNode
	namespace canvas {
		template <numsky::MESH_BUILTIN_ENUM mesh_enum> void TagParse_recursive_put_mesh(std::map<std::string, IAstNode*(*)(BaseAstNode*, ParseContext*, rapidxml::xml_node<>*)>& nameToFunc) {
			nameToFunc[numsky::MeshEnumVariable<mesh_enum>::mesh_name] = [](BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>* xchild)->IAstNode*{
				return node->xparse_child_mesh(ctx, xchild, mesh_enum);
			};
			TagParse_recursive_put_mesh<(numsky::MESH_BUILTIN_ENUM)(mesh_enum-1)>(nameToFunc);
		}
		template <> void TagParse_recursive_put_mesh<numsky::MESH_POINT>(std::map<std::string, IAstNode*(*)(BaseAstNode*, ParseContext*, rapidxml::xml_node<>*)>& nameToFunc) {
			nameToFunc[numsky::MeshEnumVariable<numsky::MESH_POINT>::mesh_name] = [](BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>* xchild)->IAstNode*{
				return node->xparse_child_mesh(ctx, xchild, numsky::MESH_POINT);
			};
		}
		class TagParse {
			std::map<std::string, IAstNode*(*)(BaseAstNode*, ParseContext*, rapidxml::xml_node<>*)> nameToFunc;
		public:
			IAstNode* parse(BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>*attr) {
				auto iter = nameToFunc.find(attr->name());
				if(iter==nameToFunc.end()) {
					ctx->raise(attr->name(), "invalid tag", attr->name());
					return NULL;
				} else {
					return iter->second(node, ctx, attr);
				}
			}
			TagParse() {
				nameToFunc["var"] = [](BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>* xchild)->IAstNode*{
					return node->xparse_child_var(ctx, xchild);
				};
				nameToFunc["proc"] = [](BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>* xchild)->IAstNode*{
					return node->xparse_child_proc(ctx, xchild);
				};
				nameToFunc["block"] = [](BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>* xchild)->IAstNode*{
					return node->xparse_child_block(ctx, xchild);
				};
				nameToFunc["Any"] = [](BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>* xchild)->IAstNode*{
					return node->xparse_child_any(ctx, xchild);
				};
				nameToFunc["Table"] = [](BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>* xchild)->IAstNode*{
					return node->xparse_child_table(ctx, xchild);
				};
				nameToFunc["Camera"] = [](BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>* xchild)->IAstNode*{
					return node->xparse_child_camera(ctx, xchild);
				};
				// arr
				nameToFunc["Arr"] = [](BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>* xchild)->IAstNode*{
					return node->xparse_child_arr(ctx, xchild, 0);
				};
				nameToFunc["Array"] = [](BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>* xchild)->IAstNode*{
					return node->xparse_child_arr(ctx, xchild, 0);
				};
				for(int i=1;i<=CANVAS_MAX_DIM;i++) {
					std::string tag = std::string("Arr") + std::to_string(i) + std::string("d");
					nameToFunc[tag] = [](BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>* xchild)->IAstNode*{
						const char* name = xchild->name();
						if(name[4] == 'd') {
							return node->xparse_child_arr(ctx, xchild, name[3] - '0');
						} else {
							return node->xparse_child_arr(ctx, xchild, 10*(name[3] - '0') + name[4]- '0');
						}
					};
				}
				// scalar
				for(size_t i=0;i<sizeof(NS_DTYPE_CHARS);i++) {
					numsky_dtype *try_dtype = numsky_get_dtype_by_char(NS_DTYPE_CHARS[i]);
					nameToFunc[try_dtype->name] = [](BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>* xchild)->IAstNode*{
						numsky_dtype *dtype = ctx->try_parse_dtype(xchild->name());
						return node->xparse_child_scalar(ctx, xchild, dtype);
					};
				}
				// mesh
				nameToFunc["Mesh"] = [](BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>* xchild)->IAstNode*{
					return node->xparse_child_mesh(ctx, xchild, -1);
				};
				TagParse_recursive_put_mesh<numsky::MESH_SECTOR>(nameToFunc);
			}
		};
		static TagParse tag_parse;

		BaseAstNode::~BaseAstNode() {
			for(auto iter=children.begin();iter!=children.end();++iter) {
				delete *iter;
			}
		}

		ExpandControl* BaseAstNode::get_expand_control() {
			return &ctrl;
		}

		TypeGuard* BaseAstNode::get_type_guard() {
			return NULL;
		}

		std::string BaseAstNode::dump_xml(int depth) {
			/*std::ostringstream stream;
			// begin
			for(int i=0;i<depth;i++) {
				stream << "  ";
			}
			switch(node_type) {
				case AST_SCALAR: stream<<"<"<<dtype->name<<">"<<std::endl; break;
				case AST_ARRAY: stream<<"<arr dtype=\""<<dtype->name<<"\">"<<std::endl; break;
				case AST_CANVAS: stream<<"<canvas>"<<std::endl; break;
				default: stream<<"<UNKNOWN>"<<std::endl; break;
			}
			// middle
			for(int i=0;i<children.size();i++) {
				stream << children[i]->dump_xml(depth + 1);
			}
			// end
			for(int i=0;i<depth;i++) {
				stream << "  ";
			}
			switch(node_type) {
				case AST_SCALAR: stream<<"</"<<dtype->name<<">"<<std::endl; break;
				case AST_ARRAY: stream<<"</arr>"<<std::endl; break;
				case AST_CANVAS: stream<<"</canvas>"<<std::endl; break;
				default: stream<<"</UNKNOWN>"<<std::endl; break;
			}
			return stream.str();*/
			return std::string("nothing");
		}
		void BaseAstNode::xparse_children(ParseContext *ctx, rapidxml::xml_node<> *xnode) {
			for(rapidxml::xml_node<> *xchild = xnode->first_node();xchild;xchild=xchild->next_sibling()){
				if(xchild->type() == rapidxml::node_element) {
					std::string tag(xchild->name(), xchild->name_size());
					IAstNode* child = tag_parse.parse(this, ctx, xchild);
					children.push_back(child);
				}
			}
		}

		void BaseAstNode::post_parse(PostParseContext *ctx) {
			for(auto iter=children.begin();iter!=children.end();iter++) {
				(*iter)->post_parse(ctx);
			}
		}
		void BaseAstNode::xparse_attr_xname(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			xname = xattr->value();
		}
		void BaseAstNode::xparse_attr_xfor(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			std::string value(xattr->value(), xattr->value_size());
			for(auto iter=value.begin();iter!=value.end();++iter) {
				if(*iter == '\t' || *iter == '\n' || *iter == '\r') {
					*iter = ' ';
				}
			}
			if(ctrl.fi_forvar > 0) {
				ctx->raise(xattr->name(), "for attr has been defined");
			} else if(value.find(" in ") != std::string::npos){
				std::tie(ctrl.fi_forvar, ctrl.fi_forgen) = ctx->put_forin(xattr, value);
			} else if(value.find("=") != std::string::npos) {
				std::tie(ctrl.fi_forvar, ctrl.fi_forseq) = ctx->put_foreq(xattr, value);
			} else {
				ctx->raise(xattr->name(), "for syntax error");
			}
		}
		void BaseAstNode::xparse_attr_xif(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			std::string value(xattr->value(), xattr->value_size());
			if(ctrl.fi_if > 0) {
				ctx->raise(xattr->name(), "if attr has been defined");
			} else {
				ctrl.fi_if = ctx->put_if(xattr, value);
			}
		}
		void BaseAstNode::xparse_attr_xsort(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			if(ctrl.fi_forvar == 0) {
				ctx->raise(xattr->name(), "use for before sort");
				return ;
			}
			ctrl.fi_forsort = ctx->put_explist<false>(xattr->value(), xattr->value_size());
		}


		void BaseAstNode::xparse_attr_len(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			TypeGuard *guard = get_type_guard();
			if(guard) {
				guard->si_len = ctx->put_static_explist(xattr);
			} else {
				ctx->raise(xattr->name(), "this node can't has len");
			}
		}
		void BaseAstNode::xparse_attr_Shape(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			TypeGuard *guard = get_type_guard();
			if(guard) {
				guard->si_shape = ctx->put_static_explist(xattr);
			} else {
				ctx->raise(xattr->name(), "this node can't has shape");
			}
		}
		void BaseAstNode::xparse_attr_count(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			TypeGuard *guard = get_type_guard();
			if(guard) {
				guard->si_count = ctx->put_static_explist(xattr);
			} else {
				ctx->raise(xattr->name(), "this node can't has count");
			}
		}

		IAstNode* BaseAstNode::xparse_child_var(ParseContext *ctx, rapidxml::xml_node<> *xnode) {
			if(!xparse_has_scope()) {
				ctx->raise(xnode->name(), "this element cannot has var as child");
				return NULL;
			} else {
				IAstNode *child = new VarAstNode(false);
				child->parse_xml(ctx, xnode);
				return child;
			}
		}
		IAstNode * BaseAstNode::xparse_child_proc(ParseContext *ctx, rapidxml::xml_node<> *xnode) {
			if(!xparse_has_scope()) {
				ctx->raise(xnode->name(), "this element cannot has var as child");
				return NULL;
			} else {
				IAstNode *child = new ProcAstNode();
				child->parse_xml(ctx, xnode);
				return child;
			}
		}
		BaseAstNode* BaseAstNode::xparse_child_table(ParseContext *ctx, rapidxml::xml_node<> *xnode) {
			ctx->raise(xnode->name(), "<table> cannot be child for this tag");
			return NULL;
		}
		BaseAstNode* BaseAstNode::xparse_child_block(ParseContext *ctx, rapidxml::xml_node<> *xnode) {
			ctx->raise(xnode->name(), "<block> cannot be child for this tag");
			return NULL;
		}
		BaseAstNode* BaseAstNode::xparse_child_any(ParseContext *ctx, rapidxml::xml_node<> *xnode) {
			ctx->raise(xnode->name(), "<any> cannot be child for this tag");
			return NULL;
		}
		BaseAstNode* BaseAstNode::xparse_child_arr(ParseContext *ctx, rapidxml::xml_node<> *xnode, int child_ndim) {
			ctx->raise(xnode->name(), "arr cannot be child for this tag");
			return NULL;
		}
		BaseAstNode* BaseAstNode::xparse_child_scalar(ParseContext *ctx, rapidxml::xml_node<> *xnode, numsky_dtype *scalar_dtype) {
			ctx->raise(xnode->name(), "scalar cannot be child for this tag");
			return NULL;
		}
		BaseAstNode* BaseAstNode::xparse_child_camera(ParseContext *ctx, rapidxml::xml_node<> *xnode) {
			ctx->raise(xnode->name(), "camera cannot be child for this tag");
			return NULL;
		}
		BaseAstNode* BaseAstNode::xparse_child_mesh(ParseContext *ctx, rapidxml::xml_node<> *xnode, int mesh_enum) {
			ctx->raise(xnode->name(), "mesh cannot be child for this tag");
			return NULL;
		}
	}

	namespace canvas {
		BaseAstNode* AbstractBlockAstNode::xparse_child_table(ParseContext *ctx, rapidxml::xml_node<> *xnode) {
			return get_parent()->xparse_child_table(ctx, xnode);
		}
		BaseAstNode* AbstractBlockAstNode::xparse_child_block(ParseContext *ctx, rapidxml::xml_node<> *xnode) {
			return get_parent()->xparse_child_block(ctx, xnode);
		}
		BaseAstNode* AbstractBlockAstNode::xparse_child_arr(ParseContext *ctx, rapidxml::xml_node<> *xnode, int child_ndim) {
			return get_parent()->xparse_child_arr(ctx, xnode, child_ndim);
		}
		BaseAstNode* AbstractBlockAstNode::xparse_child_scalar(ParseContext *ctx, rapidxml::xml_node<> *xnode, numsky_dtype *scalar_dtype) {
			return get_parent()->xparse_child_scalar(ctx, xnode, scalar_dtype);
		}
		BaseAstNode* AbstractBlockAstNode::xparse_child_mesh(ParseContext *ctx, rapidxml::xml_node<> *xnode, int mesh_enum) {
			return get_parent()->xparse_child_mesh(ctx, xnode, mesh_enum);
		}
		bool AbstractBlockAstNode::xparse_has_scope() {
			return true;
		}
	}
}
