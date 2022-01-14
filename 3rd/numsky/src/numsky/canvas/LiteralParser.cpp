
#include "numsky/canvas/LiteralParser.h"

namespace numsky {
	namespace canvas {
		enum AttrEnum {
			// x- attributes
			ATTR___begin,
			ATTR_x_name,
			ATTR_x_for,
			ATTR_x_if,
			ATTR_x_sort,
			ATTR_x_type,
			ATTR_x_local,
			ATTR_x_function,
			// 1d attributes
			ATTR_Shape,
			// 2d attributes
			ATTR_rot,
			ATTR_pos,
			ATTR_Ortho,
			ATTR_Perspective,
			// 2d mesh
			ATTR_scale,
			ATTR_layer,
			ATTR_Pivot,
			// mesh build args
			ATTR_Size,
			ATTR_Vertices,
			ATTR_Indices,
			ATTR___end,
		};
		template <int A> struct AttrDesc;
		/*****************/
		/* x- attribute  */
		/*****************/
		template <> struct AttrDesc<ATTR_x_name> {
			static std::string name() { return "x-name"; }
			static std::string type_desc() { return "string without quotes"; }
			static std::string desc() { return "field for Table"; }
			static void parse(IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
				node->xparse_attr_xname(ctx, attr);
			}
		};
		template <> struct AttrDesc<ATTR_x_for> {
			static std::string name() { return "x-for"; }
			static std::string type_desc() { return "for statement"; }
			static std::string desc() { return "loop control"; }
			static void parse(IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
				node->xparse_attr_xfor(ctx, attr);
			}
		};
		template <> struct AttrDesc<ATTR_x_if> {
			static std::string name() { return "x-if"; }
			static std::string type_desc() { return "lua expression"; }
			static std::string desc() { return "if control"; }
			static void parse(IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
				node->xparse_attr_xif(ctx, attr);
			}
		};
		template <> struct AttrDesc<ATTR_x_sort> {
			static std::string name() { return "x-sort"; }
			static std::string type_desc() { return "lua expression"; }
			static std::string desc() { return ""; }
			static void parse(IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
				node->xparse_attr_xsort(ctx, attr);
			}
		};
		template <> struct AttrDesc<ATTR_x_type> {
			static std::string name() { return "x-type"; }
			static std::string type_desc() { return "string without quotes. dtype for array, point or line or triangle for mesh, "; }
			static std::string desc() { return "some builtin-type"; }
			static void parse(IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
				node->xparse_attr_xtype(ctx, attr);
			}
		};
		template <> struct AttrDesc<ATTR_x_local> {
			static std::string name() { return "x-local"; }
			static std::string type_desc() { return ""; }
			static std::string desc() { return ""; }
			static void parse(IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
				node->xparse_attr_xlocal(ctx, attr);
			}
		};
		template <> struct AttrDesc<ATTR_x_function> {
			static std::string name() { return "x-function"; }
			static std::string type_desc() { return ""; }
			static std::string desc() { return ""; }
			static void parse(IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
				node->xparse_attr_xfunction(ctx, attr);
			}
		};
		/***************************/
		/* 1d element's attribute  */
		/***************************/
		template <> struct AttrDesc<ATTR_Shape> {
			static std::string name() { return "Shape"; }
			static std::string type_desc() { return "int * for Array, int dim1,dim2,dim3=1 for Camera"; }
			static std::string desc() { return "Shape for Array or Camera"; }
			static void parse(IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
				node->xparse_attr_Shape(ctx, attr);
			}
		};
		/***************************/
		/* 2d element's attribute  */
		/***************************/
		template <> struct AttrDesc<ATTR_rot> {
			static std::string name() { return "rot"; }
			static std::string type_desc() { return "float zDegree or float xDegree,yDegree"; }
			static std::string desc() { return "Mesh rotation"; }
			static void parse(IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
				node->xparse_attr_rot(ctx, attr);
			}
		};
		template <> struct AttrDesc<ATTR_pos> {
			static std::string name() { return "pos"; }
			static std::string type_desc() { return "float x,y,z=0"; }
			static std::string desc() { return "Mesh position"; }
			static void parse(IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
				node->xparse_attr_pos(ctx, attr);
			}
		};
		template <> struct AttrDesc<ATTR_Ortho> {
			static std::string name() { return "Ortho"; }
			static std::string type_desc() { return "float left,right,bottom,top,near=--10.far=10"; }
			static std::string desc() { return "Ortho transform"; }
			static void parse(IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
				node->xparse_attr_Ortho(ctx, attr);
			}
		};
		template <> struct AttrDesc<ATTR_Perspective> {
			static std::string name() { return "Perspective"; }
			static std::string type_desc() { return "float fovyDegree, aspectDegree, zNear, zFar"; }
			static std::string desc() { return "Perspective transform"; }
			static void parse(IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
				node->xparse_attr_Perspective(ctx, attr);
			}
		};
		template <> struct AttrDesc<ATTR_scale> {
			static std::string name() { return "scale"; }
			static std::string type_desc() { return "float x,y,z=0"; }
			static std::string desc() { return "Dynamic scale for mesh"; }
			static void parse(IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
				node->xparse_attr_scale(ctx, attr);
			}
		};
		template <> struct AttrDesc<ATTR_layer> {
			static std::string name() { return "layer"; }
			static std::string type_desc() { return "int"; }
			static std::string desc() { return "Mesh's color fill start at layer, layer start with 1"; }
			static void parse(IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
				node->xparse_attr_layer(ctx, attr);
			}
		};
		template <> struct AttrDesc<ATTR_Pivot> {
			static std::string name() { return "Pivot"; }
			static std::string type_desc() { return "float x,y,z=0"; }
			static std::string desc() { return "Rotation position for mesh."; }
			static void parse(IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
				node->xparse_attr_Pivot(ctx, attr);
			}
		};
		template <> struct AttrDesc<ATTR_Size> {
			static std::string name() { return "Size"; }
			static std::string type_desc() { return "float x,y,z=0"; }
			static std::string desc() { return "Static scale for mesh."; }
			static void parse(IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
				node->xparse_attr_Size(ctx, attr);
			}
		};
		template <> struct AttrDesc<ATTR_Vertices> {
			static std::string name() { return "Vertices"; }
			static std::string type_desc() { return "numsky.ndarray or table"; }
			static std::string desc() { return "Vertex array of Mesh."; }
			static void parse(IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
				node->xparse_attr_Vertices(ctx, attr);
			}
		};
		template <> struct AttrDesc<ATTR_Indices> {
			static std::string name() { return "Indices"; }
			static std::string type_desc() { return "numsky.ndarray or table"; }
			static std::string desc() { return "Index array of Mesh, index must start with 1."; }
			static void parse(IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
				node->xparse_attr_Indices(ctx, attr);
			}
		};
	}
}

namespace numsky {
	namespace canvas {
		enum TagEnum {
			TAG___begin,
			// control
			TAG_var,
			TAG_proc,
			TAG_block,
			//
			TAG_Array,
			TAG_Scalar,
			//
			TAG_Mesh,
			TAG_Camera,
			//
			TAG_Table,
			TAG_Any,
			TAG___end,
		};
		template <int A> struct TagDesc;
		template <> struct TagDesc<TAG_var> {
			static std::string name() { return "var"; }
			static std::string desc() { return "define a value"; }
			static void put(TagParse & tagParse) {
				tagParse.nameToFunc["var"] = [](BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>* xchild)->IAstNode*{
					return node->xparse_child_var(ctx, xchild);
				};
			}
		};
		template <> struct TagDesc<TAG_proc> {
			static std::string name() { return "proc"; }
			static std::string desc() { return "define a value"; }
			static void put(TagParse & tagParse) {
				tagParse.nameToFunc["proc"] = [](BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>* xchild)->IAstNode*{
					return node->xparse_child_proc(ctx, xchild);
				};
			}
		};
		template <> struct TagDesc<TAG_block> {
			static std::string name() { return "block"; }
			static void put(TagParse & tagParse) {
				tagParse.nameToFunc["block"] = [](BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>* xchild)->IAstNode*{
					return node->xparse_child_block(ctx, xchild);
				};
			}
		};
		// array & scalar
		template <> struct TagDesc<TAG_Array> {
			static std::string name() { return "Array,Arr*d"; }
			static void put(TagParse & tagParse) {
				tagParse.nameToFunc["Arr"] = [](BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>* xchild)->IAstNode*{
					return node->xparse_child_arr(ctx, xchild, 0);
				};
				tagParse.nameToFunc["Array"] = [](BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>* xchild)->IAstNode*{
					return node->xparse_child_arr(ctx, xchild, 0);
				};
				for(int i=1;i<=CANVAS_MAX_DIM;i++) {
					std::string tag = std::string("Arr") + std::to_string(i) + std::string("d");
					tagParse.nameToFunc[tag] = [](BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>* xchild)->IAstNode*{
						const char* name = xchild->name();
						if(name[4] == 'd') {
							return node->xparse_child_arr(ctx, xchild, name[3] - '0');
						} else {
							return node->xparse_child_arr(ctx, xchild, 10*(name[3] - '0') + name[4]- '0');
						}
					};
				}
			}
		};
		template <> struct TagDesc<TAG_Scalar> {
			static std::string name() { return "Scalar"; }
			static void put(TagParse & tagParse) {
				for(size_t i=0;i<sizeof(NS_DTYPE_CHARS);i++) {
					numsky_dtype *try_dtype = numsky_get_dtype_by_char(NS_DTYPE_CHARS[i]);
					tagParse.nameToFunc[try_dtype->name] = [](BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>* xchild)->IAstNode*{
						numsky_dtype *dtype = ctx->try_parse_dtype(xchild->name());
						return node->xparse_child_scalar(ctx, xchild, dtype);
					};
				}
			}
		};
		// table & any
		template <> struct TagDesc<TAG_Any> {
			static std::string name() { return "Any"; }
			static void put(TagParse & tagParse) {
				tagParse.nameToFunc["Any"] = [](BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>* xchild)->IAstNode*{
					return node->xparse_child_any(ctx, xchild);
				};
			}
		};
		template <> struct TagDesc<TAG_Table> {
			static std::string name() { return "Table"; }
			static void put(TagParse & tagParse) {
				tagParse.nameToFunc["Table"] = [](BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>* xchild)->IAstNode*{
					return node->xparse_child_table(ctx, xchild);
				};
			}
		};
		// camera & mesh
		template <> struct TagDesc<TAG_Camera> {
			static std::string name() { return "Camera"; }
			static void put(TagParse & tagParse) {
				tagParse.nameToFunc["Camera"] = [](BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>* xchild)->IAstNode*{
					return node->xparse_child_camera(ctx, xchild);
				};
			}
		};
		template <numsky::MESH_BUILTIN_ENUM mesh_enum> void TagParse_recursive_put_mesh(TagParse & tagParse);
		template <> struct TagDesc<TAG_Mesh> {
			static std::string name() { return "Mesh"; }
			static void put(TagParse & tagParse) {
				tagParse.nameToFunc["Mesh"] = [](BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>* xchild)->IAstNode*{
					return node->xparse_child_mesh(ctx, xchild, -1);
				};
				TagParse_recursive_put_mesh<numsky::MESH_SECTOR>(tagParse);
			}
		};

		/* mesh recursive put put */
		template <numsky::MESH_BUILTIN_ENUM mesh_enum> void TagParse_recursive_put_mesh(TagParse & tagParse) {
			tagParse.nameToFunc[numsky::MeshEnumVariable<mesh_enum>::mesh_name] = [](BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>* xchild)->IAstNode*{
				return node->xparse_child_mesh(ctx, xchild, mesh_enum);
			};
			TagParse_recursive_put_mesh<(numsky::MESH_BUILTIN_ENUM)(mesh_enum-1)>(tagParse);
		}
		template <> void TagParse_recursive_put_mesh<numsky::MESH_POINT>(TagParse & tagParse) {
			tagParse.nameToFunc[numsky::MeshEnumVariable<numsky::MESH_POINT>::mesh_name] = [](BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>* xchild)->IAstNode*{
				return node->xparse_child_mesh(ctx, xchild, numsky::MESH_POINT);
			};
		}
	}

	namespace canvas {

		// recursive put functions
		template <int A> void AttrParse_recursive_put(AttrParse & attr_parse) {
			attr_parse.nameToFunc[AttrDesc<A>::name()] = AttrDesc<A>::parse;
			AttrParse_recursive_put<A-1>(attr_parse);
		}
		template <> void AttrParse_recursive_put<ATTR___begin>(AttrParse & attr_parse) {}
		template <> void AttrParse_recursive_put<ATTR___end>(AttrParse & attr_parse) {
			AttrParse_recursive_put<ATTR___end-1>(attr_parse);
		}

		AttrParse::AttrParse() {
			/*nameToFunc["ndim"] = [](IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
				node->xparse_attr_ndim(ctx, attr);
			};
			nameToFunc["len"] = [](IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
				node->xparse_attr_len(ctx, attr);
			};
			nameToFunc["count"] = [](IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
				node->xparse_attr_count(ctx, attr);
			};*/
			AttrParse_recursive_put<ATTR___end>(*this);
		}

		// recursive put functions
		template <int A> void TagParse_recursive_put(TagParse & tag_parse) {
			TagDesc<A>::put(tag_parse);
			TagParse_recursive_put<A-1>(tag_parse);
		}
		template <> void TagParse_recursive_put<TAG___begin>(TagParse & tag_parse) {}
		template <> void TagParse_recursive_put<TAG___end>(TagParse & tag_parse) {
			TagParse_recursive_put<TAG___end-1>(tag_parse);
		}
		TagParse::TagParse() {
			TagParse_recursive_put<TAG___end>(*this);
			/*TagParse_recursive_put_mesh<numsky::MESH_SECTOR>(nameToDescFunc);*/
		}
		int enum_tag_attr(lua_State*L) {
			/*
			AttrParse attr_parse;
			TagParse tag_parse;
			lua_newtable(L);
			for(auto iter=tag_parse.nameToDescFunc.begin();iter!=tag_parse.nameToDescFunc.end();iter++) {
				lua_pushstring(L, iter->second.desc.c_str());
				lua_setfield(L, -2, iter->first.c_str());
			}
			lua_newtable(L);
			for(auto iter=attr_parse.nameToFunc.begin();iter!=attr_parse.nameToFunc.end();iter++) {
				lua_pushboolean(L, 1);
				lua_setfield(L, -2, iter->first.c_str());
			}
			return 2;
			*/
			return 0;
		}
	}
}
