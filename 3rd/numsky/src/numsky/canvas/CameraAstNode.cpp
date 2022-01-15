
#include "numsky/canvas/AstNode.h"
#include "numsky/canvas/ValNode.h"
#include <sstream>
#include <type_traits>
#include <map>

// node implement
namespace numsky {
	namespace canvas {
		void CameraAstNode::xparse_attr_rot(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			fi_rot = ctx->put_explist<false>(xattr->value(), xattr->value_size());
		}
		void CameraAstNode::xparse_attr_pos(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			fi_pos = ctx->put_explist<false>(xattr->value(), xattr->value_size());
		}
		void CameraAstNode::xparse_attr_Ortho(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			if(si_perspective != 0 || si_ortho != 0) {
				ctx->raise(xattr->name(), "ortho and perspective can only set one");
			}
			si_ortho = ctx->put_static_explist(xattr);
		}
		void CameraAstNode::xparse_attr_Perspective(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			if(si_perspective != 0 || si_ortho != 0) {
				ctx->raise(xattr->name(), "ortho and perspective can only set one");
			}
			si_perspective = ctx->put_static_explist(xattr);
		}
		void CameraAstNode::xparse_attr_Shape(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			si_shape = ctx->put_static_explist(xattr);
		}
		IValNode *CameraAstNode::eval(EvalContext *ctx) {
			return new CameraValNode(this);
		}
		BaseAstNode* CameraAstNode::xparse_child_block(ParseContext *ctx, rapidxml::xml_node<> *xnode) {
			MeshBlockAstNode *child = new MeshBlockAstNode(this);
            child->parse_xml(ctx, xnode);
            return child;
		}
		BaseAstNode* CameraAstNode::xparse_child_mesh(ParseContext *ctx, rapidxml::xml_node<> *xnode, int mesh_enum) {
			BaseAstNode *child = nullptr;
			if(mesh_enum<0) {
				child = new MeshAstNode();
			} else {
				switch(mesh_enum) {
				case numsky::MESH_POINT: {
					child = new PointAstNode();
					break;
				}
				case numsky::MESH_LINE: {
					child = new LineAstNode();
					break;
				}
				case numsky::MESH_RECT: {
					child = new RectAstNode();
					break;
				}
				case numsky::MESH_POLYGON: {
					child = new PolygonAstNode();
					break;
				}
				case numsky::MESH_CIRCLE: {
					child = new CircleAstNode();
					break;
				}
				case numsky::MESH_SECTOR: {
					child = new SectorAstNode();
					break;
				}
				default: {
					ctx->raise(xnode->name(), "TODO");
				}
				}
			}
			child->parse_xml(ctx, xnode);
			return child;
		}
		void CameraAstNode::post_parse(PostParseContext *ctx) {
			ctx->set_cur_line(line);
			if(si_shape<0) {
				int nresults = 0;
				std::unique_ptr<int64_t[]> args = ctx->eval_integers(si_shape, nresults);
				if(nresults!=3) {
					ctx->raise(line, "shape must be 3 value");
				}
				for(int i=0;i<3;i++) {
					if(args[i]<=0) {
						ctx->raise(line, "shape must be integer > 0 ");
					}
				}
				screen_shape.d[0] = args[0];
				screen_shape.d[1] = args[1];
				screen_shape.d[2] = args[2];
			}
			if(si_ortho<0) {
				int nresults = 0;
				std::unique_ptr<double[]> args = ctx->eval_numbers(si_ortho, nresults);
				if(nresults==4) {
					projection_matrix = tinygl::Matrix4f::fromOrtho(
							args[0], args[1], args[2], args[3], -10, 10);
				} else if(nresults==6) {
					projection_matrix = tinygl::Matrix4f::fromOrtho(
							args[0], args[1], args[2], args[3], args[4], args[5]);
				} else {
					ctx->raise(line, "ortho must be 4 or 6 value");
				}
			} else if(si_perspective<0) {
				int nresults = 0;
				std::unique_ptr<double[]> args = ctx->eval_numbers(si_perspective, nresults);
				if(nresults!=4) {
					ctx->raise(line, "perspective must be 4 value");
				}
				projection_matrix = tinygl::Matrix4f::fromPerspective(args[0], args[1], args[2], args[3]);
			}
			BaseAstNode::post_parse(ctx);
		}
        bool CameraAstNode::xparse_has_scope() {
            return true;
        }
	}

	// AbstractMeshAstNode
	namespace canvas {
		void AbstractMeshAstNode::xparse_data(ParseContext *ctx, const char *data, int data_len, bool isScope) {
			if(fi_data!=0) {
				ctx->raise(data, "mesh's data has been setted");
			} else {
				if(isScope) {
					fi_data = ctx->put_explist<true>(data, data_len);
				} else {
					fi_data = ctx->put_explist<false>(data, data_len);
				}
			}
		}
		void AbstractMeshAstNode::xparse_attr_rot(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			fi_rot = ctx->put_explist<false>(xattr->value(), xattr->value_size());
		}
		void AbstractMeshAstNode::xparse_attr_pos(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			fi_pos = ctx->put_explist<false>(xattr->value(), xattr->value_size());
		}
		void AbstractMeshAstNode::xparse_attr_scale(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			fi_scale = ctx->put_explist<false>(xattr->value(), xattr->value_size());
		}
		void AbstractMeshAstNode::xparse_attr_layer(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			fi_layer = ctx->put_static_explist(xattr);
		}
		void AbstractMeshAstNode::xparse_attr_xtype(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			std::string fill_type_str(xattr->value(), xattr->value_size());
			if(fill_type_str=="point") {
				fill_type = tinygl::FILL_POINT;
			} else if(fill_type_str=="line") {
				fill_type = tinygl::FILL_LINE;
			} else if(fill_type_str=="triangle") {
				fill_type = tinygl::FILL_TRIANGLE;
			}
		}
		IValNode *AbstractMeshAstNode::eval(EvalContext *ctx) {
			return new MeshValNode(this);
		}
		void AbstractMeshAstNode::post_parse(PostParseContext *ctx) {
			if(mesh_ptr.get() == nullptr){
				ctx->raise(line, "mesh can't be NULL");
			}
			BaseAstNode::post_parse(ctx);
		}
	}

	// MeshAstNode
	namespace canvas {
		void MeshAstNode::xparse_attr_Vertices(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			si_vertices = ctx->put_static_explist(xattr);
		}
		void MeshAstNode::xparse_attr_Indices(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			si_indices = ctx->put_static_explist(xattr);
		}
		void MeshAstNode::post_parse(PostParseContext *ctx) {
			mesh_ptr = ctx->eval_mesh(si_vertices, si_indices);
			AbstractMeshAstNode::post_parse(ctx);
		}
	}

	// BuiltinMeshAstNode
	namespace canvas {
		void BuiltinMeshAstNode::xparse_attr_Pivot(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			si_pivot = ctx->put_static_explist(xattr);
		}
		void BuiltinMeshAstNode::xparse_attr_Size(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) {
			si_size = ctx->put_static_explist(xattr);
		}
		void BuiltinMeshAstNode::post_parse(PostParseContext *ctx) {
			ctx->set_cur_line(line);
			tinygl::V3f pivot(0,0,0);
			if(si_pivot<0) {
				int nresults = 0;
				std::unique_ptr<double[]> args = ctx->eval_numbers(si_pivot, nresults);
				if(nresults == 2) {
					pivot.v[0] = args[0];
					pivot.v[1] = args[1];
				} else if(nresults == 3) {
					pivot.v[0] = args[0];
					pivot.v[1] = args[1];
					pivot.v[2] = args[2];
				} else {
					ctx->raise(line, "pivot must be 2 or 3 value");
				}
			}
			int size_len=0;
			std::unique_ptr<double []> size(new double[3]{0,0,0});
			if(si_size<0) {
				size = ctx->eval_numbers(si_size, size_len);
			}
			mesh_ptr = create_mesh(ctx, size.get(), size_len);
			for(auto iter=mesh_ptr->getVertices().begin();iter!=mesh_ptr->getVertices().end();iter++) {
				iter->X -= pivot.X;
				iter->Y -= pivot.Y;
				iter->Z -= pivot.Z;
			}
			AbstractMeshAstNode::post_parse(ctx);
		}
	}

	// PointAstNode, LineAstNode, other builtin mesh ...
	namespace canvas {
		std::unique_ptr<tinygl::Mesh> PointAstNode::create_mesh(PostParseContext *ctx, double *size, int size_len) {
			if(si_size != 0) {
				ctx->raise(line, "point can't take size attr");
			}
			return tinygl::Mesh::create_point(0,0);
		}
		std::unique_ptr<tinygl::Mesh> LineAstNode::create_mesh(PostParseContext *ctx, double *size, int size_len) {
			float from_x = -0.5;
			float from_y = 0;
			float to_x = 0.5;
			float to_y = 0;
			if(size_len == 1) {
				from_x = -size[0]/2;
				from_y = size[0]/2;
			} else if (size_len != 0) {
				ctx->raise(line, "line's size must be 0 or 1 value");
			}
			std::unique_ptr<tinygl::Mesh> mesh = tinygl::Mesh::create_line(from_x, from_y, to_x, to_y);
			return mesh;
		}
		std::unique_ptr<tinygl::Mesh> RectAstNode::create_mesh(PostParseContext *ctx, double *size, int size_len) {
			float xmin = -0.5;
			float ymin = -0.5;
			float xsize = 1;
			float ysize = 1;
			if(size_len == 2) {
				xmin = -size[0]/2;
				ymin = -size[1]/2;
				xsize = size[0];
				ysize = size[1];
			} else if(size_len == 1) {
				xmin = -size[0]/2;
				ymin = -size[0]/2;
				xsize = size[0];
				ysize = size[0];
			} else if(size_len != 0){
				ctx->raise(line, "rect's size must be 0 or 1 or 2 value");
			}
			return tinygl::Mesh::create_rect(xmin, ymin, xsize, ysize);
		}
		std::unique_ptr<tinygl::Mesh> PolygonAstNode::create_mesh(PostParseContext *ctx, double *size, int size_len) {
			float radius = 0.5;
			float edge_num = 5;
			if(size_len == 2) {
				radius = size[0]/2;
				edge_num = std::floor(size[1]);
			} else if(size_len == 1) {
				radius = size[0]/2;
			} else if(size_len != 0){
				ctx->raise(line, "polygon's size must be 0 or 1 or 2 value");
			}
			return tinygl::Mesh::create_polygon(0,0,radius,edge_num);
		}
		std::unique_ptr<tinygl::Mesh> CircleAstNode::create_mesh(PostParseContext *ctx, double *size, int size_len) {
			float radius = 0.5;
			if(size_len == 1) {
				radius = size[0]/2;
			} else if(size_len != 0){
				ctx->raise(line, "circle's size must be 0 or 1 value");
			}
			return tinygl::Mesh::create_circle(0, 0, radius);
		}
		std::unique_ptr<tinygl::Mesh> SectorAstNode::create_mesh(PostParseContext *ctx, double *size, int size_len) {
			float radius = 0.5;
			float degree = 45;
			if(size_len == 2) {
				radius = size[0]/2;
				degree = size[1];
			} else if(size_len == 1) {
				radius = size[0]/2;
			} else if(size_len != 0){
				ctx->raise(line, "sector's size must be 0 or 1 or 2 value");
			}
			return tinygl::Mesh::create_sector(0, 0, radius, degree);
		}
	}
	// MeshBlockAstNode
	namespace canvas {
		IValNode* MeshBlockAstNode::eval(EvalContext *ctx) {
			return new MeshBlockValNode(this);
		}
		BaseAstNode* MeshBlockAstNode::get_parent() {
			return camera_parent;
		}
	}
}
