
#pragma once

#include "numsky/canvas/IAstNode.h"
#include "tinygl/Screen.h"
#include "tinygl/Mesh.h"

namespace numsky {
	namespace canvas {
		class CameraAstNode: public BaseAstNode {
		public:
			int fi_rot;
			int fi_pos;
			tinygl::ScreenShape screen_shape;
			tinygl::Matrix4f projection_matrix;
		protected:
			int si_ortho;
			int si_perspective;
			int si_shape;
			void xparse_attr_rot(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) final;
			void xparse_attr_pos(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) final;
			void xparse_attr_Ortho(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) final;
			void xparse_attr_Perspective(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) final;
			void xparse_attr_Shape(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) final;
		public:
			friend class CameraValNode;
			CameraAstNode(): BaseAstNode(), fi_rot(0), fi_pos(0), screen_shape(4,4,1), si_ortho(0), si_perspective(0), si_shape(0) {}
			IValNode* eval(EvalContext *ctx) final;
			void post_parse(PostParseContext *ctx) final;
			BaseAstNode* xparse_child_mesh(ParseContext *ctx, rapidxml::xml_node<> *xnode, int mesh_enum) final;
			BaseAstNode* xparse_child_block(ParseContext *ctx, rapidxml::xml_node<> *xnode) final;
			bool xparse_has_scope() final;
		};

		class AbstractMeshAstNode: public BaseAstNode {
		public:
			int fi_rot;
			int fi_pos;
			int fi_scale;
			int fi_data;
			int fi_layer;
			tinygl::FILL_TYPE fill_type;
			std::unique_ptr<tinygl::Mesh> mesh_ptr;
		protected:
			void xparse_data(ParseContext *ctx, const char *data, int data_len, bool isPI) override;
			void xparse_attr_xtype(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) final;
			void xparse_attr_rot(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) final;
			void xparse_attr_pos(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) final;
			void xparse_attr_scale(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) final;
			void xparse_attr_layer(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) final;
		public:
			friend class MeshValNode;
			AbstractMeshAstNode(): BaseAstNode(), fi_rot(0), fi_pos(0), fi_scale(0), fi_data(0),
			fi_layer(0), fill_type(tinygl::FILL_TRIANGLE), mesh_ptr(nullptr) {}
			IValNode* eval(EvalContext *ctx) final;
			void post_parse(PostParseContext *ctx) override;
		};

		class MeshAstNode : public AbstractMeshAstNode {
		public:
			int si_vertices;
			int si_indices;
		protected:
			void xparse_attr_Vertices(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) final;
			void xparse_attr_Indices(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) final;
		public:
			MeshAstNode() : AbstractMeshAstNode(), si_vertices(0), si_indices(0) {}
			void post_parse(PostParseContext *ctx) final;
		};

		class BuiltinMeshAstNode : public AbstractMeshAstNode {
		public:
			int si_pivot;
			int si_size;
		protected:
			void xparse_attr_Pivot(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) final;
			void xparse_attr_Size(ParseContext *ctx, rapidxml::xml_attribute<> *xattr) final;
			virtual std::unique_ptr<tinygl::Mesh> create_mesh(PostParseContext *ctx, double* size, int size_len) = 0;
		public:
			void post_parse(PostParseContext *ctx) final;
			BuiltinMeshAstNode() : AbstractMeshAstNode(), si_pivot(0), si_size(0) {}
		};

		class PointAstNode : public BuiltinMeshAstNode {
		protected:
			std::unique_ptr<tinygl::Mesh> create_mesh(PostParseContext *ctx, double *size, int size_len) final;
		public:
			PointAstNode() {
				fill_type = tinygl::FILL_POINT;
			}
		};

		class LineAstNode : public BuiltinMeshAstNode {
		protected:
			std::unique_ptr<tinygl::Mesh> create_mesh(PostParseContext *ctx, double *size, int size_len) final;
		public:
			LineAstNode() {
				fill_type = tinygl::FILL_LINE;
			}
		};

		class RectAstNode : public BuiltinMeshAstNode {
		protected:
			std::unique_ptr<tinygl::Mesh> create_mesh(PostParseContext *ctx, double *size, int size_len) final;
		public:
			RectAstNode() {
				fill_type = tinygl::FILL_TRIANGLE;
			}
		};

		class PolygonAstNode : public BuiltinMeshAstNode {
		protected:
			std::unique_ptr<tinygl::Mesh> create_mesh(PostParseContext *ctx, double *size, int size_len) final;
		public:
			PolygonAstNode() {
				fill_type = tinygl::FILL_TRIANGLE;
			}
		};

		class CircleAstNode : public BuiltinMeshAstNode {
		protected:
			std::unique_ptr<tinygl::Mesh> create_mesh(PostParseContext *ctx, double *size, int size_len) final;
		public:
			CircleAstNode() {
				fill_type = tinygl::FILL_TRIANGLE;
			}
		};

		class SectorAstNode : public BuiltinMeshAstNode {
		protected:
			std::unique_ptr<tinygl::Mesh> create_mesh(PostParseContext *ctx, double *size, int size_len) final;
		public:
			SectorAstNode() {
				fill_type = tinygl::FILL_TRIANGLE;
			}
		};

		class MeshBlockAstNode : public AbstractBlockAstNode {
			CameraAstNode* camera_parent;
		public:
			MeshBlockAstNode(CameraAstNode* parent): AbstractBlockAstNode(), camera_parent(parent) {}
			BaseAstNode *get_parent() final;
			IValNode* eval(EvalContext *ctx) final;
		};

	}

}
