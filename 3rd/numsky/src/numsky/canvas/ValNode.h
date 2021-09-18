
#ifndef __VAL_NODE_H__
#define __VAL_NODE_H__
#include <memory>
#include <string>
#include <vector>
#include <map>

#include "numsky/canvas/AstNode.h"
#include "numsky/canvas/EvalContext.h"
#include "numsky/lua-numsky.h"
#include "numsky/tinygl/lua-numsky_tinygl.h"
namespace numsky {
	namespace canvas {

		class IValNode {
			public:
				IValNode(BaseAstNode* an) {}
				virtual int pre_eval(EvalContext *ctx, numsky_ndarray *arr);
				virtual void draw_eval(EvalContext *ctx, tinygl::Camera *came);
				virtual char* cpy_eval(EvalContext *ctx, char* ptr);
				virtual void ret_eval(EvalContext *ctx, int tablei);
				virtual ~IValNode() {};
		};

		template <typename TAstNode> class BaseValNode : public IValNode {
			protected:
				TAstNode* ast_node;
			public:
				BaseValNode(TAstNode *an) : IValNode(an), ast_node(an) {}
				~BaseValNode() override {};
		};

		class ScalarValNode : public BaseValNode<ScalarAstNode> {
			protected:
				int len;
				std::unique_ptr<char []> datamem;
			public:
				ScalarValNode(ScalarAstNode* an) : BaseValNode<ScalarAstNode>(an) {}
				int pre_eval(EvalContext *ctx, numsky_ndarray *arr) override;
				char* cpy_eval(EvalContext *ctx, char* ptr) override;
				void ret_eval(EvalContext *ctx, int tablei) override;
		};

		class AnyValNode : public BaseValNode<AnyAstNode> {
			protected:
				int val_stacki;
				int val_num;
			public:
				AnyValNode(AnyAstNode* an) : BaseValNode<AnyAstNode>(an), val_stacki(0), val_num(0) {}
				int pre_eval(EvalContext *ctx, numsky_ndarray *arr) override;
				void ret_eval(EvalContext *ctx, int tablei) override;
		};

		template <typename TAstNode> class ChildableValNode : public BaseValNode<TAstNode> {
			protected:
				std::vector<IValNode*> children;
			public:
				ChildableValNode(TAstNode *an) : BaseValNode<TAstNode>(an) {}
				int expand_children(EvalContext *ctx, numsky_ndarray *arr);
				~ChildableValNode() override {
					for(auto iter=children.begin();iter!=children.end();++iter) {
						delete *iter;
					}
				}
		};

		class LuaBlockValNode : public ChildableValNode<LuaBlockAstNode> {
			public:
				LuaBlockValNode(LuaBlockAstNode * an) : ChildableValNode<LuaBlockAstNode>(an) {}
				int pre_eval(EvalContext *ctx, numsky_ndarray *arr) override;
				void ret_eval(EvalContext *ctx, int tablei) override;
		};

		class ArrBlockValNode : public ChildableValNode<ArrBlockAstNode> {
			public:
				ArrBlockValNode(ArrBlockAstNode * an) : ChildableValNode<ArrBlockAstNode>(an) {}
				int pre_eval(EvalContext *ctx, numsky_ndarray *arr) override;
				char* cpy_eval(EvalContext *ctx, char* ptr) override;
		};

		class ListValNode : public ChildableValNode<ListAstNode> {
			public:
				ListValNode(ListAstNode * an) : ChildableValNode<ListAstNode>(an) {}
				int pre_eval(EvalContext *ctx, numsky_ndarray *arr) override;
				char* cpy_eval(EvalContext *ctx, char* ptr) override;
		};

		class ArrayValNode : public ChildableValNode<ArrayAstNode> {
			protected:
				std::unique_ptr<numsky_ndarray, void(*)(numsky_ndarray*)> arr_ptr;
			public:
				ArrayValNode(ArrayAstNode * an) : ChildableValNode<ArrayAstNode>(an),
				arr_ptr(numsky::ndarray_new_preinit<false>(NULL, an->ndim, an->dtype->typechar)) {
					for(int i=0;i<an->ndim;i++) {
						arr_ptr->dimensions[i] = an->shape[i];
					}
				}
				int pre_eval(EvalContext *ctx, numsky_ndarray *arr) override;
				void ret_eval(EvalContext *ctx, int tablei) override;
		};

		class CameraValNode : public ChildableValNode<CameraAstNode> {
			protected:
				std::unique_ptr<tinygl::Camera> camera_ptr;
				tinygl::V3f pos;
				tinygl::V3f rot;
			public:
				CameraValNode(CameraAstNode * an) : ChildableValNode<CameraAstNode>(an),
				camera_ptr(numsky::tinygl_camera_newunsafe(an->screen_shape)), pos(0,0,0), rot(0,0,0) {
					camera_ptr->set_projection_matrix(an->projection_matrix);
				}
				int pre_eval(EvalContext *ctx, numsky_ndarray *arr) final;
				void ret_eval(EvalContext *ctx, int tablei) final;
		};

		class MeshValNode : public BaseValNode<AbstractMeshAstNode> {
			protected:
				int color_pixelsize;
				std::unique_ptr<uint8_t[]> color_data;
				tinygl::V3f pos;
				tinygl::V3f rot;
				tinygl::V3f scale;
				int layer;
			public:
				MeshValNode(AbstractMeshAstNode * an) : BaseValNode<AbstractMeshAstNode>(an), color_pixelsize(0),
				pos(0,0,0), rot(0,0,0), scale(1,1,1), layer(0) {}
				int pre_eval(EvalContext *ctx, numsky_ndarray *arr) final;
				void draw_eval(EvalContext *ctx, tinygl::Camera *came) final;
		};

		class MeshBlockValNode : public ChildableValNode<MeshBlockAstNode> {
			public:
				MeshBlockValNode(MeshBlockAstNode * an) : ChildableValNode<MeshBlockAstNode>(an) {}
				int pre_eval(EvalContext *ctx, numsky_ndarray *arr) final;
				void draw_eval(EvalContext *ctx, tinygl::Camera *came) final;
		};

		class TableValNode : public ChildableValNode<TableAstNode> {
			public:
				TableValNode(TableAstNode* an) : ChildableValNode<TableAstNode>(an) {}
				int pre_eval(EvalContext *ctx, numsky_ndarray *arr) final;
				void ret_eval(EvalContext *ctx, int tablei) final;
		};

		class ReturnValNode : public ChildableValNode<numsky_canvas> {
			public:
				ReturnValNode(numsky_canvas* an) : ChildableValNode<numsky_canvas>(an) {}
				int pre_eval(EvalContext *ctx, numsky_ndarray *arr) final;
				void ret_eval(EvalContext *ctx, int tablei) final;
		};

	}
}


#endif
