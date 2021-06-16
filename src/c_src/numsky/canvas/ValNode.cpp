
#include <algorithm>
#include "numsky/canvas/ValNode.h"
namespace numsky {

	namespace canvas {
		int IValNode::pre_eval(EvalContext *ctx, numsky_ndarray *arr) {
			return luaL_error(ctx->L, "pre eval not implement for this node");
		}
		void IValNode::draw_eval(EvalContext *ctx, tinygl::Camera *came) {
			luaL_error(ctx->L, "draw eval not implement for this node");
		}
		char* IValNode::cpy_eval(EvalContext *ctx, char* ptr) {
			luaL_error(ctx->L, "cpy eval not implement for this node");
			return NULL;
		}
		void IValNode::ret_eval(EvalContext *ctx, int tablei) {
			luaL_error(ctx->L, "ret eval not implement for this node");
		}
	}

	// ScalarValNode
	namespace canvas {
		int ScalarValNode::pre_eval(EvalContext *ctx, numsky_ndarray *arr) {
			if(ast_node->fi_data != 0) {
				ctx->scalar(ast_node->fi_data, [&](int nresults) {
					len = nresults;
					datamem.reset(new char[ast_node->dtype->elsize*len]);
					int error_index = ast_node->lcpy_func(ctx->L, datamem.get(), nresults);
					if(error_index > 0) {
						std::string istr = std::to_string(error_index);
						ctx->raise("scalar eval error:", istr);
					}
				});
			} else {
				// TODO throw exception ?
				len = 0;
				ctx->raise("scalar must has at least 1 item");
			}
			if(ast_node->type_guard.len>0) {
				ctx->assert_length(ast_node->type_guard.len, len);
			}
			return len;
		}
		char* ScalarValNode::cpy_eval(EvalContext *ctx, char* ptr) {
			return ast_node->cpy_func(ctx->L, ptr, datamem.get(), len);
		}
		void ScalarValNode::ret_eval(EvalContext *ctx, int tablei) {
			char *dataptr = datamem.get();
			for(int i=0;i<len;i++) {
				ast_node->dtype->dataptr_push(ctx->L, dataptr);
				ctx->ret_top(ast_node, tablei);
				dataptr += ast_node->dtype->elsize;
			}
		}
	}

	// AnyValNode
	namespace canvas {
		int AnyValNode::pre_eval(EvalContext *ctx, numsky_ndarray *arr) {
			if(ast_node->fi_data != 0) {
				std::tie(val_stacki, val_num) = ctx->eval_any(ast_node->fi_data);
			}
			return 0;
		}
		void AnyValNode::ret_eval(EvalContext *ctx, int tablei) {
			for(int i=0;i<val_num;i++) {
				lua_pushvalue(ctx->L, val_stacki + i);
				ctx->ret_top(ast_node, tablei);
			}
		}
	}

	namespace canvas {
		template <typename T> int ChildableValNode<T>::expand_children(EvalContext *ctx, numsky_ndarray *arr) {
			int dim = 0;
			for(auto iter=this->ast_node->children.begin();iter!=this->ast_node->children.end();++iter) {
				IAstNode* childAst = *iter;
				ctx->set_cursor(childAst);
				ExpandControl *ctrl = childAst->get_expand_control();
				if(ctrl == NULL) {
					childAst->eval(ctx);
					continue;
				}
				if(ctrl->fi_if > 0 && (ctrl->fi_if < ctrl->fi_forvar || ctrl->fi_forvar == 0)) {
					bool existed = ctx->boolean(ctrl->fi_if);
					if(!existed) {
						continue;
					}
				}
				if(ctrl->fi_forvar > 0) {
					int count = 0;
					if(ctrl->fi_forsort > 0) {
						std::vector<std::pair<double, IValNode*>> sort_vec;
						auto lam = [&]() {
							bool existed = true;
							if(ctrl->fi_if > 0) {
								existed = ctx->boolean(ctrl->fi_if);
							}
							if(existed) {
								IValNode *val_node = childAst->eval(ctx);
								dim += val_node->pre_eval(ctx, arr);
								count ++;
								double sort_key = ctx->number(ctrl->fi_forsort);
								sort_vec.push_back(std::make_pair(sort_key, val_node));
							}
						};
						if(ctrl->fi_forgen > 0) {
							ctx->forin(ctrl->fi_forvar, ctrl->fi_forgen, lam);
						} else if(ctrl->fi_forseq > 0) {
							ctx->foreq(ctrl->fi_forvar, ctrl->fi_forseq, lam);
						} else {
							luaL_error(ctx->L, "[fatal error]:impossible branch in expand children");
						}
						std::sort(sort_vec.begin(), sort_vec.end(), [](std::pair<double, IValNode*> &a, std::pair<double, IValNode*> &b) ->bool {
								return a.first < b.first;
						});
						for(auto pair_iter=sort_vec.begin();pair_iter!=sort_vec.end();++pair_iter) {
							children.push_back(pair_iter->second);
						}
					} else {
						auto lam = [&]() {
							bool existed = true;
							if(ctrl->fi_if > 0) {
								existed = ctx->boolean(ctrl->fi_if);
							}
							if(existed) {
								IValNode *val_node = childAst->eval(ctx);
								dim += val_node->pre_eval(ctx, arr);
								count ++;
								children.push_back(val_node);
							}
						};
						if(ctrl->fi_forgen > 0) {
							ctx->forin(ctrl->fi_forvar, ctrl->fi_forgen, lam);
						} else if(ctrl->fi_forseq > 0) {
							ctx->foreq(ctrl->fi_forvar, ctrl->fi_forseq, lam);
						} else {
							luaL_error(ctx->L, "[fatal error]:impossible branch in expand children");
						}
					}
					TypeGuard *child_guard = childAst->get_type_guard();
					if(child_guard!=NULL) {
						if(child_guard->count > 0) {
							if(child_guard->count != count) {
								ctx->raise("control count not match");
							}
						}
					}
				} else {
					IValNode *val_node = childAst->eval(ctx);
					if(val_node != NULL) {
						dim += val_node->pre_eval(ctx, arr);
						children.push_back(val_node);
					}
				}
			}
			return dim;
		}
	}

	// BlockValNode
	namespace canvas {
		int LuaBlockValNode::pre_eval(EvalContext *ctx, numsky_ndarray *arr) {
			int count = this->expand_children(ctx, arr);
			return count;
		}
		void LuaBlockValNode::ret_eval(EvalContext *ctx, int tablei) {
			for(auto iter=this->children.begin();iter!=this->children.end();++iter) {
				(*iter)->ret_eval(ctx, tablei);
			}
		}
	}

	namespace canvas {
		int ArrBlockValNode::pre_eval(EvalContext *ctx, numsky_ndarray *arr) {
			int count = this->expand_children(ctx, arr);
			if(this->ast_node->type_guard.len>0) {
				ctx->assert_length(this->ast_node->type_guard.len, count);
			}
			return count;
		}
		char* ArrBlockValNode::cpy_eval(EvalContext *ctx, char* ptr) {
			for(auto iter=this->children.begin();iter!=this->children.end();++iter) {
				ptr = (*iter)->cpy_eval(ctx, ptr);
			}
			return ptr;
		}
	}

	// ListValNode
	namespace canvas {
		int ListValNode::pre_eval(EvalContext *ctx, numsky_ndarray *arr) {
			int dim = expand_children(ctx, arr);
			if(ast_node->type_guard.len>0) {
				ctx->assert_length(ast_node->type_guard.len, dim);
			}
			// check if dim match
			int dimi = arr->nd-ast_node->ndim;
			if(arr->dimensions[dimi]==0){
				arr->dimensions[dimi] = dim;
			} else if(arr->dimensions[dimi] != dim) {
				ctx->raise("dim i not match");
			}
			return 1;
		}
		char* ListValNode::cpy_eval(EvalContext *ctx, char* ptr) {
			for(auto iter=children.begin();iter!=children.end();++iter) {
				ptr = (*iter)->cpy_eval(ctx, ptr);
			}
			return ptr;
		}
	}

	// ArrayValNode
	namespace canvas {
		int ArrayValNode::pre_eval(EvalContext *ctx, numsky_ndarray *arr) {
			if(arr!=NULL) {
				luaL_error(ctx->L, "[fatal error]:array val's arr not NULL exception");
			}
			arr = arr_ptr.get();
			int dim = expand_children(ctx, arr);
			if(ast_node->type_guard.len>0) {
				ctx->assert_length(ast_node->type_guard.len, dim);
			}
			if(arr->dimensions[0] == 0) {
				arr->dimensions[0] = dim;
			} else if(arr->dimensions[0] != dim) {
				ctx->raise("dim 0 not match");
			}
			// alloc
			numsky_ndarray_autostridecountalloc(arr);
			return 1;
		}
		void ArrayValNode::ret_eval(EvalContext *ctx, int tablei) {
			char *ptr = arr_ptr->dataptr;
			for(auto iter=children.begin();iter!=children.end();++iter) {
				ptr = (*iter)->cpy_eval(ctx, ptr);
			}
			numsky::ndarray_mem2lua(ctx->L, arr_ptr);
			ctx->ret_top(ast_node, tablei);
		}
	}

	// CameraValNode
	namespace canvas {
		int CameraValNode::pre_eval(EvalContext *ctx, numsky_ndarray *arr) {
			if(ast_node->fi_pos > 0) {
				int nresults;
				std::unique_ptr<double[]> nargs = ctx->eval_numbers(ast_node->fi_pos, nresults);
				if(nresults == 2) {
					pos.X = nargs[0];
					pos.Y = nargs[1];
					pos.Z = 0;
				} else if(nresults == 3) {
					pos.X = nargs[0];
					pos.Y = nargs[1];
					pos.Z = nargs[2];
				} else {
					ctx->raise("position must be 2 or 3 value");
				}
			}
			if(ast_node->fi_rot > 0) {
				int nresults;
				std::unique_ptr<double[]> nargs = ctx->eval_numbers(ast_node->fi_rot, nresults);
				if(nresults == 1) {
					rot.X = 0;
					rot.Y = 0;
					rot.Z = nargs[0];
				} else if(nresults == 3) {
					rot.X = nargs[0];
					rot.Y = nargs[1];
					rot.Z = nargs[2];
				} else {
					ctx->raise("rotation must be 1 or 3 value");
				}
			}
			expand_children(ctx, NULL);
			return 1;
		}
		void CameraValNode::ret_eval(EvalContext *ctx, int tablei) {
			tinygl::Camera *came = camera_ptr.get();
			came->setPosition(pos.X, pos.Y, pos.Z);
			came->setRotation(rot.X, rot.Y, rot.Z);
			for(auto iter=children.begin();iter!=children.end();++iter) {
				(*iter)->draw_eval(ctx, came);
			}
			//luabinding::ClassUtil<tinygl::Camera>::newwrap(ctx->L, camera_ptr.release());
			numsky::ltinygl_camera_pixel_array<true>(ctx->L, camera_ptr.get());
			ctx->ret_top(ast_node, tablei);
		}
	}

	// MeshValNode
	namespace canvas {
		int MeshValNode::pre_eval(EvalContext *ctx, numsky_ndarray *arr) {
			if(ast_node->fi_pos > 0) {
				int nresults;
				std::unique_ptr<double[]> nargs = ctx->eval_numbers(ast_node->fi_pos, nresults);
				if(nresults == 2) {
					pos.X = nargs[0];
					pos.Y = nargs[1];
					pos.Z = 0;
				} else if(nresults == 3) {
					pos.X = nargs[0];
					pos.Y = nargs[1];
					pos.Z = nargs[2];
				} else {
					ctx->raise("position must be 2 or 3 value");
				}
			}
			if(ast_node->fi_rot > 0) {
				int nresults;
				std::unique_ptr<double[]> nargs = ctx->eval_numbers(ast_node->fi_rot, nresults);
				if(nresults == 1) {
					rot.X = 0;
					rot.Y = 0;
					rot.Z = nargs[0];
				} else if(nresults == 3) {
					rot.X = nargs[0];
					rot.Y = nargs[1];
					rot.Z = nargs[2];
				} else {
					ctx->raise("rotation must be 1 or 3 value");
				}
			}
			if(ast_node->fi_scale > 0) {
				int nresults;
				std::unique_ptr<double[]> nargs = ctx->eval_numbers(ast_node->fi_scale, nresults);
				if(nresults == 2) {
					scale.X = nargs[0];
					scale.Y = nargs[1];
					scale.Z = 0;
				} else if(nresults == 3) {
					scale.X = nargs[0];
					scale.Y = nargs[1];
					scale.Z = nargs[2];
				} else {
					ctx->raise("scale must be 2 or 3 value");
				}
			}
			if(ast_node->fi_data != 0) {
				color_data = ctx->eval_bytes(ast_node->fi_data, color_pixelsize);
			}
			if(ast_node->fi_layer != 0) {
				layer = ctx->integer(ast_node->fi_layer) - 1;
				if(layer < 0) {
					ctx->raise("layer must be >= 1");
				}
			}
			return 0;
		}
		void MeshValNode::draw_eval(EvalContext *ctx, tinygl::Camera *came) {
			tinygl::Mesh* mesh = ast_node->mesh_ptr.get();
			mesh->shader.setColor(color_pixelsize, color_data.get());
			mesh->shader.layer = layer;
			mesh->setRotation(rot.X, rot.Y, rot.Z);
			mesh->setPosition(pos.X, pos.Y, pos.Z);
			mesh->setScale(scale.X, scale.Y, scale.Z);
			came->draw(mesh);
		}
	}

	// MeshBlockValNode
	namespace canvas {
		int MeshBlockValNode::pre_eval(EvalContext *ctx, numsky_ndarray *arr) {
			this->expand_children(ctx, arr);
			return 1;
		}
		void MeshBlockValNode::draw_eval(EvalContext *ctx, tinygl::Camera *came) {
			for(auto iter=children.begin();iter!=children.end();++iter) {
				(*iter)->draw_eval(ctx, came);
			}
		}
	}

	// TableValNode
	namespace canvas {
		int TableValNode::pre_eval(EvalContext *ctx, numsky_ndarray *arr) {
			expand_children(ctx, NULL);
			return 1;
		}
		void TableValNode::ret_eval(EvalContext *ctx, int parent_tablei) {
			lua_newtable(ctx->L);
			int self_tablei = lua_gettop(ctx->L);
			for(auto iter=children.begin();iter!=children.end();++iter) {
				IValNode *child = (*iter);
				child->ret_eval(ctx, self_tablei);
			}
			ctx->ret_top(ast_node, parent_tablei);
		}
	}

	// ReturnValNode
	namespace canvas {
		int ReturnValNode::pre_eval(EvalContext *ctx, numsky_ndarray *arr) {
			expand_children(ctx, NULL);
			return 1;
		}
		void ReturnValNode::ret_eval(EvalContext *ctx, int tablei) {
			for(auto iter=children.begin();iter!=children.end();++iter) {
				(*iter)->ret_eval(ctx, 0);
			}
		}

	}
}
