

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <sstream>

#include "numsky/lua-numsky.h"
#include "numsky/ndarray/lua-numsky_ndarray.h"
#include "numsky/canvas/ParseContext.h"
#include "numsky/canvas/ExpandControl.h"

namespace numsky {
	namespace canvas {
		class BaseAstNode;

		struct TypeGuard {
			int si_len;
			int si_count;
			int si_shape;
			int len;
			int count;
			npy_intp* shape;
			ExpandControl *ctrl;
			TypeGuard(ExpandControl *v_ctrl) : si_len(0), si_count(0), si_shape(0), len(0), count(0), shape(nullptr), ctrl(v_ctrl) {}
			inline void point_shape(npy_intp* v_shape) {
				shape = v_shape;
			}
			inline int len_count(PostParseContext *ctx, int line) {
				if(len == 0) {
					return 0;
				}
				if(ctrl->fi_if == 0 && ctrl->fi_forvar == 0) {
					if(si_count != 0) {
						ctx->raise(line, "attribute count can't stay without for or if attribute");
						return 0;
					} else {
						return len;
					}
				} else {
					if(si_count == 0) {
						return 0;
					} else {
						return len * count;
					}
				}
			}
			inline void eval(PostParseContext *ctx, int line, int ndim) {
				len = 0;
				count = 0;
				if(si_shape != 0) {
					ctx->eval(si_shape, [&](int nresults){
						if(nresults!=ndim) {
							ctx->raise(line, "shape must match ndim");
						} else {
							for(int i=0;i<ndim;i++) {
								int dim = ctx->check_length<0>(line, i-ndim);
								if(dim == 0) {
									continue;
								} else if(shape[i] == 0) {
									shape[i] = dim;
								} else if(shape[i]!=dim) {
									ctx->raise(line, "dim not match");
								}
							}
						}
					});
				}
				if(si_len != 0) {
					ctx->eval(si_len, [&](int nresults){
						if(nresults!=1) {
							ctx->raise(line, "len must be only 1 value");
						} else {
							len = ctx->check_length<1>(line, -1);
						}
					});
				}
				if(si_count != 0) {
					ctx->eval(si_count, [&](int nresults){
						if(nresults!=1) {
							ctx->raise(line, "count must be only 1 value");
						} else {
							count = ctx->check_length<1>(line, -1);
						}
					});
				}
			}
		};

	}
}

