
#pragma once

#include <tuple>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>

#include "numsky/lua-numsky.h"

#define lnumsky_template_fp2t2(L, typechar1, T2, tfunc) ([] (lua_State* _L, char tc1) -> decltype(tfunc<bool, T2>)* { \
decltype(tfunc<bool, T2>) * name = NULL;\
switch(tc1){ \
	case '?': { name = tfunc<bool, T2>; break; } \
	case 'b': { name = tfunc<int8_t, T2>; break; } \
	case 'B': { name = tfunc<uint8_t, T2>; break; } \
	case 'h': { name = tfunc<int16_t, T2>; break; } \
	case 'H': { name = tfunc<uint16_t, T2>; break; } \
	case 'i': { name = tfunc<int32_t, T2>; break; } \
	case 'I': { name = tfunc<uint32_t, T2>; break; } \
	case 'l': { name = tfunc<int64_t, T2>; break; } \
	case 'L': { luaL_error(_L, "uint64 not support"); break; } \
	case 'f': { name = tfunc<float, T2>; break; } \
	case 'd': { name = tfunc<double, T2>; break; } \
	default: { luaL_error(_L, "ERROR!!!!!, dtype_pushdata unexcept branch\n"); break; } \
} return name;})(L, typechar1)

#define lnumsky_template_fp2t1(L, T1, typechar2, tfunc) ([] (lua_State* _L, char tc2) -> decltype(tfunc<T1, bool>)* { \
decltype(tfunc<T1, bool>) * name = NULL;\
switch(tc2){ \
	case '?': { name = tfunc<T1, bool>; break; } \
	case 'b': { name = tfunc<T1, int8_t>; break; } \
	case 'B': { name = tfunc<T1, uint8_t>; break; } \
	case 'h': { name = tfunc<T1, int16_t>; break; } \
	case 'H': { name = tfunc<T1, uint16_t>; break; } \
	case 'i': { name = tfunc<T1, int32_t>; break; } \
	case 'I': { name = tfunc<T1, uint32_t>; break; } \
	case 'l': { name = tfunc<T1, int64_t>; break; } \
	case 'L': { luaL_error(_L, "uint64 not support"); break; } \
	case 'f': { name = tfunc<T1, float>; break; } \
	case 'd': { name = tfunc<T1, double>; break; } \
	default: { luaL_error(_L, "ERROR!!!!!, dtype_pushdata unexcept branch\n"); break; } \
} return name;})(L, typechar2)

#define lnumsky_template_fp2(L, typechar1, typechar2, tfunc) ([] (lua_State* _L, char tc1, char tc2) -> decltype(tfunc<bool, bool>) * { \
decltype(tfunc<bool, bool>) * func = NULL;\
switch(tc2){ \
	case '?': { func = lnumsky_template_fp2t2(_L, tc1, bool, tfunc); break; } \
	case 'b': { func = lnumsky_template_fp2t2(_L, tc1, int8_t, tfunc); break; } \
	case 'B': { func = lnumsky_template_fp2t2(_L, tc1, uint8_t, tfunc); break; } \
	case 'h': { func = lnumsky_template_fp2t2(_L, tc1, int16_t, tfunc); break; } \
	case 'H': { func = lnumsky_template_fp2t2(_L, tc1, uint16_t, tfunc); break; } \
	case 'i': { func = lnumsky_template_fp2t2(_L, tc1, int32_t, tfunc); break; } \
	case 'I': { func = lnumsky_template_fp2t2(_L, tc1, uint32_t, tfunc); break; } \
	case 'l': { func = lnumsky_template_fp2t2(_L, tc1, int64_t, tfunc); break; } \
	case 'L': { luaL_error(_L, "uint64 not support"); break; } \
	case 'f': { func = lnumsky_template_fp2t2(_L, tc1, float, tfunc); break; } \
	case 'd': { func = lnumsky_template_fp2t2(_L, tc1, double, tfunc); break; } \
	default: { luaL_error(_L, "ERROR!!!!!, dtype_pushdata unexcept branch\n"); break; } \
} return func;})(L, typechar1, typechar2)

namespace numsky {

	class ThrowableContext {
	public:
		lua_State*L;
		explicit ThrowableContext(lua_State*l): L(l) {}
		virtual void throw_func(const std::string& data);
	};

	static inline std::unique_ptr<numsky_nditer, void (*)(numsky_nditer*)> ndarray_nditer(numsky_ndarray* arr) {
		std::unique_ptr<numsky_nditer, void (*)(numsky_nditer*)> ptr(numsky_nditer_create(arr), numsky_nditer_destroy);
		return ptr;
	}

    static inline void ndarray_foreach(numsky_ndarray* arr, const std::function<void(numsky_nditer*)> & func) {
		auto ptr = ndarray_nditer(arr);
		for(npy_intp n=0;n<arr->count;n++) {
			func(ptr.get());
			numsky_nditer_next(ptr.get());
		}
	}

	template <typename TArr, typename TBuffer> void ndarray_t_copyfrom(numsky_ndarray* arr, char *dataptr) {
		auto iter = ndarray_nditer(arr);
		for(npy_intp n=0;n<arr->count;n++) {
			numsky::dataptr_cast<TArr>(iter->dataptr) = numsky::dataptr_cast<TBuffer>(dataptr);
			dataptr += sizeof(TBuffer);
			numsky_nditer_next(iter.get());
		}
	}

	template <typename TArr, typename TBuffer> void ndarray_t_copyto(numsky_ndarray* arr, char *dataptr) {
		auto iter = ndarray_nditer(arr);
		for(npy_intp n=0;n<arr->count;n++) {
			numsky::dataptr_cast<TBuffer>(dataptr) = numsky::dataptr_cast<TArr>(iter->dataptr);
			dataptr += sizeof(TBuffer);
			numsky_nditer_next(iter.get());
		}
	}

	// axis : start from 0
    static inline void ndarray_axis_foreach(numsky_ndarray* arr, std::vector<int>& axis, const std::function<void(numsky_nditer*)> & func) {
		auto sub_arr = ndarray_new_preinit<false>(nullptr, arr->nd - axis.size(), arr->dtype->typechar);
		numsky_ndarray_refdata(sub_arr.get(), NULL, arr->dataptr);
		for(int i=0;i<sub_arr->nd;i++) {
			sub_arr->dimensions[i] = arr->dimensions[axis[i]];
			sub_arr->strides[i] = arr->strides[axis[i]];
		}
		numsky_ndarray_autocount(sub_arr.get());
		ndarray_foreach(sub_arr.get(), func);
	}

	static inline std::vector<npy_intp> ndarray_broadcasting_dimensions(lua_State *L, numsky_ndarray *arr_a, numsky_ndarray *arr_b) {
		std::vector<npy_intp> dimensions;
		dimensions.resize(std::max(arr_a->nd, arr_b->nd));
		int ia=arr_a->nd-1;
		int ib=arr_b->nd-1;
		int inew=std::max(ia, ib);
		while(ia >= 0 && ib >= 0) {
			if(arr_a->dimensions[ia] == arr_b->dimensions[ib]) {
				dimensions[inew] = arr_a->dimensions[ia];
			} else if(arr_a->dimensions[ia] == 1) {
				dimensions[inew] = arr_b->dimensions[ib];
			} else if(arr_b->dimensions[ib] == 1) {
				dimensions[inew] = arr_a->dimensions[ia];
			} else {
				luaL_error(L, "bop not broadcasting able");
				return dimensions;
			}
			inew--;
			ia--;
			ib--;
		}
		while(ib >= 0) {
			dimensions[inew] = arr_b->dimensions[ib];
			inew--;
			ib--;
		}
		while(ia >= 0) {
			dimensions[inew] = arr_a->dimensions[ia];
			inew--;
			ia--;
		}
		return dimensions;
	}

	static inline void nditer_broadcasting_next(numsky_nditer *iter_a, numsky_nditer *iter_b) {
		int ia=iter_a->nd-1;
		int ib=iter_b->nd-1;
		while(true) {
			if(ia >= 0 && ib >= 0) {
				int dim_ia_m1 = iter_a->ao->dimensions[ia] - 1;
				int dim_ib_m1 = iter_b->ao->dimensions[ib] - 1;
				if(iter_a->coordinates[ia] < dim_ia_m1 && iter_b->coordinates[ib] < dim_ib_m1) {
					iter_a->coordinates[ia] ++;
					iter_a->dataptr += iter_a->ao->strides[ia];

					iter_b->coordinates[ib] ++;
					iter_b->dataptr += iter_b->ao->strides[ib];
					return ;
				} else if(iter_a->coordinates[ia] < dim_ia_m1){
					iter_a->coordinates[ia] ++;
					iter_a->dataptr += iter_a->ao->strides[ia];
					return ;
				} else if(iter_b->coordinates[ib] < dim_ib_m1) {
					iter_b->coordinates[ib] ++;
					iter_b->dataptr += iter_b->ao->strides[ib];
					return ;
				} else {
					iter_a->coordinates[ia] = 0;
					iter_a->dataptr -= iter_a->ao->strides[ia] * dim_ia_m1;
					ia --;

					iter_b->coordinates[ib] = 0;
					iter_b->dataptr -= iter_b->ao->strides[ib] * dim_ib_m1;
					ib --;
				}
			} else if(ia>=0) {
				int dim_ia_m1 = iter_a->ao->dimensions[ia] - 1;
				if(iter_a->coordinates[ia] < dim_ia_m1){
					iter_a->coordinates[ia] ++;
					iter_a->dataptr += iter_a->ao->strides[ia];
					return ;
				} else {
					iter_a->coordinates[ia] = 0;
					iter_a->dataptr -= iter_a->ao->strides[ia] * dim_ia_m1;
					ia --;
				}
			} else if(ib>=0) {
				int dim_ib_m1 = iter_b->ao->dimensions[ib] - 1;
				if(iter_b->coordinates[ib] < dim_ib_m1){
					iter_b->coordinates[ib] ++;
					iter_b->dataptr += iter_b->ao->strides[ib];
					return ;
				} else {
					iter_b->coordinates[ib] = 0;
					iter_b->dataptr -= iter_b->ao->strides[ib] * dim_ib_m1;
					ib --;
				}
			} else {
				return ;
			}
		}
	}

	static inline void ndarray_broadcasting_foreach(numsky_ndarray* later_arr, numsky_ndarray* arr1, numsky_ndarray* arr2, const std::function<void(numsky_nditer*, numsky_nditer*)> & func) {
		auto ptr1 = ndarray_nditer(arr1);
		auto ptr2 = ndarray_nditer(arr2);
		for(npy_intp n=0;n<later_arr->count;n++) {
			func(ptr1.get(), ptr2.get());
			nditer_broadcasting_next(ptr1.get(), ptr2.get());
		}
	}


	/*******************
	* ndarray_ctor.cpp *
	*******************/
    template <bool InLua> extern std::unique_ptr<numsky_ndarray, void(*)(numsky_ndarray*)>
	table_to_array(ThrowableContext *ctx, int table_idx, char typechar);

    inline std::unique_ptr<numsky_ndarray, void(*)(numsky_ndarray*)>
	check_temp_ndarray(ThrowableContext *ctx, int table_idx, char default_typechar) {
		lua_State* L = ctx->L;
		int type1 = lua_type(L, table_idx);
		if(type1 == LUA_TUSERDATA) {
			numsky_ndarray *sub_arr = luabinding::ClassUtil<numsky_ndarray>::test(L, table_idx);
			if(sub_arr!=NULL) {
				std::unique_ptr<numsky_ndarray, void(*)(numsky_ndarray*)> ptr(sub_arr, [](numsky_ndarray*l){});
				return ptr;
			}
		} else if(type1 == LUA_TTABLE){
			return table_to_array<false>(ctx, table_idx, default_typechar);
		}
		ctx->throw_func("only table or numsky.ndarray can be checked to numsky.ndarray");
		return std::unique_ptr<numsky_ndarray, void(*)(numsky_ndarray*)>(nullptr, [](numsky_ndarray*l){});
	}

	int ctor_empty(lua_State *L);
	int ctor_zeros(lua_State *L);
	int ctor_ones(lua_State *L);

	int ctor_arange(lua_State *L);
	int ctor_linspace(lua_State *L);
	//int ctor_logspace(lua_State *L);

	/**********************
	* ndarray_methods.cpp *
	***********************/
	int ndarray_methods_flatten(lua_State *L);
	int ndarray_methods_reshape(lua_State *L);
	int ndarray_methods_copy(lua_State *L);
	int ndarray_methods_roll(lua_State *L);
	int ndarray_methods_astype(lua_State *L);


	// http://lua-users.org/wiki/MetatableEvents

	/***********************
	* ndarray_tostring.cpp *
	***********************/
    int ndarray__tostring(lua_State* L);

	/***********************
	* ndarray_indexing.cpp *
	***********************/
	int ndarray__newindex(lua_State *L);
	int ndarray__index(lua_State *L);

	//int ndarray__unm(lua_State *L); // -a
	// int ndarray__concat(lua_State *L); // a..b*/
	//int ndarray__bnot(lua_State *L); // ~a

} // namespace numsky

