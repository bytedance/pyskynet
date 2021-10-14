#include <vector>

#include "numsky/lua-numsky.h"
#include "numsky/ndarray/lua-numsky_ndarray.h"

#define SLICE_NIL 0

// for recording stride offset when indexing array
class ListStride {
private:
	npy_intp *mStrides;
	npy_intp mLen;
	int mIDim;
public:
	ListStride(npy_intp vLen, int vIDim) : mStrides(new npy_intp[vLen]), mLen(vLen), mIDim(vIDim) {}
	~ListStride() { delete [] mStrides; }
	npy_intp & stride(int i){
		return mStrides[i];
	}
	npy_intp len(){
		return mLen;
	}
	char *shift(numsky_nditer *nditer){
		return nditer->dataptr + mStrides[nditer->coordinates[mIDim]];
	}
};

namespace _ndarray_dim {
	// cut a dim by integer, return dataptr offset
	static inline npy_intp integer_cut(lua_State*L, int64_t integer, numsky_ndarray *arr, int arr_i){
		npy_intp dim = arr->dimensions[arr_i];
		npy_intp stride = arr->strides[arr_i];
		int64_t start = integer;
		if(start < 0) {
			start = dim + 1 + start;
		}
		if(start < 1 || start > dim) {
			luaL_error(L, "slice index %d not in range [%d, %d] or [%d, %d]",
					integer, dim, -1, 1, dim);
		}
		return (start - 1) * stride;
	}

	// cut a dim by slice, return dataptr offset, save dim & stride in reference
	static inline npy_intp slice_cut(lua_State*L, numsky_slice* slice,
			numsky_ndarray* arr, int arr_i, numsky_ndarray* new_arr, int new_arr_i) {
		npy_intp dim = arr->dimensions[arr_i];
		npy_intp stride = arr->strides[arr_i];
		npy_intp & new_dim = new_arr->dimensions[new_arr_i];
		npy_intp & new_stride = new_arr->strides[new_arr_i];
		new_stride = stride * slice->step;
		int start = slice->start;
		if(start > dim || start < -dim) {
			luaL_error(L, "slice.start %d not in range [%d, %d] or [%d, %d]",
					slice->start, -dim, -1, 1, dim);
		}
		if(start < 0) {
			start = dim + 1 + start;
		}
		int stop = slice->stop;
		if(stop > dim || stop < -dim) {
			luaL_error(L, "slice.stop %d not in range [%d, %d] or [%d, %d]",
					slice->stop, -dim, -1, 1, dim);
		}
		if(stop < 0) {
			stop = dim + 1 + stop;
		}
		if(slice->step > 0){
			if(start == SLICE_NIL) {
				start = 1;
			}
			if(stop == SLICE_NIL) {
				stop = dim;
			}
			if(stop - start >= 0) {
				new_dim = (stop - start) / slice->step + 1;
			} else {
				new_dim = 0;
			}
		} else {
			if(start == SLICE_NIL) {
				start = dim;
			}
			if(stop == SLICE_NIL) {
				stop = 1;
			}
			if(start - stop >= 0) {
				new_dim = (start - stop) / (- slice->step) + 1;
			} else {
				new_dim = 0;
			}
		}
		return (start - 1) * new_stride;
	}

	template <typename T> static npy_intp dataptr_cut(lua_State*L, char *dataptr, numsky_ndarray* arr, int arr_i){
		return _ndarray_dim::integer_cut(L, numsky::dataptr_cast<T>(dataptr), arr, arr_i);
	}

	template <typename T> ListStride* integer_array_cut_first(lua_State*L, numsky_ndarray *index_arr,
			numsky_ndarray *arr, int arr_i, numsky_ndarray *new_arr, int new_arr_i){
		luaUtils::lassert(index_arr->nd == 1, L, "numsky.ndarray can only index array with ndim == 1 (bool array indexing TODO)");
		// create list stride
		ListStride *list_stride = new ListStride(index_arr->dimensions[0], new_arr_i);
		for(int list_i=0;list_i<list_stride->len();list_i++) {
			char *index_dataptr = index_arr->dataptr + list_i*index_arr->strides[0];
			list_stride->stride(list_i) = integer_cut(L, numsky::dataptr_cast<T>(index_dataptr), arr, arr_i);
		}
		// add dim in new_arr
		new_arr->dimensions[new_arr_i] = list_stride->len();
		new_arr->strides[new_arr_i] = 0;
		return list_stride;
	}

	template <typename T> void integer_array_cut_later(lua_State*L, numsky_ndarray *index_arr, ListStride* list_stride, numsky_ndarray *arr, int arr_i) {
		luaUtils::lassert(std::is_integral<T>::value, L, "numsky.ndarray can only index bool array or integral array");
		luaUtils::lassert(index_arr->nd == 1, L, "numsky.ndarray can only index array with ndim == 1 (bool array indexing TODO)");
		luaUtils::lassert(list_stride->len() == index_arr->dimensions[0], L, "numsky.ndarray indexing two array with dim conflict");
		for(int list_i=0;list_i<list_stride->len();list_i++) {
			char *index_dataptr = index_arr->dataptr + list_i*index_arr->strides[0];
			list_stride->stride(list_i) += integer_cut(L, numsky::dataptr_cast<T>(index_dataptr), arr, arr_i);
		}
	}

	template <bool FIRST> ListStride* boolean_array_cut(lua_State*L, numsky_ndarray *index_arr, ListStride* list_stride,
			numsky_ndarray *arr, int arr_i, numsky_ndarray *new_arr, int new_arr_i){
		luaUtils::lassert(index_arr->nd <= arr->nd - arr_i, L, "ndarray don't have enough dim when indexing bool array");
		for(int i=0;i<index_arr->nd;i++) {
			luaUtils::lassert(index_arr->dimensions[i] == arr->dimensions[arr_i + i],
					L, "dim not match when indexing bool array");
		}
		std::vector<npy_intp> strides;
		strides.reserve(16);
		numsky::ndarray_foreach(index_arr, [&](numsky_nditer* iter){
			if(iter->dataptr[0]!=0){
				luaUtils::lassert(iter->dataptr[0] == 1, L, "[ERROR]inner error... bool must be 0 or 1");
				npy_intp cur_stride = 0;
				for(int i=0;i<index_arr->nd;i++) {
					cur_stride += iter->coordinates[i] * arr->strides[arr_i + i];
				}
				strides.push_back(cur_stride);
			}
		});
		if(FIRST) {
			list_stride = new ListStride(strides.size(), new_arr_i);
			for(int list_i=0;list_i<list_stride->len();list_i++) {
				list_stride->stride(list_i) = strides[list_i];
			}
			new_arr->dimensions[new_arr_i] = list_stride->len();
			new_arr->strides[new_arr_i] = 0;
		} else {
			luaUtils::lassert(strides.size() == list_stride->len(), L, "ndarray select count not equal when indexing bool array");
			for(int list_i=0;list_i<list_stride->len();list_i++) {
				list_stride->stride(list_i) += strides[list_i];
			}
		}
		return list_stride;
	}
} // namespace _ndarray_dim

// check assignable with broadcasting
static inline bool _ndarray_check_assignable(numsky_ndarray *arr_a, numsky_ndarray*arr_b) {
	if(arr_a->count == 0 || arr_b->count == 0) {
		return false;
	}
	int ia=arr_a->nd-1;
	int ib=arr_b->nd-1;
	if(ia < 0 || ib < 0) {
		return false;
	}
	while(ia >= 0 && ib >= 0) {
		bool assignable = (arr_b->dimensions[ib] == 1) ||
			(arr_a->dimensions[ia] == arr_b->dimensions[ib]);
		if(!assignable) {
			return false;
		}
		ia--;
		ib--;
	}
	while(ib >= 0) {
		if(arr_b->dimensions[ib] != 1) {
			return false;
		}
		ib--;
	}
	return true;
}

namespace _ndarray {
	// index like 'new_arr = arr[integer]' but not incref refcount, do incref in lua function, or just free outside
	static inline void _indexing_integer(lua_State* L, numsky_ndarray* arr, npy_intp integer, numsky_ndarray* new_arr) {
		npy_intp offset = _ndarray_dim::integer_cut(L, integer, arr, 0);
		for(int arr_i=1;arr_i<arr->nd;arr_i++){
			new_arr->dimensions[arr_i-1] = arr->dimensions[arr_i];
			new_arr->strides[arr_i-1] = arr->strides[arr_i];
		}
		new_arr->nd = arr->nd - 1;
		numsky_ndarray_autocount(new_arr);
		numsky_ndarray_refdata(new_arr, NULL, arr->dataptr + offset);
	}

	// index like 'new_arr = arr[slice]' but not incref refcount, do incref in lua function, or just free outside
	static inline void _indexing_slice(lua_State *L, numsky_ndarray* arr, numsky_slice* slice, numsky_ndarray* new_arr) {
		npy_intp offset = _ndarray_dim::slice_cut(L, slice, arr, 0, new_arr, 0);
		for(int arr_i=1;arr_i<arr->nd;arr_i++){
			new_arr->dimensions[arr_i] = arr->dimensions[arr_i];
			new_arr->strides[arr_i] = arr->strides[arr_i];
		}
		numsky_ndarray_autocount(new_arr);
		numsky_ndarray_refdata(new_arr, NULL, arr->dataptr + offset);
	}

	static std::unique_ptr<ListStride> _indexing_array(lua_State *L, numsky_ndarray* arr, numsky_ndarray* index_arr, numsky_ndarray* new_arr) {
		luaUtils::lassert(arr->nd >= index_arr->nd, L, "bool array has too many dims when indexing");
		ListStride* list_stride=NULL;
		if(index_arr->dtype->kind == 'i' || index_arr->dtype->kind == 'u') {
			auto array_cut = lnumsky_template_fp(L, index_arr->dtype->typechar, _ndarray_dim::integer_array_cut_first);
			list_stride = array_cut(L, index_arr, arr, 0, new_arr, 0);
		} else if(index_arr->dtype->typechar == '?'){
			list_stride = _ndarray_dim::boolean_array_cut<true>(L, index_arr, NULL, arr, 0, new_arr, 0);
		} else {
			luaL_error(L, "numsky.ndarray can only index array with dtype is bool or integer ");
			return std::unique_ptr<ListStride>(nullptr);
		}
		int new_arr_i = 1;
		int arr_i = index_arr->nd;
		for(;arr_i<arr->nd;arr_i++, new_arr_i++){
			new_arr->dimensions[new_arr_i] = arr->dimensions[arr_i];
			new_arr->strides[new_arr_i] = arr->strides[arr_i];
		}
		new_arr->nd = new_arr_i;
		numsky_ndarray_autocount(new_arr);
		numsky_ndarray_refdata(new_arr, NULL, arr->dataptr);
		return std::unique_ptr<ListStride>(list_stride);
	}

	static std::unique_ptr<ListStride> _indexing_table(lua_State *L, numsky_ndarray* arr, int table_idx, numsky_ndarray* new_arr) {
		int table_len = luaL_len(L, table_idx);
		npy_intp offset = 0;
		std::unique_ptr<ListStride> list_stride_ptr(nullptr);
		int arr_i = 0;
		int new_arr_i = 0;
		for(int table_i=1;table_i<=table_len;lua_pop(L, 1), arr_i++, table_i++) {
			luaUtils::lassert(arr_i < arr->nd, L, "numsky.ndarray: too many indices for array");
			lua_geti(L, table_idx, table_i);
			int index_type = lua_type(L, -1);
			if(index_type == LUA_TNUMBER) {
				int index_integer = luaL_checkinteger(L, -1);
				offset += _ndarray_dim::integer_cut(L, index_integer, arr, arr_i);
				continue;
			} else if(index_type == LUA_TUSERDATA) {
				auto index_slice = luabinding::ClassUtil<numsky_slice>::test(L, -1);
				if(index_slice != NULL) {
					offset += _ndarray_dim::slice_cut(L, index_slice, arr, arr_i, new_arr, new_arr_i);
					new_arr_i ++;
					continue;
				}
				auto index_arr = luabinding::ClassUtil<numsky_ndarray>::test(L, -1);
				luaUtils::lassert(arr_i + index_arr->nd <= arr->nd, L, "ndarray don't have enough dim when indexing bool array");
				if(index_arr != NULL) {
					if(index_arr->dtype->kind == 'i' || index_arr->dtype->kind == 'u') {
						if(list_stride_ptr.get() != NULL) {
							auto later_cut = lnumsky_template_fp(L, index_arr->dtype->typechar, _ndarray_dim::integer_array_cut_later);
							later_cut(L, index_arr, list_stride_ptr.get(), arr, arr_i);
						} else {
							auto first_cut = lnumsky_template_fp(L, index_arr->dtype->typechar, _ndarray_dim::integer_array_cut_first);
							// create list stride, increase new_arr index
							list_stride_ptr.reset(first_cut(L, index_arr, arr, arr_i, new_arr, new_arr_i));
							new_arr_i ++;
						}
					} else if(index_arr->dtype->kind == 'b'){
						if(list_stride_ptr.get() != NULL) {
							_ndarray_dim::boolean_array_cut<false>(L, index_arr, list_stride_ptr.get(), arr, arr_i, new_arr, new_arr_i);
						} else {
							list_stride_ptr.reset(_ndarray_dim::boolean_array_cut<true>(L, index_arr, NULL, arr, arr_i, new_arr, new_arr_i));
							new_arr_i ++;
						}
						arr_i += index_arr->nd - 1;
					} else {
						luaL_error(L, "numsky.ndarray can only index array with bool or integer dtype ");
					}
					continue;
				}
			}
			luaL_error(L, "numsky.ndarray indexing with unexcepted type %s", luaL_typename(L, -1));
		}
		for(;arr_i < arr->nd;arr_i++){
			new_arr->dimensions[new_arr_i] = arr->dimensions[arr_i];
			new_arr->strides[new_arr_i] = arr->strides[arr_i];
			new_arr_i ++;
		}
		new_arr->nd = new_arr_i; // reset nd
		numsky_ndarray_autocount(new_arr);
		numsky_ndarray_refdata(new_arr, NULL, arr->dataptr + offset);
		return list_stride_ptr;
	}

	// combine _indexing_integer, _indexing_slice, _indexing_table, _indexing_array
	static std::unique_ptr<ListStride> indexing_inplace(lua_State *L, numsky_ndarray *arr_obj, int idx, numsky_ndarray *new_arr) {
		int first_type = lua_type(L, idx);
		luaUtils::lassert(arr_obj->nd>=1, L, "numsky.ndarray: too many indices for indexing");
		if(first_type == LUA_TNUMBER) {
			_indexing_integer(L, arr_obj, luaL_checkinteger(L, idx), new_arr);
			return std::unique_ptr<ListStride>(nullptr);
		} else if(first_type == LUA_TUSERDATA) {
			auto index_slice = luabinding::ClassUtil<numsky_slice>::test(L, idx);
			if(index_slice != NULL) {
				_indexing_slice(L, arr_obj, index_slice, new_arr);
				return std::unique_ptr<ListStride>(nullptr);
			}
			auto index_arr = luabinding::ClassUtil<numsky_ndarray>::test(L, idx);
			if(index_arr != NULL) {
				return _indexing_array(L, arr_obj, index_arr, new_arr);
			}
			luaL_error(L, "numsky.ndarray indexing with unexcepted type %s", luaL_typename(L, idx));
		} else if(first_type == LUA_TTABLE){
			return _indexing_table(L, arr_obj, idx, new_arr);
		} else {
			luaL_error(L, "numsky.ndarray can't index type=%s ", lua_typename(L, first_type));
		}
		return std::unique_ptr<ListStride>(nullptr);
	}
} // namespace _ndarray

// used when return one value
template <typename T> static void dataptr_push(lua_State*L, char* dataptr){
	numsky::generic<T>::push(L, numsky::dataptr_cast<T>(dataptr));
}

// e.g arr[index] = arr[1,2,3]
template <typename T> static void numsky_ndarray__index_copy(numsky_ndarray *new_arr, ListStride *list_stride) {
	auto foreign_base = skynet_foreign_newbytes(new_arr->count*sizeof(T));
	char *dataptr = foreign_base->data;
	numsky::ndarray_foreach(new_arr, [&](numsky_nditer* iter) {
		numsky::dataptr_cast<T>(dataptr) = numsky::dataptr_cast<T>(list_stride->shift(iter));
		dataptr += sizeof(T) ;
	});
	numsky_ndarray_refdata(new_arr, foreign_base, foreign_base->data);
	numsky_ndarray_autostridecount(new_arr);
}

// e.g arr[index] = arr[1,2,3]
template <typename T1, typename T2> static void _ndarray__newindex_assign_array(numsky_ndarray* left_arr, numsky_ndarray *right_arr, ListStride *list_stride) {
	if(list_stride==NULL) {
		numsky::ndarray_broadcasting_foreach(left_arr, left_arr, right_arr, [&](numsky_nditer* left_iter, numsky_nditer* right_iter){
			numsky::dataptr_cast<T1>(left_iter->dataptr) = numsky::dataptr_cast<T2>(right_iter->dataptr);
		});
	} else {
		numsky::ndarray_broadcasting_foreach(left_arr, left_arr, right_arr, [&](numsky_nditer* left_iter, numsky_nditer* right_iter){
			numsky::dataptr_cast<T1>(list_stride->shift(left_iter)) = numsky::dataptr_cast<T2>(right_iter->dataptr);
		});
	}
}

// e.g arr[index] = 123
template <typename T1, typename T2> static void _ndarray__newindex_assign_data(numsky_ndarray* left_arr, T2 right_value, ListStride *list_stride) {
	if(list_stride==NULL) {
		numsky::ndarray_foreach(left_arr, [&](numsky_nditer* iter) {
			numsky::dataptr_cast<T1>(iter->dataptr) = right_value;
		});
	} else {
		numsky::ndarray_foreach(left_arr, [&](numsky_nditer* iter) {
			numsky::dataptr_cast<T1>(list_stride->shift(iter)) = right_value;
		});
	}
}

// API
int numsky::ndarray__index(lua_State *L){
	auto arr_obj = luabinding::ClassUtil<numsky_ndarray>::check(L, 1);
	char typechar = arr_obj->dtype->typechar;
	int first_type = lua_type(L,2);
	if(first_type != LUA_TSTRING) {
		luaUtils::lassert(arr_obj->nd>=1, L, "numsky.ndarray: too many indices for indexing");
		auto new_arr = numsky::ndarray_new_preinit<false>(L, arr_obj->nd, typechar);
		auto list_stride_ptr = _ndarray::indexing_inplace(L, arr_obj, 2, new_arr.get());
		if(list_stride_ptr.get() == NULL) {
			if(new_arr->nd >= 1) {
				skynet_foreign_incref(arr_obj->foreign_base);
				numsky_ndarray_refdata(new_arr.get(), arr_obj->foreign_base, new_arr->dataptr);
				ndarray_mem2lua(L, new_arr);
				return 1;
			} else {
				lnumsky_template_fp(L, typechar, dataptr_push)(L, new_arr->dataptr);
				// if return value, clear reference to foreign_base
				numsky_ndarray_refdata(new_arr.get(), NULL, NULL);
				return 1;
			}
		} else {
			lnumsky_template_fp(L, typechar, numsky_ndarray__index_copy)(new_arr.get(), list_stride_ptr.get());
			ndarray_mem2lua(L, new_arr);
			return 1;
		}
	} else {
		luabinding::ClassUtil<numsky_ndarray>::upget_function_or_property(L, arr_obj);
		return 1;
	}
}

// API
int numsky::ndarray__newindex(lua_State *L){
	auto arr_obj = luabinding::ClassUtil<numsky_ndarray>::check(L, 1);
	char typechar = arr_obj->dtype->typechar;
    //int key_type = lua_type(L,2);
    int value_type = lua_type(L,3);
	luaUtils::lassert(arr_obj->nd>=1, L, "numsky.ndarray: too many indices for indexing");
	// step 1. make temp left array, list stride
	auto left_arr_keeper = numsky::ndarray_new_preinit<false>(L, arr_obj->nd, typechar);
	auto left_arr = left_arr_keeper.get();
	// step 2. checkout temp left array by indexing
	auto list_stride_ptr = _ndarray::indexing_inplace(L, arr_obj, 2, left_arr);
	// step 3. checkout right data or right array
	if(value_type == LUA_TNUMBER) {
		luaUtils::lassert(typechar!='b', L, "number value can not be assign to bool array");
		if(lua_isinteger(L, 3)) {
			auto fp_assign = lnumsky_template_fp2t2(L, typechar, lua_Integer, _ndarray__newindex_assign_data);
			fp_assign(left_arr, lua_tointeger(L, 3), list_stride_ptr.get());
		} else {
			auto fp_assign = lnumsky_template_fp2t2(L, typechar, lua_Number, _ndarray__newindex_assign_data);
			fp_assign(left_arr, lua_tonumber(L, 3), list_stride_ptr.get());
		}
		return 0;
	} else if(value_type == LUA_TBOOLEAN){
		luaUtils::lassert(typechar=='b', L, "bool value can only be assign to bool array");
		_ndarray__newindex_assign_data<bool, bool>(left_arr, lua_toboolean(L, 3), list_stride_ptr.get());
		return 0;
	} else {
		numsky::ThrowableContext ctx(L);
		auto value_arr_ptr = numsky::check_temp_ndarray(&ctx, 3, typechar);
		if(!_ndarray_check_assignable(left_arr, value_arr_ptr.get())){
			return luaL_error(L, "right arr not assignable to left");
		} else {
			auto fp_assign = lnumsky_template_fp2(L, typechar, value_arr_ptr->dtype->typechar, _ndarray__newindex_assign_array);
			fp_assign(left_arr, value_arr_ptr.get(), list_stride_ptr.get());
			return 0;
		}
	}
}
