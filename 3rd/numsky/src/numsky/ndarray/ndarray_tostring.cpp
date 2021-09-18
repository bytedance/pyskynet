#include <sstream>

#include "numsky/lua-numsky.h"
#include "numsky/lua-numsky_module.h"
#include "numsky/ndarray/lua-numsky_ndarray.h"

template <typename T> int _ndarray__tostring(lua_State *L, struct numsky_ndarray* arr) {
	std::ostringstream stream;
	auto iter_ptr = numsky::ndarray_nditer(arr);
	auto iter = iter_ptr.get();
	stream << "array(";
	for(int i=0;i<iter->nd;i++) {
		stream<<'{';
	}
	for(int i=0;i<iter->ao->count;numsky_nditer_next(iter), i++) {
		char *dataptr = iter->dataptr;
		int sub_ndim = numsky_nditer_sub_ndim(iter);
		if(sub_ndim == 0) {
			stream << ',';
		} else if(sub_ndim > 0 && sub_ndim < iter->nd) {
			for(int i=0;i<sub_ndim;i++) {
				stream << '}';
			}
			stream << ",\n";
			if(sub_ndim > 1) {
				stream << '\n';
			}
			for(int i=0;i<sub_ndim;i++) {
				stream << '{';
			}
		}
		switch(arr->dtype->typechar) {
			case '?': {
						  if(dataptr[0]) {
							  stream << "true";
						  } else {
							  stream << "false";
						  }
						  break;
					  }
			case 'b':
			case 'B':
			case 'h':
			case 'H':
			case 'i':
			case 'I':
			case 'l':
			case 'L':
					  stream<<(int64_t)(numsky::dataptr_cast<T>(dataptr));
					  break;
			case 'f':
			case 'd': {
					  double f_value = numsky::dataptr_cast<T>(dataptr);
					  if(f_value - (int64_t)f_value == 0) {
						  stream<<(f_value)<<'.';
					  } else {
						  stream<<f_value;
					  }
					  break;
			  }
			default:
					  stream<<("wrong...");
					  break;
		}
    }
    for(int i=0;i<iter->nd;i++) {
		stream<<'}';
    }
    stream<<',';
    stream<<arr->dtype->name;
    stream<<')';
	std::string s = stream.str();
    lua_pushlstring(L, s.c_str(), s.size());
    return 1;
}

// API
int numsky::ndarray__tostring(lua_State *L){
	auto arr_obj = luabinding::ClassUtil<numsky_ndarray>::check(L, 1);
	auto fp_tostring = lnumsky_template_fp(L, arr_obj->dtype->typechar, _ndarray__tostring);
	return fp_tostring(L, arr_obj);
}

