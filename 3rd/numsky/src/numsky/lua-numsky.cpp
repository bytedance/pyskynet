#include "numsky/lua-numsky.h"

namespace luabinding {
    template <> const char* Class_<numsky_dtype>::metaname= "numsky.dtype";
    template <> const char* Class_<numsky_ndarray>::metaname= NS_ARR_METANAME;
    template <> const char* Class_<numsky_nditer>::metaname= "numsky.nditer";
    template <> const char* Class_<numsky_slice>::metaname= "numsky.slice";
    template <> const char* Class_<numsky_ufunc>::metaname= "numsky.ufunc";
}

namespace numsky {

}
