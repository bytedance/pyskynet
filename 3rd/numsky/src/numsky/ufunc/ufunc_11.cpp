#include <type_traits>
#include <typeinfo>
#include <string>
#include <cmath>

#include "numsky/lua-numsky.h"
#include "numsky/ndarray/lua-numsky_ndarray.h"
#include "numsky/ufunc/lua-numsky_ufunc.h"

using TError = uint64_t;
namespace _ufunc_11 {
	template <numsky::UFUNC_ENUM ufunc_enum, typename TArg> struct ufunc_rawitem;
	template <numsky::UFUNC_ENUM ufunc_enum> struct ufunc_item {
		template <typename TArg> static void oper(lua_State *L, char *re, char *arg) {
			numsky::dataptr_cast<typename ufunc_rawitem<ufunc_enum, TArg>::TRet>(re) =
				ufunc_rawitem<ufunc_enum, TArg>::oper(L, numsky::dataptr_cast<TArg>(arg));
		}
		template <typename TArg> static char template_check_type(lua_State *L) {
			using TRet = typename ufunc_rawitem<ufunc_enum, TArg>::TRet;
			using TValidRet = typename std::conditional<
					std::is_same<TRet, uint64_t>::value,
					bool, TRet>::type;

			if(std::is_same<TRet, uint64_t>::value){
				char a = numsky::generic<TArg>::typechar;
				luaL_error(L, "%c not support", a);
				return 'L';
			} else {
				return numsky::generic<TValidRet>::typechar;
			}
		}
		static char check_type(lua_State *L, char typechar) {
			return lnumsky_template_fp(L, typechar, template_check_type)(L);
		}
		static numsky::generic_ufunc<1, 1>::T_oper check_oper(lua_State *L, char typechar) {
			return lnumsky_template_fp(L, typechar, oper);
		}
	};

	template <typename TArg> struct generic_uop {
		using TSelf=TArg;
		using TSelfNotBool=typename std::conditional<
				std::is_same<TArg, bool>::value,
				TError, TSelf>::type;
		using TSelfNotFloating=typename std::conditional<
				std::is_floating_point<TArg>::value,
				TError, TSelf>::type;
		using TFloatingNotBool=typename std::conditional<
				std::is_same<TArg, bool>::value,
				TError,
				typename std::conditional<
					std::is_floating_point<TArg>::value,
					TArg,
					typename std::conditional<sizeof(TArg) >= 4, double, float>::type
				>::type
			>::type;
	};
} // namespace _ufunc_11

namespace _ufunc_11 {
	template <typename TArg> struct ufunc_rawitem<numsky::UFUNC_unm, TArg> {
		using TRet=typename generic_uop<TArg>::TSelfNotBool;
		static inline TRet oper(lua_State*L, TArg arg) {
			return -arg;
		}
	};

	template <typename TArg> struct ufunc_rawitem<numsky::UFUNC_bnot, TArg> {
		using TRet=typename generic_uop<TArg>::TSelfNotFloating;
		template <typename T> static inline typename std::enable_if<std::is_same<T, bool>::value, T>::type my_bnot(T arg) {
			return !arg;
		}
		template <typename T> static inline typename std::enable_if<(!std::is_same<T, bool>::value) && std::is_integral<T>::value, T>::type my_bnot(T arg) {
			return ~arg;
		}
		template <typename T> static inline typename std::enable_if<std::is_floating_point<T>::value, T>::type my_bnot(T arg) {
			return 0;
		}
		static inline TRet oper(lua_State*L, TArg arg) {
			return my_bnot<TArg>(arg);
		}
	};

	template <typename TArg> struct ufunc_rawitem<numsky::UFUNC_abs, TArg> {
		using TRet=typename generic_uop<TArg>::TSelfNotBool;
		template <typename T> static inline typename std::enable_if<std::is_unsigned<T>::value, T>::type my_abs(T arg) {
			return arg;
		}
		template <typename T> static inline typename std::enable_if<!std::is_unsigned<T>::value, T>::type my_abs(T arg) {
			return std::abs(arg);
		}
		static inline TRet oper(lua_State*L, TArg arg) {
			return ufunc_rawitem<numsky::UFUNC_abs, TArg>::my_abs<TArg>(arg);
		}
	};

	template <typename TArg> struct ufunc_rawitem<numsky::UFUNC_ceil, TArg> {
		using TRet=typename generic_uop<TArg>::TSelfNotBool;
		static inline TRet oper(lua_State*L, TArg arg) {
			return std::ceil(arg);
		}
	};

	template <typename TArg> struct ufunc_rawitem<numsky::UFUNC_floor, TArg> {
		using TRet=typename generic_uop<TArg>::TSelfNotBool;
		static inline TRet oper(lua_State*L, TArg arg) {
			return std::floor(arg);
		}
	};

	template <typename TArg> struct ufunc_rawitem<numsky::UFUNC_acos, TArg> {
		using TRet=typename generic_uop<TArg>::TFloatingNotBool;
		static inline TRet oper(lua_State*L, TArg arg) {
			return std::acos(arg);
		}
	};

	template <typename TArg> struct ufunc_rawitem<numsky::UFUNC_asin, TArg> {
		using TRet=typename generic_uop<TArg>::TFloatingNotBool;
		static inline TRet oper(lua_State*L, TArg arg) {
			return std::asin(arg);
		}
	};

	template <typename TArg> struct ufunc_rawitem<numsky::UFUNC_atan, TArg> {
		using TRet=typename generic_uop<TArg>::TFloatingNotBool;
		static inline TRet oper(lua_State*L, TArg arg) {
			return std::atan(arg);
		}
	};

	template <typename TArg> struct ufunc_rawitem<numsky::UFUNC_cos, TArg> {
		using TRet=typename generic_uop<TArg>::TFloatingNotBool;
		static inline TRet oper(lua_State*L, TArg arg) {
			return std::cos(arg);
		}
	};

	template <typename TArg> struct ufunc_rawitem<numsky::UFUNC_sin, TArg> {
		using TRet=typename generic_uop<TArg>::TFloatingNotBool;
		static inline TRet oper(lua_State*L, TArg arg) {
			return std::sin(arg);
		}
	};

	template <typename TArg> struct ufunc_rawitem<numsky::UFUNC_tan, TArg> {
		using TRet=typename generic_uop<TArg>::TFloatingNotBool;
		static inline TRet oper(lua_State*L, TArg arg) {
			return std::tan(arg);
		}
	};

	template <typename TArg> struct ufunc_rawitem<numsky::UFUNC_deg, TArg> {
		using TRet=typename generic_uop<TArg>::TFloatingNotBool;
		static inline TRet oper(lua_State*L, TArg arg) {
			return arg*180.0/3.14159265358979;
		}
	};

	template <typename TArg> struct ufunc_rawitem<numsky::UFUNC_rad, TArg> {
		using TRet=typename generic_uop<TArg>::TFloatingNotBool;
		static inline TRet oper(lua_State*L, TArg arg) {
			return arg*3.14159265358979/180.0;
		}
	};

	template <typename TArg> struct ufunc_rawitem<numsky::UFUNC_log, TArg> {
		using TRet=typename generic_uop<TArg>::TFloatingNotBool;
		static inline TRet oper(lua_State*L, TArg arg) {
			return std::log(arg);
		}
	};

	template <typename TArg> struct ufunc_rawitem<numsky::UFUNC_exp, TArg> {
		using TRet=typename generic_uop<TArg>::TFloatingNotBool;
		static inline TRet oper(lua_State*L, TArg arg) {
			return std::exp(arg);
		}
	};

	template <typename TArg> struct ufunc_rawitem<numsky::UFUNC_sqrt, TArg> {
		using TRet=typename generic_uop<TArg>::TFloatingNotBool;
		static inline TRet oper(lua_State*L, TArg arg) {
			return std::sqrt(arg);
		}
	};

} // namespace _ufunc_11

#define make_ufunc(vName) \
template <> const numsky_ufunc numsky::ufunc_instance<numsky::UFUNC_##vName>::ufunc = { \
	1, \
	1, \
	reinterpret_cast<void*>(_ufunc_11::ufunc_item<numsky::UFUNC_##vName>::check_type), \
	reinterpret_cast<void*>(_ufunc_11::ufunc_item<numsky::UFUNC_##vName>::check_oper), \
	NULL, \
	#vName, \
	numsky::UFUNC_##vName, \
}

make_ufunc(unm);
make_ufunc(bnot);
make_ufunc(abs);
make_ufunc(ceil);
make_ufunc(floor);

make_ufunc(acos);
make_ufunc(asin);
make_ufunc(atan);
make_ufunc(cos);
make_ufunc(sin);
make_ufunc(tan);
make_ufunc(deg);
make_ufunc(rad);

make_ufunc(log);
make_ufunc(exp);
make_ufunc(sqrt);


