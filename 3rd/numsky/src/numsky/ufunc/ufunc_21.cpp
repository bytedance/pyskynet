#include <type_traits>
#include <typeinfo>
#include <string>
#include <cmath>

#include "numsky/lua-numsky.h"
#include "numsky/lua-numsky_module.h"

#include "numsky/ndarray/lua-numsky_ndarray.h"
#include "numsky/ufunc/lua-numsky_ufunc.h"

using TError = uint64_t;
namespace _ufunc_21 {
	template <typename TLeft, typename TRight> struct generic_bop_floating_point {
		using Tadd=typename std::conditional<
				std::is_same<TLeft, double>::value || std::is_same<TRight, double>::value,
				double,
				typename std::conditional<
					(std::is_integral<TLeft>::value && sizeof(TLeft) >= 4) ||
					(std::is_integral<TRight>::value && sizeof(TRight) >= 4),
					double,
					float
				>::type
			>::type;
	};

	template <typename TLeft, typename TRight> struct generic_bop_integral {
		using TBig=typename std::conditional<sizeof(TLeft) >= sizeof(TRight), TLeft, TRight>::type;
		using TSmall=typename std::conditional<sizeof(TLeft) < sizeof(TRight), TLeft, TRight>::type;

		using Tadd=typename std::conditional<
				(std::is_same<TBig, TSmall>::value) ||
				(std::is_signed<TBig>::value && sizeof(TBig) > sizeof(TSmall)) ||
				(std::is_unsigned<TBig>::value && std::is_unsigned<TSmall>::value && sizeof(TBig) > sizeof(TSmall)),
				TBig,
				typename std::conditional<
					std::is_same<TSmall,bool>::value,
					TBig,
					typename std::conditional<
						std::is_same<TBig,bool>::value,
						TSmall,
						typename std::conditional<
							std::is_same<TBig,uint8_t>::value || std::is_same<TBig,int8_t>::value,
							int16_t,
							typename std::conditional<
								std::is_same<TBig,uint16_t>::value || std::is_same<TBig,int16_t>::value,
								int32_t,
								typename std::conditional<
									std::is_same<TBig,uint32_t>::value || std::is_same<TBig,int32_t>::value,
									int64_t,
									uint64_t// unexcepted branch
									//std::string // unexcepted branch
								>::type
							>::type
						>::type
					>::type
				>::type
			>::type;
	};

	template <typename TLeft, typename TRight> struct generic_bop {
		using TValidAll=typename std::conditional<
				std::is_floating_point<TLeft>::value || std::is_floating_point<TRight>::value,
				typename generic_bop_floating_point<TLeft, TRight>::Tadd,
				typename generic_bop_integral<TLeft, TRight>::Tadd
			>::type;
		using TValidFloating=typename std::conditional<
				std::is_floating_point<TLeft>::value || std::is_floating_point<TRight>::value,
				typename generic_bop_floating_point<TLeft, TRight>::Tadd,
				double
			>::type;
		// mathematic
		// add fmax fmin matmul maximum minimum multiply
		// floor_divide fmod mod power remainder subtract
		using TRet_add=typename std::conditional<
				std::is_same<TLeft, bool>::value || std::is_same<TRight, bool>::value,
				TError, TValidAll>::type;

		// true_divide divide arctan2 copysign heaviside hypot logaddexp logaddexp2 nextafter
		using TRet_div=typename std::conditional<
				std::is_same<TLeft, bool>::value || std::is_same<TRight, bool>::value,
				TError, TValidFloating>::type;
		using T__div=TRet_div;

		// bitwise
		// bitwise_and bitwise_or bitwise_xor
		using TRet_bitwise=typename std::conditional<
				(std::is_integral<TLeft>::value && std::is_integral<TRight>::value),
				TValidAll, TError>::type;

		// gcd lcm left_shift right_shift
		using TRet_shift=typename std::conditional<
				std::is_integral<TLeft>::value && std::is_integral<TRight>::value &&
				(!std::is_same<TLeft, bool>::value) && (!std::is_same<TRight, bool>::value),
				TValidAll, TError>::type;

		// equal greater greater_equal less less_equal not_equal
		using TRet_comparison=bool;
	};

	template <typename TLeft, typename TRight> struct ufunc_rawerr {
		// this struct is only used for avoid compile warning...
		using TRet=uint64_t; // using uint64_t as error
		static inline TRet oper(lua_State*L, TLeft left, TRight right) {
			char l = numsky::generic<TLeft>::typechar;
			char r = numsky::generic<TRight>::typechar;
			luaL_error(L, "%c op %c not support", l, r);
			return 0;
		}
	};

	// implement this later
	template <numsky::UFUNC_ENUM ufunc_enum, typename TLeft, typename TRight> struct ufunc_rawitem;
	template <numsky::UFUNC_ENUM ufunc_enum, typename T> struct ufunc_rawitem_reduce {
		static inline T init(lua_State*L, T first) {
			luaL_error(L, "this ufunc not allow reduce");
			return first;
		}
	};

	template <numsky::UFUNC_ENUM ufunc_enum> struct ufunc_item {
		template <typename TLeft, typename TRight> static void oper(lua_State *L, char *re, char *a, char *b) {
			numsky::dataptr_cast<typename ufunc_rawitem<ufunc_enum, TLeft, TRight>::TRet>(re) =
				ufunc_rawitem<ufunc_enum, TLeft, TRight>::oper(L,
						numsky::dataptr_cast<TLeft>(a), numsky::dataptr_cast<TRight>(b));
		}
		template <typename T> static void init(lua_State*L, char *re, char *first) {
			numsky::dataptr_cast<T>(re) = ufunc_rawitem_reduce<ufunc_enum, T>::init(L, numsky::dataptr_cast<T>(first));
		}
		template <typename TLeft, typename TRight> static char template_check_type(lua_State *L) {
			using TRet = typename ufunc_rawitem<ufunc_enum, TLeft, TRight>::TRet;
			using TValidRet = typename std::conditional<
					std::is_same<TRet, uint64_t>::value,
					bool, TRet>::type;

			if(std::is_same<TRet, uint64_t>::value){
				char l = numsky::generic<TLeft>::typechar;
				char r = numsky::generic<TRight>::typechar;
				luaL_error(L, "%c op %c not support", l, r);
				return 'L';
			} else {
				return numsky::generic<TValidRet>::typechar;
			}
		}
		static char check_type(lua_State *L, char ltc, char rtc) {
			return lnumsky_template_fp2(L, ltc, rtc, template_check_type)(L);
		}
		static numsky::generic_ufunc<2, 1>::T_oper check_oper(lua_State *L, char ltc, char rtc) {
			return lnumsky_template_fp2(L, ltc, rtc, oper);
		}
		static numsky::generic_ufunc<2, 1>::T_init check_init(lua_State *L, char tc) {
			return lnumsky_template_fp(L, tc, init);
		}
	};

} // namespace _ufunc_21

// reduce item
namespace _ufunc_21 {
	template <typename T> struct ufunc_rawitem_reduce<numsky::UFUNC_add, T> {
		static inline T init(lua_State*L, T first) {
			return 0;
		}
	};

	template <typename T> struct ufunc_rawitem_reduce<numsky::UFUNC_mul, T> {
		static inline T init(lua_State*L, T first) {
			return 1;
		}
	};

	template <typename T> struct ufunc_rawitem_reduce<numsky::UFUNC_bor, T> {
		static inline T init(lua_State*L, T first) {
			return 0;
		}
	};

	template <typename T> struct ufunc_rawitem_reduce<numsky::UFUNC_band, T> {
		static inline T init(lua_State*L, T first) {
			T zero = 0;
			return ~zero;
		}
	};

	template <> struct ufunc_rawitem_reduce<numsky::UFUNC_band, bool> {
		static inline bool init(lua_State*L, bool first) {
			return true;
		}
	};

	template <> struct ufunc_rawitem_reduce<numsky::UFUNC_band, float> {
		static inline float init(lua_State*L, float first) {
			luaL_error(L, "band(float) not allow ");
			return 0;
		}
	};

	template <> struct ufunc_rawitem_reduce<numsky::UFUNC_band, double> {
		static inline double init(lua_State*L, double first) {
			luaL_error(L, "band(double) not allow ");
			return 0;
		}
	};

	template <typename T> struct ufunc_rawitem_reduce<numsky::UFUNC_bxor, T> {
		static inline T init(lua_State*L, T first) {
			return 0;
		}
	};

	template <typename T> struct ufunc_rawitem_reduce<numsky::UFUNC_fmax, T> {
		static inline T init(lua_State*L, T first) {
			return first;
		}
	};

	template <typename T> struct ufunc_rawitem_reduce<numsky::UFUNC_fmin, T> {
		static inline T init(lua_State*L, T first) {
			return first;
		}
	};
} // namespace _ufunc_21

// meta function
namespace _ufunc_21 {

	template <typename TLeft, typename TRight> struct ufunc_rawitem<numsky::UFUNC_add, TLeft, TRight> {
		using TRet=typename _ufunc_21::generic_bop<TLeft, TRight>::TRet_add;
		static inline TRet oper(lua_State*L, TLeft left, TRight right) {
			return (TRet)left + (TRet)right;
		}
	};

	template <typename TLeft, typename TRight> struct ufunc_rawitem<numsky::UFUNC_sub, TLeft, TRight> {
		using TRet=typename _ufunc_21::generic_bop<TLeft, TRight>::TRet_add;
		static inline TRet oper(lua_State*L, TLeft left, TRight right) {
			return (TRet)left - (TRet)right;
		}
	};

	template <typename TLeft, typename TRight> struct ufunc_rawitem<numsky::UFUNC_mul, TLeft, TRight> {
		using TRet=typename _ufunc_21::generic_bop<TLeft, TRight>::TRet_add;
		static inline TRet oper(lua_State*L, TLeft left, TRight right) {
			return (TRet)left * (TRet)right;
		}
	};

	template <typename TLeft, typename TRight> struct ufunc_rawitem<numsky::UFUNC_div, TLeft, TRight> {
		using TRet=typename _ufunc_21::generic_bop<TLeft, TRight>::TRet_div;
		static inline TRet oper(lua_State*L, TLeft left, TRight right) {
			return 1.0 * (TRet)left / (TRet)right;
		}
	};

	template <typename TLeft, typename TRight> struct ufunc_rawitem<numsky::UFUNC_idiv, TLeft, TRight> {
		using TRet=typename _ufunc_21::generic_bop<TLeft, TRight>::TRet_add;
		static inline TRet oper(lua_State*L, TLeft left, TRight right) {
			if(std::is_integral<TLeft>::value && std::is_integral<TRight>::value && right == 0) {
				luaL_error(L, "error : divide by zero when __idiv");
			}
			return std::floor(1.0*(TRet)left/(TRet)right);
		}
	};

	template <typename TLeft, typename TRight> struct ufunc_rawitem<numsky::UFUNC_mod, TLeft, TRight> {
		using TRet=typename _ufunc_21::generic_bop<TLeft, TRight>::TRet_add;
		static inline TRet oper(lua_State*L, TLeft left, TRight right) {
			if(std::is_integral<TLeft>::value && std::is_integral<TRight>::value) {
				if(right == 0) {
					luaL_error(L, "error : __mod zero when integer");
				}
				int64_t ileft = left;
				int64_t iright = right;
				return (int64_t)ileft % (int64_t)iright;
			} else {
				return std::fmod((TRet)left, (TRet)right);
			}
		}
	};

	template <typename TLeft, typename TRight> struct ufunc_rawitem<numsky::UFUNC_pow, TLeft, TRight> {
		using TRet=typename _ufunc_21::generic_bop<TLeft, TRight>::TRet_add;
		static inline TRet oper(lua_State*L, TLeft left, TRight right) {
			return std::pow((TRet)left, (TRet)right);
		}
	};

	// bitwise
	template <typename TLeft, typename TRight> struct ufunc_rawitem<numsky::UFUNC_band, TLeft, TRight> {
		using TRet=typename _ufunc_21::generic_bop<TLeft, TRight>::TRet_bitwise;
		static inline TRet oper(lua_State*L, TLeft left, TRight right) {
			return (TRet)left & (TRet)right;
		}
	};

	template <typename TLeft, typename TRight> struct ufunc_rawitem<numsky::UFUNC_bor, TLeft, TRight> {
		using TRet=typename _ufunc_21::generic_bop<TLeft, TRight>::TRet_bitwise;
		static inline TRet oper(lua_State*L, TLeft left, TRight right) {
			return (TRet)left | (TRet)right;
		}
	};

	template <typename TLeft, typename TRight> struct ufunc_rawitem<numsky::UFUNC_bxor, TLeft, TRight> {
		using TRet=typename _ufunc_21::generic_bop<TLeft, TRight>::TRet_bitwise;
		static inline TRet oper(lua_State*L, TLeft left, TRight right) {
			return (TRet)left ^ (TRet)right;
		}
	};

	template <typename TLeft, typename TRight> struct ufunc_rawitem<numsky::UFUNC_shl, TLeft, TRight> {
		using TRet=typename _ufunc_21::generic_bop<TLeft, TRight>::TRet_shift;
		static inline TRet oper(lua_State*L, TLeft left, TRight right) {
			return (TRet)left << (TRet)right;
		}
	};

	template <typename TLeft, typename TRight> struct ufunc_rawitem<numsky::UFUNC_shr, TLeft, TRight> {
		using TRet=typename _ufunc_21::generic_bop<TLeft, TRight>::TRet_shift;
		static inline TRet oper(lua_State*L, TLeft left, TRight right) {
			return (TRet)left >> (TRet)right;
		}
	};

} // namespace _ufunc_21

// not meta function
namespace _ufunc_21 {
	template <typename TLeft, typename TRight> struct ufunc_rawitem<numsky::UFUNC_eq, TLeft, TRight> {
		using TRet=typename _ufunc_21::generic_bop<TLeft, TRight>::TRet_comparison;
		using TCast=typename _ufunc_21::generic_bop<TLeft, TRight>::TRet_add;
		static inline TRet oper(lua_State*L, TLeft left, TRight right) {
			return static_cast<TCast>(left) == static_cast<TCast>(right);
		}
	};

	template <typename TLeft, typename TRight> struct ufunc_rawitem<numsky::UFUNC_lt, TLeft, TRight> {
		using TRet=typename _ufunc_21::generic_bop<TLeft, TRight>::TRet_comparison;
		using TCast=typename _ufunc_21::generic_bop<TLeft, TRight>::TRet_add;
		static inline TRet oper(lua_State*L, TLeft left, TRight right) {
			return static_cast<TCast>(left) < static_cast<TCast>(right);
		}
	};

	template <typename TLeft, typename TRight> struct ufunc_rawitem<numsky::UFUNC_le, TLeft, TRight> {
		using TRet=typename _ufunc_21::generic_bop<TLeft, TRight>::TRet_comparison;
		using TCast=typename _ufunc_21::generic_bop<TLeft, TRight>::TRet_add;
		static inline TRet oper(lua_State*L, TLeft left, TRight right) {
			return static_cast<TCast>(left) <= static_cast<TCast>(right);
		}
	};

	template <typename TLeft, typename TRight> struct ufunc_rawitem<numsky::UFUNC_ne, TLeft, TRight> {
		using TRet=typename _ufunc_21::generic_bop<TLeft, TRight>::TRet_comparison;
		using TCast=typename _ufunc_21::generic_bop<TLeft, TRight>::TRet_add;
		static inline TRet oper(lua_State*L, TLeft left, TRight right) {
			return static_cast<TCast>(left) != static_cast<TCast>(right);
		}
	};

	template <typename TLeft, typename TRight> struct ufunc_rawitem<numsky::UFUNC_gt, TLeft, TRight> {
		using TRet=typename _ufunc_21::generic_bop<TLeft, TRight>::TRet_comparison;
		using TCast=typename _ufunc_21::generic_bop<TLeft, TRight>::TRet_add;
		static inline TRet oper(lua_State*L, TLeft left, TRight right) {
			return static_cast<TCast>(left) > static_cast<TCast>(right);
		}
	};

	template <typename TLeft, typename TRight> struct ufunc_rawitem<numsky::UFUNC_ge, TLeft, TRight> {
		using TRet=typename _ufunc_21::generic_bop<TLeft, TRight>::TRet_comparison;
		using TCast=typename _ufunc_21::generic_bop<TLeft, TRight>::TRet_add;
		static inline TRet oper(lua_State*L, TLeft left, TRight right) {
			return static_cast<TCast>(left) >= static_cast<TCast>(right);
		}
	};

	// TODO(cz) for nan case ??
	template <typename TLeft, typename TRight> struct ufunc_rawitem<numsky::UFUNC_fmax, TLeft, TRight> {
		using TRet=typename _ufunc_21::generic_bop<TLeft, TRight>::TRet_add;
		static inline TRet oper(lua_State*L, TLeft left, TRight right) {
			return static_cast<TRet>(left) >= static_cast<TRet>(right) ? left : right;
		}
	};

	template <typename TLeft, typename TRight> struct ufunc_rawitem<numsky::UFUNC_fmin, TLeft, TRight> {
		using TRet=typename _ufunc_21::generic_bop<TLeft, TRight>::TRet_add;
		static inline TRet oper(lua_State*L, TLeft left, TRight right) {
			return static_cast<TRet>(left) <= static_cast<TRet>(right) ? left : right;
		}
	};

	template <typename TLeft, typename TRight> struct ufunc_rawitem<numsky::UFUNC_atan2, TLeft, TRight> {
		using TRet=typename _ufunc_21::generic_bop<TLeft, TRight>::TRet_div;
		static inline TRet oper(lua_State*L, TLeft left, TRight right) {
			return std::atan2(static_cast<TRet>(left), static_cast<TRet>(right));
		}
	};
} // namespace _ufunc_21

#define make_ufunc(vName) \
template <> const numsky_ufunc numsky::ufunc_instance<numsky::UFUNC_##vName>::ufunc = { \
	2, \
	1, \
	reinterpret_cast<void*>(_ufunc_21::ufunc_item<numsky::UFUNC_##vName>::check_type), \
	reinterpret_cast<void*>(_ufunc_21::ufunc_item<numsky::UFUNC_##vName>::check_oper), \
	reinterpret_cast<void*>(_ufunc_21::ufunc_item<numsky::UFUNC_##vName>::check_init), \
	#vName, \
	numsky::UFUNC_##vName, \
}

// mathematic
make_ufunc(add);
make_ufunc(sub);
make_ufunc(mul);
make_ufunc(div);
make_ufunc(idiv);
make_ufunc(mod);
make_ufunc(pow);
// bitwise
make_ufunc(band);
make_ufunc(bor);
make_ufunc(bxor);
make_ufunc(shl);
make_ufunc(shr);
// comparison
make_ufunc(eq);
make_ufunc(lt);
make_ufunc(le);
make_ufunc(ne);
make_ufunc(gt);
make_ufunc(ge);
// other
make_ufunc(fmax);
make_ufunc(fmin);
make_ufunc(atan2);
