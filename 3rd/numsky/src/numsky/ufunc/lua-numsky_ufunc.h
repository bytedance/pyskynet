#ifndef LUA_NUMSKY_UFUNC_H
#define LUA_NUMSKY_UFUNC_H

#include "numsky/lua-numsky.h"

namespace numsky {
	enum UFUNC_ENUM {
		// 2->1 meta
		// mathematic
		UFUNC_add,
		UFUNC_sub,
		UFUNC_mul,
		UFUNC_div,
		UFUNC_idiv,
		UFUNC_mod,
		UFUNC_pow,
		// bitwise
		UFUNC_band,
		UFUNC_bor,
		UFUNC_bxor,
		UFUNC_shl,
		UFUNC_shr,
		// 2->1 not meta
		// comparison
		UFUNC_eq,
		UFUNC_lt,
		UFUNC_le,
		// comparison duplicating
		UFUNC_ne,
		UFUNC_gt,
		UFUNC_ge,
		// max min
		UFUNC_fmax,
		UFUNC_fmin,
		//UFUNC_maximum,
		//UFUNC_minimum,
		// integer
		//UFUNC_gcd,
		//UFUNC_lcm,
		// other floating 2-1 ufunc
		UFUNC_atan2,
		//UFUNC_copysign,
		//UFUNC_heaviside,
		//UFUNC_hypot,
		//UFUNC_logaddexp,
		//UFUNC_logaddexp2,
		//UFUNC_nextafter,
		// float + integer ufunc
		//UFUNC_ldexp,
		// special 2-1 ufunc
		UFUNC_matmul,

		// 1->1 meta
		UFUNC_unm,
		UFUNC_bnot,
		// 1->1 not meta
		UFUNC_abs,
		UFUNC_ceil,
		UFUNC_floor,
		// triangle
		UFUNC_acos,
		UFUNC_asin,
		UFUNC_atan,
		UFUNC_cos,
		UFUNC_sin,
		UFUNC_tan,
		UFUNC_deg,
		UFUNC_rad,
		// other
		UFUNC_log,
		UFUNC_exp,
		UFUNC_sqrt,

		UFUNC_isfinite,
		UFUNC_isinf,
		UFUNC_isnan,
		UFUNC_isnat,
	};
	template <int NIN, int NOUT> struct generic_ufunc;
	template <> struct generic_ufunc<1,1> {
	   using T_oper=void (*)(lua_State*, char*, char*);
	   using T_check_oper=T_oper (*)(lua_State*, char);
	   using T_check_type=char (*)(lua_State*, char);
	};
	template <> struct generic_ufunc<1,2> {
	   using T_oper=void (*)(lua_State*, char*, char*, char*);
	   using T_check_oper=T_oper (*)(lua_State*, char);
	   using T_check_type=char (*)(lua_State*, char);
	};
	template <> struct generic_ufunc<2,1> {
	   using T_oper=void (*)(lua_State*, char*, char*, char*);
	   using T_init=void (*)(lua_State*, char*, char*);
	   using T_check_oper=T_oper (*)(lua_State*, char, char);
	   using T_check_type=char (*)(lua_State*, char, char);
	   using T_check_init=T_init (*)(lua_State*, char);
	};
	template <> struct generic_ufunc<2,2> {
	   using T_oper=void (*)(lua_State*, char*, char*, char*, char*);
	   using T_check_oper=T_oper (*)(lua_State*, char, char);
	   using T_check_type=char (*)(lua_State*, char, char);
	};

	template <UFUNC_ENUM ufunc_num> struct ufunc_instance {
		static const numsky_ufunc ufunc;
	};

	int ufunc__call_21(lua_State *L, numsky_ufunc* ufunc, int lefti, int righti);
	int ufunc__call_11(lua_State *L, numsky_ufunc* ufunc, int argi);
	int ufunc_reduce(lua_State *L);

	int methods_sum(lua_State *L);
	int methods_prod(lua_State *L);
	int methods_any(lua_State *L);
	int methods_all(lua_State *L);
	int methods_max(lua_State *L);
	int methods_min(lua_State *L);
}

#endif
