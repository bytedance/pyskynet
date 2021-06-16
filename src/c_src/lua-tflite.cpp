#define LUA_LIB

#include <memory>
#include <sstream>
#include "c_api.h"
#include "lua-binding.h"
#include "numsky/lua-numsky.h"

namespace luabinding {
	template <> const char* Class_<TfLiteModel>::metaname= "tflite.model";
	template <> const char* Class_<TfLiteInterpreter>::metaname= "tflite.interpreter";
	template <> const char* Class_<TfLiteInterpreterOptions>::metaname= "tflite.options";
	template <> const char* Class_<TfLiteTensor>::metaname= "tflite.tensor";
}


namespace lua_tflite {

	static void error_reporter(void* L, const char* format, va_list args){
		char buf[1024];
		vsnprintf(buf, sizeof(buf), format, args);
		luaL_error(reinterpret_cast<lua_State*>(L), buf);
	}

	static int OptionsCreate(lua_State*L){
		auto obj = TfLiteInterpreterOptionsCreate();
		TfLiteInterpreterOptionsSetErrorReporter(obj, error_reporter, reinterpret_cast<void*>(L));
		TfLiteInterpreterOptionsSetNumThreads(obj, 1);
		// TODO(cz) how to use TfLiteInterpreterOptionsAddDelegate??
		luabinding::ClassUtil<TfLiteInterpreterOptions>::newwrap(L, obj);
		return 1;
	}

	static int OptionsSetNumThreads(lua_State*L) {
		auto obj = luabinding::ClassUtil<TfLiteInterpreterOptions>::check(L, 1);
		int n = luaL_checkinteger(L, 2);
		TfLiteInterpreterOptionsSetNumThreads(obj, n);
		return 0;
	}

	static int ModelCreate(lua_State*L) {
		size_t sz;
		const char *data = luaL_checklstring(L, 1, &sz);
		auto obj = TfLiteModelCreate(reinterpret_cast<const void*>(data), sz);
		luabinding::ClassUtil<TfLiteModel>::newwrap(L, obj);
		// model string must be refered
		lua_pushvalue(L, 1);
		lua_setuservalue(L, -2);
		return 1;
	}

	template <typename T> void noDelete(T* t) {}

namespace Tensor {
	static char get_typechar(const TfLiteTensor *tensor) {
		TfLiteType t = TfLiteTensorType(tensor);
		switch(t) {
			case kTfLiteBool:
				return '?';
			case kTfLiteInt8:
				return 'b';
			case kTfLiteUInt8:
				return 'B';
			case kTfLiteInt16:
				return 'h';
			case kTfLiteInt32:
				return 'i';
			case kTfLiteInt64:
				return 'l';
			case kTfLiteFloat16:
				return 'e';
			case kTfLiteFloat32:
				return 'f';
			case kTfLiteFloat64:
				return 'd';
			case kTfLiteString:
			case kTfLiteNoType:
			case kTfLiteComplex64:
			default:
				return '\0';
		}
	}
	static void name_getter(lua_State* L, TfLiteTensor* tensor) {
		const char* name = TfLiteTensorName(tensor);
		lua_pushstring(L, name);
	}
	static void data_getter(lua_State* L, TfLiteTensor* obj) {
		lua_pushlightuserdata(L, TfLiteTensorData(obj));
	}
	static void ndim_getter(lua_State* L, TfLiteTensor* obj){
		lua_pushinteger(L, TfLiteTensorNumDims(obj));
	}
	static void shape_getter(lua_State* L, TfLiteTensor* tensor){
		int32_t ndim = TfLiteTensorNumDims(tensor);
		std::unique_ptr<int[]> ptr(new int[ndim]);
		for(int i=0;i<ndim;i++) {
			ptr.get()[i] = TfLiteTensorDim(tensor, i);
		}
		numsky::new_tuple(L, ndim, ptr.get());
	}
	static void dtype_getter(lua_State*L, TfLiteTensor* tensor) {
		char typechar = Tensor::get_typechar(tensor);
		luaL_getmetatable(L, luabinding::Class_<numsky_dtype>::metaname);
		lua_geti(L, -1, typechar);
	}
	static int get(lua_State* L) {
		auto tensor = luabinding::ClassUtil<TfLiteTensor>::check(L, 1);
		char* data = reinterpret_cast<char*>(TfLiteTensorData(tensor));
		if(data == NULL) {
			return luaL_error(L, "tensor:get but tensor's data is NULL, maybe call interpreter:allocate_tensors first?");
		}
		char typechar = Tensor::get_typechar(tensor);
		int32_t ndim = TfLiteTensorNumDims(tensor);
		auto arr = numsky::ndarray_new_preinit<true>(L, ndim, typechar).get();
		for(int i=0;i<ndim;i++) {
			arr->dimensions[i] = TfLiteTensorDim(tensor, i);
		}
		numsky_ndarray_autostridecountalloc(arr);
		memcpy(arr->dataptr, data, arr->count*arr->dtype->elsize);
		return 1;
	}
	static int set(lua_State* L) {
		auto tensor = luabinding::ClassUtil<TfLiteTensor>::check(L, 1);
		auto arr = luabinding::ClassUtil<numsky_ndarray>::check(L, 2);
		// 1. check data
		char* data = reinterpret_cast<char*>(TfLiteTensorData(tensor));
		if(data == NULL) {
			return luaL_error(L, "set tensor but tensor's data is NULL");
		}
		// 2. check dim
		int32_t ndim = TfLiteTensorNumDims(tensor);
		if(arr->nd != ndim) {
			return luaL_error(L, "set tensor but array's dims not match %d, %d", arr->nd, ndim);
		}
		// 3. check shape
		for(int i=0;i<ndim;i++) {
			if(arr->dimensions[i] != TfLiteTensorDim(tensor, i)) {
			   return luaL_error(L, "set tensor but array's %d dim not match", i);
			}
		}
		char typechar = get_typechar(tensor);
		if(typechar != arr->dtype->typechar) {
			return luaL_error(L, "tensor '%c' dtype not match array '%c'", typechar, arr->dtype->typechar);
		}
		if(TfLiteTensorByteSize(tensor) != arr->dtype->elsize * arr->count) {
			return luaL_error(L, "tensor bytesize %d not match array %d", TfLiteTensorByteSize(tensor), arr->dtype->typechar);
		}
		numsky_ndarray_copyto(arr, data);
		return 0;
	}
	static int tostring(lua_State* L) {
		auto tensor = luabinding::ClassUtil<TfLiteTensor>::check(L, 1);
		std::stringstream ss;
		int32_t ndim = TfLiteTensorNumDims(tensor);
		char typechar = get_typechar(tensor);
		numsky_dtype *dtype = numsky_get_dtype_by_char(typechar);
		ss<<"tflite.tensor: "<<tensor<<"(name="<<TfLiteTensorName(tensor)<<", shape=(";
		for(int i=0;i<ndim;i++) {
			ss<<TfLiteTensorDim(tensor, i);
			if(i!=ndim-1) {
				ss<<", ";
			}
		}
		ss<<"), dtype="<<dtype->name<<")";
		lua_pushstring(L, ss.str().c_str());
		return 1;
	}

} // namespace Tensor


namespace Interpreter {
	static const int INPUT_TENSORS_INDEX = 1;
	static const int OUTPUT_TENSORS_INDEX = 2;
	static int create(lua_State* L) {
		 // 1. check arg 1 model buffer
		int arg1_type = lua_type(L, 1);
		std::unique_ptr<TfLiteModel, void (*)(TfLiteModel*)> model(nullptr, noDelete);
		if(arg1_type == LUA_TSTRING) {
			size_t sz;
			const char *data = luaL_checklstring(L, 1, &sz);
			model = decltype(model)(TfLiteModelCreate(reinterpret_cast<const void*>(data), sz), TfLiteModelDelete);
		} else if(arg1_type == LUA_TUSERDATA) {
			model = decltype(model)(luabinding::ClassUtil<TfLiteModel>::check(L, 1), noDelete);
		} else {
			return luaL_error(L, "interpreter(model, options) arg2 must be threads num or options but get %s", lua_typename(L, 1));
		}
		 // 2. check arg 2 threads num, default = 1
		int arg2_type = lua_type(L, 2);
		std::unique_ptr<TfLiteInterpreterOptions, void (*)(TfLiteInterpreterOptions*)> options(nullptr, noDelete);
		switch(arg2_type) {
			case LUA_TNONE:{
			   options = decltype(options)(TfLiteInterpreterOptionsCreate(), TfLiteInterpreterOptionsDelete);
			   TfLiteInterpreterOptionsSetErrorReporter(options.get(), lua_tflite::error_reporter, reinterpret_cast<void*>(L));
			   TfLiteInterpreterOptionsSetNumThreads(options.get(), 1);
			   break;
			}
			case LUA_TNUMBER:{
			   int32_t num_threads = luaL_checkinteger(L, 2);
			   options = decltype(options)(TfLiteInterpreterOptionsCreate(), TfLiteInterpreterOptionsDelete);
			   TfLiteInterpreterOptionsSetErrorReporter(options.get(), lua_tflite::error_reporter, reinterpret_cast<void*>(L));
			   TfLiteInterpreterOptionsSetNumThreads(options.get(), num_threads);
			   break;
			}
			case LUA_TUSERDATA:{
			   options = decltype(options)(luabinding::ClassUtil<TfLiteInterpreterOptions>::check(L, 2), noDelete);
			   break;
			}
			default:
			   return luaL_error(L, "interpreter(model, options) arg2 must be threads num or options but get %s", lua_typename(L, 2));
		}
		// 3. create interpreter
		auto interpreter = TfLiteInterpreterCreate(model.get(), options.get());
		luabinding::ClassUtil<TfLiteInterpreter>::newwrap(L, interpreter);
		lua_newtable(L); // uservalue
		if(arg1_type == LUA_TSTRING) {
			lua_pushvalue(L, 1);
		} else if(arg1_type == LUA_TUSERDATA) {
			lua_getuservalue(L, 1);
		}
		lua_setfield(L, -2, "model_buffer");
		lua_setuservalue(L, -2);
		return 1;
	}
	static int resize_input_tensor(lua_State*L){
		return luaL_error(L, "resize input tensor TODO");
	}
	static int invoke(lua_State*L){
		auto interpreter = luabinding::ClassUtil<TfLiteInterpreter>::check(L, 1);
		auto re = TfLiteInterpreterInvoke(interpreter);
		if(re != kTfLiteOk) {
			return luaL_error(L, "interpreter:invoke failed with status=%d", re);
		}
		return 0;
	}
	static int allocate_tensors(lua_State*L){
		auto interpreter = luabinding::ClassUtil<TfLiteInterpreter>::check(L, 1);
		auto re = TfLiteInterpreterAllocateTensors(interpreter);
		if(re != kTfLiteOk) {
			return luaL_error(L, "interpreter:allocate_tensors failed with status=%d", re);
		}
		// cache tensor obj in uservalue table
		lua_getuservalue(L, 1);
		// input tensor list
		lua_newtable(L);
		int32_t input_count = TfLiteInterpreterGetInputTensorCount(interpreter);
		for(int i=0;i<input_count;i++){
			auto tensor = TfLiteInterpreterGetInputTensor(interpreter, i);
			if(Tensor::get_typechar(tensor) == '\0') {
				luaL_error(L, "tensor with unsupported type : %s", TfLiteTypeGetName(TfLiteTensorType(tensor)));
			}
			auto name = TfLiteTensorName(tensor);
			luabinding::ClassUtil<TfLiteTensor>::newwrap(L, tensor);
			lua_pushvalue(L, -1);
			lua_seti(L, -3, i+1);
			lua_setfield(L, -2, name);
		}
		lua_seti(L, -2, Interpreter::INPUT_TENSORS_INDEX);
		// output tensor list
		lua_newtable(L);
		int32_t output_count = TfLiteInterpreterGetOutputTensorCount(interpreter);
		for(int i=0;i<output_count;i++){
			auto tensor = TfLiteInterpreterGetOutputTensor(interpreter, i);
			if(Tensor::get_typechar(tensor) == '\0') {
				luaL_error(L, "tensor with unsupported type : %s", TfLiteTypeGetName(TfLiteTensorType(tensor)));
			}
			auto name = TfLiteTensorName(tensor);
			luabinding::ClassUtil<TfLiteTensor>::newwrap(L, const_cast<TfLiteTensor*>(tensor));
			lua_pushvalue(L, -1);
			lua_seti(L, -3, i+1);
			lua_setfield(L, -2, name);
		}
		lua_seti(L, -2, Interpreter::OUTPUT_TENSORS_INDEX);
		return 0;
	}
	static void input_tensors_getter(lua_State*L, TfLiteInterpreter *tensor){
		lua_getuservalue(L, 1);
		lua_geti(L, -1, Interpreter::INPUT_TENSORS_INDEX);
	}
	static void output_tensors_getter(lua_State*L, TfLiteInterpreter *tensor){
		lua_getuservalue(L, 1);
		lua_geti(L, -1, Interpreter::OUTPUT_TENSORS_INDEX);
	}
} // namespace Interpreter

} // namespace lua_tflite

extern "C" {

	LUAMOD_API int luaopen_tflite(lua_State* L) {

		luaL_getmetatable(L, luabinding::Class_<numsky_ndarray>::metaname);
		if(!lua_istable(L, -1)) {
			luaL_error(L, "require \"numsky\" first...");
		}
		luabinding::Module_ m(L);

		m.start();

		// options
		luabinding::Class_<TfLiteInterpreterOptions> c1(L);
		c1.start()
			.setFieldFunction("set_num_threads", lua_tflite::OptionsSetNumThreads)
			.setMetaDefaultGC(TfLiteInterpreterOptionsDelete)
			.setMetaDefaultIndex()
			.finish();
		m.setFunction("options", lua_tflite::OptionsCreate);

		// model
		luabinding::Class_<TfLiteModel> c2(L);
		c2.start()
			.setMetaDefaultGC(TfLiteModelDelete)
			.setMetaDefaultIndex()
			.finish();
		m.setFunction("model", lua_tflite::ModelCreate);

		// tensor
		luabinding::Class_<TfLiteTensor> c3(L);
		c3.start()
			.setFieldProperty("name", lua_tflite::Tensor::name_getter, NULL)
			.setFieldProperty("shape", lua_tflite::Tensor::shape_getter, NULL)
			.setFieldProperty("dtype", lua_tflite::Tensor::dtype_getter, NULL)
			.setFieldProperty("data", lua_tflite::Tensor::data_getter, NULL)
			.setFieldProperty("ndim", lua_tflite::Tensor::ndim_getter, NULL)
			.setFieldFunction("get", lua_tflite::Tensor::get)
			.setFieldFunction("set", lua_tflite::Tensor::set)
			.setMetaDefaultIndex()
			.setMetaFunction("__tostring", lua_tflite::Tensor::tostring)
			.finish();

		// interpreter
		luabinding::Class_<TfLiteInterpreter> c4(L);
		c4.start()
			.setFieldProperty("output_tensors", lua_tflite::Interpreter::output_tensors_getter, NULL)
			.setFieldProperty("input_tensors", lua_tflite::Interpreter::input_tensors_getter, NULL)
			.setFieldFunction("resize_input_tensor", lua_tflite::Interpreter::resize_input_tensor)
			.setFieldFunction("allocate_tensors", lua_tflite::Interpreter::allocate_tensors)
			.setFieldFunction("invoke", lua_tflite::Interpreter::invoke)
			.setMetaDefaultGC(TfLiteInterpreterDelete)
			.setMetaDefaultIndex()
			.finish();
		m.setFunction("interpreter", lua_tflite::Interpreter::create);

		m.finish();
		return 1;
	}

}
