using System;
using System.IO;
using System.Text;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Reflection;

namespace Drlua {
// #if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN || (UNITY_WSA && !UNITY_EDITOR) ? TODO needed?
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void DrluaCallback(IntPtr L, string data);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate byte [] CustomLoader(string path);

    public class StaticLuaCallbacks {
        //[MonoPInvokeCallback(typeof(DrluaCallback))]
		public static void Loader(IntPtr L, string path) {
			foreach(var customLoader in DrluaEnv.Instance.GetLoaderList()) {
				byte [] data = customLoader(path);
				if(data != null) {
					DrluaAPI.drlua_pushbuffer(L, data, data.Length);
					return ;
				}
			}
			DrluaAPI.drlua_pushbuffer(L, null, 0);
		}
        //[MonoPInvokeCallback(typeof(DrluaCallback))]
        public static void Panic(IntPtr L, string reason) {
            throw new Exception(reason);
        }
        //[MonoPInvokeCallback(typeof(DrluaCallback))]
        public static void Print(IntPtr L, string data) {
#if DRLUA_TEST // for build without unity
			Console.WriteLine("[DRLUA] "+data);
#else
			UnityEngine.Debug.Log("[DRLUA] "+data);
#endif
		}
	}

	class DrluaAPI {

#if DRLUA_TEST // for build without unity
		public const string LUADLL = "libdrlua.so";
#else
#if (UNITY_IPHONE || UNITY_TVOS || UNITY_WEBGL || UNITY_SWITCH) && !UNITY_EDITOR
		public const string LUADLL = "__Internal";
#else
		public const string LUADLL = "drlua";
#endif
#endif
		[DllImport(LUADLL, CallingConvention = CallingConvention.Cdecl)]
        public static extern void drlua_pushbuffer(IntPtr L, byte[] str, int len);

        [DllImport(LUADLL,CallingConvention=CallingConvention.Cdecl)]
		public static extern IntPtr drlua_new(DrluaCallback panic, DrluaCallback print, DrluaCallback loader);

        [DllImport(LUADLL,CallingConvention=CallingConvention.Cdecl)]
		public static extern int drlua_boot(IntPtr L, byte[] script, int scriptLen);

        [DllImport(LUADLL,CallingConvention=CallingConvention.Cdecl)]
		public static extern int drlua_newObj(IntPtr L, string objName, byte[] arg, int argLen);

        [DllImport(LUADLL,CallingConvention=CallingConvention.Cdecl)]
		public static extern IntPtr drlua_objCall(IntPtr L, int objId, string funcName, byte[] arg, int argLen, out IntPtr outLen);

        [DllImport(LUADLL,CallingConvention=CallingConvention.Cdecl)]
		public static extern void drlua_delObj(IntPtr L, int objId);

        [DllImport(LUADLL,CallingConvention=CallingConvention.Cdecl)]
		public static extern void drlua_popall(IntPtr L);


	}

	public class DrluaEnv {
		private IntPtr L = IntPtr.Zero;
		private List<CustomLoader> customLoaderList = new List<CustomLoader>();
		private DrluaEnv(){
			L = DrluaAPI.drlua_new(StaticLuaCallbacks.Panic, StaticLuaCallbacks.Print, StaticLuaCallbacks.Loader);
		}

		private static DrluaEnv _Instance = null;

        /// <summary> DrluaEnv使用单例模式 </summary>
		public static DrluaEnv Instance {
			get {
				if(_Instance == null) {
					_Instance = new DrluaEnv();
				}
				return _Instance;
			}
		}

        /// <summary> 等价于load(buf, "boot script")()，推荐用作lua的启动入口 </summary>
		public void boot(string buf){
			byte [] bytes = Encoding.UTF8.GetBytes(buf);
			DrluaAPI.drlua_boot(L, bytes, bytes.Length);
		}

        /// <summary> drlua接口，调用lua层创建一个对象，具体实现取决于lua层 </summary>
        /// <param name="objName"> 对象名 </param>
        /// <param name="arg"> 构造对象的bytes参数。可以为null。 </param>
        /// <param name="argLen"> 构造对象的bytes参数的长度。</param>
        /// <returns> 返回对象id </returns>
		public int newObj(string objName, byte [] arg, int argLen=-1) {
			if(arg == null) {
				argLen = 0;
			} else if(argLen < 0) {
				argLen = arg.Length;
			}
			int objId = DrluaAPI.drlua_newObj(L, objName, arg, argLen);
			DrluaAPI.drlua_popall(L);
			return objId;
		}

        /// <summary> drlua接口，调用lua层对象的方法，具体实现取决于lua层 </summary>
        /// <param name="objId"> 对象id </param>
        /// <param name="funcName"> 方法名 </param>
        /// <param name="arg"> 调用方法时传入的bytes参数。可以为null。</param>
        /// <param name="argLen"> 调用方法时传入的bytes参数的长度。</param>
        /// <returns> 返回的bytes </returns>
		public byte[] objCall(int objId, string funcName, byte [] arg, int argLen=-1){
			if(arg == null) {
				argLen = 0;
			} else if(argLen < 0) {
				argLen = arg.Length;
			}
			IntPtr outputSize;
			IntPtr ptr = DrluaAPI.drlua_objCall(L, objId, funcName, arg, argLen, out outputSize);
			int outputLen = outputSize.ToInt32();
			byte [] outputBytes = new byte[outputLen];
			Marshal.Copy(ptr, outputBytes, 0, outputLen);
			DrluaAPI.drlua_popall(L);
			return outputBytes;
		}

        /// <summary> drlua接口，删除一个lua层的对象，具体实现取决于lua层 </summary>
        /// <param name="objId"> 对象id </param>
		public void delObj(int objId) {
			DrluaAPI.drlua_delObj(L, objId);
		}

        /// <summary> 用于添加一个加载lua脚本的回调，在lua层调用require时触发 </summary>
        /// <example>
		/// <code>
		/// DrluaEnv.Instance.AddLoader(path => {
		///			string filepath = path.Replace('.', '/') + ".lua";
		///			if(File.Exists(filepath)){
		///				var bytes = File.ReadAllBytes(filepath);
		///				return bytes;
		///			}
		///			return null;
		///	});
		/// </code>
		/// </example>
        /// <param name="loader"> 加载函数 </param>
		public void AddLoader(CustomLoader loader){
			customLoaderList.Add(loader);
		}

        /// <summary> 获取所有添加的加载函数，供StaticLuaCallbacks.Loader使用 </summary>
		public List<CustomLoader> GetLoaderList(){
			return customLoaderList;
		}

	}
}
