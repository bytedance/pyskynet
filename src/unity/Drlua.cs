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

        /// <summary> DrluaEnv?????????????????? </summary>
		public static DrluaEnv Instance {
			get {
				if(_Instance == null) {
					_Instance = new DrluaEnv();
				}
				return _Instance;
			}
		}

        /// <summary> ?????????load(buf, "boot script")()???????????????lua??????????????? </summary>
		public void boot(string buf){
			byte [] bytes = Encoding.UTF8.GetBytes(buf);
			DrluaAPI.drlua_boot(L, bytes, bytes.Length);
		}

        /// <summary> drlua???????????????lua?????????????????????????????????????????????lua??? </summary>
        /// <param name="objName"> ????????? </param>
        /// <param name="arg"> ???????????????bytes??????????????????null??? </param>
        /// <param name="argLen"> ???????????????bytes??????????????????</param>
        /// <returns> ????????????id </returns>
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

        /// <summary> drlua???????????????lua??????????????????????????????????????????lua??? </summary>
        /// <param name="objId"> ??????id </param>
        /// <param name="funcName"> ????????? </param>
        /// <param name="arg"> ????????????????????????bytes??????????????????null???</param>
        /// <param name="argLen"> ????????????????????????bytes??????????????????</param>
        /// <returns> ?????????bytes </returns>
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

        /// <summary> drlua?????????????????????lua????????????????????????????????????lua??? </summary>
        /// <param name="objId"> ??????id </param>
		public void delObj(int objId) {
			DrluaAPI.drlua_delObj(L, objId);
		}

        /// <summary> ????????????????????????lua?????????????????????lua?????????require????????? </summary>
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
        /// <param name="loader"> ???????????? </param>
		public void AddLoader(CustomLoader loader){
			customLoaderList.Add(loader);
		}

        /// <summary> ???????????????????????????????????????StaticLuaCallbacks.Loader?????? </summary>
		public List<CustomLoader> GetLoaderList(){
			return customLoaderList;
		}

	}
}
