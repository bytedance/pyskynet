using System;
using System.IO;
using System.Text;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Reflection;
using Drlua;

namespace Temp {
	public class Test {
		public static void TestXOR(){
			var drlua = DrluaEnv.Instance;
			drlua.AddLoader(path => {
					string filepath = path.Replace('.', '/') + ".lua";
					if(File.Exists(filepath)){
						var bytes = File.ReadAllBytes(filepath);
						return bytes;
					}
					return null;
			});
			drlua.boot("require 'example'");
			var modelbytes = File.ReadAllBytes("xor.tflite");
			int a = drlua.newObj("interpreter", modelbytes);
			for(int x1=0;x1<=1;x1++) {
				for(int x2=0;x2<=1;x2++) {
					string s = String.Format("[{0},{1}]", x1, x2);
					byte [] re = drlua.objCall(a, "invoke", Encoding.UTF8.GetBytes(s));
					Console.WriteLine(s + "->" + Encoding.UTF8.GetString(re));
				}
			}
		}
		public static void TestScript(string [] args){
			var drlua = DrluaEnv.Instance;
			drlua.AddLoader(path => {
					string filepath = path.Replace('.', '/') + ".lua";
					if(File.Exists(filepath)){
						var bytes = File.ReadAllBytes(filepath);
						return bytes;
					}
					return null;
			});
			if(args.Length!=1) {
				Console.WriteLine("usage: test xxx.lua");
			} else {
				var mainScript = File.ReadAllText(args[0]);
				drlua.boot(mainScript);
			}
		}
		public static void Main(string[] args) {
			TestXOR();
		}
	}
}
