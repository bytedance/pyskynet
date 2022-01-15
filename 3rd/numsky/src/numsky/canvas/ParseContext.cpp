
#include "numsky/canvas/ParseContext.h"


namespace numsky {
	namespace canvas {
		ParseContext::ParseContext(lua_State*l, std::string& xml_script): NAME_FUNCS("____"), L(l), fi_counter(0), si_counter(0) {
			text = xml_script.c_str();
			int text_len = xml_script.size();
			int line_counter = 0;
			pos2line[0] = line_counter ++;
			for(int i=0;i<text_len;i++){
				if(text[i] == '\n') {
					pos2line[i] = line_counter ++;
				}
			}
			pos2line[text_len] = line_counter ++;
			streamw<<"local "<<NAME_FUNCS<<"={} ";

		}
		int ParseContext::calc_line(const char *ptr) {
			int pos = ptr - text;
			auto iter = pos2line.upper_bound(pos);
			if(iter==pos2line.end()) {
				printf("[CANVAS PARSER WARNING] invalid line for pos=%d\n", pos);
				return -1;
			} else {
				return iter->second;
			}
		}
		void ParseContext::raise(const char *where, const DefinedException & e) {
			std::string msg = e.tostring();
			luaL_error(L, "xml:ParseError:line:%d, %s", calc_line(where), msg.c_str());
		}
		void ParseContext::raise(const char *where, const char* what) {
			luaL_error(L, "xml:ParseError:line:%d, %s", calc_line(where), what);
		}
		void ParseContext::raise(const char *where, const char* what, const std::string &after) {
			luaL_error(L, "xml:ParseError:line:%d, %s: %s", calc_line(where), what, after.c_str());
		}
	}

	namespace canvas {
		void PostParseContext::throw_func(const std::string &s) {
			luaL_error(L, "xml:PostParseError:line:%d, throw:%s", cur_line, s.c_str());
		}
	}
}
