
extern "C" {
#include <lua.h>
#include <lauxlib.h>
#include <stdlib.h>
}

#include "rapidxml.hpp"
#include <map>

class LineContext {
	private:
	std::map<int, int> pos2line;
	const char* base;
	int len;
	public:
	LineContext(const char* text, int text_len) : base(text), len(text_len) {
		int line_counter = 0;
		pos2line[0] = line_counter++;
		for(int i=0;i<len;i++){
			if(text[i] == '\n') {
				pos2line[i+1] = line_counter++;
			}
		}
		pos2line[len] = line_counter++;
	}
	int calc_line(char* ptr){
		int pos = ptr-base;
		auto iter = pos2line.lower_bound(pos);
		if(iter==pos2line.end()) {
			return -1;
		} else {
			return iter->second;
		}
	}
};

static void recursive(lua_State*L, rapidxml::xml_node<> *node, LineContext* context) {
	if(node->type()==rapidxml::node_document || node->type() == rapidxml::node_element) {
		lua_newtable(L);
		// set __tag
		lua_pushlstring(L, node->name(), node->name_size());
		lua_setfield(L, -2, "__tag");
		// set __line
		if(node->type() == rapidxml::node_document) {
			lua_pushinteger(L, 1);
		} else {
			lua_pushinteger(L, context->calc_line(node->name()) + 1);
		}
		lua_setfield(L, -2, "__line");
		for(rapidxml::xml_attribute<> *attr=node->first_attribute();attr;attr=attr->next_attribute()) {
			lua_pushlstring(L, attr->name(), attr->name_size());
			lua_pushlstring(L, attr->value(), attr->value_size());
			lua_settable(L, -3);
		}
		int count = 1;
		for(rapidxml::xml_node<> *child = node->first_node();child;child=child->next_sibling()){
			recursive(L, child, context);
			lua_seti(L, -2, count++);
		}
	} else if(node->type()==rapidxml::node_data) {
		lua_pushlstring(L, node->value(), node->value_size());
	} else if(node->type()==rapidxml::node_pi){
		lua_pushlstring(L, node->name(), node->value() - node->name() + node->value_size());
	} else {
		luaL_error(L, "other xml node type=%d TODO", node->type());
	}
}

static int parse(lua_State*L){
	size_t data_len=0;
    const char *data = luaL_checklstring(L, 1, &data_len);
	rapidxml::xml_document<> doc;
	LineContext context(data, data_len);
	try {
		doc.parse<rapidxml::parse_non_destructive|rapidxml::parse_pi_nodes>(const_cast<char*>(data));
	} catch (rapidxml::parse_error err) {
		luaL_error(L, "xml error:%d, %s", context.calc_line(err.where<char>()) + 1, err.what());
	}
	recursive(L, &doc, &context);
	return 1;
}

extern "C" {

	LUAMOD_API int luaopen_rapidxml(lua_State* L) {
		luaL_Reg libs[] = {
			{ "parse", parse},
			{ NULL, NULL }
		};
		luaL_newlib(L, libs);
		return 1;
	}

}
