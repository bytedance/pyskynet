
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <sstream>

#include "numsky/canvas/IAstNode.h"
#include "numsky/canvas/ParseContext.h"

namespace numsky {
	namespace canvas {
	}
}

namespace numsky {
	namespace canvas {
		class AttrParse {
		public:
			std::map<std::string, void(*)(IAstNode*, ParseContext*, rapidxml::xml_attribute<>*)> nameToFunc;
			void parse(IAstNode*node, ParseContext*ctx, rapidxml::xml_attribute<>*attr) {
				auto iter = nameToFunc.find(attr->name());
				if(iter==nameToFunc.end()) {
					ctx->raise(attr->name(), "invalid attr", attr->name());
				} else {
					iter->second(node, ctx, attr);
				}
		   	}
			AttrParse();
		};
		class TagParse {
		public:
			std::map<std::string, IAstNode*(*)(BaseAstNode*, ParseContext*, rapidxml::xml_node<>*)> nameToFunc;
			IAstNode* parse(BaseAstNode*node, ParseContext*ctx, rapidxml::xml_node<>*attr) {
				auto iter = nameToFunc.find(attr->name());
				if(iter==nameToFunc.end()) {
					ctx->raise(attr->name(), "invalid tag", attr->name());
					return NULL;
				} else {
					return iter->second(node, ctx, attr);
				}
			}
			TagParse();
		};
	}
}
