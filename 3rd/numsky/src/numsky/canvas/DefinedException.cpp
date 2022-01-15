
#include <sstream>
#include "numsky/canvas/DefinedException.h"

namespace numsky {
	namespace canvas {
		const char root_name[] = "xml";
		static inline const char* _xml_name(rapidxml::xml_node<> *xnode) {
			if(xnode->type()==rapidxml::node_document) {
				return root_name;
			} else {
				return xnode->name();
			}
		}

		static inline const char* _xml_name(rapidxml::xml_attribute<> *xattr) {
			return xattr->name();
		}

		std::string TagWrongTagException::tostring() const {
			std::stringstream ss;
			ss<<"wrong child tag:"<<_xml_name(parent_xnode)<<"~"<<_xml_name(xnode);
			return ss.str();
		}

		std::string TagWrongAttrException::tostring() const {
			std::stringstream ss;
			ss<<"can't set attribute "<<_xml_name(xattr)<<" in "<<_xml_name(xnode);
			return ss.str();
		}

		std::string AttrWrongTypeException::tostring() const {
			std::stringstream ss;
			ss<<"attribute set wrong type "<<_xml_name(xattr);
			return ss.str();
		}
	}
}
