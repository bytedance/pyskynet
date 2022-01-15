
#pragma once

#include <memory>
#include <string>

#include "rapidxml.hpp"

extern "C" {
#include "lua.h"
#include "lauxlib.h"
}

namespace numsky {
	namespace canvas {
		class DefinedException {
		public:
			virtual std::string tostring() const = 0;
		};

		class TagWrongTagException : public DefinedException {
			rapidxml::xml_node<> *parent_xnode;
			rapidxml::xml_node<> *xnode;
		public:
			TagWrongTagException(rapidxml::xml_node<> *v_p_xnode, rapidxml::xml_node<> *v_xnode) : parent_xnode(v_p_xnode), xnode(v_xnode) {}
			std::string tostring() const final;
		};

		class TagWrongAttrException: public DefinedException {
			rapidxml::xml_node<> *xnode;
			rapidxml::xml_attribute<> *xattr;
		public:
			TagWrongAttrException(rapidxml::xml_node<> *v_xnode, rapidxml::xml_attribute<>* v_xattr) : xnode(v_xnode), xattr(v_xattr) {}
			std::string tostring() const final;
		};

		class AttrWrongTypeException: public DefinedException {
			rapidxml::xml_attribute<> *xattr;
		public:
			AttrWrongTypeException(rapidxml::xml_attribute<> *v_xattr) : xattr(v_xattr) {}
			std::string tostring() const final;
		};
	}
}
