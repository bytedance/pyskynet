

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace numsky {
	namespace canvas {
		struct ExpandControl {
			int fi_if;
			int fi_forvar;
			int fi_forgen;
			int fi_forseq;
			int fi_forsort;
			ExpandControl () : fi_if(0), fi_forvar(0), fi_forgen(0), fi_forseq(0), fi_forsort(0) {}
		};

	}
}

