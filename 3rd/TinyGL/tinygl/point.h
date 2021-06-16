#pragma once

#include "tinygl/math.h"

namespace tinygl {
	struct ZBufferPoint {
	  int x,y,z;     /* integer coordinates in the screen */
	};

	/* computed vertex */
	struct GLVertex {
		V4f pc;               /* coordinates in the normalized volume */
		int clip_code;        /* clip code */
		ZBufferPoint zp;      /* integer coordinates for the rasterization */
	};
}
