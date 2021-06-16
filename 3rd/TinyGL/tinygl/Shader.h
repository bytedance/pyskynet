#pragma once

#define MAX_PIXEL_BYTES 16
#include <memory>
#include <functional>

namespace tinygl {
	enum FILL_TYPE {
		FILL_POINT=0,
		FILL_LINE=1,
		FILL_TRIANGLE=2,
	};

	struct Shader {
	private:
		int color_pixelsize;
		unsigned char color_data[MAX_PIXEL_BYTES];
	public:
		template <int N> friend struct PIXEL;
	public:
		unsigned char layer; // fill screen's channel from layerth byte, default layer = 0
		FILL_TYPE fill_type;
		Shader(): color_pixelsize(1), layer(0), fill_type(FILL_TRIANGLE) {
			memset(color_data, 0, MAX_PIXEL_BYTES);
		}
		inline void setColor(int pixelsize, unsigned char *data) {
			color_pixelsize = pixelsize;
			if(pixelsize>MAX_PIXEL_BYTES) {
				color_pixelsize = MAX_PIXEL_BYTES;
			}
			for(int i=0;i<color_pixelsize;i++) {
				color_data[i] = data[i];
			}
		}
	};

}
