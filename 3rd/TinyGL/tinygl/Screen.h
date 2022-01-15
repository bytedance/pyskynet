#pragma once

/*
 * Z buffer
 */
#include <string.h>
#include <stdio.h>

#include "point.h"
#include "Shader.h"


#define ZB_Z_BITS 16

#define ZB_POINT_Z_FRAC_BITS 14


namespace tinygl {

	union ScreenShape {
		int d[3];
		struct {
			int ysize;
			int xsize;
			int pixelsize;
		};
		ScreenShape(int v_ysize, int v_xsize, int v_pixelsize) : ysize(v_ysize), xsize(v_xsize), pixelsize(v_pixelsize) {}
	};

	class Screen {
	private:
		V3f scale;
		V3f trans;
		int ysize,xsize;
		int pixelsize;		// channel
		bool dirty;
	public:
		unsigned short *zbuf; // depth buffer, shape=(ysize, xsize), dtype=uint16
		unsigned char *pbuf; // pixel buffer, shape=(ysize, xsize, pixelsize), dtype=uint8
		Screen(int v_ysize, int v_xsize, int v_pixelsize, unsigned short *v_zbuf, unsigned char *v_pbuf);

		// pixel array shape : (ysize, xsize, channel)
		inline ScreenShape getShape() { return ScreenShape(ysize, xsize, pixelsize); }
		friend class Camera;
		friend class Mesh;

	private:
		// set zbuf & pbuf with 0
		void clear();

		/* draw functions */
		void plot(Shader *shader, ZBufferPoint *p);

		void line(Shader *shader, ZBufferPoint *p1,ZBufferPoint *p2);

		void triangle(Shader* shader, ZBufferPoint *p1,ZBufferPoint *p2,ZBufferPoint *p3);

		void transform_to_screen(Shader *shader, GLVertex *v);

		template <int N> void plot_N(Shader *shader, ZBufferPoint * p);
		template <int N> void line_N(Shader *shader, ZBufferPoint * p1, ZBufferPoint * p2);
		template <int N> void triangle_N(Shader *shader, ZBufferPoint *p0,ZBufferPoint *p1,ZBufferPoint *p2);
	};
}
