#include <stdlib.h>
#include "tinygl/Screen.h"

#define pixel_call(pixelsize, tfunc, ...) \
switch(pixelsize){ \
	case 1: { tfunc<1>(__VA_ARGS__); break; } \
	case 2: { tfunc<2>(__VA_ARGS__); break; } \
	case 3: { tfunc<3>(__VA_ARGS__); break; } \
	case 4: { tfunc<4>(__VA_ARGS__); break; } \
	case 5: { tfunc<5>(__VA_ARGS__); break; } \
	case 6: { tfunc<6>(__VA_ARGS__); break; } \
	case 7: { tfunc<7>(__VA_ARGS__); break; } \
	case 8: { tfunc<8>(__VA_ARGS__); break; } \
	case 9: { tfunc<9>(__VA_ARGS__); break; } \
	case 10: { tfunc<10>(__VA_ARGS__); break; } \
	case 11: { tfunc<11>(__VA_ARGS__); break; } \
	case 12: { tfunc<12>(__VA_ARGS__); break; } \
	case 13: { tfunc<13>(__VA_ARGS__); break; } \
	case 14: { tfunc<14>(__VA_ARGS__); break; } \
	case 15: { tfunc<15>(__VA_ARGS__); break; } \
	case 16: { tfunc<16>(__VA_ARGS__); break; } \
};


namespace tinygl {
	template <int N> struct PIXEL {
		unsigned char data[N];
		inline void assign(Shader *shader) {
			int pixelsize = shader->color_pixelsize;
			if(pixelsize + shader->layer > N) {
				pixelsize = N - shader->layer;
			}
			for(int i=0;i<pixelsize;i++) {
				data[i+shader->layer] = shader->color_data[i];
			}
		}
	};

	template <int N> static inline PIXEL<N> PCHAR_TO_PIXEL(unsigned char *pchar) {
		return *reinterpret_cast<PIXEL<N>*>(pchar);
	}

	template <int N> static inline PIXEL<N>* PCHAR_TO_PPIXEL(unsigned char *pchar) {
		return reinterpret_cast<PIXEL<N>*>(pchar);
	}

	template <int N> void Screen::plot_N(Shader *shader, ZBufferPoint * p) {

		PIXEL<N> *pp = PCHAR_TO_PPIXEL<N>(this->pbuf) + (p->y * this->xsize + p->x);
		unsigned short *pz = this->zbuf + (p->y * this->xsize + p->x);

		int zz = p->z >> ZB_POINT_Z_FRAC_BITS;
		if (zz>=*pz) {
			pp->assign(shader);
			*pz = zz;
		}
	}

	template <int N> void Screen::line_N(Shader *shader, ZBufferPoint * p1, ZBufferPoint * p2) {

		if (p1->y > p2->y || (p1->y == p2->y && p1->x > p2->x)) {
			ZBufferPoint *tmp = p1;
			p1 = p2;
			p2 = tmp;
		}
		int sx = this->xsize;

		PIXEL<N> *pp = PCHAR_TO_PPIXEL<N>(this->pbuf) + (p1->y * sx + p1->x);
		unsigned short *pz = this->zbuf + (p1->y * sx + p1->x);

		int z = p1->z;

		auto DRAWLINE = [&] (int dx, int dy, int inc_1, int inc_2) {
			int n=dx;
			int zinc=(p2->z-p1->z)/n;
			int a=2*dy-dx;
			dy=2*dy;
			dx=2*dx-dy;
			do {
				// PUT_PIXEL
				int zz = z >> ZB_POINT_Z_FRAC_BITS;
				if (zz>=*pz)  {
					pp->assign(shader);
					*pz=zz;
				}
				// go forward
				z+=zinc;
				if (a>0) {
					pp+=(inc_1);
					pz+=(inc_1);
					a-=dx;
				}
				else {
					pp+=(inc_2);
					pz+=(inc_2);
					a+=dy;
				}
			} while (--n >= 0);
		};

		int dx = p2->x - p1->x;
		int dy = p2->y - p1->y;

		if (dx == 0 && dy == 0) {
			int zz = z >> ZB_POINT_Z_FRAC_BITS;
			if (zz>=*pz)  {
				pp->assign(shader);
				*pz=zz;
			}
		} else if (dx > 0) {
			if (dx >= dy) {
				DRAWLINE(dx, dy, sx + 1, 1);
			} else {
				DRAWLINE(dy, dx, sx + 1, sx);
			}
		} else {
			dx = -dx;
			if (dx >= dy) {
				DRAWLINE(dx, dy, sx - 1, -1);
			} else {
				DRAWLINE(dy, dx, sx - 1, sx);
			}
		}
	}

	template <int N> void Screen::triangle_N(Shader *shader, ZBufferPoint *p0,ZBufferPoint *p1,ZBufferPoint *p2) {


		int error=0,derror=0;
		int x1=0,dxdy_min=0,dxdy_max=0;
		/* warning: x2 is multiplied by 2^16 */
		int x2=0,dx2dy2=0;

		int z1=0,dzdl_min=0,dzdl_max=0;

		/* we sort the vertex with increasing y */
		if (p1->y < p0->y) {
			ZBufferPoint *t = p0;
			p0 = p1;
			p1 = t;
		}
		if (p2->y < p0->y) {
			ZBufferPoint *t = p2;
			p2 = p1;
			p1 = p0;
			p0 = t;
		} else if (p2->y < p1->y) {
			ZBufferPoint *t = p1;
			p1 = p2;
			p2 = t;
		}

		/* we compute dXdx and dXdy for all interpolated values */

		float fdx1 = p1->x - p0->x;
		float fdy1 = p1->y - p0->y;

		float fdx2 = p2->x - p0->x;
		float fdy2 = p2->y - p0->y;

		float fz = fdx1 * fdy2 - fdx2 * fdy1;
		if (fz == 0) {
			this->line_N<N>(shader, p0, p1);
			this->line_N<N>(shader, p0, p2);
			return;
		}
		fz = 1.0 / fz;

		fdx1 *= fz;
		fdy1 *= fz;
		fdx2 *= fz;
		fdy2 *= fz;

		float d1 = p1->z - p0->z;
		float d2 = p2->z - p0->z;
		int dzdx = (int) (fdy2 * d1 - fdy1 * d2);
		int dzdy = (int) (fdx1 * d2 - fdx2 * d1);

		/* screen coordinates */

		PIXEL<N> *pp1 = PCHAR_TO_PPIXEL<N>(this->pbuf) + p0->y * this->xsize;
		unsigned short *pz1 = this->zbuf + p0->y * this->xsize;

		for(int part=0;part<2;part++) {
			ZBufferPoint *pr1,*pr2,*l1,*l2;
			int update_left,update_right;
			int nb_lines;
			if (part == 0) {
				if (fz > 0) {
					update_left=1;
					update_right=1;
					l1=p0;
					l2=p2;
					pr1=p0;
					pr2=p1;
				} else {
					update_left=1;
					update_right=1;
					l1=p0;
					l2=p1;
					pr1=p0;
					pr2=p2;
				}
				nb_lines = p1->y - p0->y;
			} else {
				/* second part */
				if (fz > 0) {
					update_left=0;
					update_right=1;
					pr1=p1;
					pr2=p2;
				} else {
					update_left=1;
					update_right=0;
					l1=p1;
					l2=p2;
				}
				nb_lines = p2->y - p1->y + 1;
			}

			/* compute the values for the left edge */

			if (update_left) {
				int dy1 = l2->y - l1->y;
				int dx1 = l2->x - l1->x;
				int tmp;
				if (dy1 > 0)
					tmp = (dx1 << 16) / dy1;
				else
					tmp = 0;
				x1 = l1->x;
				error = 0;
				derror = tmp & 0x0000ffff;
				dxdy_min = tmp >> 16;
				dxdy_max = dxdy_min + 1;

				z1=l1->z;
				dzdl_min=(dzdy + dzdx * dxdy_min);
				dzdl_max=dzdl_min + dzdx;
			}

			/* compute values for the right edge */

			if (update_right) {
				int dx2 = (pr2->x - pr1->x);
				int dy2 = (pr2->y - pr1->y);
				if (dy2>0)
					dx2dy2 = ( dx2 << 16) / dy2;
				else
					dx2dy2 = 0;
				x2 = pr1->x << 16;
			}

			/* we draw all the scan line of the part */

			while (nb_lines>0) {
				nb_lines--;
				/* generic draw line */
				{

					int n=(x2 >> 16) - x1;

					PIXEL<N> *pp=pp1+x1;
					unsigned short *pz=pz1+x1;
					unsigned int z=z1;
					while (n>=0) {
						// draw pixel
						unsigned int zz = z >> ZB_POINT_Z_FRAC_BITS;
						if (zz >= pz[0]) {
							pp[0].assign(shader);
							pz[0]=zz;
						}
						z+=dzdx;
						// move forward
						pp=pp+1;
						pz+=1;
						n-=1;
					}
				}

				/* left edge */
				error+=derror;
				if (error > 0) {
					error-=0x10000;
					x1+=dxdy_max;
					z1+=dzdl_max;
				} else {
					x1+=dxdy_min;
					z1+=dzdl_min;
				}

				/* right edge */
				x2+=dx2dy2;

				/* screen coordinates */
				pp1+=this->xsize;
				pz1+=this->xsize;
			}
		}
	}

	inline void clip_min_max(int &num, int left, int right) {
		if(num < left) {
			num = left;
		} else if (num > right) {
			num = right;
		}
	}
	void Screen::transform_to_screen(Shader *shader, GLVertex *v) {
		float winv = 1.0/v->pc.W;
		v->zp.x = (int) ( v->pc.X * winv * scale.X
				+ trans.X );
		v->zp.y = (int) ( v->pc.Y * winv * scale.Y
				+ trans.Y );
		v->zp.z = (int) ( v->pc.Z * winv * scale.Z
				+ trans.Z );
		clip_min_max(v->zp.x, 0, xsize - 1);
		clip_min_max(v->zp.y, 0, ysize - 1);
	}

}

namespace tinygl {
	Screen::Screen(int v_ysize, int v_xsize, int v_pixelsize, unsigned short *v_zbuf, unsigned char *v_pbuf) :
		ysize(v_ysize), xsize(v_xsize), pixelsize(v_pixelsize), zbuf(v_zbuf), pbuf(v_pbuf) {

		float zsize = (1 << (ZB_Z_BITS + ZB_POINT_Z_FRAC_BITS));

		trans.X = xsize / 2.0;
		trans.Y = ysize / 2.0;
		trans.Z = ((zsize - 0.5) / 2.0) + ((1 << ZB_POINT_Z_FRAC_BITS)) / 2;

		scale.X = (xsize - 1) / 2.0;
		scale.Y = -(ysize - 1) / 2.0;
		scale.Z = -((zsize - 0.5) / 2.0);

		dirty = true;
		clear();
	}

	void Screen::clear() {
		if(dirty) {
			memset(zbuf, 0, xsize*ysize*sizeof(unsigned short));
			memset(pbuf, 0, xsize*ysize*pixelsize);
			dirty = false;
		}
	}

	void Screen::plot(Shader *shader, ZBufferPoint *p) {
		dirty = true;
		pixel_call(pixelsize, plot_N, shader, p);
	}

	void Screen::line(Shader *shader, ZBufferPoint *p1, ZBufferPoint *p2) {
		dirty = true;
		pixel_call(pixelsize, line_N, shader, p1, p2);
	}

	void Screen::triangle(Shader *shader, ZBufferPoint *p0, ZBufferPoint *p1, ZBufferPoint *p2) {
		dirty = true;
		pixel_call(pixelsize, triangle_N, shader, p0, p1, p2);
	}
}
