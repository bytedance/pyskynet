#include "tinygl/Camera.h"

/* fill triangle profile */
/* #define PROFILE */

#define CLIP_XMIN   (1<<0)
#define CLIP_XMAX   (1<<1)
#define CLIP_YMIN   (1<<2)
#define CLIP_YMAX   (1<<3)
#define CLIP_ZMIN   (1<<4)
#define CLIP_ZMAX   (1<<5)

/* point & line */
namespace tinygl {

	/* point */

	void Camera::gl_draw_point(GLVertex *p0) {
		if (p0->clip_code == 0) {
			screen->plot(shader, &p0->zp);
		}
	}

	/* line */

	static inline void interpolate(GLVertex *q,GLVertex *p0,GLVertex *p1,float t) {
		q->pc.X=p0->pc.X+(p1->pc.X-p0->pc.X)*t;
		q->pc.Y=p0->pc.Y+(p1->pc.Y-p0->pc.Y)*t;
		q->pc.Z=p0->pc.Z+(p1->pc.Z-p0->pc.Z)*t;
		q->pc.W=p0->pc.W+(p1->pc.W-p0->pc.W)*t;
	}

	/*
	 * Line Clipping
	 */

	/* Line Clipping algorithm from 'Computer Graphics', Principles and
	   Practice */
	static inline int ClipLine1(float denom,float num,float *tmin,float *tmax) {
		float t;

		if (denom>0) {
			t=num/denom;
			if (t>*tmax) return 0;
			if (t>*tmin) *tmin=t;
		} else if (denom<0) {
			t=num/denom;
			if (t<*tmin) return 0;
			if (t<*tmax) *tmax=t;
		} else if (num>0) return 0;
		return 1;
	}

	void Camera::gl_draw_line(GLVertex *p1,GLVertex *p2) {
		float dx,dy,dz,dw,x1,y1,z1,w1;
		float tmin,tmax;
		GLVertex q1,q2;
		int cc1,cc2;

		cc1=p1->clip_code;
		cc2=p2->clip_code;

		if ( (cc1 | cc2) == 0) {
			screen->line(shader, &p1->zp,&p2->zp);
		} else if ( (cc1&cc2) != 0 ) {
			return;
		} else {
			dx=p2->pc.X-p1->pc.X;
			dy=p2->pc.Y-p1->pc.Y;
			dz=p2->pc.Z-p1->pc.Z;
			dw=p2->pc.W-p1->pc.W;
			x1=p1->pc.X;
			y1=p1->pc.Y;
			z1=p1->pc.Z;
			w1=p1->pc.W;

			tmin=0;
			tmax=1;
			if (ClipLine1(dx+dw,-x1-w1,&tmin,&tmax) &&
					ClipLine1(-dx+dw,x1-w1,&tmin,&tmax) &&
					ClipLine1(dy+dw,-y1-w1,&tmin,&tmax) &&
					ClipLine1(-dy+dw,y1-w1,&tmin,&tmax) &&
					ClipLine1(dz+dw,-z1-w1,&tmin,&tmax) &&
					ClipLine1(-dz+dw,z1-w1,&tmin,&tmax)) {

				interpolate(&q1,p1,p2,tmin);
				interpolate(&q2,p1,p2,tmax);
				screen->transform_to_screen(shader, &q1);
				screen->transform_to_screen(shader, &q2);
				screen->line(shader, &q1.zp,&q2.zp);
			}
		}
	}
}


/* triangle */
namespace tinygl {

	/*
	 * Clipping
	 */

	/* We clip the segment [a,b] against the 6 planes of the normal volume.
	 * We compute the point 'c' of intersection and the value of the parameter 't'
	 * of the intersection if x=a+t(b-a).
	 */

#define clip_func(name,sign,dir,dir1,dir2) \
	static float name(V4f *c,V4f *a,V4f *b) \
	{\
	  float t,dX,dY,dZ,dW,den;\
	  dX = (b->X - a->X);\
	  dY = (b->Y - a->Y);\
	  dZ = (b->Z - a->Z);\
	  dW = (b->W - a->W);\
	  den = -(sign d ## dir) + dW;\
	  if (den == 0) t=0;\
	  else t = ( sign a->dir - a->W) / den;\
	  c->dir1 = a->dir1 + t * d ## dir1;\
	  c->dir2 = a->dir2 + t * d ## dir2;\
	  c->W = a->W + t * dW;\
	  c->dir = sign c->W;\
	  return t;\
	}


	clip_func(clip_xmin,-,X,Y,Z)

	clip_func(clip_xmax,+,X,Y,Z)

	clip_func(clip_ymin,-,Y,X,Z)

	clip_func(clip_ymax,+,Y,X,Z)

	clip_func(clip_zmin,-,Z,X,Y)

	clip_func(clip_zmax,+,Z,X,Y)


	float (*clip_proc[6])(V4f *,V4f *,V4f *)=  {
		clip_xmin,clip_xmax,
		clip_ymin,clip_ymax,
		clip_zmin,clip_zmax
	};

	void Camera::updateTmp(GLVertex *q,GLVertex *p0,GLVertex *p1,float t, int clip_mask) {
		q->clip_code=q->pc.calc_clipcode() & ~((clip_mask-1) | clip_mask);
		if (q->clip_code==0) {
			screen->transform_to_screen(shader, q);
		}
	}

	void Camera::gl_draw_triangle(GLVertex *p0,GLVertex *p1,GLVertex *p2) {
		int cc[3] = {
			p0->clip_code,
			p1->clip_code,
			p2->clip_code
		};

		/* we handle the non clipped case here to go faster */
		if ((cc[0] | cc[1] | cc[2]) == 0) {
			screen->triangle(shader, &p0->zp,&p1->zp,&p2->zp);
		} else {
			if ((cc[0] & cc[1] & cc[2]) == 0) {
				gl_draw_triangle_clip(p0,p1,p2,0);
			}
		}
	}

	void Camera::gl_draw_triangle_clip(GLVertex *p0,GLVertex *p1,GLVertex *p2,int clip_bit) {
		int co,co1,cc[3],clip_mask;
		GLVertex tmp1,tmp2,*q[3];
		float tt;

		cc[0]=p0->clip_code;
		cc[1]=p1->clip_code;
		cc[2]=p2->clip_code;

		co=cc[0] | cc[1] | cc[2];
		if (co == 0) {
			gl_draw_triangle(p0,p1,p2);
		} else {
			/* the triangle is completely outside */
			if((cc[0] & cc[1] & cc[2]) != 0) return ;

			/* find the next direction to clip */
			while (clip_bit < 6 && (co & (1 << clip_bit)) == 0)  {
				clip_bit++;
			}

			/* this test can be true only in case of rounding errors */
			if (clip_bit == 6) {
#if 0
				printf("Error:\n");
				printf("%f %f %f %f\n",p0->pc.X,p0->pc.Y,p0->pc.Z,p0->pc.W);
				printf("%f %f %f %f\n",p1->pc.X,p1->pc.Y,p1->pc.Z,p1->pc.W);
				printf("%f %f %f %f\n",p2->pc.X,p2->pc.Y,p2->pc.Z,p2->pc.W);
#endif
				return;
			}

			clip_mask = 1 << clip_bit;
			co1=(cc[0] ^ cc[1] ^ cc[2]) & clip_mask;

			if (co1)  {
				/* one point outside */

				if (cc[0] & clip_mask) { q[0]=p0; q[1]=p1; q[2]=p2; }
				else if (cc[1] & clip_mask) { q[0]=p1; q[1]=p2; q[2]=p0; }
				else { q[0]=p2; q[1]=p0; q[2]=p1; }

				tt=clip_proc[clip_bit](&tmp1.pc,&q[0]->pc,&q[1]->pc);
				updateTmp(&tmp1,q[0],q[1],tt,clip_mask);

				tt=clip_proc[clip_bit](&tmp2.pc,&q[0]->pc,&q[2]->pc);
				updateTmp(&tmp2,q[0],q[2],tt,clip_mask);

				gl_draw_triangle_clip(&tmp1,q[1],q[2],clip_bit+1);

				gl_draw_triangle_clip(&tmp2,&tmp1,q[2],clip_bit+1);
			} else {
				/* two points outside */

				if ((cc[0] & clip_mask)==0) { q[0]=p0; q[1]=p1; q[2]=p2; }
				else if ((cc[1] & clip_mask)==0) { q[0]=p1; q[1]=p2; q[2]=p0; }
				else { q[0]=p2; q[1]=p0; q[2]=p1; }

				tt=clip_proc[clip_bit](&tmp1.pc,&q[0]->pc,&q[1]->pc);
				updateTmp(&tmp1,q[0],q[1],tt,clip_mask);

				tt=clip_proc[clip_bit](&tmp2.pc,&q[0]->pc,&q[2]->pc);
				updateTmp(&tmp2,q[0],q[2],tt,clip_mask);

				gl_draw_triangle_clip(q[0],&tmp1,&tmp2,clip_bit+1);
			}
		}
	}
}

/* matrix */
namespace tinygl {
	void Camera::clearScreen() {
		screen->clear();
	}

	void Camera::update_matrix() {
		final_matrix = projection_matrix *
			Matrix4f::fromRotation(rotation.X, rotation.Y, -rotation.Z)*
			Matrix4f::fromTranslation(-position.X, -position.Y, position.Z);
		clearScreen();
	}

	void Camera::perspective(float fovy, float aspect, float near, float farp) {
		projection_matrix = Matrix4f::fromPerspective(fovy, aspect, near, farp);
		update_matrix();
	}

	void Camera::ortho(float left, float right, float bottom, float top, float near, float far) {
		projection_matrix = Matrix4f::fromOrtho(left, right, bottom, top, near, far);
		update_matrix();
	}

}

/* some api functions */
namespace tinygl {
	Camera::Camera(int ysize, int xsize, int channel)
		: zbuf_base(malloc(xsize*ysize*sizeof(unsigned short)), [](void * data){free(data);}),
		  pbuf_base(malloc(xsize*ysize*channel), [](void * data){free(data);}),
		  screen(new Screen(
					  ysize, xsize, channel,
					  reinterpret_cast<unsigned short*>(zbuf_base.get()),
					  reinterpret_cast<unsigned char*>(pbuf_base.get()))
				  ) {
		projection_matrix = Matrix4f::fromIdentity();
		final_matrix = Matrix4f::fromIdentity();
		update_matrix();
	}

	Camera::Camera(int ysize, int xsize, int channel,
			void* v_zbuf_base, void(*zbuf_del)(void*), void* zbuf_data,
			void* v_pbuf_base, void(*pbuf_del)(void*), void* pbuf_data)
		: zbuf_base(v_zbuf_base, zbuf_del),
		  pbuf_base(v_pbuf_base, pbuf_del),
		  screen(new Screen(ysize, xsize, channel,
					  reinterpret_cast<unsigned short*>(zbuf_data),
					  reinterpret_cast<unsigned char*>(pbuf_data))) {
		projection_matrix = Matrix4f::fromIdentity();
		final_matrix = Matrix4f::fromIdentity();
		update_matrix();
	}

	/* draw functions */
	void Camera::draw(Mesh *mesh) {
		shader = mesh->prepare_draw(final_matrix, screen.get());
		switch(shader->fill_type) {
			case FILL_POINT: {
				for(auto iter=mesh->gl_cache().begin();iter!=mesh->gl_cache().end();iter++) {
					gl_draw_point(&(*iter));
				}
				break;
			}
			case FILL_LINE: {
				int size = mesh->gl_cache().size();
				if(size == 2){
					gl_draw_line(&mesh->gl_cache()[0], &mesh->gl_cache()[1]);
				} else if (size == 1) {
					gl_draw_point(&mesh->gl_cache()[0]);
				} else if(size >= 3) {
					for(int i=0;i<size-1;i++) {
						gl_draw_line(&mesh->gl_cache()[i], &mesh->gl_cache()[i+1]);
					}
					gl_draw_line(&mesh->gl_cache()[size-1], &mesh->gl_cache()[0]);
				}
				break;
			}
			case FILL_TRIANGLE: {
				for(auto iter=mesh->getIndices().begin();iter!=mesh->getIndices().end();iter++) {
					gl_draw_triangle(&mesh->gl_cache()[iter->X], &mesh->gl_cache()[iter->Y], &mesh->gl_cache()[iter->Z]);
				}
				break;
			}
			default:
				break;
		}
		shader = nullptr;
	}

}
