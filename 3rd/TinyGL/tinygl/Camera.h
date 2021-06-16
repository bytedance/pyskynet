
#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <assert.h>
#include "tinygl/Screen.h"
#include "tinygl/Transform.h"
#include "tinygl/math.h"
#include "tinygl/Mesh.h"

namespace tinygl {
	class Camera : public Transform {

	private:
		/* matrix */
		Matrix4f projection_matrix;
		Matrix4f final_matrix;

		/* screen with Z buffer */
		std::unique_ptr<void, void(*)(void*)> zbuf_base;
		std::unique_ptr<void, void(*)(void*)> pbuf_base;
		std::unique_ptr<Screen> screen;

	public:

		// normal constructor, malloc buffer & create screen
		Camera(int ysize, int xsize, int channel);



		inline Screen * getScreen() { return screen.get(); }

		// for 3d
		void perspective(float fovy, float aspect, float near, float farp);

		// for 2d
		void ortho(float left, float right, float bottom, float top, float near=-10, float far=10);

		// clear screen
		void clearScreen();

		// draw function
		void draw(Mesh* mesh);


	private:
		void update_matrix() final;
		Shader *shader;
		void gl_draw_point(GLVertex *p0);
		void gl_draw_line(GLVertex *p0, GLVertex *p1);
		void gl_draw_triangle(GLVertex *p0, GLVertex *p1, GLVertex *p2);
		void gl_draw_triangle_clip(GLVertex *p0,GLVertex *p1,GLVertex *p2,int clip_bit);
		void updateTmp(GLVertex *q,GLVertex *p0,GLVertex *p1,float t, int clip_mask);

	public:
		// some functions only used by lua
		Camera(int ysize, int xsize, int channel,
				void* v_zbuf_base, void(*zbuf_del)(void*), void* zbuf_data,
				void* v_pbuf_base, void(*pbuf_del)(void*), void* pbuf_data);

		inline void* get_zbuf_base() { return zbuf_base.get(); }
		inline void* get_pbuf_base() { return pbuf_base.get(); }
		inline void set_projection_matrix(const Matrix4f & matrix) {
			projection_matrix = matrix;
			update_matrix();
		}
	};
}

