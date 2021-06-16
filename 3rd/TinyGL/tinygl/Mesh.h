
#pragma once

#include "tinygl/math.h"
#include "tinygl/Screen.h"
#include "tinygl/Shader.h"
#include "tinygl/Transform.h"
#include <vector>
#include <functional>
#include <memory>

namespace tinygl {
	class Mesh : public Transform {
	private:
		std::vector<V3f> vertices;
		std::vector<V3i> indices;
		std::vector<GLVertex> cache;
		Matrix4f final_matrix;
		bool matrix_modified;
	public:
		Shader shader;
		Mesh(int vlen, std::function<void(V3f&, int)> vget, int ilen, std::function<void(V3i&, int)> iget);

		// 2d builtin mesh
		static std::unique_ptr<Mesh> create_rect(float xmin, float ymin, float xsize, float ysize);
		static std::unique_ptr<Mesh> create_polygon(float xmin, float ymin, float radius, unsigned char edge_num);
		static std::unique_ptr<Mesh> create_circle(float xmin, float ymin, float radius);
		static std::unique_ptr<Mesh> create_sector(float center_x, float center_y, float radius, float degree);
		static std::unique_ptr<Mesh> create_line(float from_x, float from_y, float to_x, float to_y);
		static std::unique_ptr<Mesh> create_point(float x, float y);

		// 3d builtin mesh TODO


		/* functions for drawing */

		// transform from world space to clip space, result save in cache;
		Shader* prepare_draw(Matrix4f& camera_matrix, Screen* screen);
		inline std::vector<V3f> & getVertices() { return vertices; }
		inline std::vector<V3i> & getIndices() { return indices; }
		inline std::vector<GLVertex> & gl_cache() { return cache; }
		void update_matrix() final;


	};
}
