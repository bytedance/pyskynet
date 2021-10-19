
#include "tinygl/point.h"
#include "tinygl/Screen.h"
#include "tinygl/Mesh.h"
#include "tinygl/Shader.h"
#include <vector>
#include <functional>
#include <memory>

#define LEN(a) sizeof(a)/(3*sizeof(a[0][0]))

namespace tinygl {
	Mesh::Mesh(int vlen, std::function<void(V3f&, int)> vget, int ilen, std::function<void(V3i&, int)> iget) {
		final_matrix = Matrix4f::fromIdentity();
		vertices.resize(vlen);
		cache.resize(vlen);
		for(int i=0;i<vlen;i++) {
			vget(vertices[i], i);
		}
		indices.resize(ilen);
		for(int i=0;i<ilen;i++) {
			iget(indices[i], i);
		}
		update_matrix();
	}

	void Mesh::update_matrix() {
		matrix_modified = true;
	}

	// transform from world space to clip space, result save in cache;
	Shader* Mesh::prepare_draw(Matrix4f& camera_matrix, Screen* screen) {
		if(matrix_modified) {
			final_matrix = Matrix4f::fromTranslation(position.X, position.Y, -position.Z)*
			Matrix4f::fromRotation(-rotation.X, -rotation.Y, rotation.Z)*
			Matrix4f::fromScale(scale.X, scale.Y, scale.Z);
			matrix_modified = false;
		}

		Matrix4f matrix = camera_matrix*final_matrix;

		for(size_t i=0;i<vertices.size();i++) {
			// from unity to gl
			auto temp = vertices[i];
			temp.Z = - temp.Z;
			matrix.mul3to4(temp, cache[i].pc);
			cache[i].clip_code = cache[i].pc.calc_clipcode();
			if(cache[i].clip_code == 0) {
				screen->transform_to_screen(&shader, &cache[i]);
			}
		}
		return &shader;
	}

	static inline std::unique_ptr<Mesh> create_mesh(int vlen, float (*vertices)[3], int ilen, int (*indices)[3]) {
		std::unique_ptr<Mesh> re(new Mesh(vlen, [&](V3f& vertex, int i){
			for(int j=0;j<3;j++) {
				vertex.v[j] = vertices[i][j];
			}
		}, ilen, [&](V3i& index, int i) {
			for(int j=0;j<3;j++) {
				index.v[j] = indices[i][j];
			}
		}));
		unsigned char color[3] = {255,0,255};
		re->shader.setColor(3, color);
		return re;
	}

	std::unique_ptr<Mesh> Mesh::create_rect(float xmin, float ymin, float xsize, float ysize) {
		float vertices[][3] = {
			{xmin+xsize,ymin,0},{xmin+xsize,ymin+ysize,0},
			{xmin,ymin,0},{xmin,ymin+ysize,0},
		};
		int indices[][3] = {
			{0,1,2}, {2,1,3}
		};
		return create_mesh(LEN(vertices), vertices, LEN(indices), indices);
	}

	std::unique_ptr<Mesh> Mesh::create_sector(float center_x, float center_y, float radius, float degree) {
		int split_num = std::ceil(degree / 15);
		if(split_num < 2) {
			split_num = 2;
		}
		float per_angle = degree*(3.14159265358979323846/180)/split_num;
		float start_angle = -degree*(3.14159265358979323846/180)/2;
		std::unique_ptr<Mesh> re(new Mesh(split_num + 2, [&](V3f& vertex, int i){
			if(i==0) {
				vertex.X = center_x;
				vertex.Y = center_y;
				vertex.Z = 0;
			} else {
				float angle = start_angle + (i-1) * per_angle;
				vertex.X = center_x + radius*std::cos(angle);
				vertex.Y = center_y + radius*std::sin(angle);
				vertex.Z = 0;
			}
		}, split_num, [&](V3i& index, int i) {
			index.X = 0;
			index.Y = i + 1;
			index.Z = i + 2;
		}));
		unsigned char color[3] = {255,0,255};
		re->shader.setColor(3, color);
		return re;
	}

	std::unique_ptr<Mesh> Mesh::create_polygon(float center_x, float center_y, float radius, unsigned char edge_num) {
		if(edge_num < 3) {
			edge_num = 24;
		}
		float per_angle = 2*3.14159265358979323846/edge_num;
		std::unique_ptr<Mesh> re(new Mesh(edge_num, [&](V3f& vertex, int i){
			vertex.X = center_x + radius*std::cos(per_angle*i);
			vertex.Y = center_y + radius*std::sin(per_angle*i);
			vertex.Z = 0;
		}, edge_num-2, [&](V3i& index, int i) {
			index.X = 0;
			index.Y = i + 1;
			index.Z = i + 2;
		}));
		unsigned char color[3] = {255,0,255};
		re->shader.setColor(3, color);
		return re;
	}

	std::unique_ptr<Mesh> Mesh::create_circle(float center_x, float center_y, float radius) {
		return Mesh::create_polygon(center_x, center_y, radius, 24);
	}

	std::unique_ptr<Mesh> Mesh::create_line(float from_x, float from_y, float to_x, float to_y) {
		std::unique_ptr<Mesh> re(new Mesh(2, [&](V3f& vertex, int i){
			if(i==0) {
				vertex.X = from_x;
				vertex.Y = from_y;
				vertex.Z = 0;
			} else {
				vertex.X = to_x;
				vertex.Y = to_y;
				vertex.Z = 0;
			}
		}, 0, [&](V3i& index, int i) {
		}));
		unsigned char color[3] = {255,0,255};
		re->shader.setColor(3, color);
		re->shader.fill_type = FILL_LINE;
		return re;
	}

	std::unique_ptr<Mesh> Mesh::create_point(float x, float y) {
		std::unique_ptr<Mesh> re(new Mesh(1, [&](V3f& vertex, int i){
			vertex.X = x;
			vertex.Y = y;
			vertex.Z = 0;
		}, 0, [&](V3i& index, int i) {
		}));
		unsigned char color[3] = {255,0,255};
		re->shader.setColor(3, color);
		re->shader.fill_type = FILL_POINT;
		return re;
	}
}
