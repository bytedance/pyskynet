

#pragma once

namespace tinygl {
	class Transform {
	protected:
		V3f position;
		V3f rotation;
		V3f scale;
		virtual void update_matrix() = 0;
	public:
		Transform () : position(0,0,0), rotation(0,0,0), scale(1,1,1) {}
		virtual ~Transform() {};

		inline void setScale(float x, float y, float z) {
			scale.X = x;
			scale.Y = y;
			scale.Z = z;
			update_matrix();
		}

		inline void setPosition(float x, float y, float z) {
			position.X = x;
			position.Y = y;
			position.Z = z;
			update_matrix();
		}

		inline void setRotation(float x, float y, float z) {
			rotation.X = x;
			rotation.Y = y;
			rotation.Z = z;
			update_matrix();
		}

		inline V3f getRotation() {
			return rotation;
		}

		inline V3f getPosition() {
			return position;
		}

		inline V3f getScale() {
			return scale;
		}

	};
}
