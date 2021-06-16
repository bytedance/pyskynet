
#pragma once

#include <cmath>
#include <stdlib.h>

/* Vector & Matrix */

namespace tinygl {
	enum AXIS_ENUM {
		AXIS_X = 0,
		AXIS_Y = 1,
		AXIS_Z = 2,
	};

	union V2f {
		float v[2];
		struct {
			float U;
			float V;
		};
		V2f() {}
		V2f(float u, float v) : U(u), V(v) {}
	};

	union V3f {
		 float v[3];
		 struct {
			 float X;
			 float Y;
			 float Z;
		 };
		 V3f() {}
		 V3f(float x, float y, float z) : X(x), Y(y), Z(z) {}
	};

	union V3i {
		 int v[3];
		 struct {
			 int X;
			 int Y;
			 int Z;
		 };
		 V3i() {}
		 V3i(int x, int y, int z) : X(x), Y(y), Z(z) {}
	};

	union V4f {
		 float v[4];
		 struct {
			 float X;
			 float Y;
			 float Z;
			 float W;
		 };
		 V4f() {}
		 V4f(float x, float y, float z, float w) : X(x), Y(y), Z(z), W(w) {}
		 inline int calc_clipcode() {
			 const double CLIP_EPSILON = 1e-5;
			 float w1=W * (1.0 + CLIP_EPSILON);
			 return (X<-w1) |
				 ((X>w1)<<1) |
				 ((Y<-w1)<<2) |
				 ((Y>w1)<<3) |
				 ((Z<-w1)<<4) |
				 ((Z>w1)<<5) ;
		 }
	};

	class Matrix4f {
	private:
		float m[4][4];
	public:
		/* member functions */
		inline void operator*=(const Matrix4f& right) {
			Matrix4f left = *this;
			for(int i=0;i<4;i++) {
				for(int j=0;j<4;j++) {
					float s=0.0;
					for(int k=0;k<4;k++) {
						s+=left.m[i][k]*right.m[k][j];
					}
					m[i][j]=s;
				}
			}
		}
		inline Matrix4f operator*(const Matrix4f& right){
			Matrix4f ret;
			for(int i=0;i<4;i++) {
				for(int j=0;j<4;j++) {
					float s=0.0;
					for(int k=0;k<4;k++) {
						s+=m[i][k]*right.m[k][j];
					}
					ret.m[i][j]=s;
				}
			}
			return ret;
		}
		inline void mul3to4(V3f & v3, V4f & v4){
			float *a = &m[0][0];

			v4.X = (v3.X * a[0] + v3.Y * a[1] +
				  v3.Z * a[2] + a[3]);
			v4.Y = (v3.X * a[4] + v3.Y * a[5] +
				  v3.Z * a[6] + a[7]);
			v4.Z = (v3.X * a[8] + v3.Y * a[9] +
				  v3.Z * a[10] + a[11]);
			v4.W = (v3.X * a[12] + v3.Y * a[13] +
				  v3.Z * a[14] + a[15]);
		}
		static inline Matrix4f fromFrustum(float left,float right,float bottom,float top, float near,float farp) {
			float x = (2.0*near) / (right-left);
			float y = (2.0*near) / (top-bottom);
			float A = (right+left) / (right-left);
			float B = (top+bottom) / (top-bottom);
			float C = -(farp+near) / ( farp-near);
			float D = -(2.0*farp*near) / (farp-near);

			Matrix4f matrix;
			float *r=&matrix.m[0][0];
			r[0]= x; r[1]=0; r[2]=A; r[3]=0;
			r[4]= 0; r[5]=y; r[6]=B; r[7]=0;
			r[8]= 0; r[9]=0; r[10]=C; r[11]=D;
			r[12]= 0; r[13]=0; r[14]=-1; r[15]=0;
			return matrix;
		}
		static inline Matrix4f fromPerspective(float fovy, float aspect, float near, float farp) {
			float ymax = near * std::tan( fovy * 3.14159265358979323846 / 360.0 );
			float ymin = -ymax;

			float xmin = ymin * aspect;
			float xmax = ymax * aspect;
			return Matrix4f::fromFrustum(xmin, xmax, ymin, ymax, near, farp);
		}
		static inline Matrix4f fromOrtho(float left,float right,float bottom,float top, float near,float farp) {
			float x = 2.0 / (right-left);
			float y = 2.0 / (top-bottom);
			float z = -2.0 / (farp-near);
			float A = -(right+left) / (right-left);
			float B = -(top+bottom) / (top-bottom);
			float C = -(farp+near) / ( farp-near);

			Matrix4f matrix;
			float *r=&matrix.m[0][0];
			r[0]= x; r[1]=0; r[2]=0; r[3]=A;
			r[4]= 0; r[5]=y; r[6]=0; r[7]=B;
			r[8]= 0; r[9]=0; r[10]=z; r[11]=C;
			r[12]= 0; r[13]=0; r[14]=0; r[15]=1;
			return matrix;
		}
		static inline Matrix4f fromIdentity() {
			Matrix4f matrix;
			for(int i=0;i<4;i++) {
				for(int j=0;j<4;j++) {
					if (i==j) {
						matrix.m[i][j]=1.0;
					}else {
						matrix.m[i][j]=0.0;
					}
				}
			}
			return matrix;
		}
		static inline Matrix4f fromScale(float x, float y, float z){
			Matrix4f re = fromIdentity();
			re.m[0][0] = x;
			re.m[1][1] = y;
			re.m[2][2] = z;
			return re;
		}
		static inline Matrix4f fromTranslation(float x, float y, float z) {
			Matrix4f re = fromIdentity();
			re.m[0][3] = x;
			re.m[1][3] = y;
			re.m[2][3] = z;
			return re;
		}
		template <AXIS_ENUM axis> static inline Matrix4f fromAxisRotate(float degree) {
			float rad = degree * 3.141592653589793 / 180.0;
			Matrix4f matrix = fromIdentity();
			int v=(axis+1)%3;
			int w=(axis+2)%3;
			float s=std::sin(rad);
			float c=std::cos(rad);
			matrix.m[v][v]=c;	matrix.m[v][w]=-s;
			matrix.m[w][v]=s;	matrix.m[w][w]=c;
			return matrix;
		}
		static inline Matrix4f fromAxisRotate(float degree, float x, float y, float z) {
			float u[3] = {x, y, z};

			float angle = degree * 3.141592653589793 / 180.0;
			Matrix4f matrix;

			/* normalize vector */
			float len = u[0]*u[0]+u[1]*u[1]+u[2]*u[2];
			if (len == 0.0f) {
				return fromIdentity();
			}
			len = 1.0f / std::sqrt(len);
			u[0] *= len;
			u[1] *= len;
			u[2] *= len;

			/* store cos and sin values */
			float cost=std::cos(angle);
			float sint=std::sin(angle);

			/* fill in the values */
			matrix.m[3][0]=matrix.m[3][1]=matrix.m[3][2]=
				matrix.m[0][3]=matrix.m[1][3]=matrix.m[2][3]=0.0f;
			matrix.m[3][3]=1.0f;

			/* do the math */
			matrix.m[0][0]=u[0]*u[0]+cost*(1-u[0]*u[0]);
			matrix.m[1][0]=u[0]*u[1]*(1-cost)-u[2]*sint;
			matrix.m[2][0]=u[2]*u[0]*(1-cost)+u[1]*sint;
			matrix.m[0][1]=u[0]*u[1]*(1-cost)+u[2]*sint;
			matrix.m[1][1]=u[1]*u[1]+cost*(1-u[1]*u[1]);
			matrix.m[2][1]=u[1]*u[2]*(1-cost)-u[0]*sint;
			matrix.m[0][2]=u[2]*u[0]*(1-cost)-u[1]*sint;
			matrix.m[1][2]=u[1]*u[2]*(1-cost)+u[0]*sint;
			matrix.m[2][2]=u[2]*u[2]+cost*(1-u[2]*u[2]);
			return matrix;
		}
		static inline Matrix4f fromRotation(float eulerX, float eulerY, float eulerZ) {
			Matrix4f re = fromAxisRotate<AXIS_Y>(eulerY)*fromAxisRotate<AXIS_X>(eulerX)*fromAxisRotate<AXIS_Z>(eulerZ);
			return re;
		}
	};


}
