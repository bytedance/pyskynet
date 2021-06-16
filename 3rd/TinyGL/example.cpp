
#include <memory>
#include "tinygl/Mesh.h"
#include "tinygl/Camera.h"
#include "tinygl/Screen.h"
#include "bitmap_image.hpp"
#include <cstring>

int height = 50;
int width = 100;
int pixelsize = 5;

int main(){
	// 1. create camera
	std::unique_ptr<tinygl::Camera> came(new tinygl::Camera(height,width,pixelsize)); // put height before width, just like buffer's shape
	came->ortho(-10,10,-10,10); // screen auto scale for ortho,height,width
	came->setPosition(2,2,0); // set camera's position, z=0 for 2d

	// 2. create mesh
	std::unique_ptr<tinygl::Mesh> rect1 = tinygl::Mesh::create_rect(0,0,5,5);
	rect1->shader.layer = 1; // rect1's color will fill from layerth channel
	float green_blue = 250.250f;
	rect1->shader.setColor(4,reinterpret_cast<unsigned char*>(&green_blue)); // set color with a 4-byte float
	rect1->setPosition(-1,-1,0); // set position, z=0 for 2d

	std::unique_ptr<tinygl::Mesh> rect2 = tinygl::Mesh::create_rect(0,0,5,5);
	unsigned char red[1] = {255};
	rect2->shader.layer = 0; // default layer is 0
	rect2->shader.setColor(1,red);
	rect2->setRotation(0,0,90); // set angleZ rotation for 2d
	rect2->setScale(0.5,0.5,1); // set scale, z=1 for 2d

	// 3. draw camera
	came->draw(rect1.get());
	came->draw(rect2.get());

	// 4. get buffer // buffer shape is (height, width, pixelsize)
	unsigned char *pbuf = came->getScreen()->pbuf;

	// 5. save buffer to hello.bmp
	bitmap_image image(width,height);
	for(int y=0;y<height;y++) {
		for(int x=0;x<width;x++) {
			unsigned char *offset = pbuf + (y*width + x)*pixelsize;
			float green_blue = *reinterpret_cast<float*>(offset+1);
			int green = green_blue;
			int blue = 1000*(green_blue - green);
			image.set_pixel(x,y,offset[0],green,blue);
		}
	}
	image.save_image(std::string("hello.bmp"));
	return 0;
}
