# TinyGL

## 简介
本工程是一个基于CPU绘制、用于实现多通道特征的轻量图形库，支持绘制2D/3D图形。
修改自开源库[TinyGL](https://bellard.org/TinyGL/)。

### 若干设计
* 为简化设计，该图形库接口的坐标系、旋转表示等和unity一致，为左手系，按照左手定则旋转角度。
* rotation为角度制，相机默认朝向为z轴正方向。
* 绘制2D图片时，建议使用xoy平面，z坐标置0. rotation只设置z的旋转。
* 像素buffer最多支持16个byte的通道

### 基础类
* Transform
	* Transform是Mesh和Camera的基类，具有改变位置、旋转和缩放的功能
	* 缩放仅对Mesh生效
	* 旋转以欧拉角表示，暂不支持四元数
* Mesh
	* Mesh是图形类，使用顶点(vertices)+面数组(indices)表示图形，类似于obj格式。
	* Shader是着色类，每个Mesh都包含一个Shader实例表示该Mesh的着色方式
	* 通过Shader可以配置绘制Mesh的颜色、绘制方式(点/线/面)。
	* 如果是以点的方式绘制图形，则会绘制顶点数组中的所有点。
	* 如果以线绘制图形，则会按照顶点数组依次连接成一个闭环。
	* 如果以面绘制图形，则会按照根据面数组绘制三角形面片。
	* Shader的layer表示颜色从第layer个通道开始绘制
	* 内置了若干2D图形，矩形、圆形、扇形、正多边形。
	* 3D图形TODO
* Camera
	* Camera是相机类，默认方向为z轴正方向，ortho用做正交投影，perspective用做透视投影，一般情况下，ortho用于2d，perspective用于3d。
	* Screen是屏幕类，每个Camera都包含一个Screen实例用来维护绘制图形的buffer，buffer分为像素buffer(pbuf)和深度buffer(zbuf)
	* Camera初始化时传入的长宽即Screen的长宽，会自动拉伸绘制的图形，但不会改变Camera的视野范围。
	* 当Camera的变换矩阵(Camera的坐标、旋转、ortho、perspective)被设置时，Screen会自动清空，其他情况下需要手动清空Screen。

### 使用示例
* 参见[example.cpp](example.cpp)
