

You can build from the source code˙
    - [OpenCV](https://opencv.org/releases/) (v4.0 and later, only the main modules are needed).
    安装完后请在tmc3/CMakeLists.txt的3行将路径配置为您的目录，确保目录下有OpenCVConfig.cmake
	
	- [Open3D](https://www.open3d.org/) (请下载最新的release版本).
    安装完后请在tmc3/CMakeLists.txt的3行将路径配置为您的目录
	
之后可以运行make.sh（修改为release可调式模式））

-a=C:\Users\31046\Desktop\city3D\Block.ply -s=C:\Users\31046\Desktop\city3D\pointCloud.ply