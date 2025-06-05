cmake_minimum_required(VERSION 3.0)
# 项目名称o3d_01， 编程语言C++
project(TMC3 LANGUAGES CXX)

# 使用C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(OpenCV_DIR "C:/little_boy/software/opencv/opencv/build/x64/vc16/lib")
# 添加Open3D根目录路径到CMake的搜索路径
list(APPEND CMAKE_PREFIX_PATH "C:/little_boy/software/Open3D-devel-windows-amd64-0.19.0")

# 2. 查找 包
find_package(OpenCV REQUIRED
    COMPONENTS core highgui imgproc videoio  # 指定你需要的模块
    PATHS ${OpenCV_DIR}  # 指定搜索路径
)

find_package(Open3D REQUIRED)

# 检查是否找到包
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found! Please set OpenCV_DIR correctly.")
else()
    # 打印找到的 OpenCV 版本信息
    message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
endif()

if(NOT Open3D_FOUND)
    message(FATAL_ERROR "Open3D not found! Please set Open3D_DIR correctly.")
else()
    # 打印找到的 OpenCV 版本信息
    message(STATUS "Found Open3D: ${Open3D_VERSION}")
    message(STATUS "Open3D include dirs: ${Open3D_INCLUDE_DIRS}")
    message(STATUS "Open3D libraries: ${Open3D_LIBS}")
endif()




if (UNIX)
  add_definitions(-D_POSIX_C_SOURCE=200809L)
elseif (WIN32)
  add_definitions(-D_USE_MATH_DEFINES)
endif()



find_package(Git)



file(GLOB PROJECT_INC_FILES
"PCCMisc.h"
"PCCPointSet.h"
"PCCMath.h"
"PCCMisc.h"
 "my_function.h"
 "ply.h"
"tables.h"
)

file(GLOB PROJECT_CPP_FILES
 "my_function.cpp"
 "TMC3.cpp"
 "ply.cpp"
)


add_executable (tmc3
  ${PROJECT_CPP_FILES}
  ${PROJECT_INC_FILES}
)

# 5. 链接库
target_link_libraries(tmc3 PRIVATE ${OpenCV_LIBS})
target_include_directories(tmc3 PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(tmc3 PRIVATE Open3D::Open3D)

# 由于下载的Open3D是动态库，需要将Open3D.dll拷贝到执行文件的目录下，以下自定义指令完成了这一工作。
add_custom_command(TARGET tmc3 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    "${Open3D_DIR}/../bin/Open3D.dll"
    $<TARGET_FILE_DIR:tmc3>)

install (TARGETS tmc3 DESTINATION bin)
