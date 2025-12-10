cmake_minimum_required(VERSION 3.8)
project(perception_interfaces)

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# 인터페이스(SRV) 생성
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/PickObject.srv"
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
