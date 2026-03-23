cmake_minimum_required (VERSION 3.10)
project("bytetrack")

set(CMAKE_CXX_STANDARD 20)
# 设置为 ON 表示指定的 C++ 标准是必需的
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 设置为 OFF 表示禁用编译器特定的语言扩展
# 具体来说：
#   当设置为 OFF 时，编译器会使用纯标准的 C++ 标志，例如 -std=c++17
#   当设置为 ON（默认值）时，编译器会使用带扩展的标志，例如 -std=gnu++17
# 禁用扩展可以确保你的代码具有更好的可移植性，能够在不同的编译器之间兼容
set(CMAKE_CXX_EXTENSIONS OFF)

# 如果支持，请为 MSVC 编译器启用热重载。
if (POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

# opencv
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
    message("OpenCV_LIBS: " ${OpenCV_LIBS})
endif()

# Eigen3
find_package(Eigen3 REQUIRED)
if(Eigen3_FOUND)
    # 获取目标接口的包含目录
    get_target_property(Eigen3_INCLUDE_DIR Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    message("Eigen3 Target Include Dirs: ${Eigen3_INCLUDE_DIR}")
endif()

# Source files from src/
file(GLOB SRC_FILES src/ByteTrack/*.cpp)

# Test source files from test/
file(GLOB TEST_SRC_FILES test/*.cpp)

# Create the test executable
add_executable(test_app ${SRC_FILES} ${TEST_SRC_FILES})

# Link OpenCV and Eigen3 (header-only, so Eigen3 doesn't need linking)
target_link_libraries(test_app
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

target_include_directories(test_app PRIVATE
    include
    src/ByteTrack
)
