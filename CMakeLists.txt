
project(hipCPU)

cmake_minimum_required (VERSION 3.5)

find_package(OpenMP REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

include_directories(${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/include/hipCPU)
subdirs(tests)

install(DIRECTORY include/hipCPU DESTINATION include
        FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY include/hipCPU DESTINATION include
        FILES_MATCHING PATTERN "*.h")

# Uninstall target
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
