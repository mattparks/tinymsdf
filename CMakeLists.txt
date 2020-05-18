cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)
project(tinymsdf VERSION 0.1.0 LANGUAGES CXX)

option(BUILD_TINYMSDF_EXAMPLE "Build the tinymsdf example" ON)

include(GNUInstallDirs)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

if(NOT FREETYPE_FOUND)
	find_package(Freetype REQUIRED)
endif()

add_library(tinymsdf tinymsdf.hpp tinymsdf.cpp)
target_compile_features(tinymsdf PUBLIC cxx_std_17)
target_include_directories(tinymsdf PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
		${FREETYPE_INCLUDE_DIRS}
		)
target_link_libraries(tinymsdf PUBLIC ${FREETYPE_LIBRARIES})

if(BUILD_TINYMSDF_EXAMPLE)
	add_executable(tinymsdf-example lodepng.h lodepng.cpp main.cpp)
	set_target_properties(tinymsdf-example PROPERTIES OUTPUT_NAME tinymsdf)
	target_include_directories(tinymsdf-example PUBLIC
			$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
			)
	target_link_libraries(tinymsdf-example PUBLIC tinymsdf)
endif()
