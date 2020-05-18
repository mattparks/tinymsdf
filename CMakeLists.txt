cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)
project(tinymsdf VERSION 0.1.0 LANGUAGES C)

option(BUILD_TINYMSDF_EXAMPLE "Build the tinymsdf example" ON)

include(GNUInstallDirs)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

if(NOT FREETYPE_FOUND)
	find_package(Freetype REQUIRED)
endif()

add_library(tinymsdf tinymsdf.h tinymsdf.c)
target_compile_features(tinymsdf PUBLIC c_std_11)
target_include_directories(tinymsdf PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
		${FREETYPE_INCLUDE_DIRS}
		)
target_link_libraries(tinymsdf PUBLIC ${FREETYPE_LIBRARIES})

if(BUILD_TINYMSDF_EXAMPLE)
	add_executable(tinymsdf-example main.c)
	set_target_properties(tinymsdf-example PROPERTIES OUTPUT_NAME tinymsdf)
	target_include_directories(tinymsdf-example PUBLIC
			$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/example>
			)
	target_link_libraries(tinymsdf-example PUBLIC tinymsdf)
endif()
