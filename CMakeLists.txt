# The name of my project is "PeopleTrainer". CMakeLists files in this project can 
# refer to the root source directory of the project as ${PEOPLETRAINER_SOURCE_DIR} and 
# to the root binary directory of the project as ${PEOPLETRAINER_BINARY_DIR}. 
cmake_minimum_required (VERSION 2.8) 

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# The version number.
# must go before the project command
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
if(DEFINED CMAKE_BUILD_TYPE)
  set_property( CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES} )
endif()
set (Tutorial_VERSION_MAJOR 1)
set (Tutorial_VERSION_MINOR 0)
set(CMAKE_BUILD_TYPE Release)
if(NOT CMAKE_BUILD_TYPE) 
    set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

project ("People Trainer")

find_package(OpenCV REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(PeopleTrainer src/Main.cpp) 
target_link_libraries(PeopleTrainer ${OpenCV_LIBS})

install(TARGETS PeopleTrainer DESTINATION bin)
