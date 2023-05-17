# Pangolin (https://github.com/stevenlovegrove/Pangolin)
find_package(Pangolin 0.8 REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(mesh_renderer main.cpp)
target_link_libraries(mesh_renderer Abc::Alembic pango_display pango_plot pango_glgeometry pango_video ${OpenCV_LIBS} ${OpenEXR_LIBRARIES})
target_include_directories(mesh_renderer PRIVATE .)
target_include_directories(mesh_renderer SYSTEM PRIVATE
  ${OpenEXR_INCLUDE_DIRS}
)
install(TARGETS mesh_renderer)
