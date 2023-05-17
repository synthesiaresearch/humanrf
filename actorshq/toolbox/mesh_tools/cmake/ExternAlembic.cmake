include(ExternalProject)

ExternalProject_Add(ExternAlembic
  PREFIX alembic
  SOURCE_DIR "${CMAKE_SOURCE_DIR}/third_party/alembic"
  CMAKE_ARGS -DALEMBIC_SHARED_LIBS=OFF -DUSE_BINARIES=OFF -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -DUSE_TESTS=OFF
  BUILD_BYPRODUCTS alembic/lib/libAlembic.a
)

ExternalProject_Get_Property(ExternAlembic install_dir)
add_library(Abc::Alembic STATIC IMPORTED)
file(MAKE_DIRECTORY "${install_dir}/include")
set_target_properties(Abc::Alembic PROPERTIES
  IMPORTED_LOCATION "${install_dir}/lib/libAlembic.a"
  INTERFACE_INCLUDE_DIRECTORIES "${install_dir}/include"
)
add_dependencies(Abc::Alembic ExternAlembic)
