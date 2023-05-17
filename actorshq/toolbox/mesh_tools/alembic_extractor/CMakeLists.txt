find_package(Threads REQUIRED)
add_executable(alembic_extractor main.cpp)
target_link_libraries(alembic_extractor
  Abc::Alembic
  ${OpenEXR_LIBRARIES}
  Threads::Threads
)
target_include_directories(alembic_extractor PRIVATE .)
target_include_directories(alembic_extractor SYSTEM PRIVATE
  ${OpenEXR_INCLUDE_DIRS}
)
install(TARGETS alembic_extractor)
