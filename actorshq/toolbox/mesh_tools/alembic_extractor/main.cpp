#include "CLI/App.hpp"
#include "CLI/Formatter.hpp"
#include "CLI/Config.hpp"

#include <Alembic/AbcGeom/All.h>
#include <Alembic/AbcCoreOgawa/All.h>

#include <cstdint>
#include <cinttypes>
#include <cstdio>
#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <optional>
#include <future>
#include <memory>
#include <filesystem>
#include <future>

namespace fs = std::filesystem;
namespace AA = Alembic::Abc;
namespace AAG = Alembic::AbcGeom;


void write_obj(const std::string& output_file, const AAG::IPolyMeshSchema::Sample& mesh_sample){
    std::ofstream output(output_file);
    // Write vertices
    for(size_t i=0; i < mesh_sample.getPositions()->size(); i++){
      auto& position = (*(mesh_sample.getPositions()))[i];
      output << "v " << position[0] << " " << position[1] << " " << position[2] << "\n";
    }

    // Validate that all faces use three vertices
    for(size_t i=0; i < mesh_sample.getFaceCounts()->size(); i++){
      auto& count = (*(mesh_sample.getFaceCounts()))[i];
      if (count != 3){
          std::cerr << "Encountered non-triangle face, this is not supported" << std::endl;
          exit(1);
      }
    }
    // Write faces
    for(size_t i=0; i < mesh_sample.getFaceIndices()->size(); i+=3){
        output << "f " << (*(mesh_sample.getFaceIndices()))[i]+1 << " " << (*(mesh_sample.getFaceIndices()))[i+1]+1 << " " << (*(mesh_sample.getFaceIndices()))[i+2]+1 << "\n";
    }
}


void convert_alembic_to_objs(
  const fs::path& path_alembic,
  const fs::path& path_objs
){
  // See: https://github.com/alembic/alembic/blob/master/lib/Alembic/AbcGeom/Tests/PolyMeshTest.cpp
  AA::IArchive archive( Alembic::AbcCoreOgawa::ReadArchive(), path_alembic );
  AAG::IPolyMesh poly_mesh(AA::IObject( archive, AA::kTop ), "object" );
  AAG::IPolyMeshSchema& mesh = poly_mesh.getSchema();
  AAG::IN3fGeomParam N = mesh.getNormalsParam();
  AAG::IV2fGeomParam uv = mesh.getUVsParam();
  std::cout << "Extracting " << mesh.getNumSamples() << " .obj files ..." << std::endl;

  AAG::IPolyMeshSchema::Sample mesh_sample;
  std::vector<std::future<void>> futures;
  for(size_t i=0; i<mesh.getNumSamples(); i++){
    mesh.get( mesh_sample, i );
    if(mesh_sample.getPositions()->size()==0)
      continue;
    std::stringstream index_stream;
    index_stream << std::setfill('0') << std::setw(6) << i;
    futures.push_back(std::async(std::launch::async, write_obj, path_objs / ("Frame" + index_stream.str() + ".obj"), mesh_sample));
  }
  for (auto& f : futures)
    f.wait();
}

int main(int argc, const char** argv) {
  std::string object_name = "object";
  fs::path alembic_file;
  fs::path output_directory;

  CLI::App app{"alembic_extractor"};
  app.add_option("--alembic", alembic_file, "Path to Alembic (.abc) animated mesh file.")->required();
  app.add_option("--output", output_directory, "Output directory where files of the pattern Frame\%06d.obj are created.")->required();
  CLI11_PARSE(app, argc, argv);
  fs::create_directories(output_directory);
  convert_alembic_to_objs(alembic_file, output_directory);

  return 0;
}
