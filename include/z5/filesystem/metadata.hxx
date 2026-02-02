#pragma once

#include "z5/metadata.hxx"
#include "z5/filesystem/attributes.hxx"

namespace z5 {
namespace filesystem {

namespace metadata_detail {

    inline void writeMetadata(const fs::path & path, const nlohmann::json & j) {
        std::ofstream file(path);
        file << std::setw(4) << j << std::endl;
        file.close();
    }

    inline void readMetadata(const fs::path & path, nlohmann::json & j) {
        std::ifstream file(path);
        file >> j;
        file.close();
    }

    inline types::FileFormat getMetadataPath(const handle::Dataset & handle, fs::path & path) {
        fs::path zarrV3Path = handle.path() / "zarr.json";
        fs::path zarrPath = handle.path() / ".zarray";
        fs::path n5Path = handle.path() / "attributes.json";

        // check for v3 first
        if(fs::exists(zarrV3Path)) {
            nlohmann::json j;
            readMetadata(zarrV3Path, j);
            if(j.value("node_type", "") == "array") {
                path = zarrV3Path;
                return types::zarr_v3;
            }
        }

        if(fs::exists(zarrPath) && fs::exists(n5Path)) {
            throw std::runtime_error("Zarr and N5 specification are not both supported");
        }
        if(!fs::exists(zarrPath) && !fs::exists(n5Path)){
            throw std::runtime_error("Invalid path: no metadata existing");
        }
        const bool isZarr = fs::exists(zarrPath);
        path = isZarr ? zarrPath : n5Path;
        return isZarr ? types::zarr_v2 : types::n5;
    }
}

    template<class GROUP>
    inline void writeMetadata(const z5::handle::File<GROUP> & handleBase, const Metadata & metadata) {
        const auto & handle = handleBase;

        if(metadata.isZarrV3()) {
            const auto path = handle.path() / "zarr.json";
            nlohmann::json j;
            j["zarr_format"] = 3;
            j["node_type"] = "group";
            metadata_detail::writeMetadata(path, j);
            return;
        }

        const bool isZarr = metadata.isZarr();
        const auto path = handle.path() / (isZarr ? ".zgroup" : "attributes.json");
        nlohmann::json j;
        if(isZarr) {
            j["zarr_format"] = metadata.zarrFormat;
        } else {
            // n5 stores attributes and metadata in the same file,
            // so we need to make sure that we don't overwrite attributes
            try {
                readAttributes(handle, j);
            } catch(std::runtime_error) {}  // read attributes throws RE if there are no attributes, we can just ignore this
            j["n5"] = metadata.n5Format();
        }
        metadata_detail::writeMetadata(path, j);
    }


    template<class GROUP>
    inline void writeMetadata(const z5::handle::Group<GROUP> & handle, const Metadata & metadata) {

        if(metadata.isZarrV3()) {
            const auto path = handle.path() / "zarr.json";
            nlohmann::json j;
            j["zarr_format"] = 3;
            j["node_type"] = "group";
            metadata_detail::writeMetadata(path, j);
            return;
        }

        const bool isZarr = metadata.isZarr();
        const auto path = handle.path() / (isZarr ? ".zgroup" : "attributes.json");
        nlohmann::json j;
        if(isZarr) {
            j["zarr_format"] = metadata.zarrFormat;
        } else {
            // we don't need to write metadata for n5 groups
            return;
        }
        metadata_detail::writeMetadata(path, j);
    }


    inline void writeMetadata(const handle::Dataset & handle, const DatasetMetadata & metadata) {

        if(metadata.isZarrV3()) {
            const auto path = handle.path() / "zarr.json";
            nlohmann::json j;
            metadata.toJson(j);
            metadata_detail::writeMetadata(path, j);
            return;
        }

        const auto path = handle.path() / (metadata.isZarr() ? ".zarray" : "attributes.json");
        nlohmann::json j;
        metadata.toJson(j);
        metadata_detail::writeMetadata(path, j);
    }


    template<class GROUP>
    inline void readMetadata(const z5::handle::Group<GROUP> & handle, nlohmann::json & j) {

        // check for v3 first
        const auto v3Path = handle.path() / "zarr.json";
        if(fs::exists(v3Path)) {
            nlohmann::json jTmp;
            metadata_detail::readMetadata(v3Path, jTmp);
            j["zarr_format"] = jTmp.value("zarr_format", 3);
            if(jTmp.find("node_type") != jTmp.end()) {
                j["node_type"] = jTmp["node_type"];
            }
            return;
        }

        const bool isZarr = handle.isZarr();
        const auto path = handle.path() / (isZarr ? ".zgroup" : "attributes.json");
        nlohmann::json jTmp;
        metadata_detail::readMetadata(path, jTmp);
        if(isZarr) {
            j["zarr_format"] = jTmp["zarr_format"];
        } else {
            auto jIt = jTmp.find("n5");
            if(jIt != jTmp.end()) {
                j["n5"] = jIt.value();
            }
        }
    }


    inline void readMetadata(const handle::Dataset & handle, DatasetMetadata & metadata) {
        nlohmann::json j;
        fs::path path;
        auto fmt = metadata_detail::getMetadataPath(handle, path);
        metadata_detail::readMetadata(path, j);
        metadata.fromJson(j, fmt);
    }

}
}
