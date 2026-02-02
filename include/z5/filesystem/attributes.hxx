#pragma once

#include <fstream>
#include "z5/filesystem/handle.hxx"


namespace z5 {
namespace filesystem {


namespace attrs_detail {

    inline void readAttributes(const fs::path & path, nlohmann::json & j) {
        if(!fs::exists(path)) {
            return;
        }
        std::ifstream file(path);
        file >> j;
        file.close();
    }

    inline void writeAttributes(const fs::path & path, const nlohmann::json & j) {
        nlohmann::json jOut;
        // if we already have attributes, read them
        if(fs::exists(path)) {
            std::ifstream file(path);
            file >> jOut;
            file.close();
        }
        for(auto jIt = j.begin(); jIt != j.end(); ++jIt) {
            jOut[jIt.key()] = jIt.value();
        }
        std::ofstream file(path);
        file << jOut;
        file.close();
    }

    inline void removeAttribute(const fs::path & path, const std::string & key) {
        nlohmann::json jOut;
        // if we already have attributes, read them
        if(fs::exists(path)) {
            std::ifstream file(path);
            file >> jOut;
            file.close();
        }
        else {
            return;
        }
        jOut.erase(key);
        std::ofstream file(path);
        file << jOut;
        file.close();
    }

    // v3 helpers: attributes are embedded in zarr.json["attributes"]
    inline void readV3Attributes(const fs::path & zarrJsonPath, nlohmann::json & j) {
        if(!fs::exists(zarrJsonPath)) {
            return;
        }
        std::ifstream file(zarrJsonPath);
        nlohmann::json root;
        file >> root;
        file.close();
        if(root.find("attributes") != root.end()) {
            j = root["attributes"];
        }
    }

    inline void writeV3Attributes(const fs::path & zarrJsonPath, const nlohmann::json & j) {
        nlohmann::json root;
        if(fs::exists(zarrJsonPath)) {
            std::ifstream file(zarrJsonPath);
            file >> root;
            file.close();
        }
        if(root.find("attributes") == root.end()) {
            root["attributes"] = nlohmann::json::object();
        }
        for(auto jIt = j.begin(); jIt != j.end(); ++jIt) {
            root["attributes"][jIt.key()] = jIt.value();
        }
        std::ofstream file(zarrJsonPath);
        file << root;
        file.close();
    }

    inline void removeV3Attribute(const fs::path & zarrJsonPath, const std::string & key) {
        if(!fs::exists(zarrJsonPath)) {
            return;
        }
        nlohmann::json root;
        std::ifstream infile(zarrJsonPath);
        infile >> root;
        infile.close();
        if(root.find("attributes") != root.end()) {
            root["attributes"].erase(key);
        }
        std::ofstream outfile(zarrJsonPath);
        outfile << root;
        outfile.close();
    }
}

    template<class GROUP>
    inline void readAttributes(const z5::handle::Group<GROUP> & group, nlohmann::json & j
    ) {
        if(group.isZarrV3()) {
            const auto path = group.path() / "zarr.json";
            attrs_detail::readV3Attributes(path, j);
            return;
        }
        const auto path = group.path() / (group.isZarr() ? ".zattrs" : "attributes.json");
        attrs_detail::readAttributes(path, j);
    }

    template<class GROUP>
    inline void writeAttributes(const z5::handle::Group<GROUP> & group, const nlohmann::json & j) {
        if(group.isZarrV3()) {
            const auto path = group.path() / "zarr.json";
            attrs_detail::writeV3Attributes(path, j);
            return;
        }
        const auto path = group.path() / (group.isZarr() ? ".zattrs" : "attributes.json");
        attrs_detail::writeAttributes(path, j);
    }

    template<class GROUP>
    inline void removeAttribute(const z5::handle::Group<GROUP> & group, const std::string & key) {
        if(group.isZarrV3()) {
            const auto path = group.path() / "zarr.json";
            attrs_detail::removeV3Attribute(path, key);
            return;
        }
        const auto path = group.path() / (group.isZarr() ? ".zattrs" : "attributes.json");
        attrs_detail::removeAttribute(path, key);
    }


    template<class DATASET>
    inline void readAttributes(const z5::handle::Dataset<DATASET> & ds, nlohmann::json & j
    ) {
        if(ds.isZarrV3()) {
            const auto path = ds.path() / "zarr.json";
            attrs_detail::readV3Attributes(path, j);
            return;
        }
        const auto path = ds.path() / (ds.isZarr() ? ".zattrs" : "attributes.json");
        attrs_detail::readAttributes(path, j);
    }

    template<class DATASET>
    inline void writeAttributes(const z5::handle::Dataset<DATASET> & ds, const nlohmann::json & j) {
        if(ds.isZarrV3()) {
            const auto path = ds.path() / "zarr.json";
            attrs_detail::writeV3Attributes(path, j);
            return;
        }
        const auto path = ds.path() / (ds.isZarr() ? ".zattrs" : "attributes.json");
        attrs_detail::writeAttributes(path, j);
    }

    template<class DATASET>
    inline void removeAttribute(const z5::handle::Dataset<DATASET> & ds, const std::string & key) {
        if(ds.isZarrV3()) {
            const auto path = ds.path() / "zarr.json";
            attrs_detail::removeV3Attribute(path, key);
            return;
        }
        const auto path = ds.path() / (ds.isZarr() ? ".zattrs" : "attributes.json");
        attrs_detail::removeAttribute(path, key);
    }


    template<class GROUP>
    inline bool isSubGroup(const z5::handle::Group<GROUP> & group, const std::string & key){
        fs::path path = group.path() / key;
        if(!fs::exists(path)) {
            return false;
        }

        // v3: check zarr.json for node_type
        if(group.isZarrV3()) {
            fs::path zarrJson = path / "zarr.json";
            if(!fs::exists(zarrJson)) {
                return true;  // directory without zarr.json in v3 context is a group
            }
            nlohmann::json j;
            attrs_detail::readAttributes(zarrJson, j);
            return j.value("node_type", "group") == "group";
        }

        if(group.isZarr()) {
            path /= ".zgroup";
            return fs::exists(path);
        } else {
            path /= "attributes.json";
            if(!fs::exists(path)) {
                return true;
            }
            nlohmann::json j;
            attrs_detail::readAttributes(path, j);
            return !z5::handle::hasAllN5DatasetAttributes(j);
        }
    }

}
}
