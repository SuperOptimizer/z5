#pragma once

#include "z5/s3/handle.hxx"
#include "z5/s3/attributes.hxx"


namespace z5 {
namespace s3 {

    template<class GROUP>
    inline void writeMetadata(const z5::handle::File<GROUP> & handle, const Metadata & metadata) {
        // Get S3 handle implementation
        const auto & s3Handle = static_cast<const typename std::remove_reference<decltype(handle)>::type &>(handle);
        const std::string prefix = s3Handle.nameInBucket();

        if(metadata.isZarrV3()) {
            nlohmann::json j;
            j["zarr_format"] = 3;
            j["node_type"] = "group";
            const std::string objectKey = prefix.empty() ? "zarr.json" : prefix + "/zarr.json";
            s3Handle.writeObjectImpl(objectKey, j.dump(4));
        } else if(metadata.isZarr()) {
            nlohmann::json j;
            j["zarr_format"] = metadata.zarrFormat;
            const std::string objectKey = prefix.empty() ? ".zgroup" : prefix + "/.zgroup";
            s3Handle.writeObjectImpl(objectKey, j.dump(4));
        } else {
            nlohmann::json j;
            j["n5"] = metadata.n5Format();
            const std::string objectKey = prefix.empty() ? "attributes.json" : prefix + "/attributes.json";
            s3Handle.writeObjectImpl(objectKey, j.dump(4));
        }
    }


    template<class GROUP>
    inline void writeMetadata(const z5::handle::Group<GROUP> & handle, const Metadata & metadata) {
        const auto & s3Handle = static_cast<const typename std::remove_reference<decltype(handle)>::type &>(handle);
        const std::string prefix = s3Handle.nameInBucket();

        if(metadata.isZarrV3()) {
            nlohmann::json j;
            j["zarr_format"] = 3;
            j["node_type"] = "group";
            const std::string objectKey = prefix + "/zarr.json";
            s3Handle.writeObjectImpl(objectKey, j.dump(4));
        } else if(metadata.isZarr()) {
            nlohmann::json j;
            j["zarr_format"] = metadata.zarrFormat;
            const std::string objectKey = prefix + "/.zgroup";
            s3Handle.writeObjectImpl(objectKey, j.dump(4));
        }
        // n5 groups don't need metadata written
    }


    inline void writeMetadata(const handle::Dataset & handle, const DatasetMetadata & metadata) {
        const std::string prefix = handle.nameInBucket();

        nlohmann::json j;
        metadata.toJson(j);

        if(metadata.isZarrV3()) {
            const std::string objectKey = prefix + "/zarr.json";
            handle.writeObjectImpl(objectKey, j.dump(4));
        } else if(metadata.isZarr()) {
            const std::string objectKey = prefix + "/.zarray";
            handle.writeObjectImpl(objectKey, j.dump(4));
        } else {
            const std::string objectKey = prefix + "/attributes.json";
            handle.writeObjectImpl(objectKey, j.dump(4));
        }
    }


    inline void readMetadata(const handle::Dataset & handle, DatasetMetadata & metadata) {
        const std::string prefix = handle.nameInBucket();

        // Try v3 first
        {
            std::string content;
            const std::string v3Key = prefix + "/zarr.json";
            if(handle.readObjectImpl(v3Key, content)) {
                nlohmann::json j = nlohmann::json::parse(content);
                if(j.value("node_type", "") == "array") {
                    metadata.fromJson(j, types::zarr_v3);
                    return;
                }
            }
        }

        // Try v2 zarr
        {
            std::string content;
            const std::string zarrKey = prefix + "/.zarray";
            if(handle.readObjectImpl(zarrKey, content)) {
                nlohmann::json j = nlohmann::json::parse(content);
                metadata.fromJson(j, types::zarr_v2);
                return;
            }
        }

        // Try n5
        {
            std::string content;
            const std::string n5Key = prefix + "/attributes.json";
            if(handle.readObjectImpl(n5Key, content)) {
                nlohmann::json j = nlohmann::json::parse(content);
                metadata.fromJson(j, types::n5);
                return;
            }
        }

        throw std::runtime_error("Could not read dataset metadata from S3");
    }


}
}
