#pragma once

#include <sstream>

#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/Object.h>
#include <aws/s3/model/GetObjectRequest.h>

#include "z5/s3/handle.hxx"


namespace z5 {
namespace s3 {

namespace attrs_detail {
    inline void readAttributes(const std::string & bucketName,
                               const std::string & objectName,
                               nlohmann::json & j) {
        Aws::SDKOptions options;
        Aws::InitAPI(options);

        Aws::S3::S3Client client;
        Aws::S3::Model::GetObjectRequest request;
        request.SetBucket(Aws::String(bucketName.c_str(), bucketName.size()));
        request.SetKey(Aws::String(objectName.c_str(), objectName.size()));

        auto outcome = client.GetObject(request);
        if(outcome.IsSuccess()) {
            auto & retrieved = outcome.GetResultWithOwnership().GetBody();
            std::stringstream stream;
            stream << retrieved.rdbuf();
            const std::string content = stream.str();
            j = nlohmann::json::parse(content);
        }

        Aws::ShutdownAPI(options);
    }

    inline void writeAttributes(const std::string & bucketName,
                                const std::string & objectName,
                                const nlohmann::json & j) {
        Aws::SDKOptions options;
        Aws::InitAPI(options);

        Aws::S3::S3Client client;
        Aws::S3::Model::PutObjectRequest request;
        request.SetBucket(Aws::String(bucketName.c_str(), bucketName.size()));
        request.SetKey(Aws::String(objectName.c_str(), objectName.size()));

        const std::string content = j.dump(4);
        auto inputData = Aws::MakeShared<std::stringstream>("PutObjectBody");
        *inputData << content;
        request.SetBody(inputData);

        client.PutObject(request);
        Aws::ShutdownAPI(options);
    }

    // Read V3 attributes from zarr.json["attributes"]
    inline void readV3Attributes(const std::string & bucketName,
                                 const std::string & zarrJsonKey,
                                 nlohmann::json & j) {
        nlohmann::json root;
        readAttributes(bucketName, zarrJsonKey, root);
        if(root.find("attributes") != root.end()) {
            j = root["attributes"];
        }
    }

    // Write V3 attributes into zarr.json["attributes"]
    inline void writeV3Attributes(const std::string & bucketName,
                                  const std::string & zarrJsonKey,
                                  const nlohmann::json & j) {
        // Read existing zarr.json
        nlohmann::json root;
        readAttributes(bucketName, zarrJsonKey, root);
        if(root.find("attributes") == root.end()) {
            root["attributes"] = nlohmann::json::object();
        }
        for(auto jIt = j.begin(); jIt != j.end(); ++jIt) {
            root["attributes"][jIt.key()] = jIt.value();
        }
        writeAttributes(bucketName, zarrJsonKey, root);
    }

    // Remove a V3 attribute from zarr.json["attributes"]
    inline void removeV3Attribute(const std::string & bucketName,
                                  const std::string & zarrJsonKey,
                                  const std::string & key) {
        nlohmann::json root;
        readAttributes(bucketName, zarrJsonKey, root);
        if(root.find("attributes") != root.end()) {
            root["attributes"].erase(key);
        }
        writeAttributes(bucketName, zarrJsonKey, root);
    }
}

    template<class GROUP>
    inline void readAttributes(const z5::handle::Group<GROUP> & group, nlohmann::json & j) {
        std::string objectName = group.nameInBucket();

        // Check v3 first
        if(group.isZarrV3()) {
            const std::string zarrJsonKey = objectName.empty() ? "zarr.json" : objectName + "/zarr.json";
            attrs_detail::readV3Attributes(group.bucketName(), zarrJsonKey, j);
            return;
        }

        if(group.isZarr()) {
            objectName += "/.zattrs";
        } else {
            objectName += "/attributes.json";
        }
        attrs_detail::readAttributes(group.bucketName(), objectName, j);
    }

    template<class GROUP>
    inline void writeAttributes(const z5::handle::Group<GROUP> & group, const nlohmann::json & j) {
        std::string objectName = group.nameInBucket();

        // v3: attributes embedded in zarr.json
        if(group.isZarrV3()) {
            const std::string zarrJsonKey = objectName.empty() ? "zarr.json" : objectName + "/zarr.json";
            attrs_detail::writeV3Attributes(group.bucketName(), zarrJsonKey, j);
            return;
        }

        // Read existing, merge, write back
        const std::string attrKey = group.isZarr() ?
            objectName + "/.zattrs" : objectName + "/attributes.json";
        nlohmann::json existing;
        attrs_detail::readAttributes(group.bucketName(), attrKey, existing);
        for(auto jIt = j.begin(); jIt != j.end(); ++jIt) {
            existing[jIt.key()] = jIt.value();
        }
        attrs_detail::writeAttributes(group.bucketName(), attrKey, existing);
    }

    template<class GROUP>
    inline void removeAttribute(const z5::handle::Group<GROUP> & group, const std::string & key) {
        std::string objectName = group.nameInBucket();

        if(group.isZarrV3()) {
            const std::string zarrJsonKey = objectName.empty() ? "zarr.json" : objectName + "/zarr.json";
            attrs_detail::removeV3Attribute(group.bucketName(), zarrJsonKey, key);
            return;
        }

        const std::string attrKey = group.isZarr() ?
            objectName + "/.zattrs" : objectName + "/attributes.json";
        nlohmann::json existing;
        attrs_detail::readAttributes(group.bucketName(), attrKey, existing);
        existing.erase(key);
        attrs_detail::writeAttributes(group.bucketName(), attrKey, existing);
    }


    template<class DATASET>
    inline void readAttributes(const z5::handle::Dataset<DATASET> & ds, nlohmann::json & j) {
        std::string objectName = ds.nameInBucket();

        if(ds.isZarrV3()) {
            const std::string zarrJsonKey = objectName + "/zarr.json";
            attrs_detail::readV3Attributes(ds.bucketName(), zarrJsonKey, j);
            return;
        }

        if(ds.isZarr()) {
            objectName += "/.zattrs";
        } else {
            objectName += "/attributes.json";
        }
        attrs_detail::readAttributes(ds.bucketName(), objectName, j);
    }

    template<class DATASET>
    inline void writeAttributes(const z5::handle::Dataset<DATASET> & ds, const nlohmann::json & j) {
        std::string objectName = ds.nameInBucket();

        if(ds.isZarrV3()) {
            const std::string zarrJsonKey = objectName + "/zarr.json";
            attrs_detail::writeV3Attributes(ds.bucketName(), zarrJsonKey, j);
            return;
        }

        const std::string attrKey = ds.isZarr() ?
            objectName + "/.zattrs" : objectName + "/attributes.json";
        nlohmann::json existing;
        attrs_detail::readAttributes(ds.bucketName(), attrKey, existing);
        for(auto jIt = j.begin(); jIt != j.end(); ++jIt) {
            existing[jIt.key()] = jIt.value();
        }
        attrs_detail::writeAttributes(ds.bucketName(), attrKey, existing);
    }

    template<class DATASET>
    inline void removeAttribute(const z5::handle::Dataset<DATASET> & ds, const std::string & key) {
        std::string objectName = ds.nameInBucket();

        if(ds.isZarrV3()) {
            const std::string zarrJsonKey = objectName + "/zarr.json";
            attrs_detail::removeV3Attribute(ds.bucketName(), zarrJsonKey, key);
            return;
        }

        const std::string attrKey = ds.isZarr() ?
            objectName + "/.zattrs" : objectName + "/attributes.json";
        nlohmann::json existing;
        attrs_detail::readAttributes(ds.bucketName(), attrKey, existing);
        existing.erase(key);
        attrs_detail::writeAttributes(ds.bucketName(), attrKey, existing);
    }


    template<class GROUP>
    inline bool isSubGroup(const z5::handle::Group<GROUP> & group, const std::string & key){
        // v3: check zarr.json for node_type
        if(group.isZarrV3()) {
            const std::string prefix = group.nameInBucket();
            const std::string zarrJsonKey = prefix.empty() ?
                key + "/zarr.json" : prefix + "/" + key + "/zarr.json";
            std::string content;
            if(group.readObjectImpl(zarrJsonKey, content)) {
                nlohmann::json j = nlohmann::json::parse(content);
                return j.value("node_type", "group") == "group";
            }
            return true;  // no zarr.json means it's a group
        }

        if(group.isZarr()) {
            return group.in(key + "/.zgroup");
        } else {
            nlohmann::json j;
            attrs_detail::readAttributes(group.bucketName(),
                                         group.nameInBucket() + "/attributes.json", j);
            return !z5::handle::hasAllN5DatasetAttributes(j);
        }
    }

}
}
