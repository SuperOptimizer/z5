#pragma once
// aws includes
#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/Object.h>
#include <aws/s3/model/ListObjectsV2Request.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/DeleteObjectRequest.h>
#include <aws/core/utils/stream/PreallocatedStreamBuf.h>

#include <sstream>

#include "z5/handle.hxx"


namespace z5 {
namespace s3 {
namespace handle {

    // TODO need to support more options
    // - different regions than us-east-1 (the default)
    // common functionality for S3 File and Group handles
    class S3HandleImpl {
    public:
        S3HandleImpl(const std::string & bucketName, const std::string & nameInBucket)
            : bucketName_(bucketName.c_str(), bucketName.size()),
              nameInBucket_(nameInBucket),
              options_(){}
        ~S3HandleImpl() {}

        // check if this handle exists
        inline bool existsImpl() const {
            Aws::InitAPI(options_);
            Aws::S3::S3Client client;
            Aws::S3::Model::ListObjectsV2Request request;
            request.WithBucket(Aws::String(bucketName_.c_str(), bucketName_.size()));
            request.WithPrefix(Aws::String(nameInBucket_.c_str(), nameInBucket_.size()));
            request.WithMaxKeys(1);
            const auto object_list = client.ListObjectsV2(request);
            const bool res = object_list.IsSuccess() && object_list.GetResult().GetKeyCount() > 0;
            Aws::ShutdownAPI(options_);
            return res;
        }

        inline void keysImpl(std::vector<std::string> & out) const {
            Aws::InitAPI(options_);
            Aws::S3::S3Client client;
            Aws::S3::Model::ListObjectsV2Request request;
            request.WithBucket(Aws::String(bucketName_.c_str(), bucketName_.size()));
            // add delimiter to the prefix
            const std::string bucketPrefix = nameInBucket_ == "" ? "" : nameInBucket_ + "/";
            request.WithPrefix(Aws::String(bucketPrefix.c_str(), bucketPrefix.size()));
            request.WithDelimiter("/");

            Aws::S3::Model::ListObjectsV2Result object_list;

            do {
                object_list = client.ListObjectsV2(request).GetResult();
                for(const auto & common_prefix : object_list.GetCommonPrefixes()) {
                    const std::string prefix(common_prefix.GetPrefix().c_str(),
                                             common_prefix.GetPrefix().size());
                    if(!prefix.empty() && prefix.back() == '/') {
                        std::vector<std::string> prefixSplit;
                        util::split(prefix, prefixSplit, "/");
                        out.emplace_back(prefixSplit.rbegin()[1]);
                    }
                }
            } while(object_list.GetIsTruncated());
            Aws::ShutdownAPI(options_);
        }

        inline bool inImpl(const std::string & name) const {
            Aws::InitAPI(options_);
            Aws::S3::S3Client client;
            Aws::S3::Model::ListObjectsV2Request request;
            request.WithBucket(Aws::String(bucketName_.c_str(), bucketName_.size()));
            const std::string prefix = nameInBucket_ == "" ? name : (nameInBucket_ + "/" + name);
            request.WithPrefix(Aws::String(prefix.c_str(), prefix.size()));
            request.WithMaxKeys(1);
            const auto object_list = client.ListObjectsV2(request);
            const bool res = object_list.IsSuccess() && object_list.GetResult().GetKeyCount() > 0;
            Aws::ShutdownAPI(options_);
            return res;
        }

        inline bool isZarrGroup() const {
            return inImpl(".zgroup");
        }
        inline bool isZarrDataset() const {
            return inImpl(".zarray");
        }
        inline bool isZarrV3Group() const {
            return inImpl("zarr.json");
        }
        inline bool isZarrV3Dataset() const {
            return inImpl("zarr.json");
        }

        // Read an S3 object into a string
        inline bool readObjectImpl(const std::string & objectKey, std::string & content) const {
            Aws::SDKOptions options;
            Aws::InitAPI(options);
            Aws::S3::S3Client client;
            Aws::S3::Model::GetObjectRequest request;
            request.SetBucket(Aws::String(bucketName_.c_str(), bucketName_.size()));
            request.SetKey(Aws::String(objectKey.c_str(), objectKey.size()));

            auto outcome = client.GetObject(request);
            bool success = false;
            if(outcome.IsSuccess()) {
                std::stringstream stream;
                stream << outcome.GetResultWithOwnership().GetBody().rdbuf();
                content = stream.str();
                success = true;
            }
            Aws::ShutdownAPI(options);
            return success;
        }

        // Read an S3 object into a binary buffer
        inline bool readBinaryObjectImpl(const std::string & objectKey, std::vector<char> & buffer) const {
            Aws::SDKOptions options;
            Aws::InitAPI(options);
            Aws::S3::S3Client client;
            Aws::S3::Model::GetObjectRequest request;
            request.SetBucket(Aws::String(bucketName_.c_str(), bucketName_.size()));
            request.SetKey(Aws::String(objectKey.c_str(), objectKey.size()));

            auto outcome = client.GetObject(request);
            bool success = false;
            if(outcome.IsSuccess()) {
                auto & body = outcome.GetResultWithOwnership().GetBody();
                std::stringstream stream;
                stream << body.rdbuf();
                const std::string content = stream.str();
                buffer.assign(content.begin(), content.end());
                success = true;
            }
            Aws::ShutdownAPI(options);
            return success;
        }

        // Write a string to an S3 object
        inline bool writeObjectImpl(const std::string & objectKey, const std::string & content) const {
            Aws::SDKOptions options;
            Aws::InitAPI(options);
            Aws::S3::S3Client client;
            Aws::S3::Model::PutObjectRequest request;
            request.SetBucket(Aws::String(bucketName_.c_str(), bucketName_.size()));
            request.SetKey(Aws::String(objectKey.c_str(), objectKey.size()));

            auto inputData = Aws::MakeShared<std::stringstream>("PutObjectBody");
            *inputData << content;
            request.SetBody(inputData);

            auto outcome = client.PutObject(request);
            bool success = outcome.IsSuccess();
            Aws::ShutdownAPI(options);
            return success;
        }

        // Write binary data to an S3 object
        inline bool writeBinaryObjectImpl(const std::string & objectKey, const std::vector<char> & data) const {
            Aws::SDKOptions options;
            Aws::InitAPI(options);
            Aws::S3::S3Client client;
            Aws::S3::Model::PutObjectRequest request;
            request.SetBucket(Aws::String(bucketName_.c_str(), bucketName_.size()));
            request.SetKey(Aws::String(objectKey.c_str(), objectKey.size()));

            auto inputData = Aws::MakeShared<std::stringstream>("PutObjectBody");
            inputData->write(data.data(), data.size());
            request.SetBody(inputData);
            request.SetContentLength(data.size());

            auto outcome = client.PutObject(request);
            bool success = outcome.IsSuccess();
            Aws::ShutdownAPI(options);
            return success;
        }

        // Delete an S3 object
        inline bool deleteObjectImpl(const std::string & objectKey) const {
            Aws::SDKOptions options;
            Aws::InitAPI(options);
            Aws::S3::S3Client client;
            Aws::S3::Model::DeleteObjectRequest request;
            request.SetBucket(Aws::String(bucketName_.c_str(), bucketName_.size()));
            request.SetKey(Aws::String(objectKey.c_str(), objectKey.size()));

            auto outcome = client.DeleteObject(request);
            bool success = outcome.IsSuccess();
            Aws::ShutdownAPI(options);
            return success;
        }

        inline const std::string & bucketNameImpl() const {
            return bucketName_;
        }

        inline const std::string & nameInBucketImpl() const {
            return nameInBucket_;
        }

    private:
        std::string bucketName_;
        std::string nameInBucket_;
        Aws::SDKOptions options_;
    };


    class File : public z5::handle::File<File>, public S3HandleImpl {
    public:
        typedef z5::handle::File<File> BaseType;

        // for now we only support vanilla SDKOptions
        File(const std::string & bucketName,
             const std::string & nameInBucket="",
             const FileMode mode=FileMode())
            : BaseType(mode),
              S3HandleImpl(bucketName, nameInBucket){}

        // Implement the handle API
        inline bool isS3() const {return true;}
        inline bool isGcs() const {return false;}
        // dummy impl
        const fs::path & path() const {}

        inline types::FileFormat fileFormat() const {
            if(isZarrV3Group()) return types::zarr_v3;
            return isZarrGroup() ? types::zarr_v2 : types::n5;
        }
        inline bool isZarr() const {
            return fileFormat() != types::n5;
        }

        inline bool exists() const {
            return existsImpl();
        }

        inline void create() const {
            if(!mode().canCreate()) {
                const std::string err = "Cannot create new file in file mode " + mode().printMode();
                throw std::invalid_argument(err.c_str());
            }
            // make sure that the file does not exist already
            if(exists()) {
                throw std::invalid_argument("Creating new file failed because it already exists.");
            }
        }

        inline void remove() const {
            if(!mode().canWrite()) {
                const std::string err = "Cannot remove file in file mode " + mode().printMode();
                throw std::invalid_argument(err.c_str());
            }
            if(!exists()) {
                throw std::invalid_argument("Cannot remove non-existing file.");
            }
        }

        // Implement the group handle API
        inline void keys(std::vector<std::string> & out) const {
            keysImpl(out);
        }
        inline bool in(const std::string & key) const {
            return inImpl(key);
        }

        inline const std::string & bucketName() const {
            return bucketNameImpl();
        }
        inline const std::string & nameInBucket() const {
            return nameInBucketImpl();
        }
    };


    class Group : public z5::handle::Group<Group>, public S3HandleImpl {
    public:
        typedef z5::handle::Group<Group> BaseType;

        template<class GROUP>
        Group(const z5::handle::Group<GROUP> & group, const std::string & key)
            : BaseType(group.mode()),
              S3HandleImpl(group.bucketName(),
                           (group.nameInBucket() == "") ? key : group.nameInBucket() + "/" + key){}

        // Implement th handle API
        inline bool isS3() const {return true;}
        inline bool isGcs() const {return false;}
        inline bool exists() const {return existsImpl();}
        inline types::FileFormat fileFormat() const {
            if(isZarrV3Group()) return types::zarr_v3;
            return isZarrGroup() ? types::zarr_v2 : types::n5;
        }
        inline bool isZarr() const {return fileFormat() != types::n5;}
        const fs::path & path() const {}

        inline void create() const {
            if(mode().mode() == FileMode::modes::r) {
                const std::string err = "Cannot create new group in file mode " + mode().printMode();
                throw std::invalid_argument(err.c_str());
            }
            // make sure that the file does not exist already
            if(exists()) {
                throw std::invalid_argument("Creating new group failed because it already exists.");
            }
        }

        inline void remove() const {
            if(!mode().canWrite()) {
                const std::string err = "Cannot remove group in group mode " + mode().printMode();
                throw std::invalid_argument(err.c_str());
            }
            if(!exists()) {
                throw std::invalid_argument("Cannot remove non-existing group.");
            }
        }

        // Implement the group handle API
        inline void keys(std::vector<std::string> & out) const {keysImpl(out);}
        inline bool in(const std::string & key) const {return inImpl(key);}
        inline const std::string & bucketName() const {
            return bucketNameImpl();
        }
        inline const std::string & nameInBucket() const {
            return nameInBucketImpl();
        }
    };


    class Dataset : public z5::handle::Dataset<Dataset>, public S3HandleImpl {
    public:
        typedef z5::handle::Dataset<Dataset> BaseType;

        template<class GROUP>
        Dataset(const z5::handle::Group<GROUP> & group, const std::string & key)
            : BaseType(group.mode()),
              S3HandleImpl(group.bucketName(),
                           (group.nameInBucket() == "") ? key : group.nameInBucket() + "/" + key){}

        // Implement th handle API
        inline bool isS3() const {return true;}
        inline bool isGcs() const {return false;}
        inline bool exists() const {return existsImpl();}
        inline types::FileFormat fileFormat() const {
            if(isZarrV3Dataset()) return types::zarr_v3;
            return isZarrDataset() ? types::zarr_v2 : types::n5;
        }
        inline bool isZarr() const {return fileFormat() != types::n5;}
        // dummy implementation
        const fs::path & path() const {}

        inline void create() const {
            // check if we have permissions to create a new dataset
            if(mode().mode() == FileMode::modes::r) {
                const std::string err = "Cannot create new dataset in mode " + mode().printMode();
                throw std::invalid_argument(err.c_str());
            }
            // make sure that the file does not exist already
            if(exists()) {
                throw std::invalid_argument("Creating new dataset failed because it already exists.");
            }
        }

        inline void remove() const {
            if(!mode().canWrite()) {
                const std::string err = "Cannot remove dataset in dataset mode " + mode().printMode();
                throw std::invalid_argument(err.c_str());
            }
            if(!exists()) {
                throw std::invalid_argument("Cannot remove non-existing dataset.");
            }
        }

        inline const std::string & bucketName() const {
            return bucketNameImpl();
        }
        inline const std::string & nameInBucket() const {
            return nameInBucketImpl();
        }
    };


    class Chunk : public z5::handle::Chunk<Chunk>, public S3HandleImpl {
    public:
        typedef z5::handle::Chunk<Chunk> BaseType;

        Chunk(const Dataset & ds,
              const types::ShapeType & chunkIndices,
              const types::ShapeType & chunkShape,
              const types::ShapeType & shape) : BaseType(chunkIndices, chunkShape, shape, ds.mode()),
                                                dsHandle_(ds),
                                                S3HandleImpl(ds.bucketName(),
                                                    ds.nameInBucket() + "/" +
                                                    (ds.fileFormat() == types::zarr_v3
                                                        ? getChunkKeyV3("/")
                                                        : getChunkKey(ds.isZarr()))){}

        inline void remove() const {
        }

        inline const Dataset & datasetHandle() const {
            return dsHandle_;
        }

        inline types::FileFormat fileFormat() const {
            return dsHandle_.fileFormat();
        }

        inline bool isZarr() const {
            return dsHandle_.isZarr();
        }

        inline bool exists() const {return existsImpl();}

        // dummy impl
        const fs::path & path() const {}

        inline bool isS3() const {return true;}
        inline bool isGcs() const {return false;}
        inline const std::string & bucketName() const {return bucketNameImpl();}
        inline const std::string & nameInBucket() const {return nameInBucketImpl();}

    private:
        const Dataset & dsHandle_;
    };


    //
    // additional handle factory functions for compatibility with C
    //

    // TODO
    // implement handle factories for File and Group
    /*
    // get z5::filesystem::handle::File from char pointer corresponding
    // to the file on filesystem
    inline File getFileHandle(const char *) {
        File ret();
        return ret;
    }

    // get z5::filesystem::handle::File from char pointer corresponding
    // to the file on filesystem and char pointer corresponding to key of the group
    inline Group getGroupHandle(const char *, const char *) {
        Group ret();
        return ret;
    }
    */

}
}
}
