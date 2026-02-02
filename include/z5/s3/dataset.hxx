#pragma once

#include <fstream>
#include <map>
#include <mutex>

#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/Object.h>
#include <aws/s3/model/GetObjectRequest.h>

#include "z5/dataset.hxx"
#include "z5/shard.hxx"
#include "z5/compression/crc32c_codec.hxx"
#include "z5/compression/transpose_codec.hxx"
#include "z5/s3/handle.hxx"


namespace z5 {
namespace s3 {

    template<typename T>
    class Dataset : public z5::Dataset, private z5::MixinTyped<T> {

    public:
        typedef T value_type;
        typedef types::ShapeType shape_type;
        typedef z5::MixinTyped<T> Mixin;

        // create a new array with metadata
        Dataset(const handle::Dataset & handle,
                const DatasetMetadata & metadata) : z5::Dataset(metadata),
                                                    Mixin(metadata),
                                                    handle_(handle){
        }

        //
        // Implement Dataset API
        //

        inline void writeChunk(const types::ShapeType & chunkIndices, const void * dataIn,
                               const bool isVarlen=false, const std::size_t varSize=0) const {
            // check if we are allowed to write
            if(!handle_.mode().canWrite()) {
                const std::string err = "Cannot write data in file mode " + handle_.mode().printMode();
                throw std::invalid_argument(err.c_str());
            }

            if(isSharded_) {
                writeChunkSharded(chunkIndices, dataIn, isVarlen, varSize);
                return;
            }

            // create chunk handle
            handle::Chunk chunk(handle_, chunkIndices, defaultChunkShape(), shape());

            // Prepare data — apply transpose and byte-swap
            const void * writeData = dataIn;
            std::vector<T> transposedBuf;
            if(hasTranspose_ && !transposeOrder_.empty() && !isVarlen) {
                transposedBuf.resize(chunkSize_);
                compression::TransposeCodec codec(transposeOrder_);
                codec.encode(static_cast<const T*>(dataIn), transposedBuf.data(), chunkShape_);
                writeData = transposedBuf.data();
            }

            std::vector<T> swapBuf;
            if(needsByteSwap_ && !isVarlen) {
                swapBuf.assign(static_cast<const T*>(writeData), static_cast<const T*>(writeData) + chunkSize_);
                byteSwapInPlace(swapBuf.data(), chunkSize_);
                writeData = swapBuf.data();
            }

            // Compress
            std::vector<char> buffer;
            util::data_to_buffer(chunk, writeData, buffer, Mixin::compressor_, Mixin::fillValue_, isVarlen, varSize);

            // Apply crc32c
            if(hasCrc32c_) {
                compression::crc32c_encode(buffer);
            }

            // Write to S3
            const std::string objectKey = handle_.nameInBucket() + "/" +
                (handle_.fileFormat() == types::zarr_v3
                    ? chunk.getChunkKeyV3("/")
                    : chunk.getChunkKey(handle_.isZarr(), handle_.zarrDelimiter()));
            handle_.writeBinaryObjectImpl(objectKey, buffer);
        }


        // read a chunk
        inline bool readChunk(const types::ShapeType & chunkIndices, void * dataOut) const {

            if(isSharded_) {
                return readChunkSharded(chunkIndices, dataOut);
            }

            // get the chunk handle
            handle::Chunk chunk(handle_, chunkIndices, defaultChunkShape(), shape());

            // throw runtime error if trying to read non-existing chunk
            if(!chunk.exists()) {
                throw std::runtime_error("Trying to read a chunk that does not exist");
            }

            // load the data from S3
            std::vector<char> buffer;
            read(chunk, buffer);

            // Apply crc32c decode
            if(hasCrc32c_) {
                compression::crc32c_decode(buffer);
            }

            // format the data
            if(hasTranspose_ && !transposeOrder_.empty()) {
                std::vector<T> tmpBuf(chunkSize_);
                const bool is_varlen = util::buffer_to_data<T>(chunk, buffer, tmpBuf.data(), Mixin::compressor_);
                if(needsByteSwap_) {
                    byteSwapInPlace(tmpBuf.data(), chunkSize_);
                }
                compression::TransposeCodec codec(transposeOrder_);
                codec.decode(tmpBuf.data(), static_cast<T*>(dataOut), chunkShape_);
                return is_varlen;
            }

            const bool is_varlen = util::buffer_to_data<T>(chunk, buffer, dataOut, Mixin::compressor_);
            if(needsByteSwap_ && !is_varlen) {
                byteSwapInPlace(dataOut, chunkSize_);
            }
            return is_varlen;
        }


        inline void readRawChunk(const types::ShapeType & chunkIndices,
                                 std::vector<char> & buffer) const {
            if(isSharded_) {
                auto shardIdx = shard_utils::computeShardIndex(chunkIndices, chunksPerShard_);
                auto innerIdx = shard_utils::computeInnerIndex(chunkIndices, chunksPerShard_);
                std::size_t linearIdx = shard_utils::linearInnerIndex(innerIdx, chunksPerShard_);
                std::string shardKey = getShardObjectKey(shardIdx);
                readChunkFromShardOnS3(shardKey, linearIdx, buffer);
                return;
            }
            handle::Chunk chunk(handle_, chunkIndices, defaultChunkShape(), shape());
            read(chunk, buffer);
        }


        inline void checkRequestType(const std::type_info & type) const {
            if(type != typeid(T)) {
                std::cout << "Mytype: " << typeid(T).name() << " your type: " << type.name() << std::endl;
                throw std::runtime_error("Request has wrong type");
            }
        }


        inline bool chunkExists(const types::ShapeType & chunkId) const {
            if(isSharded_) {
                auto shardIdx = shard_utils::computeShardIndex(chunkId, chunksPerShard_);
                std::string shardKey = getShardObjectKey(shardIdx);
                return handle_.inImpl(shardKey.substr(handle_.nameInBucket().size() + 1));
            }
            handle::Chunk chunk(handle_, chunkId, defaultChunkShape(), shape());
            return chunk.exists();
        }

        inline std::size_t getChunkSize(const types::ShapeType & chunkId) const {
            if(isSharded_) return chunkSize_;
            handle::Chunk chunk(handle_, chunkId, defaultChunkShape(), shape());
            return chunk.size();
        }

        inline void getChunkShape(const types::ShapeType & chunkId,
                                  types::ShapeType & chunkShape,
                                  const bool fromHeader=false) const {
            if(isSharded_) { chunkShape = chunkShape_; return; }
            handle::Chunk chunk(handle_, chunkId, defaultChunkShape(), shape());
            const auto & cshape = chunk.shape();
            chunkShape.resize(cshape.size());
            std::copy(cshape.begin(), cshape.end(), chunkShape.begin());
        }

        inline std::size_t getChunkShape(const types::ShapeType & chunkId,
                                         const unsigned dim,
                                         const bool fromHeader=false) const {
            if(isSharded_) return chunkShape_[dim];
            handle::Chunk chunk(handle_, chunkId, defaultChunkShape(), shape());
            return chunk.shape()[dim];
        }


        // compression options
        inline types::Compressor getCompressor() const {return Mixin::compressor_->type();}
        inline void getCompressor(std::string & compressor) const {
            auto compressorType = getCompressor();
            compressor = isZarr_ ? types::Compressors::compressorToZarr()[compressorType] : types::Compressors::compressorToN5()[compressorType];
        }

        inline void getCompressionOptions(types::CompressionOptions & opts) const {
            Mixin::compressor_->getOptions(opts);
        }

        inline void getFillValue(void * fillValue) const {
            *((T*) fillValue) = Mixin::fillValue_;
        }

        inline void decompress(const std::vector<char> & buffer,
                               void * dataOut,
                               const std::size_t data_size) const {
            util::decompress<T>(buffer, dataOut, data_size, Mixin::compressor_);
        }

        inline bool checkVarlenChunk(const types::ShapeType & chunkId, std::size_t & chunkSize) const {
            if(isSharded_) { chunkSize = chunkSize_; return false; }
            handle::Chunk chunk(handle_, chunkId, defaultChunkShape(), shape());
            if(isZarr_ || !chunk.exists()) {
                chunkSize = chunk.size();
                return false;
            }
            chunkSize = chunk.size();
            return false;
        }

        inline const FileMode & mode() const {
            return handle_.mode();
        }

        inline void removeChunk(const types::ShapeType & chunkId) const {
            if(isSharded_) {
                auto shardIdx = shard_utils::computeShardIndex(chunkId, chunksPerShard_);
                std::string shardKey = getShardObjectKey(shardIdx);
                handle_.deleteObjectImpl(shardKey);
                return;
            }
            handle::Chunk chunk(handle_, chunkId, defaultChunkShape(), shape());
            chunk.remove();
        }

        inline void remove() const {
            handle_.remove();
        }

        // dummy impls for path (S3 doesn't have filesystem paths)
        inline const fs::path & path() const {
            static const fs::path dummy;
            return dummy;
        }
        inline void chunkPath(const types::ShapeType & chunkId, fs::path & p) const {
            // Return the S3 object key as a path for debugging
            if(isSharded_) {
                auto shardIdx = shard_utils::computeShardIndex(chunkId, chunksPerShard_);
                p = getShardObjectKey(shardIdx);
            } else {
                handle::Chunk chunk(handle_, chunkId, defaultChunkShape(), shape());
                p = chunk.nameInBucket();
            }
        }

        // Flush buffered shards to S3
        inline void flushShards() const override {
            std::lock_guard<std::mutex> lock(shardMutex_);
            for(auto & [shardIdx, buffer] : shardBuffers_) {
                flushShardToS3(shardIdx, buffer);
            }
            shardBuffers_.clear();
            shardIndexCache_.clear();
        }

        // Flush a specific shard to S3
        inline void flushShard(const types::ShapeType & shardIndices) const {
            std::lock_guard<std::mutex> lock(shardMutex_);
            auto it = shardBuffers_.find(shardIndices);
            if(it != shardBuffers_.end()) {
                flushShardToS3(shardIndices, it->second);
                shardBuffers_.erase(it);
                shardIndexCache_.erase(shardIndices);
            }
        }

        // delete copy constructor and assignment operator
        Dataset(const Dataset & that) = delete;
        Dataset & operator=(const Dataset & that) = delete;

    private:

        inline void byteSwapInPlace(void * data, std::size_t numElements) const {
            if(sizeof(T) <= 1) return;
            T * arr = static_cast<T*>(data);
            util::reverseEndiannessInplace<T>(arr, arr + numElements);
        }

        inline std::string getShardObjectKey(const types::ShapeType & shardIdx) const {
            std::string sep = zarrDelimiter_.empty() ? "/" : zarrDelimiter_;
            std::string shardKeyStr = shard_utils::shardKey(shardIdx, sep);
            return handle_.nameInBucket() + "/" + shardKeyStr;
        }

        inline void read(const handle::Chunk & chunk, std::vector<char> & buffer) const {
            Aws::SDKOptions options;
            Aws::InitAPI(options);

            Aws::S3::S3Client client;
            Aws::S3::Model::GetObjectRequest request;

            const auto & bucketName = chunk.bucketName();
            const auto & objectName = chunk.nameInBucket();
            request.SetBucket(Aws::String(bucketName.c_str(), bucketName.size()));
            request.SetKey(Aws::String(objectName.c_str(), objectName.size()));

            auto outcome = client.GetObject(request);
            if(outcome.IsSuccess()) {
                auto & retrieved = outcome.GetResultWithOwnership().GetBody();
                std::stringstream stream;
                stream << retrieved.rdbuf();
                const std::string content = stream.str();
                buffer.assign(content.begin(), content.end());
            } else {
                Aws::ShutdownAPI(options);
                throw std::runtime_error("Could not read chunk from S3.");
            }

            Aws::ShutdownAPI(options);
        }

        // Read an entire shard file from S3 and extract a single chunk
        inline void readChunkFromShardOnS3(const std::string & shardObjectKey,
                                            std::size_t linearIdx,
                                            std::vector<char> & chunkData) const {
            // Read the entire shard binary
            std::vector<char> shardData;
            if(!handle_.readBinaryObjectImpl(shardObjectKey, shardData)) {
                throw std::runtime_error("Could not read shard from S3: " + shardObjectKey);
            }

            const std::size_t indexSize = numInnerChunks_ * 16;
            if(shardData.size() < indexSize) {
                throw std::runtime_error("Shard file too small to contain index");
            }

            // Read index
            const char * indexData;
            if(indexLocation_ == "start") {
                indexData = shardData.data();
            } else {
                indexData = shardData.data() + shardData.size() - indexSize;
            }

            ShardIndex index;
            index.read(indexData, numInnerChunks_);

            if(index.isEmpty(linearIdx)) {
                throw std::runtime_error("Trying to read empty chunk from shard");
            }

            uint64_t offset = index.chunkOffset(linearIdx);
            uint64_t nbytes = index.chunkSize(linearIdx);
            chunkData.assign(shardData.begin() + offset, shardData.begin() + offset + nbytes);
        }

        // Flush a shard buffer to S3 as a single binary object
        inline void flushShardToS3(const types::ShapeType & shardIdx, const ShardBuffer & buf) const {
            const std::size_t numChunks = numInnerChunks_;
            ShardIndex index(numChunks);

            // Build the data blob
            std::vector<char> dataBlob;
            uint64_t currentOffset = 0;
            if(indexLocation_ == "start") {
                currentOffset = index.byteSize();
            }

            for(std::size_t i = 0; i < numChunks; ++i) {
                if(buf.hasChunk(i)) {
                    const auto & chunkData = buf.getChunk(i);
                    index.setChunk(i, currentOffset, chunkData.size());
                    dataBlob.insert(dataBlob.end(), chunkData.begin(), chunkData.end());
                    currentOffset += chunkData.size();
                }
            }

            // Serialize index
            std::vector<char> indexBytes;
            index.write(indexBytes);

            // Assemble full shard
            std::vector<char> shardFile;
            if(indexLocation_ == "start") {
                shardFile.insert(shardFile.end(), indexBytes.begin(), indexBytes.end());
                shardFile.insert(shardFile.end(), dataBlob.begin(), dataBlob.end());
            } else {
                shardFile.insert(shardFile.end(), dataBlob.begin(), dataBlob.end());
                shardFile.insert(shardFile.end(), indexBytes.begin(), indexBytes.end());
            }

            // Write to S3
            const std::string objectKey = getShardObjectKey(shardIdx);
            handle_.writeBinaryObjectImpl(objectKey, shardFile);
        }

        inline void writeChunkSharded(const types::ShapeType & chunkIndices, const void * dataIn,
                                       const bool isVarlen, const std::size_t varSize) const {
            auto shardIdx = shard_utils::computeShardIndex(chunkIndices, chunksPerShard_);
            auto innerIdx = shard_utils::computeInnerIndex(chunkIndices, chunksPerShard_);
            std::size_t linearIdx = shard_utils::linearInnerIndex(innerIdx, chunksPerShard_);

            const std::size_t dataSize = chunkSize_;
            const T * srcData = static_cast<const T*>(dataIn);

            // Apply transpose if needed
            std::vector<T> transposedBuf;
            if(hasTranspose_ && !transposeOrder_.empty()) {
                transposedBuf.resize(dataSize);
                compression::TransposeCodec codec(transposeOrder_);
                codec.encode(srcData, transposedBuf.data(), chunkShape_);
                srcData = transposedBuf.data();
            }

            // Byte-swap if needed
            std::vector<T> swapBuf;
            if(needsByteSwap_) {
                swapBuf.assign(srcData, srcData + dataSize);
                byteSwapInPlace(swapBuf.data(), dataSize);
                srcData = swapBuf.data();
            }

            // Compress
            std::vector<char> buffer;
            util::compress(srcData, dataSize, buffer, Mixin::compressor_);

            // Apply crc32c if needed
            if(hasCrc32c_) {
                compression::crc32c_encode(buffer);
            }

            // Buffer the compressed chunk
            {
                std::lock_guard<std::mutex> lock(shardMutex_);
                auto it = shardBuffers_.find(shardIdx);
                if(it == shardBuffers_.end()) {
                    auto emplaced = shardBuffers_.emplace(shardIdx, ShardBuffer(numInnerChunks_));
                    it = emplaced.first;
                    // Load existing shard from S3 if it exists
                    loadExistingShardFromS3(shardIdx, it->second);
                }
                it->second.writeChunk(linearIdx, std::move(buffer));
            }
        }

        inline bool readChunkSharded(const types::ShapeType & chunkIndices, void * dataOut) const {
            auto shardIdx = shard_utils::computeShardIndex(chunkIndices, chunksPerShard_);
            auto innerIdx = shard_utils::computeInnerIndex(chunkIndices, chunksPerShard_);
            std::size_t linearIdx = shard_utils::linearInnerIndex(innerIdx, chunksPerShard_);

            std::string shardKey = getShardObjectKey(shardIdx);

            std::vector<char> buffer;
            readChunkFromShardOnS3(shardKey, linearIdx, buffer);

            // Apply crc32c decode if needed
            if(hasCrc32c_) {
                compression::crc32c_decode(buffer);
            }

            // Decompress
            if(hasTranspose_ && !transposeOrder_.empty()) {
                std::vector<T> tmpBuf(chunkSize_);
                util::decompress<T>(buffer, tmpBuf.data(), chunkSize_, Mixin::compressor_);
                if(needsByteSwap_) {
                    byteSwapInPlace(tmpBuf.data(), chunkSize_);
                }
                compression::TransposeCodec codec(transposeOrder_);
                codec.decode(tmpBuf.data(), static_cast<T*>(dataOut), chunkShape_);
            } else {
                util::decompress<T>(buffer, dataOut, chunkSize_, Mixin::compressor_);
                if(needsByteSwap_) {
                    byteSwapInPlace(dataOut, chunkSize_);
                }
            }

            return false;
        }

        // Load existing shard chunks from S3 into buffer for partial update
        inline void loadExistingShardFromS3(const types::ShapeType & shardIdx, ShardBuffer & buf) const {
            std::string shardKey = getShardObjectKey(shardIdx);
            std::vector<char> shardData;
            if(!handle_.readBinaryObjectImpl(shardKey, shardData)) {
                return;  // Shard doesn't exist yet
            }

            const std::size_t indexSize = numInnerChunks_ * 16;
            if(shardData.size() < indexSize) {
                return;
            }

            const char * indexData;
            if(indexLocation_ == "start") {
                indexData = shardData.data();
            } else {
                indexData = shardData.data() + shardData.size() - indexSize;
            }

            ShardIndex index;
            index.read(indexData, numInnerChunks_);

            for(std::size_t i = 0; i < numInnerChunks_; ++i) {
                if(!index.isEmpty(i)) {
                    uint64_t offset = index.chunkOffset(i);
                    uint64_t nbytes = index.chunkSize(i);
                    std::vector<char> chunkData(shardData.begin() + offset, shardData.begin() + offset + nbytes);
                    buf.writeChunk(i, std::move(chunkData));
                }
            }
        }

        handle::Dataset handle_;
        mutable std::mutex shardMutex_;
        mutable std::map<types::ShapeType, ShardBuffer> shardBuffers_;
        mutable std::map<types::ShapeType, ShardIndex> shardIndexCache_;
    };


}
}
