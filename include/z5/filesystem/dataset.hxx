#pragma once

#include <ios>
#include <map>
#include <mutex>

#include "z5/dataset.hxx"
#include "z5/shard.hxx"
#include "z5/compression/crc32c_codec.hxx"
#include "z5/compression/transpose_codec.hxx"
#include "z5/filesystem/handle.hxx"


namespace z5 {
namespace filesystem {


    template<typename T>
    class Dataset : public z5::Dataset, private z5::MixinTyped<T> {

    public:

        typedef T value_type;
        typedef types::ShapeType shape_type;
        typedef z5::MixinTyped<T> Mixin;
        typedef z5::Dataset BaseType;

        // create a new array with metadata
        Dataset(const handle::Dataset & handle,
                const DatasetMetadata & metadata) : BaseType(metadata),
                                                    Mixin(metadata),
                                                    handle_(handle){
            // disable sync of c++ and c streams for potentially faster I/O
            std::ios_base::sync_with_stdio(false);
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

            // create chunk handle and check if this chunk is valid
            handle::Chunk chunk(handle_, chunkIndices, defaultChunkShape(), shape());
            checkChunk(chunk, isVarlen);
            const auto & path = chunk.path();

            // Apply transpose and byte-swap if needed (non-sharded path)
            const void * writeData = dataIn;
            std::vector<T> transposedBuf;
            if(hasTranspose_ && !transposeOrder_.empty() && !isVarlen) {
                const std::size_t chunkSz = isZarr_ ? chunk.defaultSize() : chunk.size();
                transposedBuf.resize(chunkSz);
                compression::TransposeCodec codec(transposeOrder_);
                codec.encode(static_cast<const T*>(dataIn), transposedBuf.data(), chunk.defaultShape());
                writeData = transposedBuf.data();
            }

            // Byte-swap if codec endianness != host endianness
            std::vector<T> swapBuf;
            if(needsByteSwap_ && !isVarlen) {
                const std::size_t chunkSz = isZarr_ ? chunk.defaultSize() : chunk.size();
                swapBuf.assign(static_cast<const T*>(writeData), static_cast<const T*>(writeData) + chunkSz);
                byteSwapInPlace(swapBuf.data(), chunkSz);
                writeData = swapBuf.data();
            }

            // create the output buffer and format the data
            std::vector<char> buffer;
            // data_to_buffer will return false if there's nothing to write
            if(!util::data_to_buffer(chunk, writeData, buffer, Mixin::compressor_, Mixin::fillValue_, isVarlen, varSize)) {
                // if we have data on disc for the chunk, delete it
                if(fs::exists(path)) {
                    fs::remove(path);
                }
                return;
            }

            // apply codec pipeline: crc32c
            if(hasCrc32c_) {
                compression::crc32c_encode(buffer);
            }

            // write the chunk to disc
            if(!isZarr_ || zarrDelimiter_ == "/") {
                // need to make sure we have the root directory if this is an nested chunk
                chunk.create();
            }
            write(path, buffer);
        }


        // read a chunk
        // IMPORTANT we assume that the data pointer is already initialized up to chunkSize_
        inline bool readChunk(const types::ShapeType & chunkIndices, void * dataOut) const {

            if(isSharded_) {
                return readChunkSharded(chunkIndices, dataOut);
            }

            // get the chunk handle
            handle::Chunk chunk(handle_, chunkIndices, defaultChunkShape(), shape());

            // make sure that we have a valid chunk
            checkChunk(chunk);

            // throw runtime error if trying to read non-existing chunk
            if(!chunk.exists()) {
                throw std::runtime_error("Trying to read a chunk that does not exist");
            }

            // load the data from disc
            std::vector<char> buffer;
            read(chunk.path(), buffer);

            // apply codec pipeline: crc32c decode
            if(hasCrc32c_) {
                compression::crc32c_decode(buffer);
            }

            // format the data
            if(hasTranspose_ && !transposeOrder_.empty()) {
                // Decompress to temp, then byte-swap, then inverse transpose
                std::vector<T> tmpBuf(chunk.defaultSize());
                const bool is_varlen = util::buffer_to_data<T>(chunk, buffer, tmpBuf.data(), Mixin::compressor_);
                if(needsByteSwap_) {
                    byteSwapInPlace(tmpBuf.data(), chunk.defaultSize());
                }
                compression::TransposeCodec codec(transposeOrder_);
                codec.decode(tmpBuf.data(), static_cast<T*>(dataOut), chunk.defaultShape());
                return is_varlen;
            }

            const bool is_varlen = util::buffer_to_data<T>(chunk, buffer, dataOut, Mixin::compressor_);

            // Byte-swap after decompression
            if(needsByteSwap_ && !is_varlen) {
                byteSwapInPlace(dataOut, chunk.size());
            }

            return is_varlen;
        }

        inline void readRawChunk(const types::ShapeType & chunkIndices,
                                 std::vector<char> & buffer) const {
            if(isSharded_) {
                auto shardIdx = shard_utils::computeShardIndex(chunkIndices, chunksPerShard_);
                auto innerIdx = shard_utils::computeInnerIndex(chunkIndices, chunksPerShard_);
                std::size_t linearIdx = shard_utils::linearInnerIndex(innerIdx, chunksPerShard_);
                std::string shardKeyStr = shard_utils::shardKey(shardIdx, zarrDelimiter_.empty() ? "/" : zarrDelimiter_);
                fs::path shardPath = handle_.path() / shardKeyStr;
                if(!ShardBuffer::readChunk(shardPath, linearIdx, numInnerChunks_, indexLocation_, buffer, indexHasCrc32c_)) {
                    throw std::runtime_error("Trying to read a chunk that does not exist in shard");
                }
                return;
            }
            handle::Chunk chunk(handle_, chunkIndices, defaultChunkShape(), shape());
            read(chunk.path(), buffer);
        }

        inline void checkRequestType(const std::type_info & type) const {
            if(type != typeid(T)) {
                // TODO all in error message
                std::cout << "Mytype: " << typeid(T).name() << " your type: " << type.name() << std::endl;
                throw std::runtime_error("Request has wrong type");
            }
        }

        inline bool chunkExists(const types::ShapeType & chunkId) const {
            if(isSharded_) {
                return chunkExistsSharded(chunkId);
            }
            handle::Chunk chunk(handle_, chunkId, defaultChunkShape(), shape());
            return chunk.exists();
        }


        inline std::size_t getChunkSize(const types::ShapeType & chunkId) const {
            if(isSharded_) {
                // For sharded datasets, chunk size is always the full chunk size
                return chunkSize_;
            }
            handle::Chunk chunk(handle_, chunkId, defaultChunkShape(), shape());
            return chunk.size();
        }


        inline void getChunkShape(const types::ShapeType & chunkId,
                                  types::ShapeType & chunkShape,
                                  const bool fromHeader=false) const {
            if(isSharded_) {
                chunkShape = chunkShape_;
                return;
            }
            handle::Chunk chunk(handle_, chunkId, defaultChunkShape(), shape());
            if(!isZarr_ && fromHeader) {
                read_shape_from_n5_header(chunk.path(), chunkShape);
            } else {
                const auto & cshape = chunk.shape();
                chunkShape.resize(cshape.size());
                std::copy(cshape.begin(), cshape.end(), chunkShape.begin());
            }
        }


        inline std::size_t getChunkShape(const types::ShapeType & chunkId,
                                         const unsigned dim,
                                         const bool fromHeader=false) const {
            if(isSharded_) {
                return chunkShape_[dim];
            }
            handle::Chunk chunk(handle_, chunkId, defaultChunkShape(), shape());
            if(!isZarr_ && fromHeader) {
                types::ShapeType chunkShape;
                read_shape_from_n5_header(chunk.path(), chunkShape);
                return chunkShape[dim];
            } else {
                return chunk.shape()[dim];
            }
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

        inline void decompress(const std::vector<char> & buffer,
                               void * dataOut,
                               const std::size_t data_size) const {
            util::decompress<T>(buffer, dataOut, data_size, Mixin::compressor_);
        }

        inline void getFillValue(void * fillValue) const {
            *((T*) fillValue) = Mixin::fillValue_;
        }


        inline bool checkVarlenChunk(const types::ShapeType & chunkId, std::size_t & chunkSize) const {
            if(isSharded_) {
                chunkSize = chunkSize_;
                return false;
            }
            handle::Chunk chunk(handle_, chunkId, defaultChunkShape(), shape());
            if(isZarr_ || !chunk.exists()) {
                chunkSize = chunk.size();
                return false;
            }

            const bool is_varlen = read_varlen_from_n5_header(chunk.path(), chunkSize);
            if(!is_varlen) {
                chunkSize = chunk.size();
            }
            return is_varlen;
        }

        inline const FileMode & mode() const {
            return handle_.mode();
        }
        inline const fs::path & path() const {
            return handle_.path();
        }
        inline void chunkPath(const types::ShapeType & chunkId, fs::path & path) const {
            if(isSharded_) {
                auto shardIdx = shard_utils::computeShardIndex(chunkId, chunksPerShard_);
                std::string shardKeyStr = shard_utils::shardKey(shardIdx, zarrDelimiter_.empty() ? "/" : zarrDelimiter_);
                path = handle_.path() / shardKeyStr;
                return;
            }
            handle::Chunk chunk(handle_, chunkId, defaultChunkShape(), shape());
            path = chunk.path();
        }
        inline void removeChunk(const types::ShapeType & chunkId) const {
            if(isSharded_) {
                // For sharded datasets, we can only remove entire shard files
                auto shardIdx = shard_utils::computeShardIndex(chunkId, chunksPerShard_);
                std::string shardKeyStr = shard_utils::shardKey(shardIdx, zarrDelimiter_.empty() ? "/" : zarrDelimiter_);
                fs::path shardPath = handle_.path() / shardKeyStr;
                if(fs::exists(shardPath)) {
                    fs::remove(shardPath);
                }
                return;
            }
            handle::Chunk chunk(handle_, chunkId, defaultChunkShape(), shape());
            chunk.remove();
        }
        inline void remove() const {
            handle_.remove();
        }

        // Flush all buffered shards to disk
        inline void flushShards() const {
            std::lock_guard<std::mutex> lock(globalShardMutex_);
            for(auto & [shardIdx, buffer] : shardBuffers_) {
                std::string shardKeyStr = shard_utils::shardKey(shardIdx, zarrDelimiter_.empty() ? "/" : zarrDelimiter_);
                fs::path shardPath = handle_.path() / shardKeyStr;
                buffer.flush(shardPath, indexLocation_);
            }
            shardBuffers_.clear();
            shardIndexCache_.clear();
            shardMutexes_.clear();
        }

        // Flush a specific shard to disk
        inline void flushShard(const types::ShapeType & shardIndices) const {
            std::lock_guard<std::mutex> lock(globalShardMutex_);
            auto it = shardBuffers_.find(shardIndices);
            if(it != shardBuffers_.end()) {
                std::string shardKeyStr = shard_utils::shardKey(shardIndices, zarrDelimiter_.empty() ? "/" : zarrDelimiter_);
                fs::path shardPath = handle_.path() / shardKeyStr;
                it->second.flush(shardPath, indexLocation_);
                shardBuffers_.erase(it);
                shardIndexCache_.erase(shardIndices);
                shardMutexes_.erase(shardIndices);
            }
        }

        // delete copy constructor and assignment operator
        // because the compressor cannot be copied by default
        // and we don't really need this to be copyable afaik
        // if this changes at some point, we need to provide a proper
        // implementation here
        Dataset(const Dataset & that) = delete;
        Dataset & operator=(const Dataset & that) = delete;

    private:

        // Load an existing shard file's chunks into a ShardBuffer for merging
        inline void loadExistingShard(const fs::path & shardPath, ShardBuffer & buf) const {
            auto index = ShardBuffer::readIndex(shardPath, numInnerChunks_, indexLocation_, indexHasCrc32c_);
            for(std::size_t i = 0; i < numInnerChunks_; ++i) {
                if(!index.isEmpty(i)) {
                    std::vector<char> chunkData;
                    ShardBuffer::readChunkWithIndex(shardPath, i, index, chunkData);
                    buf.writeChunk(i, std::move(chunkData));
                }
            }
        }

        // Byte-swap an array of T elements in-place
        inline void byteSwapInPlace(void * data, std::size_t numElements) const {
            if(sizeof(T) <= 1) return;
            T * arr = static_cast<T*>(data);
            util::reverseEndiannessInplace<T>(arr, arr + numElements);
        }

        inline void write(const fs::path & path, const std::vector<char> & buffer) const {
            std::ofstream file(path, std::ios::binary);
            file.write(&buffer[0], buffer.size());
            file.close();
        }

        inline void read(const fs::path & path, std::vector<char> & buffer) const {
            // open input stream and read the filesize
            std::ifstream file(path, std::ios::binary);

            file.seekg(0, std::ios::end);
            const std::size_t file_size = file.tellg();
            file.seekg(0, std::ios::beg);

            // resize the data vector
            buffer.resize(file_size);

            // read the file
            file.read(&buffer[0], file_size);
            file.close();
        }


        // check that the chunk handle is valid
        inline void checkChunk(const handle::Chunk & chunk,
                               const bool isVarlen=false) const {
            // check dimension
            const auto & chunkIndices = chunk.chunkIndices();
            if(!chunking_.checkBlockCoordinate(chunkIndices)) {
                throw std::runtime_error("Invalid chunk");
            }
            // varlen chunks are only supported in n5
            if(isVarlen && isZarr_) {
                throw std::runtime_error("Varlength chunks are not supported in zarr");
            }
        }


        inline bool read_varlen_from_n5_header(const fs::path & path,
                                               std::size_t & chunkSize) const {
            std::ifstream file(path, std::ios::binary);

            // read the mode
            uint16_t mode;
            file.read((char *) &mode, 2);
            util::reverseEndiannessInplace(mode);

            if(mode == 0) {
                return false;
            }

            // read the number of dimensions
            uint16_t ndim;
            file.read((char *) &ndim, 2);
            util::reverseEndiannessInplace(ndim);

            // advance the file by ndim * 4 to skip the shape
            file.seekg((ndim + 1) * 4);

            uint32_t varlength;
            file.read((char*) &varlength, 4);
            util::reverseEndiannessInplace(varlength);
            chunkSize = varlength;

            file.close();
            return true;
        }


        inline void read_shape_from_n5_header(const fs::path & path,
                                              types::ShapeType & chunkShape) const {
            std::ifstream file(path, std::ios::binary);

            // advance the file by 2 to skip the mode
            file.seekg(2);

            // read the number of dimensions
            uint16_t ndim;
            file.read((char *) &ndim, 2);
            util::reverseEndiannessInplace(ndim);

            // read temporary shape with uint32 entries
            std::vector<uint32_t> shapeTmp(ndim);
            for(int d = 0; d < ndim; ++d) {
                file.read((char *) &shapeTmp[d], 4);
            }
            util::reverseEndiannessInplace<uint32_t>(shapeTmp.begin(), shapeTmp.end());

            // // N5-Axis order: we need to reverse the chunk shape read from the header
            std::reverse(shapeTmp.begin(), shapeTmp.end());

            chunkShape.resize(ndim);
            std::copy(shapeTmp.begin(), shapeTmp.end(), chunkShape.begin());

            file.close();
        }

        inline void writeChunkSharded(const types::ShapeType & chunkIndices, const void * dataIn,
                                       const bool isVarlen, const std::size_t varSize) const {
            // Compute shard and inner chunk index
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

            // Compress the chunk data
            std::vector<char> buffer;
            util::compress(srcData, dataSize, buffer, Mixin::compressor_);

            // Apply crc32c if needed
            if(hasCrc32c_) {
                compression::crc32c_encode(buffer);
            }

            // Buffer the compressed chunk with per-shard locking
            {
                std::unique_lock<std::mutex> globalLock(globalShardMutex_);
                auto it = shardBuffers_.find(shardIdx);
                if(it == shardBuffers_.end()) {
                    std::string shardKeyStr = shard_utils::shardKey(shardIdx, zarrDelimiter_.empty() ? "/" : zarrDelimiter_);
                    fs::path shardPath = handle_.path() / shardKeyStr;
                    auto emplaced = shardBuffers_.emplace(shardIdx, ShardBuffer(numInnerChunks_));
                    it = emplaced.first;
                    if(fs::exists(shardPath)) {
                        loadExistingShard(shardPath, it->second);
                    }
                }
                std::mutex & shardLock = getShardMutex(shardIdx);
                globalLock.unlock();
                std::lock_guard<std::mutex> guard(shardLock);
                it->second.writeChunk(linearIdx, std::move(buffer));
            }
        }

        inline bool readChunkSharded(const types::ShapeType & chunkIndices, void * dataOut) const {
            auto shardIdx = shard_utils::computeShardIndex(chunkIndices, chunksPerShard_);
            auto innerIdx = shard_utils::computeInnerIndex(chunkIndices, chunksPerShard_);
            std::size_t linearIdx = shard_utils::linearInnerIndex(innerIdx, chunksPerShard_);

            std::string shardKeyStr = shard_utils::shardKey(shardIdx, zarrDelimiter_.empty() ? "/" : zarrDelimiter_);
            fs::path shardPath = handle_.path() / shardKeyStr;

            // Use cached shard index if available, otherwise read and cache it
            std::vector<char> buffer;
            {
                std::lock_guard<std::mutex> lock(globalShardMutex_);
                auto cacheIt = shardIndexCache_.find(shardIdx);
                if(cacheIt == shardIndexCache_.end()) {
                    auto idx = ShardBuffer::readIndex(shardPath, numInnerChunks_, indexLocation_, indexHasCrc32c_);
                    cacheIt = shardIndexCache_.emplace(shardIdx, std::move(idx)).first;
                }
                if(cacheIt->second.numChunks() == 0) {
                    return false;  // shard file doesn't exist
                }
                if(!ShardBuffer::readChunkWithIndex(shardPath, linearIdx, cacheIt->second, buffer)) {
                    return false;  // chunk is empty (fill value)
                }
            }

            // Apply crc32c decode if needed
            if(hasCrc32c_) {
                compression::crc32c_decode(buffer);
            }

            // Decompress
            if(hasTranspose_ && !transposeOrder_.empty()) {
                // Decompress into temporary buffer, byte-swap, then inverse transpose
                std::vector<T> tmpBuf(chunkSize_);
                util::decompress<T>(buffer, tmpBuf.data(), chunkSize_, Mixin::compressor_);
                if(needsByteSwap_) {
                    byteSwapInPlace(tmpBuf.data(), chunkSize_);
                }
                compression::TransposeCodec codec(transposeOrder_);
                codec.decode(tmpBuf.data(), static_cast<T*>(dataOut), chunkShape_);
            } else {
                if(!Mixin::compressor_) {
                    throw std::runtime_error("readChunkSharded: compressor is null for shard at " + shardPath.string());
                }
                util::decompress<T>(buffer, dataOut, chunkSize_, Mixin::compressor_);
                if(needsByteSwap_) {
                    byteSwapInPlace(dataOut, chunkSize_);
                }
            }

            return true;
        }

        inline bool chunkExistsSharded(const types::ShapeType & chunkId) const {
            auto shardIdx = shard_utils::computeShardIndex(chunkId, chunksPerShard_);
            auto innerIdx = shard_utils::computeInnerIndex(chunkId, chunksPerShard_);
            std::size_t linearIdx = shard_utils::linearInnerIndex(innerIdx, chunksPerShard_);

            std::lock_guard<std::mutex> lock(globalShardMutex_);
            auto cacheIt = shardIndexCache_.find(shardIdx);
            if(cacheIt == shardIndexCache_.end()) {
                std::string shardKeyStr = shard_utils::shardKey(shardIdx, zarrDelimiter_.empty() ? "/" : zarrDelimiter_);
                fs::path shardPath = handle_.path() / shardKeyStr;
                auto idx = ShardBuffer::readIndex(shardPath, numInnerChunks_, indexLocation_, indexHasCrc32c_);
                cacheIt = shardIndexCache_.emplace(shardIdx, std::move(idx)).first;
            }
            if(cacheIt->second.numChunks() == 0) {
                return false;
            }
            return !cacheIt->second.isEmpty(linearIdx);
        }

    private:
        handle::Dataset handle_;
        mutable std::mutex globalShardMutex_;  // protects the maps themselves
        mutable std::map<types::ShapeType, std::unique_ptr<std::mutex>> shardMutexes_;
        mutable std::map<types::ShapeType, ShardBuffer> shardBuffers_;
        mutable std::map<types::ShapeType, ShardIndex> shardIndexCache_;

        // Get or create a per-shard mutex (caller must hold globalShardMutex_)
        inline std::mutex & getShardMutex(const types::ShapeType & shardIdx) const {
            auto it = shardMutexes_.find(shardIdx);
            if(it == shardMutexes_.end()) {
                it = shardMutexes_.emplace(shardIdx, std::make_unique<std::mutex>()).first;
            }
            return *it->second;
        }
    };


}
}
