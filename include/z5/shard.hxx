#pragma once

#include <vector>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <algorithm>

#include "z5/types/types.hxx"

namespace z5 {

    static constexpr uint64_t SHARD_EMPTY_SENTINEL = UINT64_MAX;

    class ShardIndex {
    public:
        ShardIndex() : numChunks_(0) {}
        explicit ShardIndex(std::size_t numChunks)
            : numChunks_(numChunks), entries_(numChunks) {
            for(auto & e : entries_) {
                e.offset = SHARD_EMPTY_SENTINEL;
                e.nbytes = SHARD_EMPTY_SENTINEL;
            }
        }

        struct Entry {
            uint64_t offset = SHARD_EMPTY_SENTINEL;
            uint64_t nbytes = SHARD_EMPTY_SENTINEL;
        };

        std::size_t numChunks() const { return numChunks_; }

        uint64_t chunkOffset(std::size_t linearIndex) const { return entries_.at(linearIndex).offset; }
        uint64_t chunkSize(std::size_t linearIndex) const { return entries_.at(linearIndex).nbytes; }
        bool isEmpty(std::size_t linearIndex) const {
            const auto & e = entries_.at(linearIndex);
            return e.offset == SHARD_EMPTY_SENTINEL && e.nbytes == SHARD_EMPTY_SENTINEL;
        }

        void setChunk(std::size_t linearIndex, uint64_t offset, uint64_t nbytes) {
            entries_.at(linearIndex).offset = offset;
            entries_.at(linearIndex).nbytes = nbytes;
        }

        void setEmpty(std::size_t linearIndex) {
            entries_.at(linearIndex).offset = SHARD_EMPTY_SENTINEL;
            entries_.at(linearIndex).nbytes = SHARD_EMPTY_SENTINEL;
        }

        std::size_t byteSize() const { return numChunks_ * 2 * sizeof(uint64_t); }

        // Serialize index to bytes (little-endian uint64 pairs)
        void write(std::vector<char> & out) const {
            out.resize(byteSize());
            for(std::size_t i = 0; i < numChunks_; ++i) {
                std::memcpy(&out[i * 16], &entries_[i].offset, 8);
                std::memcpy(&out[i * 16 + 8], &entries_[i].nbytes, 8);
            }
        }

        // Deserialize index from bytes
        void read(const char * data, std::size_t numChunks) {
            numChunks_ = numChunks;
            entries_.resize(numChunks);
            for(std::size_t i = 0; i < numChunks; ++i) {
                std::memcpy(&entries_[i].offset, data + i * 16, 8);
                std::memcpy(&entries_[i].nbytes, data + i * 16 + 8, 8);
            }
        }

    private:
        std::size_t numChunks_;
        std::vector<Entry> entries_;
    };


    class ShardBuffer {
    public:
        explicit ShardBuffer(std::size_t numInnerChunks)
            : chunkBuffers_(numInnerChunks) {}

        void writeChunk(std::size_t innerChunkIndex, const std::vector<char> & compressedData) {
            chunkBuffers_.at(innerChunkIndex) = compressedData;
        }

        void writeChunk(std::size_t innerChunkIndex, std::vector<char> && compressedData) {
            chunkBuffers_.at(innerChunkIndex) = std::move(compressedData);
        }

        bool hasChunk(std::size_t innerChunkIndex) const {
            return !chunkBuffers_.at(innerChunkIndex).empty();
        }

        const std::vector<char> & getChunk(std::size_t innerChunkIndex) const {
            return chunkBuffers_.at(innerChunkIndex);
        }

        // Flush: concatenate all chunks + write index at end (or start)
        void flush(const fs::path & path, const std::string & indexLocation) const {
            const std::size_t numChunks = chunkBuffers_.size();
            ShardIndex index(numChunks);

            // Compute the data blob
            std::vector<char> dataBlob;
            uint64_t currentOffset = 0;

            // If index is at start, we need to reserve space for the index first
            if(indexLocation == "start") {
                currentOffset = index.byteSize();
            }

            for(std::size_t i = 0; i < numChunks; ++i) {
                if(!chunkBuffers_[i].empty()) {
                    index.setChunk(i, currentOffset, chunkBuffers_[i].size());
                    dataBlob.insert(dataBlob.end(), chunkBuffers_[i].begin(), chunkBuffers_[i].end());
                    currentOffset += chunkBuffers_[i].size();
                }
            }

            // Serialize index
            std::vector<char> indexBytes;
            index.write(indexBytes);

            // Write the shard file
            // Create parent directory if needed
            fs::path parent = path.parent_path();
            if(!parent.empty() && !fs::exists(parent)) {
                try { fs::create_directories(parent); } catch(fs::filesystem_error) {}
            }

            std::ofstream file(path, std::ios::binary);
            if(indexLocation == "start") {
                file.write(indexBytes.data(), indexBytes.size());
                file.write(dataBlob.data(), dataBlob.size());
            } else {
                // "end" is default
                file.write(dataBlob.data(), dataBlob.size());
                file.write(indexBytes.data(), indexBytes.size());
            }
            file.close();
        }

        // Read a single inner chunk from a shard file
        static bool readChunk(const fs::path & path,
                              std::size_t innerChunkIndex,
                              std::size_t numInnerChunks,
                              const std::string & indexLocation,
                              std::vector<char> & chunkData,
                              bool indexHasCrc32c = false) {
            if(!fs::exists(path)) {
                return false;
            }

            std::ifstream file(path, std::ios::binary);
            file.seekg(0, std::ios::end);
            const std::size_t fileSize = file.tellg();

            const std::size_t indexSize = numInnerChunks * 16;
            // crc32c appends 4 bytes to the serialized index
            const std::size_t indexTotalSize = indexSize + (indexHasCrc32c ? 4 : 0);

            // Read the index
            std::vector<char> indexBytes(indexSize);
            if(indexLocation == "start") {
                file.seekg(0, std::ios::beg);
            } else {
                file.seekg(fileSize - indexTotalSize, std::ios::beg);
            }
            file.read(indexBytes.data(), indexSize);

            ShardIndex index;
            index.read(indexBytes.data(), numInnerChunks);

            if(index.isEmpty(innerChunkIndex)) {
                file.close();
                return false;
            }

            // Read the chunk data
            const uint64_t offset = index.chunkOffset(innerChunkIndex);
            const uint64_t nbytes = index.chunkSize(innerChunkIndex);
            chunkData.resize(nbytes);
            file.seekg(offset, std::ios::beg);
            file.read(chunkData.data(), nbytes);
            file.close();
            return true;
        }

        // Read the shard index only (for caching)
        static ShardIndex readIndex(const fs::path & path,
                                    std::size_t numInnerChunks,
                                    const std::string & indexLocation,
                                    bool indexHasCrc32c = false) {
            ShardIndex index;
            if(!fs::exists(path)) {
                return index;
            }

            std::ifstream file(path, std::ios::binary);
            file.seekg(0, std::ios::end);
            const std::size_t fileSize = file.tellg();
            const std::size_t indexSize = numInnerChunks * 16;
            const std::size_t indexTotalSize = indexSize + (indexHasCrc32c ? 4 : 0);

            std::vector<char> indexBytes(indexSize);
            if(indexLocation == "start") {
                file.seekg(0, std::ios::beg);
            } else {
                file.seekg(fileSize - indexTotalSize, std::ios::beg);
            }
            file.read(indexBytes.data(), indexSize);
            file.close();

            index.read(indexBytes.data(), numInnerChunks);
            return index;
        }

        // Read chunk using a pre-loaded index (avoids re-reading index)
        static bool readChunkWithIndex(const fs::path & path,
                                       std::size_t innerChunkIndex,
                                       const ShardIndex & index,
                                       std::vector<char> & chunkData) {
            if(index.isEmpty(innerChunkIndex)) {
                return false;
            }

            std::ifstream file(path, std::ios::binary);
            const uint64_t offset = index.chunkOffset(innerChunkIndex);
            const uint64_t nbytes = index.chunkSize(innerChunkIndex);
            chunkData.resize(nbytes);
            file.seekg(offset, std::ios::beg);
            file.read(chunkData.data(), nbytes);
            file.close();
            return true;
        }

    private:
        std::vector<std::vector<char>> chunkBuffers_;
    };


    // Utility functions for shard index computation
    namespace shard_utils {

        // Compute how many inner chunks per dimension fit in a shard
        inline types::ShapeType chunksPerShard(const types::ShapeType & shardShape,
                                                const types::ShapeType & chunkShape) {
            types::ShapeType cps(shardShape.size());
            for(std::size_t d = 0; d < shardShape.size(); ++d) {
                cps[d] = shardShape[d] / chunkShape[d];
            }
            return cps;
        }

        // Total number of inner chunks in a shard
        inline std::size_t totalInnerChunks(const types::ShapeType & chunksPerShard) {
            return std::accumulate(chunksPerShard.begin(), chunksPerShard.end(),
                                   (std::size_t)1, std::multiplies<std::size_t>());
        }

        // Given global chunk indices, compute which shard they belong to
        inline types::ShapeType computeShardIndex(const types::ShapeType & chunkIndices,
                                                   const types::ShapeType & chunksPerShard) {
            types::ShapeType shardIdx(chunkIndices.size());
            for(std::size_t d = 0; d < chunkIndices.size(); ++d) {
                shardIdx[d] = chunkIndices[d] / chunksPerShard[d];
            }
            return shardIdx;
        }

        // Given global chunk indices, compute the inner index within the shard
        inline types::ShapeType computeInnerIndex(const types::ShapeType & chunkIndices,
                                                   const types::ShapeType & chunksPerShard) {
            types::ShapeType innerIdx(chunkIndices.size());
            for(std::size_t d = 0; d < chunkIndices.size(); ++d) {
                innerIdx[d] = chunkIndices[d] % chunksPerShard[d];
            }
            return innerIdx;
        }

        // Linearize inner indices in C-order (row-major)
        inline std::size_t linearInnerIndex(const types::ShapeType & innerIndices,
                                             const types::ShapeType & chunksPerShard) {
            std::size_t linear = 0;
            std::size_t stride = 1;
            for(int d = static_cast<int>(innerIndices.size()) - 1; d >= 0; --d) {
                linear += innerIndices[d] * stride;
                stride *= chunksPerShard[d];
            }
            return linear;
        }

        // Build a chunk key (path) for a shard (uses the shard indices, not inner chunk indices)
        inline std::string shardKey(const types::ShapeType & shardIndices, const std::string & separator="/") {
            std::string name = "c";
            for(const auto & idx : shardIndices) {
                name += separator + std::to_string(idx);
            }
            return name;
        }
    }

} // namespace z5
