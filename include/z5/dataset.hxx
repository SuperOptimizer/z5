#pragma once

#include <memory>

#include "z5/metadata.hxx"
#include "z5/handle.hxx"
#include "z5/shard.hxx"
#include "z5/types/types.hxx"
#include "z5/util/util.hxx"
#include "z5/util/blocking.hxx"
#include "z5/util/format_data.hxx"

// different compression backends
#include "z5/compression/raw_compressor.hxx"
#include "z5/compression/blosc_compressor.hxx"
#include "z5/compression/zlib_compressor.hxx"
#include "z5/compression/bzip2_compressor.hxx"
#include "z5/compression/xz_compressor.hxx"
#include "z5/compression/lz4_compressor.hxx"
#include "z5/compression/zstd_compressor.hxx"


namespace z5 {

    // Abstract basis class for the dataset
    class Dataset {

    public:
        //
        // Constructor
        //
        Dataset(const DatasetMetadata & metadata) : isZarr_(metadata.isZarr()),
                                                    dtype_(metadata.dtype),
                                                    shape_(metadata.shape),
                                                    chunkShape_(metadata.chunkShape),
                                                    chunkSize_(std::accumulate(chunkShape_.begin(), chunkShape_.end(), 1, std::multiplies<std::size_t>())),
                                                    zarrDelimiter_(metadata.zarrDelimiter),
                                                    chunking_(shape_, chunkShape_),
                                                    isSharded_(metadata.isSharded),
                                                    shardShape_(metadata.shardShape),
                                                    indexLocation_(metadata.indexLocation),
                                                    indexHasCrc32c_(metadata.indexHasCrc32c),
                                                    hasCrc32c_(metadata.hasCrc32c),
                                                    hasTranspose_(metadata.hasTranspose),
                                                    transposeOrder_(metadata.transposeOrder),
                                                    bytesEndian_(metadata.bytesEndian)
        {
            if(isSharded_) {
                chunksPerShard_ = shard_utils::chunksPerShard(shardShape_, chunkShape_);
                numInnerChunks_ = shard_utils::totalInnerChunks(chunksPerShard_);
            }
            // Detect if byte-swapping is needed
            const uint16_t endianTest = 1;
            const bool hostIsLittle = (*reinterpret_cast<const char*>(&endianTest) == 1);
            needsByteSwap_ = (bytesEndian_ == "big" && hostIsLittle) ||
                             (bytesEndian_ == "little" && !hostIsLittle);
        }

        // Destructor — flush any buffered shards
        virtual ~Dataset(){ flushShards(); }

        //
        // API - already implemented and SHOULD NOT be overwritten
        //

        inline void checkRequestShape(const types::ShapeType & offset, const types::ShapeType & shape) const {
            if(offset.size() != shape_.size() || shape.size() != shape_.size()) {
                throw std::runtime_error("Request has wrong dimension");
            }
            for(int d = 0; d < shape_.size(); ++d) {
                if(offset[d] + shape[d] > shape_[d]) {
                    std::cout << "Out of range: " << offset << " + " << shape << std::endl;
                    std::cout << " = " << offset[d] + shape[d] << " > " << shape_[d] << std::endl;;
                    throw std::runtime_error("Request is out of range");
                }
                if(shape[d] == 0) {
                    throw std::runtime_error("Request shape has a zero entry");
                }
            }
        }

        // maximal chunk size and shape
        inline std::size_t defaultChunkSize() const {return chunkSize_;}
        inline const types::ShapeType & defaultChunkShape() const {return chunkShape_;}
        inline std::size_t defaultChunkShape(const unsigned d) const {return chunkShape_[d];}

        // chunking
        inline const util::Blocking & chunking() const {return chunking_;}

        // shapes and dimension
        inline unsigned dimension() const {return shape_.size();}
        inline const types::ShapeType & shape() const {return shape_;}
        inline std::size_t shape(const unsigned d) const {return shape_[d];}
        inline std::size_t size() const {
            return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<std::size_t>());
        }

        inline void getChunkOffset(const types::ShapeType & chunkId, types::ShapeType & chunkOffset) const {
            chunkOffset.resize(shape_.size());
            for(unsigned dim = 0; dim < shape_.size(); ++dim) {
                chunkOffset[dim] = chunkId[dim] * chunkShape_[dim];
            }
        }

        inline std::size_t numberOfChunks() const {return chunking_.numberOfBlocks();}
        inline const types::ShapeType & chunksPerDimension() const {return chunking_.blocksPerDimension();}
        inline std::size_t chunksPerDimension(const unsigned d) const {return chunking_.blocksPerDimension()[d];}

        inline types::Datatype getDtype() const {return dtype_;}
        inline bool isZarr() const {return isZarr_;}
        inline bool isSharded() const {return isSharded_;}
        inline const types::ShapeType & shardShape() const {return shardShape_;}
        inline const types::ShapeType & chunksPerShard() const {return chunksPerShard_;}
        inline std::size_t numInnerChunks() const {return numInnerChunks_;}
        inline const std::string & indexLocation() const {return indexLocation_;}
        inline bool hasCrc32c() const {return hasCrc32c_;}
        inline bool hasTranspose() const {return hasTranspose_;}
        inline const std::vector<int> & transposeOrder() const {return transposeOrder_;}
        inline const std::string & bytesEndian() const {return bytesEndian_;}
        inline bool needsByteSwap() const {return needsByteSwap_;}


        //
        // API - MUST implement
        //

        // we need to use void pointer here to have a generic API
        // write a chunk
        virtual void writeChunk(const types::ShapeType &, const void *,
                                const bool=false, const std::size_t=0) const = 0;
        // read a chunk - returns True if this is a varlen chunk
        virtual bool readChunk(const types::ShapeType &, void *) const = 0;
        // read a chunk; return the unformatted data
        virtual void readRawChunk(const types::ShapeType &, std::vector<char> &) const = 0;

        // check the request type
        virtual void checkRequestType(const std::type_info &) const = 0;

        // size and shape of an actual chunk
        virtual bool chunkExists(const types::ShapeType &) const = 0;
        virtual std::size_t getChunkSize(const types::ShapeType &) const = 0;
        virtual void getChunkShape(const types::ShapeType &, types::ShapeType &, const bool=false) const = 0;
        virtual std::size_t getChunkShape(const types::ShapeType &, const unsigned, const bool=false) const = 0;
        virtual bool checkVarlenChunk(const types::ShapeType &, std::size_t &) const = 0;

        // compression options and fill value
        virtual types::Compressor getCompressor() const = 0;
        virtual void getCompressor(std::string &) const = 0;
        virtual void getFillValue(void *) const = 0;
        virtual void getCompressionOptions(types::CompressionOptions &) const = 0;
        virtual void decompress(const std::vector<char> &, void *, const std::size_t) const = 0;

        // file paths, permissions and removal
        virtual const FileMode & mode() const = 0;
        virtual const fs::path & path() const = 0;
        virtual void chunkPath(const types::ShapeType &, fs::path &) const = 0;
        virtual void removeChunk(const types::ShapeType &) const = 0;
        virtual void remove() const = 0;
        virtual void flushShards() const {};

    protected:
        // private members:
        bool isZarr_;
        types::Datatype dtype_;
        types::ShapeType shape_;
        types::ShapeType chunkShape_;
        std::size_t chunkSize_;
        std::string zarrDelimiter_;

        util::Blocking chunking_;

        // sharding
        bool isSharded_ = false;
        types::ShapeType shardShape_;
        types::ShapeType chunksPerShard_;
        std::size_t numInnerChunks_ = 0;
        std::string indexLocation_ = "end";
        bool indexHasCrc32c_ = false;

        // codec pipeline
        bool hasCrc32c_ = false;
        bool hasTranspose_ = false;
        std::vector<int> transposeOrder_;
        std::string bytesEndian_ = "little";
        bool needsByteSwap_ = false;
    };


    // mixin to provide some typed functionality for typing
    template<class T>
    class MixinTyped {
    public:
        MixinTyped(const DatasetMetadata & metadata) : fillValue_(static_cast<T>(metadata.fillValue)) {
            init_compressor(metadata);
        }
        ~MixinTyped(){}

    protected:
        T fillValue_;
        // unique ptr to hold child classes of compressor
        std::unique_ptr<compression::CompressorBase<T>> compressor_;

    private:
        void init_compressor(const DatasetMetadata & metadata) {
            switch(metadata.compressor) {
                case types::raw:
            	    compressor_.reset(new compression::RawCompressor<T>()); break;
                #ifdef WITH_BLOSC
                case types::blosc:
            	    compressor_.reset(new compression::BloscCompressor<T>(metadata)); break;
                #endif
                #ifdef WITH_ZLIB
                case types::zlib:
                    compressor_.reset(new compression::ZlibCompressor<T>(metadata)); break;
                #endif
                #ifdef WITH_BZIP2
                case types::bzip2:
                    compressor_.reset(new compression::Bzip2Compressor<T>(metadata)); break;
                #endif
                #ifdef WITH_XZ
                case types::xz:
                    compressor_.reset(new compression::XzCompressor<T>(metadata)); break;
                #endif
                #ifdef WITH_LZ4
                case types::lz4:
                    compressor_.reset(new compression::Lz4Compressor<T>(metadata)); break;
                #endif
                #ifdef WITH_ZSTD
                case types::zstd:
                    compressor_.reset(new compression::ZstdCompressor<T>(metadata)); break;
                #endif
            }
        }
    };


} // namespace::z5
