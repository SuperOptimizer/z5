#pragma once

#ifdef WITH_BLOSC

#include <blosc2.h>
#include <limits>
#include "z5/compression/compressor_base.hxx"
#include "z5/metadata.hxx"

namespace z5 {
namespace compression {

    template<typename T>
    class BloscCompressor : public CompressorBase<T> {

    public:
        BloscCompressor(const DatasetMetadata & metadata) {
            init(metadata);
        }

        ~BloscCompressor() {
            if(cctx_ != nullptr) {
                blosc2_free_ctx(cctx_);
            }
            if(dctx_ != nullptr) {
                blosc2_free_ctx(dctx_);
            }
        }

        BloscCompressor(const BloscCompressor &) = delete;
        BloscCompressor & operator=(const BloscCompressor &) = delete;

        void compress(const T * dataIn, std::vector<char> & dataOut, std::size_t sizeIn) const {

            if(sizeIn > static_cast<std::size_t>(std::numeric_limits<int32_t>::max() / sizeof(T))) {
                throw std::runtime_error("Blosc2 compression input too large");
            }
            const std::size_t sizeOut = sizeIn * sizeof(T) + BLOSC2_MAX_OVERHEAD;
            if(sizeOut > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
                throw std::runtime_error("Blosc2 compression output buffer too large");
            }

            dataOut.clear();
            dataOut.resize(sizeOut);

            // compress the data
            const int32_t srcBytes = static_cast<int32_t>(sizeIn * sizeof(T));
            const int32_t destBytes = static_cast<int32_t>(sizeOut);
            int sizeCompressed = blosc2_compress_ctx(
                cctx_,
                dataIn, srcBytes,
                dataOut.data(), destBytes
            );

            // check for errors
            if(sizeCompressed <= 0) {
                throw std::runtime_error("Blosc2 compression failed");
            }

            // resize the out data
            dataOut.resize(sizeCompressed);
        }

        void decompress(const std::vector<char> & dataIn, T * dataOut, std::size_t sizeOut) const {

            // decompress the data
            if(sizeOut > static_cast<std::size_t>(std::numeric_limits<int32_t>::max() / sizeof(T))) {
                throw std::runtime_error("Blosc2 decompression output too large");
            }
            if(dataIn.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
                throw std::runtime_error("Blosc2 decompression input too large");
            }
            const int32_t srcBytes = static_cast<int32_t>(dataIn.size());
            const int32_t destBytes = static_cast<int32_t>(sizeOut * sizeof(T));
            int sizeDecompressed = blosc2_decompress_ctx(
                dctx_,
                dataIn.data(), srcBytes,
                dataOut, destBytes
            );

            // check for errors
            if(sizeDecompressed <= 0) {
                throw std::runtime_error("Blosc2 decompression failed");
            }
        }

        inline types::Compressor type() const {
            return types::blosc;
        }

        inline void getOptions(types::CompressionOptions & opts) const {
            opts["codec"] = compressor_;
            opts["shuffle"] = shuffle_;
            opts["level"] = clevel_;
            opts["blocksize"] = blocksize_;
            opts["nthreads"] = nthreads_;
        }

    private:
        // set the compression parameters from metadata
        void init(const DatasetMetadata & metadata) {
            const auto & cOpts = metadata.compressionOptions;

            // Ensure blosc2 is initialized (registers codecs including openh264)
            blosc2_init();

            clevel_     = std::get<int>(cOpts.at("level"));
            shuffle_    = std::get<int>(cOpts.at("shuffle"));
            compressor_ = std::get<std::string>(cOpts.at("codec"));
            blocksize_ = std::get<int>(cOpts.at("blocksize"));

            // set nthreads with a default value of 1
            nthreads_ = 1;
            auto threadsIt = cOpts.find("nthreads");
            if(threadsIt != cOpts.end()) {
                nthreads_ = std::get<int>(threadsIt->second);
            }

            const int compcode = blosc2_compname_to_compcode(compressor_.c_str());
            if(compcode < 0) {
                throw std::runtime_error("Unsupported Blosc2 codec: " + compressor_);
            }

            blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS;
            cparams.compcode = static_cast<uint8_t>(compcode);
            cparams.clevel = static_cast<uint8_t>(clevel_);
            cparams.typesize = static_cast<int32_t>(sizeof(T));
            cparams.nthreads = static_cast<int16_t>(nthreads_);
            cparams.blocksize = static_cast<int32_t>(blocksize_);
            for(auto & filter : cparams.filters) {
                filter = BLOSC_NOFILTER;
            }
            switch(shuffle_) {
                case 0:
                    cparams.filters[0] = BLOSC_NOFILTER;
                    break;
                case 1:
                    cparams.filters[0] = BLOSC_SHUFFLE;
                    break;
                case 2:
                    cparams.filters[0] = BLOSC_BITSHUFFLE;
                    break;
                default:
                    throw std::runtime_error("Unsupported Blosc2 shuffle option");
            }

            cctx_ = blosc2_create_cctx(cparams);
            if(cctx_ == nullptr) {
                throw std::runtime_error("Failed to create Blosc2 compression context");
            }

            blosc2_dparams dparams = BLOSC2_DPARAMS_DEFAULTS;
            dparams.nthreads = static_cast<int16_t>(nthreads_);
            dctx_ = blosc2_create_dctx(dparams);
            if(dctx_ == nullptr) {
                blosc2_free_ctx(cctx_);
                cctx_ = nullptr;
                throw std::runtime_error("Failed to create Blosc2 decompression context");
            }
        }

        // the blosc compressor
        std::string compressor_;
        // compression level
        int clevel_;
        // blosc shuffle
        int shuffle_;
        int blocksize_;
        int nthreads_;
        blosc2_context * cctx_ = nullptr;
        blosc2_context * dctx_ = nullptr;
    };

} // namespace compression
} // namespace z5

#endif
