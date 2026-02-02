#pragma once

#ifdef WITH_ZSTD

#include <zstd.h>

#include "z5/compression/compressor_base.hxx"
#include "z5/metadata.hxx"

namespace z5 {
namespace compression {

    template<typename T>
    class ZstdCompressor : public CompressorBase<T> {

    public:
        ZstdCompressor(const DatasetMetadata & metadata) {
            init(metadata);
        }

        void compress(const T * dataIn, std::vector<char> & dataOut, std::size_t sizeIn) const {

            const std::size_t srcSize = sizeIn * sizeof(T);
            const std::size_t maxDstSize = ZSTD_compressBound(srcSize);

            dataOut.clear();
            dataOut.resize(maxDstSize);

            const std::size_t compressedSize = ZSTD_compress(
                dataOut.data(), maxDstSize,
                dataIn, srcSize,
                clevel_
            );

            if(ZSTD_isError(compressedSize)) {
                throw std::runtime_error(std::string("Zstd compression failed: ") + ZSTD_getErrorName(compressedSize));
            }

            dataOut.resize(compressedSize);
        }

        void decompress(const std::vector<char> & dataIn, T * dataOut, std::size_t sizeOut) const {

            const std::size_t dstSize = sizeOut * sizeof(T);
            const std::size_t decompressedSize = ZSTD_decompress(
                dataOut, dstSize,
                dataIn.data(), dataIn.size()
            );

            if(ZSTD_isError(decompressedSize)) {
                throw std::runtime_error(std::string("Zstd decompression failed: ") + ZSTD_getErrorName(decompressedSize));
            }
        }

        inline types::Compressor type() const {
            return types::zstd;
        }

        inline void getOptions(types::CompressionOptions & opts) const {
            opts["level"] = clevel_;
        }

    private:
        void init(const DatasetMetadata & metadata) {
            clevel_ = std::get<int>(metadata.compressionOptions.at("level"));
        }

        int clevel_;
    };

} // namespace compression
} // namespace z5

#endif
