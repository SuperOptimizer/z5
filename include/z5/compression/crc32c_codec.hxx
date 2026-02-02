#pragma once

#include <cstdint>
#include <vector>
#include <stdexcept>
#include <cstring>

namespace z5 {
namespace compression {

    // Software CRC32C implementation using the Castagnoli polynomial
    namespace crc32c_detail {

        inline uint32_t crc32c_table(uint32_t index) {
            static uint32_t table[256] = {0};
            static bool initialized = false;
            if(!initialized) {
                for(uint32_t i = 0; i < 256; ++i) {
                    uint32_t crc = i;
                    for(int j = 0; j < 8; ++j) {
                        if(crc & 1)
                            crc = (crc >> 1) ^ 0x82F63B78u;
                        else
                            crc >>= 1;
                    }
                    table[i] = crc;
                }
                initialized = true;
            }
            return table[index];
        }

        inline uint32_t compute_crc32c(const void * data, std::size_t length) {
            uint32_t crc = 0xFFFFFFFF;
            const uint8_t * buf = static_cast<const uint8_t*>(data);
            for(std::size_t i = 0; i < length; ++i) {
                crc = crc32c_table((crc ^ buf[i]) & 0xFF) ^ (crc >> 8);
            }
            return crc ^ 0xFFFFFFFF;
        }
    }

    // Append a 4-byte CRC32C checksum to the buffer
    inline void crc32c_encode(std::vector<char> & buffer) {
        uint32_t checksum = crc32c_detail::compute_crc32c(buffer.data(), buffer.size());
        const std::size_t oldSize = buffer.size();
        buffer.resize(oldSize + 4);
        std::memcpy(&buffer[oldSize], &checksum, 4);
    }

    // Verify and strip the 4-byte CRC32C checksum from the buffer
    inline void crc32c_decode(std::vector<char> & buffer) {
        if(buffer.size() < 4) {
            throw std::runtime_error("Buffer too small for CRC32C checksum");
        }
        const std::size_t dataSize = buffer.size() - 4;
        uint32_t stored;
        std::memcpy(&stored, &buffer[dataSize], 4);
        uint32_t computed = crc32c_detail::compute_crc32c(buffer.data(), dataSize);
        if(stored != computed) {
            throw std::runtime_error("CRC32C checksum mismatch");
        }
        buffer.resize(dataSize);
    }

} // namespace compression
} // namespace z5
