#pragma once

#include <vector>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include "z5/types/types.hxx"

namespace z5 {
namespace compression {

    // Transpose codec: reorders array dimensions before compression
    class TransposeCodec {
    public:
        TransposeCodec(const std::vector<int> & order) : order_(order) {
            inverseOrder_.resize(order.size());
            for(std::size_t i = 0; i < order.size(); ++i) {
                inverseOrder_[order[i]] = static_cast<int>(i);
            }
        }

        // Transpose data in-place (conceptually) by copying to a new buffer
        template<typename T>
        void encode(const T * dataIn, T * dataOut,
                    const types::ShapeType & shape) const {
            const int ndim = shape.size();
            if(static_cast<int>(order_.size()) != ndim) {
                throw std::runtime_error("Transpose order size does not match dimensions");
            }

            // Compute transposed shape
            types::ShapeType transposedShape(ndim);
            for(int d = 0; d < ndim; ++d) {
                transposedShape[d] = shape[order_[d]];
            }

            // Compute strides for original and transposed layout
            std::vector<std::size_t> srcStrides(ndim);
            std::vector<std::size_t> dstStrides(ndim);
            srcStrides[ndim - 1] = 1;
            dstStrides[ndim - 1] = 1;
            for(int d = ndim - 2; d >= 0; --d) {
                srcStrides[d] = srcStrides[d + 1] * shape[d + 1];
                dstStrides[d] = dstStrides[d + 1] * transposedShape[d + 1];
            }

            const std::size_t totalSize = std::accumulate(shape.begin(), shape.end(),
                                                           (std::size_t)1, std::multiplies<std::size_t>());

            // For each element, compute src multi-index, map to dst multi-index, write
            std::vector<std::size_t> srcIdx(ndim);
            for(std::size_t i = 0; i < totalSize; ++i) {
                // Decompose flat index into multi-index (C-order)
                std::size_t remaining = i;
                for(int d = 0; d < ndim; ++d) {
                    srcIdx[d] = remaining / srcStrides[d];
                    remaining %= srcStrides[d];
                }

                // Compute destination flat index
                std::size_t dstFlat = 0;
                for(int d = 0; d < ndim; ++d) {
                    dstFlat += srcIdx[order_[d]] * dstStrides[d];
                }

                dataOut[dstFlat] = dataIn[i];
            }
        }

        // Inverse transpose
        template<typename T>
        void decode(const T * dataIn, T * dataOut,
                    const types::ShapeType & shape) const {
            const int ndim = shape.size();

            // dataIn is in transposed layout; shape is the ORIGINAL shape
            types::ShapeType transposedShape(ndim);
            for(int d = 0; d < ndim; ++d) {
                transposedShape[d] = shape[order_[d]];
            }

            std::vector<std::size_t> dstStrides(ndim);
            std::vector<std::size_t> srcStrides(ndim);
            dstStrides[ndim - 1] = 1;
            srcStrides[ndim - 1] = 1;
            for(int d = ndim - 2; d >= 0; --d) {
                dstStrides[d] = dstStrides[d + 1] * shape[d + 1];
                srcStrides[d] = srcStrides[d + 1] * transposedShape[d + 1];
            }

            const std::size_t totalSize = std::accumulate(shape.begin(), shape.end(),
                                                           (std::size_t)1, std::multiplies<std::size_t>());

            std::vector<std::size_t> dstIdx(ndim);
            for(std::size_t i = 0; i < totalSize; ++i) {
                // Decompose flat index into multi-index of original shape
                std::size_t remaining = i;
                for(int d = 0; d < ndim; ++d) {
                    dstIdx[d] = remaining / dstStrides[d];
                    remaining %= dstStrides[d];
                }

                // Compute source flat index in transposed layout
                std::size_t srcFlat = 0;
                for(int d = 0; d < ndim; ++d) {
                    srcFlat += dstIdx[order_[d]] * srcStrides[d];
                }

                dataOut[i] = dataIn[srcFlat];
            }
        }

        const std::vector<int> & order() const { return order_; }
        const std::vector<int> & inverseOrder() const { return inverseOrder_; }

    private:
        std::vector<int> order_;
        std::vector<int> inverseOrder_;
    };

} // namespace compression
} // namespace z5
