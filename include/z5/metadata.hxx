#pragma once

#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <iomanip>

#include "z5/types/types.hxx"


namespace z5 {

    // general format
    struct Metadata {
        types::FileFormat format;
        int zarrFormat;
        int n5Major;
        int n5Minor;
        int n5Patch;

        Metadata(const types::FileFormat format) : format(format),
                                      zarrFormat(format == types::zarr_v3 ? 3 : 2),
                                      n5Major(2),
                                      n5Minor(0),
                                      n5Patch(0)
        {}

        // backward compat constructor
        Metadata(const bool isZarr) : Metadata(isZarr ? types::zarr_v2 : types::n5) {}

        // backward compat accessor
        inline bool isZarr() const { return format != types::n5; }
        inline bool isZarrV3() const { return format == types::zarr_v3; }

        inline std::string n5Format() const {
            return std::to_string(n5Major) + "." + std::to_string(n5Minor) + "." + std::to_string(n5Patch);
        }
    };


    struct DatasetMetadata : public Metadata {

        //template<typename T>
        DatasetMetadata(
            const types::Datatype dtype,
            const types::ShapeType & shape,
            const types::ShapeType & chunkShape,
            const bool isZarr,
            const types::Compressor compressor=types::raw,
            const types::CompressionOptions & compressionOptions=types::CompressionOptions(),
            const double fillValue=0,
            const std::string & zarrDelimiter="."
            ) : Metadata(isZarr),
                dtype(dtype),
                shape(shape),
                chunkShape(chunkShape),
                compressor(compressor),
                compressionOptions(compressionOptions),
                fillValue(fillValue),
                zarrDelimiter(zarrDelimiter)
        {
            checkShapes();
        }

        DatasetMetadata(
            const types::Datatype dtype,
            const types::ShapeType & shape,
            const types::ShapeType & chunkShape,
            const types::FileFormat format,
            const types::Compressor compressor=types::raw,
            const types::CompressionOptions & compressionOptions=types::CompressionOptions(),
            const double fillValue=0,
            const std::string & zarrDelimiter="."
            ) : Metadata(format),
                dtype(dtype),
                shape(shape),
                chunkShape(chunkShape),
                compressor(compressor),
                compressionOptions(compressionOptions),
                fillValue(fillValue),
                zarrDelimiter(zarrDelimiter)
        {
            checkShapes();
        }


        // empty constructor
        DatasetMetadata() : Metadata(true)
        {}


        //
        void toJson(nlohmann::json & j) const {
            if(format == types::zarr_v3) {
                toJsonZarrV3(j);
            } else if(format == types::n5) {
                toJsonN5(j);
            } else {
                toJsonZarr(j);
            }
        }


        //
        void fromJson(nlohmann::json & j, const bool isZarrDs) {
            format = isZarrDs ? types::zarr_v2 : types::n5;
            isZarrDs ? fromJsonZarr(j) : fromJsonN5(j);
            checkShapes();
        }

        void fromJson(nlohmann::json & j, const types::FileFormat fmt) {
            format = fmt;
            if(fmt == types::zarr_v3) {
                fromJsonZarrV3(j);
            } else if(fmt == types::n5) {
                fromJsonN5(j);
            } else {
                fromJsonZarr(j);
            }
            checkShapes();
        }

    private:
        void toJsonZarr(nlohmann:: json & j) const {

            nlohmann::json compressionOpts;
            types::writeZarrCompressionOptionsToJson(compressor, compressionOptions, compressionOpts);
            j["compressor"] = compressionOpts;

            j["dtype"] = types::Datatypes::dtypeToZarr().at(dtype);
            j["shape"] = shape;
            j["chunks"] = chunkShape;

            if (std::isnan(fillValue)) {
              j["fill_value"] = "NaN";
            } else if (std::isinf(fillValue)) {
              j["fill_value"] = fillValue > 0 ? "Infinity" : "-Infinity";
            } else {
              j["fill_value"] = fillValue;
            }

            j["filters"] = nullptr;
            j["order"] = "C";
            j["zarr_format"] = 2;
            j["dimension_separator"] = zarrDelimiter;
        }

        void toJsonN5(nlohmann::json & j) const {

            // N5-Axis order: we need to reverse the shape when writing to metadata
            types::ShapeType rshape(shape.rbegin(), shape.rend());
            j["dimensions"] = rshape;

            // N5-Axis order: we need to reverse the block-size when writing to metadata
            types::ShapeType rchunks(chunkShape.rbegin(), chunkShape.rend());
            j["blockSize"] = rchunks;

            j["dataType"] = types::Datatypes::dtypeToN5().at(dtype);

            // write the new format
            nlohmann::json jOpts;
            types::writeN5CompressionOptionsToJson(compressor, compressionOptions, jOpts);
            j["compression"] = jOpts;
        }


        void toJsonZarrV3(nlohmann::json & j) const {
            j["zarr_format"] = 3;
            j["node_type"] = "array";
            j["data_type"] = types::Datatypes::dtypeToZarrV3().at(dtype);
            j["shape"] = shape;

            // chunk_grid - when sharded, the grid units are shards
            nlohmann::json chunkGrid;
            chunkGrid["name"] = "regular";
            chunkGrid["configuration"]["chunk_shape"] = isSharded ? shardShape : chunkShape;
            j["chunk_grid"] = chunkGrid;

            // chunk_key_encoding
            nlohmann::json chunkKeyEncoding;
            chunkKeyEncoding["name"] = "default";
            chunkKeyEncoding["configuration"]["separator"] = "/";
            j["chunk_key_encoding"] = chunkKeyEncoding;

            // Build inner codecs pipeline
            nlohmann::json innerCodecs = nlohmann::json::array();

            // bytes codec is always first
            nlohmann::json bytesCodec;
            bytesCodec["name"] = "bytes";
            bytesCodec["configuration"]["endian"] = bytesEndian;
            innerCodecs.push_back(bytesCodec);

            // optional transpose codec
            if(hasTranspose && !transposeOrder.empty()) {
                nlohmann::json transposeCodec;
                transposeCodec["name"] = "transpose";
                transposeCodec["configuration"]["order"] = transposeOrder;
                innerCodecs.push_back(transposeCodec);
            }

            // optional compressor codec
            if(compressor != types::raw) {
                nlohmann::json compressorCodec;
                types::writeZarrV3CompressionOptionsToJson(compressor, compressionOptions, compressorCodec);
                innerCodecs.push_back(compressorCodec);
            }

            // optional crc32c codec
            if(hasCrc32c) {
                nlohmann::json crc32cCodec;
                crc32cCodec["name"] = "crc32c";
                innerCodecs.push_back(crc32cCodec);
            }

            if(isSharded) {
                // Wrap in sharding_indexed codec
                nlohmann::json shardingCodec;
                shardingCodec["name"] = "sharding_indexed";
                nlohmann::json shardConfig;
                shardConfig["chunk_shape"] = chunkShape;
                shardConfig["codecs"] = innerCodecs;

                // index codecs
                nlohmann::json indexCodecs = nlohmann::json::array();
                nlohmann::json indexBytesCodec;
                indexBytesCodec["name"] = "bytes";
                indexBytesCodec["configuration"]["endian"] = "little";
                indexCodecs.push_back(indexBytesCodec);
                shardConfig["index_codecs"] = indexCodecs;
                shardConfig["index_location"] = indexLocation;
                shardingCodec["configuration"] = shardConfig;

                nlohmann::json codecs = nlohmann::json::array();
                codecs.push_back(shardingCodec);
                j["codecs"] = codecs;
            } else {
                j["codecs"] = innerCodecs;
            }

            // fill_value
            if (std::isnan(fillValue)) {
                j["fill_value"] = "NaN";
            } else if (std::isinf(fillValue)) {
                j["fill_value"] = fillValue > 0 ? "Infinity" : "-Infinity";
            } else {
                j["fill_value"] = fillValue;
            }

            // dimension_names (optional, empty by default)
            if(!dimensionNames.empty()) {
                j["dimension_names"] = dimensionNames;
            }
        }


        void fromJsonZarr(const nlohmann::json & j) {
            checkJson(j);
            try {
                dtype = types::Datatypes::zarrToDtype().at(j["dtype"]);
            } catch(std::out_of_range) {
                throw std::runtime_error("Unsupported zarr dtype: " + static_cast<std::string>(j["dtype"]));
            }
            shape = types::ShapeType(j["shape"].begin(), j["shape"].end());
            chunkShape = types::ShapeType(j["chunks"].begin(), j["chunks"].end());

            const auto & fillValJson = j["fill_value"];
            if(fillValJson.type() == nlohmann::json::value_t::string) {
                if (fillValJson == "NaN") {
                    fillValue = std::numeric_limits<double>::quiet_NaN();
                } else if (fillValJson == "Infinity") {
                    fillValue = std::numeric_limits<double>::infinity();
                } else if (fillValJson == "-Infinity") {
                    fillValue = -std::numeric_limits<double>::infinity();
                } else {
                    throw std::runtime_error("Invalid string value for fillValue");
                }
            } else if(fillValJson.type() == nlohmann::json::value_t::null) {
                fillValue = std::numeric_limits<double>::quiet_NaN();
            } else {
                fillValue = static_cast<double>(fillValJson);
            }

            auto jIt = j.find("dimension_separator");
            if(jIt != j.end()) {
                zarrDelimiter = *jIt;
            } else {
                zarrDelimiter = ".";
            }

            const auto & compressionOpts = j["compressor"];

            std::string zarrCompressorId = compressionOpts.is_null() ? "raw" : compressionOpts["id"];
            try {
                compressor = types::Compressors::zarrToCompressor().at(zarrCompressorId);
            } catch(std::out_of_range) {
                throw std::runtime_error("z5.DatasetMetadata.fromJsonZarr: wrong compressor for zarr format: " + zarrCompressorId);
            }

            types::readZarrCompressionOptionsFromJson(compressor, compressionOpts, compressionOptions);
        }


        void fromJsonN5(const nlohmann::json & j) {

            dtype = types::Datatypes::n5ToDtype().at(j["dataType"]);

            // N5-Axis order: we need to reverse the shape when reading from metadata
            shape = types::ShapeType(j["dimensions"].rbegin(), j["dimensions"].rend());

            // N5-Axis order: we need to reverse the chunk shape when reading from metadata
            chunkShape = types::ShapeType(j["blockSize"].rbegin(), j["blockSize"].rend());

            // we need to deal with two different encodings for compression in n5:
            // in the old format, we only have the field 'compressionType', indicating which compressor should be used
            // in the new format, we have the field 'type', which indicates the compressor
            // and can have additional attributes for options

            std::string n5Compressor;
            auto jIt = j.find("compression");

            if(jIt != j.end()) {
                const auto & jOpts = *jIt;
                auto j2It = jOpts.find("type");
                if(j2It != jOpts.end()) {
                    n5Compressor = *j2It;
                } else {
                    throw std::runtime_error("z5.DatasetMetadata.fromJsonN5: wrong compression format");
                }

                // get the actual compressor
                try {
                    compressor = types::Compressors::n5ToCompressor().at(n5Compressor);
                } catch(std::out_of_range) {
                    throw std::runtime_error("z5.DatasetMetadata.fromJsonN5: wrong compressor for n5 format");
                }

                readN5CompressionOptionsFromJson(compressor, jOpts, compressionOptions);
            }

            else {
                auto j2It = j.find("compressionType");
                if(j2It != j.end()) {
                    n5Compressor = *j2It;
                } else {
                    throw std::runtime_error("z5.DatasetMetadata.fromJsonN5: wrong compression format");
                }

                // get the actual compressor
                try {
                    compressor = types::Compressors::n5ToCompressor().at(n5Compressor);
                } catch(std::out_of_range) {
                    throw std::runtime_error("z5.DatasetMetadata.fromJsonN5: wrong compressor for n5 format");
                }

                // for the old compression, we just write the default gzip options
                compressionOptions["level"] = 5;
                compressionOptions["useZlib"] = false;
            }

            fillValue = 0;
        }


        void parseCodecsPipeline(const nlohmann::json & codecs) {
            for(const auto & codec : codecs) {
                const std::string codecName = codec["name"];
                if(codecName == "bytes" || codecName == "endian") {
                    if(codec.find("configuration") != codec.end()) {
                        bytesEndian = codec["configuration"].value("endian", std::string("little"));
                    }
                    continue;
                }
                if(codecName == "crc32c") {
                    hasCrc32c = true;
                    continue;
                }
                if(codecName == "transpose") {
                    hasTranspose = true;
                    if(codec.find("configuration") != codec.end()) {
                        const auto & config = codec["configuration"];
                        if(config.find("order") != config.end()) {
                            transposeOrder = config["order"].get<std::vector<int>>();
                        }
                    }
                    continue;
                }
                // try to map as a compressor
                auto it = types::Compressors::zarrV3ToCompressor().find(codecName);
                if(it != types::Compressors::zarrV3ToCompressor().end()) {
                    compressor = it->second;
                    types::readZarrV3CompressionOptionsFromJson(compressor, codec, compressionOptions);
                }
            }
        }

        void fromJsonZarrV3(const nlohmann::json & j) {
            // validate zarr_format
            if(j.value("zarr_format", 0) != 3) {
                throw std::runtime_error("Expected zarr_format 3");
            }
            if(j.value("node_type", "") != "array") {
                throw std::runtime_error("Expected node_type 'array'");
            }

            // data_type
            try {
                dtype = types::Datatypes::zarrV3ToDtype().at(j["data_type"]);
            } catch(std::out_of_range) {
                throw std::runtime_error("Unsupported zarr v3 dtype: " + j["data_type"].get<std::string>());
            }

            // shape
            shape = types::ShapeType(j["shape"].begin(), j["shape"].end());

            // chunk_grid
            const auto & chunkGrid = j["chunk_grid"];
            if(chunkGrid["name"] != "regular") {
                throw std::runtime_error("Only regular chunk grids are supported");
            }
            chunkShape = types::ShapeType(
                chunkGrid["configuration"]["chunk_shape"].begin(),
                chunkGrid["configuration"]["chunk_shape"].end()
            );

            // chunk_key_encoding
            zarrDelimiter = "/";
            if(j.find("chunk_key_encoding") != j.end()) {
                const auto & cke = j["chunk_key_encoding"];
                const std::string ckeName = cke.value("name", std::string("default"));
                if(ckeName == "v2") {
                    // v2 encoding uses "." separator by default
                    zarrDelimiter = ".";
                    if(cke.find("configuration") != cke.end()) {
                        zarrDelimiter = cke["configuration"].value("separator", std::string("."));
                    }
                } else {
                    // "default" encoding uses "/" separator
                    if(cke.find("configuration") != cke.end()) {
                        zarrDelimiter = cke["configuration"].value("separator", std::string("/"));
                    }
                }
            }

            // fill_value
            const auto & fv = j["fill_value"];
            if(fv.is_string()) {
                std::string fvs = fv;
                if(fvs == "NaN") fillValue = std::numeric_limits<double>::quiet_NaN();
                else if(fvs == "Infinity") fillValue = std::numeric_limits<double>::infinity();
                else if(fvs == "-Infinity") fillValue = -std::numeric_limits<double>::infinity();
                else throw std::runtime_error("Invalid fill_value string: " + fvs);
            } else if(fv.is_null()) {
                fillValue = std::numeric_limits<double>::quiet_NaN();
            } else {
                fillValue = fv.get<double>();
            }

            // codecs pipeline
            compressor = types::raw;
            isSharded = false;
            hasCrc32c = false;
            hasTranspose = false;
            bytesEndian = "little";

            const auto & codecs = j["codecs"];

            // Check for sharding_indexed codec
            for(const auto & codec : codecs) {
                const std::string codecName = codec["name"];
                if(codecName == "sharding_indexed") {
                    isSharded = true;
                    const auto & config = codec["configuration"];

                    // The inner chunk shape
                    chunkShape = types::ShapeType(
                        config["chunk_shape"].begin(),
                        config["chunk_shape"].end()
                    );

                    // The outer grid shape (from chunk_grid above) becomes the shard shape
                    shardShape = types::ShapeType(
                        chunkGrid["configuration"]["chunk_shape"].begin(),
                        chunkGrid["configuration"]["chunk_shape"].end()
                    );

                    // index_location
                    indexLocation = config.value("index_location", std::string("end"));

                    // Check index_codecs for crc32c
                    indexHasCrc32c = false;
                    if(config.find("index_codecs") != config.end()) {
                        for(const auto & ic : config["index_codecs"]) {
                            if(ic.value("name", "") == "crc32c") {
                                indexHasCrc32c = true;
                                break;
                            }
                        }
                    }

                    // Parse inner codecs
                    if(config.find("codecs") != config.end()) {
                        parseCodecsPipeline(config["codecs"]);
                    }
                    break;
                }
            }

            // If not sharded, parse the top-level codecs normally
            if(!isSharded) {
                parseCodecsPipeline(codecs);
            }

            // dimension_names (optional)
            if(j.find("dimension_names") != j.end() && !j["dimension_names"].is_null()) {
                dimensionNames = j["dimension_names"].get<std::vector<std::string>>();
            }
        }


    public:
        // metadata values that can be set
        types::Datatype dtype;
        types::ShapeType shape;
        types::ShapeType chunkShape;

        // compressor name and options
        types::Compressor compressor;
        types::CompressionOptions compressionOptions;

        double fillValue;
        std::string zarrDelimiter;

        // v3-specific fields
        std::vector<std::string> dimensionNames;

        // sharding fields (v3 only)
        bool isSharded = false;
        types::ShapeType shardShape;  // shape of a shard in elements (outer chunk grid)
        std::string indexLocation = "end";  // "start" or "end"
        bool indexHasCrc32c = false;  // whether index_codecs include crc32c

        // codec pipeline (v3) - for round-tripping unknown codecs
        bool hasCrc32c = false;
        bool hasTranspose = false;
        std::vector<int> transposeOrder;

        // bytes codec endianness (v3)
        std::string bytesEndian = "little";

        // metadata values that are fixed for now
        // zarr format is fixed to 2
        // const std::string order = "C";
        // const std::nullptr_t filters = nullptr;

    private:

        // make sure that shapes agree
        void checkShapes() {
            if(shape.size() != chunkShape.size()) {
                throw std::runtime_error("Dimension of shape and chunks does not agree");
            }
        }


        // make sure that fixed metadata values agree
        void checkJson(const nlohmann::json & j) {

            // check if order exists and check for the correct value
            auto jIt = j.find("order");
            if(jIt != j.end()) {
                if(*jIt != "C") {
                    throw std::runtime_error(
                        "Invalid Order: Z5 only supports C order"
                    );
                }
            }

            jIt = j.find("zarr_format");
            if(jIt != j.end()) {
                if(*jIt != 2) {
                    throw std::runtime_error(
                        "Invalid Zarr format: Z5 only supports zarr format 2"
                    );
                }
            }

            jIt = j.find("filters");
            if(jIt != j.end()) {
                if(!j["filters"].is_null() && j["filters"].size() > 0) {
                    throw std::runtime_error(
                        "Invalid Filters: Z5 does not support filters"
                    );
                }
            }
        }
    };


    inline void createDatasetMetadata(
        const std::string & dtype,
        const types::ShapeType & shape,
        const types::ShapeType & chunkShape,
        const bool createAsZarr,
        const std::string & compressor,
        const types::CompressionOptions & compressionOptions,
        const double fillValue,
        const std::string & zarrDelimiter,
        DatasetMetadata & metadata)
    {
        // get the internal data type
        types::Datatype internalDtype;
        try {
            internalDtype = types::Datatypes::n5ToDtype().at(dtype);
        } catch(const std::out_of_range & e) {
            throw std::runtime_error("z5::createDatasetMetadata: Invalid dtype for dataset");
        }

        // get the compressor
        types::Compressor internalCompressor;
        try {
            internalCompressor = types::Compressors::stringToCompressor().at(compressor);
        } catch(const std::out_of_range & e) {
            throw std::runtime_error("z5::createDatasetMetadata: Invalid compressor for dataset");
        }

        // add the default compression options if necessary
        // we need to make a copy of the compression options, because they are const
        auto internalCompressionOptions = compressionOptions;
        types::defaultCompressionOptions(internalCompressor, internalCompressionOptions, createAsZarr);

        metadata = DatasetMetadata(internalDtype, shape,
                                   chunkShape, createAsZarr,
                                   internalCompressor, internalCompressionOptions,
                                   fillValue, zarrDelimiter);
    }


    inline void createDatasetMetadata(
        const std::string & dtype,
        const types::ShapeType & shape,
        const types::ShapeType & chunkShape,
        const types::FileFormat format,
        const std::string & compressor,
        const types::CompressionOptions & compressionOptions,
        const double fillValue,
        const std::string & zarrDelimiter,
        DatasetMetadata & metadata)
    {
        types::Datatype internalDtype;
        try {
            internalDtype = types::Datatypes::n5ToDtype().at(dtype);
        } catch(const std::out_of_range & e) {
            throw std::runtime_error("z5::createDatasetMetadata: Invalid dtype for dataset");
        }

        types::Compressor internalCompressor;
        try {
            internalCompressor = types::Compressors::stringToCompressor().at(compressor);
        } catch(const std::out_of_range & e) {
            throw std::runtime_error("z5::createDatasetMetadata: Invalid compressor for dataset");
        }

        auto internalCompressionOptions = compressionOptions;
        types::defaultCompressionOptions(internalCompressor, internalCompressionOptions, format != types::n5);

        metadata = DatasetMetadata(internalDtype, shape,
                                   chunkShape, format,
                                   internalCompressor, internalCompressionOptions,
                                   fillValue, zarrDelimiter);
    }


    // Overload with sharding support
    inline void createDatasetMetadata(
        const std::string & dtype,
        const types::ShapeType & shape,
        const types::ShapeType & chunkShape,
        const types::FileFormat format,
        const std::string & compressor,
        const types::CompressionOptions & compressionOptions,
        const double fillValue,
        const std::string & zarrDelimiter,
        const types::ShapeType & shardShape,
        const std::string & indexLocation,
        DatasetMetadata & metadata)
    {
        createDatasetMetadata(dtype, shape, chunkShape, format,
                              compressor, compressionOptions,
                              fillValue, zarrDelimiter, metadata);
        if(!shardShape.empty()) {
            metadata.isSharded = true;
            metadata.shardShape = shardShape;
            metadata.indexLocation = indexLocation;
        }
    }

} // namespace::z5
