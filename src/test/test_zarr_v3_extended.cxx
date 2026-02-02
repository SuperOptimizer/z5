#include "gtest/gtest.h"

#include <random>
#include <numeric>
#include <cstring>

#include "z5/factory.hxx"
#include "z5/attributes.hxx"
#include "z5/filesystem/metadata.hxx"
#include "z5/filesystem/dataset.hxx"
#include "z5/shard.hxx"
#include "z5/compression/crc32c_codec.hxx"
#include "z5/compression/transpose_codec.hxx"


namespace z5 {

    //
    // Test ShardIndex
    //
    TEST(ShardIndexTest, BasicOperations) {
        ShardIndex index(8);
        EXPECT_EQ(index.numChunks(), 8);

        // All entries should start empty
        for(std::size_t i = 0; i < 8; ++i) {
            EXPECT_TRUE(index.isEmpty(i));
        }

        // Set a chunk
        index.setChunk(3, 100, 50);
        EXPECT_FALSE(index.isEmpty(3));
        EXPECT_EQ(index.chunkOffset(3), 100);
        EXPECT_EQ(index.chunkSize(3), 50);

        // Set empty
        index.setEmpty(3);
        EXPECT_TRUE(index.isEmpty(3));
    }

    TEST(ShardIndexTest, Serialization) {
        ShardIndex index(4);
        index.setChunk(0, 0, 100);
        index.setChunk(2, 100, 200);

        // Serialize
        std::vector<char> bytes;
        index.write(bytes);
        EXPECT_EQ(bytes.size(), 4 * 16);  // 4 entries * 16 bytes each

        // Deserialize
        ShardIndex index2;
        index2.read(bytes.data(), 4);

        EXPECT_FALSE(index2.isEmpty(0));
        EXPECT_EQ(index2.chunkOffset(0), 0);
        EXPECT_EQ(index2.chunkSize(0), 100);
        EXPECT_TRUE(index2.isEmpty(1));
        EXPECT_FALSE(index2.isEmpty(2));
        EXPECT_EQ(index2.chunkOffset(2), 100);
        EXPECT_EQ(index2.chunkSize(2), 200);
        EXPECT_TRUE(index2.isEmpty(3));
    }

    //
    // Test shard_utils
    //
    TEST(ShardUtilsTest, ChunksPerShard) {
        types::ShapeType shardShape = {64, 64};
        types::ShapeType chunkShape = {16, 16};
        auto cps = shard_utils::chunksPerShard(shardShape, chunkShape);
        EXPECT_EQ(cps[0], 4);
        EXPECT_EQ(cps[1], 4);
    }

    TEST(ShardUtilsTest, IndexComputation) {
        types::ShapeType chunkIndices = {5, 7};
        types::ShapeType cps = {4, 4};

        auto shardIdx = shard_utils::computeShardIndex(chunkIndices, cps);
        EXPECT_EQ(shardIdx[0], 1);
        EXPECT_EQ(shardIdx[1], 1);

        auto innerIdx = shard_utils::computeInnerIndex(chunkIndices, cps);
        EXPECT_EQ(innerIdx[0], 1);
        EXPECT_EQ(innerIdx[1], 3);

        auto linear = shard_utils::linearInnerIndex(innerIdx, cps);
        EXPECT_EQ(linear, 1 * 4 + 3);  // row-major: 7
    }

    //
    // Test CRC32C codec
    //
    TEST(CRC32CTest, EncodeDecodeRoundTrip) {
        std::vector<char> data = {'H', 'e', 'l', 'l', 'o', ' ', 'W', 'o', 'r', 'l', 'd'};
        std::vector<char> original = data;

        compression::crc32c_encode(data);
        EXPECT_EQ(data.size(), original.size() + 4);

        compression::crc32c_decode(data);
        EXPECT_EQ(data, original);
    }

    TEST(CRC32CTest, CorruptionDetection) {
        std::vector<char> data = {'t', 'e', 's', 't'};
        compression::crc32c_encode(data);

        // Corrupt a byte
        data[0] = 'x';

        EXPECT_THROW(compression::crc32c_decode(data), std::runtime_error);
    }

    //
    // Test Transpose codec
    //
    TEST(TransposeTest, ReverseOrder2D) {
        // 2x3 array, reverse to 3x2
        std::vector<int> order = {1, 0};
        compression::TransposeCodec codec(order);

        types::ShapeType shape = {2, 3};
        // Original (row-major): [[0,1,2],[3,4,5]]
        int dataIn[6] = {0, 1, 2, 3, 4, 5};
        int dataOut[6];

        codec.encode(dataIn, dataOut, shape);
        // Transposed (3x2): [[0,3],[1,4],[2,5]]
        EXPECT_EQ(dataOut[0], 0);
        EXPECT_EQ(dataOut[1], 3);
        EXPECT_EQ(dataOut[2], 1);
        EXPECT_EQ(dataOut[3], 4);
        EXPECT_EQ(dataOut[4], 2);
        EXPECT_EQ(dataOut[5], 5);

        // Round-trip
        int dataRoundTrip[6];
        codec.decode(dataOut, dataRoundTrip, shape);
        for(int i = 0; i < 6; ++i) {
            EXPECT_EQ(dataRoundTrip[i], dataIn[i]);
        }
    }

    TEST(TransposeTest, Identity) {
        std::vector<int> order = {0, 1, 2};
        compression::TransposeCodec codec(order);

        types::ShapeType shape = {2, 3, 4};
        std::vector<float> dataIn(24);
        std::iota(dataIn.begin(), dataIn.end(), 0.0f);
        std::vector<float> dataOut(24);

        codec.encode(dataIn.data(), dataOut.data(), shape);

        for(int i = 0; i < 24; ++i) {
            EXPECT_EQ(dataOut[i], dataIn[i]);
        }
    }

    //
    // Test sharding metadata serialization
    //
    TEST(ShardingMetadataTest, ToFromJsonV3) {
        DatasetMetadata meta;
        meta.format = types::zarr_v3;
        meta.dtype = types::float32;
        meta.shape = {128, 128, 128};
        meta.chunkShape = {32, 32, 32};
        meta.compressor = types::raw;
        meta.fillValue = 0;
        meta.isSharded = true;
        meta.shardShape = {64, 64, 64};
        meta.indexLocation = "end";

        nlohmann::json j;
        meta.toJson(j);

        // Verify structure
        EXPECT_EQ(j["zarr_format"], 3);
        EXPECT_EQ(j["node_type"], "array");

        // chunk_grid should use shard shape
        auto gridShape = j["chunk_grid"]["configuration"]["chunk_shape"].get<types::ShapeType>();
        EXPECT_EQ(gridShape, meta.shardShape);

        // codecs should contain sharding_indexed
        const auto & codecs = j["codecs"];
        EXPECT_EQ(codecs.size(), 1);
        EXPECT_EQ(codecs[0]["name"], "sharding_indexed");
        EXPECT_EQ(codecs[0]["configuration"]["chunk_shape"].get<types::ShapeType>(), meta.chunkShape);
        EXPECT_EQ(codecs[0]["configuration"]["index_location"], "end");

        // Round-trip
        DatasetMetadata meta2;
        meta2.fromJson(j, types::zarr_v3);
        EXPECT_TRUE(meta2.isSharded);
        EXPECT_EQ(meta2.chunkShape, meta.chunkShape);
        EXPECT_EQ(meta2.shardShape, meta.shardShape);
        EXPECT_EQ(meta2.indexLocation, "end");
    }

    TEST(ShardingMetadataTest, IndexLocationStart) {
        DatasetMetadata meta;
        meta.format = types::zarr_v3;
        meta.dtype = types::int32;
        meta.shape = {64, 64};
        meta.chunkShape = {16, 16};
        meta.compressor = types::raw;
        meta.fillValue = 0;
        meta.isSharded = true;
        meta.shardShape = {32, 32};
        meta.indexLocation = "start";

        nlohmann::json j;
        meta.toJson(j);

        EXPECT_EQ(j["codecs"][0]["configuration"]["index_location"], "start");

        DatasetMetadata meta2;
        meta2.fromJson(j, types::zarr_v3);
        EXPECT_EQ(meta2.indexLocation, "start");
    }

    //
    // Test CRC32C in metadata
    //
    TEST(CodecPipelineMetadataTest, Crc32cRoundTrip) {
        DatasetMetadata meta;
        meta.format = types::zarr_v3;
        meta.dtype = types::float32;
        meta.shape = {100, 100};
        meta.chunkShape = {10, 10};
        meta.compressor = types::raw;
        meta.fillValue = 0;
        meta.hasCrc32c = true;

        nlohmann::json j;
        meta.toJson(j);

        // Check that crc32c codec is in the pipeline
        bool foundCrc = false;
        for(const auto & c : j["codecs"]) {
            if(c["name"] == "crc32c") foundCrc = true;
        }
        EXPECT_TRUE(foundCrc);

        DatasetMetadata meta2;
        meta2.fromJson(j, types::zarr_v3);
        EXPECT_TRUE(meta2.hasCrc32c);
    }

    //
    // Test transpose in metadata
    //
    TEST(CodecPipelineMetadataTest, TransposeRoundTrip) {
        DatasetMetadata meta;
        meta.format = types::zarr_v3;
        meta.dtype = types::float32;
        meta.shape = {100, 100, 100};
        meta.chunkShape = {10, 10, 10};
        meta.compressor = types::raw;
        meta.fillValue = 0;
        meta.hasTranspose = true;
        meta.transposeOrder = {2, 1, 0};

        nlohmann::json j;
        meta.toJson(j);

        bool foundTranspose = false;
        for(const auto & c : j["codecs"]) {
            if(c["name"] == "transpose") {
                foundTranspose = true;
                EXPECT_EQ(c["configuration"]["order"].get<std::vector<int>>(), meta.transposeOrder);
            }
        }
        EXPECT_TRUE(foundTranspose);

        DatasetMetadata meta2;
        meta2.fromJson(j, types::zarr_v3);
        EXPECT_TRUE(meta2.hasTranspose);
        EXPECT_EQ(meta2.transposeOrder, meta.transposeOrder);
    }


    //
    // Test sharded I/O with filesystem dataset
    //
    class ShardedDatasetTest : public ::testing::Test {
    protected:
        ShardedDatasetTest()
            : fileHandle_("test_sharded.zarr", FileMode(FileMode::modes::w)),
              dsHandle_(fileHandle_, "data", "/") {}

        ~ShardedDatasetTest() override {
            // cleanup
            if(fs::exists("test_sharded.zarr")) {
                fs::remove_all("test_sharded.zarr");
            }
        }

        void SetUp() override {
            if(fs::exists("test_sharded.zarr")) {
                fs::remove_all("test_sharded.zarr");
            }
        }

        filesystem::handle::File fileHandle_;
        filesystem::handle::Dataset dsHandle_;
    };


    TEST_F(ShardedDatasetTest, WriteReadRoundTrip) {
        // Create sharded metadata
        DatasetMetadata meta;
        meta.format = types::zarr_v3;
        meta.dtype = types::int32;
        meta.shape = {64, 64};
        meta.chunkShape = {16, 16};
        meta.compressor = types::raw;
        meta.fillValue = 0;
        meta.zarrDelimiter = "/";
        meta.isSharded = true;
        meta.shardShape = {32, 32};
        meta.indexLocation = "end";

        // Create the file and dataset on disk
        fileHandle_.create();
        Metadata fmeta(types::zarr_v3);
        filesystem::writeMetadata(fileHandle_, fmeta);

        dsHandle_.create();
        filesystem::writeMetadata(dsHandle_, meta);

        // Create dataset object
        filesystem::Dataset<int32_t> ds(dsHandle_, meta);

        const std::size_t chunkSize = 16 * 16;

        // Write chunk (0,0) = first inner chunk of shard (0,0)
        std::vector<int32_t> data(chunkSize);
        std::iota(data.begin(), data.end(), 1);
        ds.writeChunk({0, 0}, data.data());

        // Write chunk (1,0) = second inner chunk of shard (0,0)
        std::vector<int32_t> data2(chunkSize);
        std::iota(data2.begin(), data2.end(), 1000);
        ds.writeChunk({1, 0}, data2.data());

        // Flush shard to disk
        ds.flushShards();

        // Read back
        std::vector<int32_t> readData(chunkSize, 0);
        ds.readChunk({0, 0}, readData.data());
        EXPECT_EQ(readData, data);

        std::vector<int32_t> readData2(chunkSize, 0);
        ds.readChunk({1, 0}, readData2.data());
        EXPECT_EQ(readData2, data2);
    }

    TEST_F(ShardedDatasetTest, IndexLocationStart) {
        DatasetMetadata meta;
        meta.format = types::zarr_v3;
        meta.dtype = types::int32;
        meta.shape = {32, 32};
        meta.chunkShape = {16, 16};
        meta.compressor = types::raw;
        meta.fillValue = 0;
        meta.zarrDelimiter = "/";
        meta.isSharded = true;
        meta.shardShape = {32, 32};
        meta.indexLocation = "start";

        fileHandle_.create();
        Metadata fmeta(types::zarr_v3);
        filesystem::writeMetadata(fileHandle_, fmeta);
        dsHandle_.create();
        filesystem::writeMetadata(dsHandle_, meta);

        filesystem::Dataset<int32_t> ds(dsHandle_, meta);

        const std::size_t chunkSize = 16 * 16;
        std::vector<int32_t> data(chunkSize);
        std::iota(data.begin(), data.end(), 42);
        ds.writeChunk({0, 0}, data.data());
        ds.flushShards();

        std::vector<int32_t> readData(chunkSize, 0);
        ds.readChunk({0, 0}, readData.data());
        EXPECT_EQ(readData, data);
    }

    TEST_F(ShardedDatasetTest, EmptyChunkInShard) {
        DatasetMetadata meta;
        meta.format = types::zarr_v3;
        meta.dtype = types::int32;
        meta.shape = {32, 32};
        meta.chunkShape = {16, 16};
        meta.compressor = types::raw;
        meta.fillValue = 0;
        meta.zarrDelimiter = "/";
        meta.isSharded = true;
        meta.shardShape = {32, 32};
        meta.indexLocation = "end";

        fileHandle_.create();
        Metadata fmeta(types::zarr_v3);
        filesystem::writeMetadata(fileHandle_, fmeta);
        dsHandle_.create();
        filesystem::writeMetadata(dsHandle_, meta);

        filesystem::Dataset<int32_t> ds(dsHandle_, meta);

        // Only write chunk (1,1), leave others empty
        const std::size_t chunkSize = 16 * 16;
        std::vector<int32_t> data(chunkSize);
        std::iota(data.begin(), data.end(), 99);
        ds.writeChunk({1, 1}, data.data());
        ds.flushShards();

        // Reading chunk (1,1) should succeed
        std::vector<int32_t> readData(chunkSize, 0);
        ds.readChunk({1, 1}, readData.data());
        EXPECT_EQ(readData, data);

        // Reading empty chunk should return false (fill value)
        std::fill(readData.begin(), readData.end(), -1);
        EXPECT_FALSE(ds.readChunk({0, 0}, readData.data()));
    }


    //
    // Test CRC32C codec with actual I/O
    //
    class Crc32cDatasetTest : public ::testing::Test {
    protected:
        Crc32cDatasetTest()
            : fileHandle_("test_crc32c.zarr", FileMode(FileMode::modes::w)),
              dsHandle_(fileHandle_, "data", "/") {}

        ~Crc32cDatasetTest() override {
            if(fs::exists("test_crc32c.zarr")) {
                fs::remove_all("test_crc32c.zarr");
            }
        }

        void SetUp() override {
            if(fs::exists("test_crc32c.zarr")) {
                fs::remove_all("test_crc32c.zarr");
            }
        }

        filesystem::handle::File fileHandle_;
        filesystem::handle::Dataset dsHandle_;
    };

    TEST_F(Crc32cDatasetTest, WriteReadRoundTrip) {
        DatasetMetadata meta;
        meta.format = types::zarr_v3;
        meta.dtype = types::float32;
        meta.shape = {20, 20};
        meta.chunkShape = {10, 10};
        meta.compressor = types::raw;
        meta.fillValue = 0;
        meta.zarrDelimiter = "/";
        meta.hasCrc32c = true;

        fileHandle_.create();
        Metadata fmeta(types::zarr_v3);
        filesystem::writeMetadata(fileHandle_, fmeta);
        dsHandle_.create();
        filesystem::writeMetadata(dsHandle_, meta);

        filesystem::Dataset<float> ds(dsHandle_, meta);

        const std::size_t chunkSize = 10 * 10;
        std::vector<float> data(chunkSize);
        for(std::size_t i = 0; i < chunkSize; ++i) data[i] = static_cast<float>(i) * 0.5f;
        ds.writeChunk({0, 0}, data.data());

        std::vector<float> readData(chunkSize, 0);
        ds.readChunk({0, 0}, readData.data());
        for(std::size_t i = 0; i < chunkSize; ++i) {
            EXPECT_FLOAT_EQ(readData[i], data[i]);
        }
    }

    //
    // Test partial shard update (write to existing shard)
    //
    class PartialShardUpdateTest : public ::testing::Test {
    protected:
        PartialShardUpdateTest()
            : fileHandle_("test_partial_shard.zarr", FileMode(FileMode::modes::w)),
              dsHandle_(fileHandle_, "data", "/") {}

        ~PartialShardUpdateTest() override {
            if(fs::exists("test_partial_shard.zarr")) {
                fs::remove_all("test_partial_shard.zarr");
            }
        }

        void SetUp() override {
            if(fs::exists("test_partial_shard.zarr")) {
                fs::remove_all("test_partial_shard.zarr");
            }
        }

        filesystem::handle::File fileHandle_;
        filesystem::handle::Dataset dsHandle_;
    };

    TEST_F(PartialShardUpdateTest, UpdateExistingShard) {
        DatasetMetadata meta;
        meta.format = types::zarr_v3;
        meta.dtype = types::int32;
        meta.shape = {32, 32};
        meta.chunkShape = {16, 16};
        meta.compressor = types::raw;
        meta.fillValue = 0;
        meta.zarrDelimiter = "/";
        meta.isSharded = true;
        meta.shardShape = {32, 32};
        meta.indexLocation = "end";

        fileHandle_.create();
        Metadata fmeta(types::zarr_v3);
        filesystem::writeMetadata(fileHandle_, fmeta);
        dsHandle_.create();
        filesystem::writeMetadata(dsHandle_, meta);

        const std::size_t chunkSize = 16 * 16;

        // First write: chunk (0,0) only
        {
            filesystem::Dataset<int32_t> ds(dsHandle_, meta);
            std::vector<int32_t> data(chunkSize);
            std::iota(data.begin(), data.end(), 1);
            ds.writeChunk({0, 0}, data.data());
            ds.flushShards();
        }

        // Second write: chunk (1,1) to same shard, should preserve (0,0)
        {
            filesystem::Dataset<int32_t> ds(dsHandle_, meta);
            std::vector<int32_t> data2(chunkSize);
            std::iota(data2.begin(), data2.end(), 500);
            ds.writeChunk({1, 1}, data2.data());
            ds.flushShards();
        }

        // Read back both chunks
        {
            filesystem::Dataset<int32_t> ds(dsHandle_, meta);

            std::vector<int32_t> readData(chunkSize, 0);
            ds.readChunk({0, 0}, readData.data());
            EXPECT_EQ(readData[0], 1);
            EXPECT_EQ(readData[chunkSize - 1], static_cast<int32_t>(chunkSize));

            std::vector<int32_t> readData2(chunkSize, 0);
            ds.readChunk({1, 1}, readData2.data());
            EXPECT_EQ(readData2[0], 500);
        }
    }


    //
    // Test bytes codec endianness metadata round-trip
    //
    TEST(BytesEndianMetadataTest, BigEndianRoundTrip) {
        DatasetMetadata meta;
        meta.format = types::zarr_v3;
        meta.dtype = types::int32;
        meta.shape = {64, 64};
        meta.chunkShape = {16, 16};
        meta.compressor = types::raw;
        meta.fillValue = 0;
        meta.bytesEndian = "big";

        nlohmann::json j;
        meta.toJson(j);

        // Check that bytes codec has endian=big
        bool foundBig = false;
        for(const auto & c : j["codecs"]) {
            if(c["name"] == "bytes") {
                foundBig = (c["configuration"]["endian"] == "big");
            }
        }
        EXPECT_TRUE(foundBig);

        // Round-trip
        DatasetMetadata meta2;
        meta2.fromJson(j, types::zarr_v3);
        EXPECT_EQ(meta2.bytesEndian, "big");
    }


    //
    // Test chunk_key_encoding v2
    //
    TEST(ChunkKeyEncodingTest, V2Encoding) {
        DatasetMetadata meta;
        meta.format = types::zarr_v3;
        meta.dtype = types::float32;
        meta.shape = {100, 100};
        meta.chunkShape = {10, 10};
        meta.compressor = types::raw;
        meta.fillValue = 0;
        meta.zarrDelimiter = "/";

        nlohmann::json j;
        meta.toJson(j);

        // Manually change to v2 encoding to test parsing
        j["chunk_key_encoding"]["name"] = "v2";
        j["chunk_key_encoding"]["configuration"]["separator"] = ".";

        DatasetMetadata meta2;
        meta2.fromJson(j, types::zarr_v3);
        EXPECT_EQ(meta2.zarrDelimiter, ".");
    }


    //
    // Test V3 group creation and detection
    //
    TEST(V3GroupTest, CreateAndDetect) {
        const std::string testDir = "test_v3_group.zarr";
        if(fs::exists(testDir)) fs::remove_all(testDir);

        filesystem::handle::File fileHandle(testDir, FileMode(FileMode::modes::w));
        fileHandle.create();
        Metadata fmeta(types::zarr_v3);
        filesystem::writeMetadata(fileHandle, fmeta);

        // Check that zarr.json was created
        EXPECT_TRUE(fs::exists(testDir + "/zarr.json"));

        // Verify format detection
        EXPECT_EQ(fileHandle.fileFormat(), types::zarr_v3);

        // Create a group
        filesystem::handle::Group group(fileHandle, "mygroup");
        group.create();
        Metadata gmeta(types::zarr_v3);
        filesystem::writeMetadata(group, gmeta);

        // Verify group detection
        EXPECT_TRUE(fs::exists(testDir + "/mygroup/zarr.json"));
        EXPECT_EQ(group.fileFormat(), types::zarr_v3);

        fs::remove_all(testDir);
    }


    //
    // Test V3 attributes in zarr.json
    //
    TEST(V3AttributesTest, GroupAttributesInZarrJson) {
        const std::string testDir = "test_v3_attrs.zarr";
        if(fs::exists(testDir)) fs::remove_all(testDir);

        filesystem::handle::File fileHandle(testDir, FileMode(FileMode::modes::w));
        fileHandle.create();
        Metadata fmeta(types::zarr_v3);
        filesystem::writeMetadata(fileHandle, fmeta);

        // Write attributes
        nlohmann::json attrs;
        attrs["description"] = "test dataset";
        attrs["version"] = 42;
        writeAttributes(fileHandle, attrs);

        // Read them back
        nlohmann::json readAttrs;
        readAttributes(fileHandle, readAttrs);
        EXPECT_EQ(readAttrs["description"], "test dataset");
        EXPECT_EQ(readAttrs["version"], 42);

        // Verify they're embedded in zarr.json
        nlohmann::json root;
        std::ifstream f(testDir + "/zarr.json");
        f >> root;
        f.close();
        EXPECT_TRUE(root.find("attributes") != root.end());
        EXPECT_EQ(root["attributes"]["description"], "test dataset");

        // Remove an attribute
        removeAttribute(fileHandle, "description");
        nlohmann::json readAttrs2;
        readAttributes(fileHandle, readAttrs2);
        EXPECT_TRUE(readAttrs2.find("description") == readAttrs2.end());
        EXPECT_EQ(readAttrs2["version"], 42);

        fs::remove_all(testDir);
    }

    TEST(V3AttributesTest, DatasetAttributesInZarrJson) {
        const std::string testDir = "test_v3_ds_attrs.zarr";
        if(fs::exists(testDir)) fs::remove_all(testDir);

        filesystem::handle::File fileHandle(testDir, FileMode(FileMode::modes::w));
        fileHandle.create();
        Metadata fmeta(types::zarr_v3);
        filesystem::writeMetadata(fileHandle, fmeta);

        filesystem::handle::Dataset dsHandle(fileHandle, "data", "/");
        DatasetMetadata meta;
        meta.format = types::zarr_v3;
        meta.dtype = types::float32;
        meta.shape = {10, 10};
        meta.chunkShape = {5, 5};
        meta.compressor = types::raw;
        meta.fillValue = 0;
        meta.zarrDelimiter = "/";
        dsHandle.create();
        filesystem::writeMetadata(dsHandle, meta);

        // Write attributes to dataset
        nlohmann::json attrs;
        attrs["units"] = "meters";
        writeAttributes(dsHandle, attrs);

        // Read them back
        nlohmann::json readAttrs;
        readAttributes(dsHandle, readAttrs);
        EXPECT_EQ(readAttrs["units"], "meters");

        // Verify embedded in zarr.json alongside array metadata
        nlohmann::json root;
        std::ifstream f(testDir + "/data/zarr.json");
        f >> root;
        f.close();
        EXPECT_EQ(root["node_type"], "array");
        EXPECT_EQ(root["attributes"]["units"], "meters");

        fs::remove_all(testDir);
    }


    //
    // Test reading a real zarr-python v3 sharded dataset
    //
    class RealV3DatasetTest : public ::testing::Test {
    protected:
        const std::string zarrPath = "/run/media/forrest/fdf1f12c-41ad-404a-bf97-0678aeadbcc8/scroll4.volpkg/volumes/PHerc332-20231117143551_masked_v3.zarr";

        bool datasetExists() const {
            return fs::exists(zarrPath + "/zarr.json");
        }
    };

    TEST_F(RealV3DatasetTest, OpenAndReadLevel0) {
        if(!datasetExists()) { GTEST_SKIP() << "V3 dataset not found"; }

        filesystem::handle::File file(zarrPath);
        auto ds = openDataset(file, "0");

        EXPECT_EQ(ds->shape()[0], 9777);
        EXPECT_EQ(ds->shape()[1], 3550);
        EXPECT_EQ(ds->shape()[2], 3400);
        EXPECT_EQ(ds->getDtype(), types::uint8);
        EXPECT_TRUE(ds->isSharded());
        EXPECT_EQ(ds->defaultChunkShape(0), 32);
        EXPECT_EQ(ds->defaultChunkShape(1), 32);
        EXPECT_EQ(ds->defaultChunkShape(2), 32);
    }

    TEST_F(RealV3DatasetTest, ReadLevel0OriginChunk) {
        if(!datasetExists()) { GTEST_SKIP() << "V3 dataset not found"; }

        filesystem::handle::File file(zarrPath);
        auto ds = openDataset(file, "0");

        // Read chunk (0,0,0) — all zeros (masked region), chunk may be empty/missing
        // If the chunk doesn't exist in the shard, readChunk returns false
        types::ShapeType chunkId({0, 0, 0});
        std::vector<uint8_t> data(32 * 32 * 32, 0);
        bool exists = ds->readChunk(chunkId, data.data());
        // Whether the chunk exists or not, data should be zero (fill value)
        for(int i = 0; i < 64; ++i) {
            EXPECT_EQ(data[i], 0) << "index " << i;
        }
    }

    TEST_F(RealV3DatasetTest, ReadLevel0MiddleChunk) {
        if(!datasetExists()) { GTEST_SKIP() << "V3 dataset not found"; }

        filesystem::handle::File file(zarrPath);
        auto ds = openDataset(file, "0");

        // Read the chunk that contains voxel (4888, 1775, 1700)
        // chunk indices: 4888/32=152, 1775/32=55, 1700/32=53
        // local offset within chunk: 4888%32=24, 1775%32=15, 1700%32=4
        types::ShapeType chunkId({152, 55, 53});
        std::vector<uint8_t> data(32 * 32 * 32, 0);
        ds->readChunk(chunkId, data.data());

        // C-order index for (24,15,4): 24*1024 + 15*32 + 4 = 25060
        uint8_t expected_first_row[] = {109, 108, 102, 94};
        for(int i = 0; i < 4; ++i) {
            EXPECT_EQ(data[24 * 1024 + 15 * 32 + 4 + i], expected_first_row[i]) << "index " << i;
        }
    }

    TEST_F(RealV3DatasetTest, ReadLevel5) {
        if(!datasetExists()) { GTEST_SKIP() << "V3 dataset not found"; }

        filesystem::handle::File file(zarrPath);
        auto ds = openDataset(file, "5");

        EXPECT_EQ(ds->shape()[0], 306);
        EXPECT_EQ(ds->shape()[1], 111);
        EXPECT_EQ(ds->shape()[2], 107);

        // Read chunk containing voxel (153, 55, 53)
        // chunk indices: 153/32=4, 55/32=1, 53/32=1
        // local: 153%32=25, 55%32=23, 53%32=21
        types::ShapeType chunkId({4, 1, 1});
        std::vector<uint8_t> data(32 * 32 * 32, 0);
        ds->readChunk(chunkId, data.data());

        // Python ref for level 5 mid [153:157, 55:59, 53:57]:
        // first value at local (25, 23, 21) = 84
        EXPECT_EQ(data[25 * 1024 + 23 * 32 + 21], 84);
        EXPECT_EQ(data[25 * 1024 + 23 * 32 + 22], 71);
        EXPECT_EQ(data[25 * 1024 + 23 * 32 + 23], 60);
        EXPECT_EQ(data[25 * 1024 + 23 * 32 + 24], 62);
    }

} // namespace z5
