#include "catch.hpp"
#include "common/binary.h"
#include "common/file_stream.h"

#include "3rd_party/mio/mio.hpp"

using namespace marian;

TEST_CASE("a few operations on binary files", "[binary]") {

  SECTION("Save two items to temporary binary file and then load and map") {

    // Create a temporary file that we will only use for the file name
    io::TemporaryFile temp("/tmp/", /*earlyUnlink=*/false);
    io::Item item1, item2;

    {
      std::vector<float>    v1 = { 3.14, 2.71, 1.0, 0.0, 1.41 };
      std::vector<uint16_t> v2 = { 5, 4, 3, 2, 1, 0 };

      item1.name  = "item1";
      item1.shape = { 5, 1 };
      item1.type  = Type::float32;
      item1.bytes.resize(v1.size() * sizeof(float));
      std::copy((char*)v1.data(), (char*)v1.data() + v1.size() * sizeof(float), item1.bytes.data());

      item2.name  = "item2";
      item2.shape = { 2, 3 };
      item2.type = Type::uint16;
      item2.bytes.resize(v2.size() * sizeof(uint32_t));
      std::copy((char*)v2.data(), (char*)v2.data() + v2.size() * sizeof(uint16_t), item2.bytes.data());
      
      std::vector<io::Item> items = {item1, item2};
      io::binary::saveItems(temp.getFileName(), items);
    }

    { // test loading
      std::vector<io::Item> items;
      io::binary::loadItems(temp.getFileName(), items);

      CHECK( item1.name == items[0].name );
      CHECK( item2.name == items[1].name );
    
      CHECK( std::equal(item1.data(), item1.data() + item1.size(), items[0].data()) );
      CHECK( std::equal(item2.data(), item2.data() + item2.size(), items[1].data()) );
    }
  
    { // test mmapping
      mio::mmap_source mmap(temp.getFileName());
      
      std::vector<io::Item> items;
      io::binary::loadItems(mmap.data(), items, /*mapped=*/true);

      CHECK( item1.name == items[0].name );
      CHECK( item2.name == items[1].name );

      CHECK( std::equal(item1.data(), item1.data() + item1.size(), items[0].data()) );
      CHECK( std::equal(item2.data(), item2.data() + item2.size(), items[1].data()) );
    }
  }
}
