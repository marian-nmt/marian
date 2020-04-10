#include "catch.hpp"
#include "common/utils.h"

using namespace marian;

TEST_CASE("utils::splitTsv", "[utils]") {
  std::string line1 = "foo bar";
  std::string line2 = "foo bar\tbazz";
  std::string line3 = "foo bar\tbazz\tfoo quux";

  std::vector<std::string> fields;

  SECTION("the tab-separated input is split") {
    utils::splitTsv(line1, fields, 1);
    CHECK( fields.size() == 1 );
    CHECK( fields[0] == "foo bar" );

    utils::splitTsv(line3, fields, 3);
    CHECK( fields == std::vector<std::string>({"foo bar", "bazz", "foo quux"}) );
  }

  SECTION("the output has at least as many elements as requested") {
    utils::splitTsv(line1, fields, 1);
    CHECK( fields.size() == 1 );

    utils::splitTsv(line1, fields, 3);
    CHECK( fields.size() == 3 );
    CHECK( fields == std::vector<std::string>({"foo bar", "", ""}) );

    utils::splitTsv(line1, fields, 2);
    CHECK( fields.size() == 2 );
    CHECK( fields == std::vector<std::string>({"foo bar", ""}) );
  }

  //SECTION("excessive tab-separated fields abort the execution") {}
}
