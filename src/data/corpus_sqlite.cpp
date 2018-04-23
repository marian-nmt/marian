#include <random>

#include "data/corpus_sqlite.h"

namespace marian {
namespace data {

CorpusSQLite::CorpusSQLite(Ptr<Config> options, bool translate /*= false*/)
    : CorpusBase(options, translate), seed_(Config::seed) {
  fillSQLite();
}

CorpusSQLite::CorpusSQLite(std::vector<std::string> paths,
                           std::vector<Ptr<Vocab>> vocabs,
                           Ptr<Config> options)
    : CorpusBase(paths, vocabs, options), seed_(Config::seed) {
  fillSQLite();
}

void CorpusSQLite::fillSQLite() {
  auto tempDir = options_->get<std::string>("tempdir");
  bool fill = false;

  // create a temporary or persistent SQLite database
  if(options_->get<std::string>("sqlite") == "temporary") {
    LOG(info, "[sqlite] Creating temporary database in {}", tempDir);

    db_.reset(
        new SQLite::Database("", SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE));
    db_->exec("PRAGMA temp_store_directory = '" + tempDir + "';");

    fill = true;
  } else {
    auto path = options_->get<std::string>("sqlite");

    if(boost::filesystem::exists(path)) {
      LOG(info, "[sqlite] Reusing persistent database {}", path);

      db_.reset(new SQLite::Database(path, SQLite::OPEN_READWRITE));
      db_->exec("PRAGMA temp_store_directory = '" + tempDir + "';");

      if(options_->get<bool>("sqlite-drop")) {
        LOG(info, "[sqlite] Dropping previous data");
        db_->exec("drop table if exists lines");
        fill = true;
      }
    } else {
      LOG(info, "[sqlite] Creating persistent database {}", path);

      db_.reset(new SQLite::Database(
          path, SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE));
      db_->exec("PRAGMA temp_store_directory = '" + tempDir + "';");

      fill = true;
    }
  }

  // populate tables with lines from text files
  if(fill) {
    std::string createStr = "create table lines (_id integer";
    std::string insertStr = "insert into lines values (?";
    for(int i = 0; i < files_.size(); ++i) {
      createStr += ", line" + std::to_string(i) + " text";
      insertStr += ", ?";
    }
    createStr += ");";
    insertStr += ");";

    db_->exec(createStr);

    SQLite::Statement ps(*db_, insertStr);

    int lines = 0;
    int report = 1000000;
    bool cont = true;

    db_->exec("begin;");
    while(cont) {
      ps.bind(1, (int)lines);

      std::string line;
      for(int i = 0; i < files_.size(); ++i) {
        cont = cont && std::getline((std::istream&)*files_[i], line);
        if(cont)
          ps.bind(i + 2, line);
      }

      if(cont) {
        ps.exec();
        ps.reset();
      }
      lines++;

      if(lines % report == 0) {
        LOG(info, "[sqlite] Inserted {} lines", lines);
        db_->exec("commit;");
        db_->exec("begin;");
        report *= 2;
      }
    }
    db_->exec("commit;");
    LOG(info, "[sqlite] Inserted {} lines", lines);
    LOG(info, "[sqlite] Creating primary index");
    db_->exec("create unique index idx_line on lines (_id);");
  }

  createRandomFunction();
}

SentenceTuple CorpusSQLite::next() {
  while(select_->executeStep()) {
    // fill up the sentence tuple with sentences from all input files
    size_t curId = select_->getColumn(0).getInt();
    SentenceTuple tup(curId);

    for(size_t i = 0; i < files_.size(); ++i) {
      auto line = select_->getColumn(i + 1);

      if(i > 0 && i == alignFileIdx_) {
        addAlignmentToSentenceTuple(line, tup);
      } else if(i > 0 && i == weightFileIdx_) {
        addWeightsToSentenceTuple(line, tup);
      } else {
        addWordsToSentenceTuple(line, i, tup);
      }
    }

    if(std::all_of(tup.begin(), tup.end(), [=](const Words& words) {
         return words.size() > 0 && words.size() <= maxLength_;
       }))
      return tup;
  }
  return SentenceTuple(0);
}

void CorpusSQLite::shuffle() {
  LOG(info, "[sqlite] Selecting shuffled data");
  select_.reset(new SQLite::Statement(
      *db_,
      "select * from lines order by random_seed(" + std::to_string(seed_)
          + ");"));
}

void CorpusSQLite::reset() {
  select_.reset(
      new SQLite::Statement(*db_, "select * from lines order by _id;"));
}

void CorpusSQLite::restore(Ptr<TrainingState> ts) {
  for(size_t i = 0; i < ts->epochs - 1; ++i) {
    select_.reset(new SQLite::Statement(
        *db_,
        "select _id from lines order by random_seed(" + std::to_string(seed_)
            + ");"));
    select_->executeStep();
    reset();
  }
}
}
}
