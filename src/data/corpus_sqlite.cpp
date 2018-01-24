#include <random>

#include "data/corpus_sqlite.h"

namespace marian {
namespace data {

CorpusSQLite::CorpusSQLite(Ptr<Config> options, bool translate)
    : options_(options),
      maxLength_(options_->get<size_t>("max-length")),
      maxLengthCrop_(options_->get<bool>("max-length-crop")),
      rightLeft_(options_->get<bool>("right-left")) {
        
  bool training = !translate;

  if(training)
    paths_ = options_->get<std::vector<std::string>>("train-sets");
  else
    paths_ = options_->get<std::vector<std::string>>("input");

  std::vector<std::string> vocabPaths;
  if(options_->has("vocabs"))
    vocabPaths = options_->get<std::vector<std::string>>("vocabs");

  if(training) {
    ABORT_IF(!vocabPaths.empty() && paths_.size() != vocabPaths.size(),
             "Number of corpus files and vocab files does not agree");
  }

  std::vector<int> maxVocabs = options_->get<std::vector<int>>("dim-vocabs");

  if(training) { // training or scoring
    std::vector<Vocab> vocabs;

    if(vocabPaths.empty()) {
      if(maxVocabs.size() < paths_.size())
        maxVocabs.resize(paths_.size(), 0);

      // Create vocabs if not provided
      for(size_t i = 0; i < paths_.size(); ++i) {
        Ptr<Vocab> vocab = New<Vocab>();
        int vocSize = vocab->loadOrCreate("", paths_[i], maxVocabs[i]);
        LOG(info,
            "[data] Setting vocabulary size for input {} to {}",
            i,
            vocSize);
        options_->get()["dim-vocabs"][i] = vocSize;

        options_->get()["vocabs"].push_back(paths_[i] + ".yml");
        vocabs_.emplace_back(vocab);
      }
    } else {
      // Load all vocabs
      if(maxVocabs.size() < vocabPaths.size())
        maxVocabs.resize(paths_.size(), 0);

      for(size_t i = 0; i < vocabPaths.size(); ++i) {
        Ptr<Vocab> vocab = New<Vocab>();
        int vocSize
            = vocab->loadOrCreate(vocabPaths[i], paths_[i], maxVocabs[i]);
        LOG(info,
            "[data] Setting vocabulary size for input {} to {}",
            i,
            vocSize);
        options_->get()["dim-vocabs"][i] = vocSize;

        vocabs_.emplace_back(vocab);
      }
    }
  } else {  // i.e., if translating
    ABORT_IF(vocabPaths.empty(), "Translating, but vocabularies are not given!");

    if(maxVocabs.size() < vocabPaths.size())
      maxVocabs.resize(paths_.size(), 0);

    for(size_t i = 0; i + 1 < vocabPaths.size(); ++i) {
      Ptr<Vocab> vocab = New<Vocab>();
      int vocSize = vocab->load(vocabPaths[i], maxVocabs[i]);
      LOG(info,
          "[data] Setting vocabulary size for input {} to {}",
          i,
          vocSize);
      options_->get()["dim-vocabs"][i] = vocSize;

      vocabs_.emplace_back(vocab);
    }
  }
  
  for(auto path : paths_) {
    if(path == "stdin")
      files_.emplace_back(new InputFileStream(std::cin));
    else {
      files_.emplace_back(new InputFileStream(path));
      ABORT_IF(files_.back()->empty(), "File '{}' is empty", path);
    }
  }

  if(training) {
    ABORT_IF(vocabs_.size() != files_.size(),
             "Number of corpus files ({}) and vocab files ({}) does not agree",
             files_.size(), vocabs_.size());
  }
  else {
    ABORT_IF(vocabs_.size() != files_.size(),
             "Number of input files ({}) and input vocab files ({}) does not agree",
             files_.size(), vocabs_.size());
  }
  
  fillSQLite();
}

CorpusSQLite::CorpusSQLite(std::vector<std::string> paths,
                           std::vector<Ptr<Vocab>> vocabs,
                           Ptr<Config> options,
                           size_t maxLength)
    : CorpusBase(paths),
      options_(options),
      vocabs_(vocabs),
      maxLength_(maxLength ? maxLength : options_->get<size_t>("max-length")),
      maxLengthCrop_(options_->get<bool>("max-length-crop")),
      rightLeft_(options_->get<bool>("right-left")) {
  ABORT_IF(paths_.size() != vocabs_.size(),
           "Number of corpus files and vocab files does not agree");

  for(auto path : paths_) {
    files_.emplace_back(new InputFileStream(path));
  }
  
  fillSQLite();
}

void CorpusSQLite::fillSQLite() {
  LOG(info, "[sqlite] Creating temporary database in {}", options_->get<std::string>("tempdir"));
  db_.reset(new SQLite::Database("", SQLite::OPEN_READWRITE|SQLite::OPEN_CREATE));
  db_->exec("PRAGMA temp_store_directory = '" + options_->get<std::string>("tempdir") + "';");  
  db_->exec("drop table if exists lines");
  
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
      
      if(lines % 1000000 == 0) {
        LOG(info, "[sqlite] Inserted {} lines", lines);
        db_->exec("commit;");
        db_->exec("begin;");
      }
  }
  db_->exec("commit;");
  LOG(info, "[sqlite] Inserted {} lines", lines);
  LOG(info, "[sqlite] Creating primary index");
  db_->exec("create unique index idx_line on lines (_id);");
}

SentenceTuple CorpusSQLite::next() {
  while(select_->executeStep()) {
    // get index of the current sentence
    pos_++;

    // fill up the sentence tuple with sentences from all input files
    
    size_t curId = select_->getColumn(0).getInt();
    SentenceTuple tup(curId);
    
    for(size_t i = 0; i < files_.size(); ++i) {
      std::string line;
      Words words = (*vocabs_[i])(select_->getColumn(i + 1));

      if(words.empty())
        words.push_back(0);

      if(maxLengthCrop_ && words.size() > maxLength_) {
        words.resize(maxLength_);
        words.back() = 0;
      }
      
      if(rightLeft_)
        std::reverse(words.begin(), words.end() - 1);

      tup.push_back(words);
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
  select_.reset(new SQLite::Statement(*db_, "select * from lines order by random();"));
}

void CorpusSQLite::reset() {
  pos_ = 0;
  select_.reset(new SQLite::Statement(*db_, "select * from lines order by _id;"));
}

}
}
