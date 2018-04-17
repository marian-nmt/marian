#include <cstdlib>
#include <iostream>
#include <string>
#include <boost/timer/timer.hpp>
#include <boost/thread/tss.hpp>
#include <boost/python.hpp>

#include "common/god.h"
#include "common/logging.h"
#include "common/threadpool.h"
#include "common/search.h"
#include "common/printer.h"
#include "common/sentence.h"
#include "common/sentences.h"
#include "common/exception.h"
#include "common/translation_task.h"

using namespace amunmt;
using namespace std;

God god_;

void init(const std::string& options) {
  god_.Init(options);
}


boost::python::list translate(boost::python::list& in)
{
  size_t miniSize = god_.Get<size_t>("mini-batch");
  size_t maxiSize = god_.Get<size_t>("maxi-batch");
  int miniWords = god_.Get<int>("mini-batch-words");

  std::vector<std::future< std::shared_ptr<Histories> >> results;
  SentencesPtr maxiBatch(new Sentences());
  SentencesPtr maxiBatchCopy(new Sentences());

  for(int lineNum = 0; lineNum < boost::python::len(in); ++lineNum) {
    std::string line = boost::python::extract<std::string>(boost::python::object(in[lineNum]));
    //cerr << "line=" << line << endl;

    maxiBatch->push_back(SentencePtr(new Sentence(god_, lineNum, line)));

    if (maxiBatch->size() >= maxiSize) {

      maxiBatch->SortByLength();
      while (maxiBatch->size()) {
        SentencesPtr miniBatch = maxiBatch->NextMiniBatch(miniSize, miniWords);
        maxiBatchCopy->push_back(miniBatch->at(0));

        results.emplace_back(
          god_.GetThreadPool().enqueue(
              [miniBatch]{ return TranslationTask(::god_, miniBatch); }
              )
        );
      }

      maxiBatch.reset(new Sentences());
    }
  }

  // last batch
  if (maxiBatch->size()) {
    maxiBatch->SortByLength();
    while (maxiBatch->size()) {
      SentencesPtr miniBatch = maxiBatch->NextMiniBatch(miniSize, miniWords);
      maxiBatchCopy->push_back(miniBatch->at(0));
      results.emplace_back(
        god_.GetThreadPool().enqueue(
            [miniBatch]{ return TranslationTask(::god_, miniBatch); }
            )
      );
    }
  }

  // resort batch into line number order
  Histories allHistories;
  for (auto&& result : results) {
    std::shared_ptr<Histories> histories = result.get();
    allHistories.Append(*histories);
  }
  allHistories.SortByLineNum();

  // output
  boost::python::list output;
  for (size_t i = 0; i < allHistories.size(); ++i) {
    const History& history = *allHistories.at(i).get();
    const Sentence& sentence = *maxiBatchCopy->at(i).get();
    std::stringstream ss;
    Printer(god_, history, ss, sentence);
    string str = ss.str();

    output.append(str);
  }

  return output;
}

BOOST_PYTHON_MODULE(libamunmt)
{
  boost::python::def("init", init);
  boost::python::def("translate", translate);
}
