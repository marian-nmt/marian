/***********************************************************************                                                                                                                 
Moses - factored phrase-based language decoder                                                                                                                                           
Copyright (C) 2010- University of Edinburgh                                                                                                                                              
                                                                                                                                                                                         
This library is free software; you can redistribute it and/or                                                                                                                            
modify it under the terms of the GNU Lesser General Public                                                                                                                               
License as published by the Free Software Foundation; either                                                                                                                             
version 2.1 of the License, or (at your option) any later version.                                                                                                                       
                                                                                                                                                                                         
This library is distributed in the hope that it will be useful,                                                                                                                          
but WITHOUT ANY WARRANTY; without even the implied warranty of                                                                                                                           
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU                                                                                                                        
Lesser General Public License for more details.                                                                                                                                          
                                                                                                                                                                                         
You should have received a copy of the GNU Lesser General Public                                                                                                                         
License along with this library; if not, write to the Free Software                                                                                                                      
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA                                                                                                           
***********************************************************************/  

#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <algorithm>

#include "file_stream.h"
#include "utils.h"
#include "extract-lex.h"

float COUNT_INCR = 1;

void fix(std::ostream& stream)
{
  stream.setf(std::ios::fixed);
  stream.precision(7);
}

int main(int argc, char* argv[])
{
  std::cerr << "Starting...\n";

  assert(argc == 6);
  char* &filePathTarget = argv[1];
  char* &filePathSource = argv[2];
  char* &filePathAlign  = argv[3];
  char* &filePathLexS2T = argv[4];
  char* &filePathLexT2S = argv[5];

  InputFileStream streamTarget(filePathTarget);
  InputFileStream streamSource(filePathSource);
  InputFileStream streamAlign(filePathAlign);

  std::ofstream streamLexS2T;
  std::ofstream streamLexT2S;
  streamLexS2T.open(filePathLexS2T);
  streamLexT2S.open(filePathLexT2S);

  fix(streamLexS2T);
  fix(streamLexT2S);

  extract::ExtractLex extractSingleton;

  size_t lineCount = 0;
  std::string lineTarget, lineSource, lineAlign;
  while (std::getline((std::istream&)streamTarget, lineTarget)) {
    if (lineCount % 10000 == 0)
      std::cerr << lineCount << " ";

    std::istream &isSource = std::getline((std::istream&)streamSource, lineSource);
    assert(isSource);
    std::istream &isAlign = std::getline((std::istream&)streamAlign, lineAlign);
    assert(isAlign);

    std::vector<std::string> toksTarget, toksSource, toksAlign;
    Split(lineTarget, toksTarget, " ");
    Split(lineSource, toksSource, " ");
    Split(lineAlign, toksAlign, " ");

    extractSingleton.Process(toksTarget, toksSource, toksAlign, lineCount);

    ++lineCount;
  }

  extractSingleton.Output(streamLexS2T, streamLexT2S);

  streamLexS2T.close();
  streamLexT2S.close();

  std::cerr << "\nFinished\n";
}

const std::string *extract::Vocab::GetOrAdd(const std::string &word)
{
  const std::string *ret = &(*m_coll.insert(word).first);
  return ret;
}

void extract::ExtractLex::Process(std::vector<std::string> &toksTarget,
                         std::vector<std::string> &toksSource,
                         std::vector<std::string> &toksAlign,
                         size_t lineCount)
{
  std::vector<bool> m_sourceAligned(toksSource.size(), false)
  , m_targetAligned(toksTarget.size(), false);

  std::vector<std::string>::const_iterator iterAlign;
  for (iterAlign = toksAlign.begin(); iterAlign != toksAlign.end(); ++iterAlign) {
    const std::string &alignTok = *iterAlign;

    std::vector<std::string> alignPosStr;
    Split(alignTok, alignPosStr, "-");
    std::vector<size_t> alignPos(alignPosStr.size());
    std::transform(alignPosStr.begin(), alignPosStr.end(),
                   alignPos.begin(),
                   [](const std::string& str) { return std::stoi(str); });

    assert(alignPos.size() == 2);

    if (alignPos[0] >= toksSource.size()) {
      std::cerr << "ERROR: alignment over source length. Alignment " << alignPos[0] << " at line " << lineCount << std::endl;
      continue;
    }
    if (alignPos[1] >= toksTarget.size()) {
      std::cerr << "ERROR: alignment over target length. Alignment " << alignPos[1] << " at line " << lineCount << std::endl;
      continue;
    }

    assert(alignPos[0] < toksSource.size());
    assert(alignPos[1] < toksTarget.size());

    m_sourceAligned[ alignPos[0] ] = true;
    m_targetAligned[ alignPos[1] ] = true;

    const std::string &tmpSource = toksSource[ alignPos[0] ];
    const std::string &tmpTarget = toksTarget[ alignPos[1] ];

    const std::string *source = m_vocab.GetOrAdd(tmpSource);
    const std::string *target = m_vocab.GetOrAdd(tmpTarget);

    Process(target, source);

  }

  ProcessUnaligned(toksTarget, toksSource, m_sourceAligned, m_targetAligned);
}

void extract::ExtractLex::Process(const std::string *target, const std::string *source)
{
  extract::WordCount &wcS2T = m_collS2T[source];
  extract::WordCount &wcT2S = m_collT2S[target];

  wcS2T.AddCount(COUNT_INCR);
  wcT2S.AddCount(COUNT_INCR);

  Process(wcS2T, target);
  Process(wcT2S, source);
}

void extract::ExtractLex::Process(extract::WordCount &wcIn, const std::string *out)
{
  std::map<const std::string*, extract::WordCount> &collOut = wcIn.GetColl();
  extract::WordCount &wcOut = collOut[out];
  wcOut.AddCount(COUNT_INCR);
}

void extract::ExtractLex::ProcessUnaligned(std::vector<std::string> &toksTarget,
                                  std::vector<std::string> &toksSource
                                  , const std::vector<bool> &m_sourceAligned, const std::vector<bool> &m_targetAligned)
{
  const std::string *nullWord = m_vocab.GetOrAdd("NULL");

  for (size_t pos = 0; pos < m_sourceAligned.size(); ++pos) {
    bool isAlignedCurr = m_sourceAligned[pos];
    if (!isAlignedCurr) {
      const std::string &tmpWord = toksSource[pos];
      const std::string *sourceWord = m_vocab.GetOrAdd(tmpWord);

      Process(nullWord, sourceWord);
    }
  }

  for (size_t pos = 0; pos < m_targetAligned.size(); ++pos) {
    bool isAlignedCurr = m_targetAligned[pos];
    if (!isAlignedCurr) {
      const std::string &tmpWord = toksTarget[pos];
      const std::string *targetWord = m_vocab.GetOrAdd(tmpWord);

      Process(targetWord, nullWord);
    }
  }
}

void extract::ExtractLex::Output(std::ofstream &streamLexS2T, std::ofstream &streamLexT2S)
{
  Output(m_collS2T, streamLexS2T);
  Output(m_collT2S, streamLexT2S);
}

void extract::ExtractLex::Output(const std::map<const std::string*, extract::WordCount> &coll, std::ofstream &outStream)
{
  std::map<const std::string*, extract::WordCount>::const_iterator iterOuter;
  for (iterOuter = coll.begin(); iterOuter != coll.end(); ++iterOuter) {
    const std::string &inStr = *iterOuter->first;
    const extract::WordCount &inWC = iterOuter->second;

    const std::map<const std::string*, extract::WordCount> &outColl = inWC.GetColl();

    std::map<const std::string*, extract::WordCount>::const_iterator iterInner;
    for (iterInner = outColl.begin(); iterInner != outColl.end(); ++iterInner) {
      const std::string &outStr = *iterInner->first;
      const extract::WordCount &outWC = iterInner->second;

      float prob = outWC.GetCount() / inWC.GetCount();
      outStream << outStr << " "  << inStr << " " << prob << std::endl;
    }
  }
}

std::ostream& operator<<(std::ostream &out, const extract::WordCount &obj)
{
  out << "(" << obj.GetCount() << ")";
  return out;
}

void extract::WordCount::AddCount(float incr)
{
  m_count += incr;
}
