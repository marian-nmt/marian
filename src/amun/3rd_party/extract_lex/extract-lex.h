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

#pragma once

#include <map>
#include <set>
#include <sstream>
#include <fstream>
#include <iostream>

namespace extract {

class WordCount
{
  friend std::ostream& operator<<(std::ostream&, const WordCount&);
public:
  float m_count;

  std::map<const std::string*, WordCount> m_coll;

  WordCount()
    :m_count(0) {
  }

  //WordCount(const WordCount &copy);

  WordCount(float count)
    :m_count(count) {
  }

  void AddCount(float incr);

  std::map<const std::string*, WordCount> &GetColl() {
    return m_coll;
  }
  const std::map<const std::string*, WordCount> &GetColl() const {
    return m_coll;
  }

  const float GetCount() const {
    return m_count;
  }

};

class Vocab
{
  std::set<std::string> m_coll;
public:
  const std::string *GetOrAdd(const std::string &word);
};

class ExtractLex
{
  Vocab m_vocab;
  std::map<const std::string*, WordCount> m_collS2T, m_collT2S;

  void Process(const std::string *target, const std::string *source);
  void Process(WordCount &wcIn, const std::string *out);
  void ProcessUnaligned(std::vector<std::string> &toksTarget, std::vector<std::string> &toksSource
                        , const std::vector<bool> &m_sourceAligned, const std::vector<bool> &m_targetAligned);

  void Output(const std::map<const std::string*, WordCount> &coll, std::ofstream &outStream);

public:
  void Process(std::vector<std::string> &toksTarget, std::vector<std::string> &toksSource, std::vector<std::string> &toksAlign, size_t lineCount);
  void Output(std::ofstream &streamLexS2T, std::ofstream &streamLexT2S);

};

}
