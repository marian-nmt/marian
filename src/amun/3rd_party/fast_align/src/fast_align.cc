// Copyright 2013 by Chris Dyer
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <utility>
#include <fstream>
#include <getopt.h>
#include <sstream>

#include "src/corpus.h"
#include "src/ttables.h"
#include "src/da.h"

using namespace std;

struct PairHash {
  size_t operator()(const pair<short, short>& x) const {
    return (unsigned short) x.first << 16 | (unsigned) x.second;
  }
};

Dict d; // integerization map

void ParseLine(const string& line,
               vector<unsigned>* src,
               vector<unsigned>* trg) {
  static const unsigned kDIV = d.Convert("|||");
  vector<unsigned> tmp;
  src->clear();
  trg->clear();
  d.ConvertWhitespaceDelimitedLine(line, &tmp);
  unsigned i = 0;
  while (i < tmp.size() && tmp[i] != kDIV) {
    src->push_back(tmp[i]);
    ++i;
  }
  if (i < tmp.size() && tmp[i] == kDIV) {
    ++i;
    for (; i < tmp.size(); ++i)
      trg->push_back(tmp[i]);
  }
}

string input;
string conditional_probability_filename = "";
string input_model_file = "";
double mean_srclen_multiplier = 1.0;
int is_reverse = 0;
int ITERATIONS = 5;
int favor_diagonal = 0;
double beam_threshold = -4.0;
double prob_align_null = 0.08;
double diagonal_tension = 4.0;
int optimize_tension = 0;
int variational_bayes = 0;
double alpha = 0.01;
int no_null_word = 0;
size_t thread_buffer_size = 10000;
bool force_align = false;
int print_scores = 0;
struct option options[] = {
    {"input",             required_argument, 0,                  'i'},
    {"reverse",           no_argument,       &is_reverse,        1  },
    {"iterations",        required_argument, 0,                  'I'},
    {"favor_diagonal",    no_argument,       &favor_diagonal,    1  },
    {"force_align",       required_argument, 0,                  'f'},
    {"mean_srclen_multiplier", required_argument, 0,             'm'},
    {"beam_threshold",    required_argument, 0,                  't'},
    {"p0",                required_argument, 0,                  'q'},
    {"diagonal_tension",  required_argument, 0,                  'T'},
    {"optimize_tension",  no_argument,       &optimize_tension,  1  },
    {"variational_bayes", no_argument,       &variational_bayes, 1  },
    {"alpha",             required_argument, 0,                  'a'},
    {"no_null_word",      no_argument,       &no_null_word,      1  },
    {"conditional_probabilities", required_argument, 0,          'p'},
    {"thread_buffer_size", required_argument, 0,                 'b'},
    {0,0,0,0}
};

bool InitCommandLine(int argc, char** argv) {
  while (1) {
    int oi;
    int c = getopt_long(argc, argv, "i:rI:df:m:t:q:T:ova:Np:b:s", options, &oi);
    if (c == -1) break;
    cerr << "ARG=" << (char)c << endl;
    switch(c) {
      case 'i': input = optarg; break;
      case 'r': is_reverse = 1; break;
      case 'I': ITERATIONS = atoi(optarg); break;
      case 'd': favor_diagonal = 1; break;
      case 'f': force_align = 1; conditional_probability_filename = optarg; break;
      case 'm': mean_srclen_multiplier = atof(optarg); break;
      case 't': beam_threshold = atof(optarg); break;
      case 'q': prob_align_null = atof(optarg); break;
      case 'T': favor_diagonal = 1; diagonal_tension = atof(optarg); break;
      case 'o': optimize_tension = 1; break;
      case 'v': variational_bayes = 1; break;
      case 'a': alpha = atof(optarg); break;
      case 'N': no_null_word = 1; break;
      case 'p': conditional_probability_filename = optarg; break;
      case 'b': thread_buffer_size = atoi(optarg); break;
      case 's': print_scores = 1; break;
      default: return false;
    }
  }
  if (input.size() == 0) return false;
  return true;
}

void UpdateFromPairs(const vector<string>& lines, const int lc, const int iter,
    const bool final_iteration, const bool use_null, const unsigned kNULL,
    const double prob_align_not_null, double* c0, double* emp_feat,
    double* likelihood, TTable* s2t, vector<string>* outputs) {
  if (final_iteration) {
    outputs->clear();
    outputs->resize(lines.size());
  }
  double emp_feat_ = 0.0;
  double c0_ = 0.0;
  double likelihood_ = 0.0;
#pragma omp parallel for schedule(dynamic) reduction(+:emp_feat_,c0_,likelihood_)
  for (int line_idx = 0; line_idx < static_cast<int>(lines.size());
      ++line_idx) {
    vector<unsigned> src, trg;
    ParseLine(lines[line_idx], &src, &trg);
    if (is_reverse)
      swap(src, trg);
    if (src.size() == 0 || trg.size() == 0) {
      cerr << "Error in line " << lc << "\n" << lines[line_idx] << endl;
      //return 1;
    }
    ostringstream oss; // collect output in last iteration
    vector<double> probs(src.size() + 1);
    bool first_al = true;  // used when printing alignments
    double local_likelihood = 0.0;
    for (unsigned j = 0; j < trg.size(); ++j) {
      const unsigned& f_j = trg[j];
      double sum = 0;
      double prob_a_i = 1.0 / (src.size() + use_null);  // uniform (model 1)
      if (use_null) {
        if (favor_diagonal)
          prob_a_i = prob_align_null;
        probs[0] = s2t->prob(kNULL, f_j) * prob_a_i;
        sum += probs[0];
      }
      double az = 0;
      if (favor_diagonal)
        az = DiagonalAlignment::ComputeZ(j + 1, trg.size(), src.size(),
            diagonal_tension) / prob_align_not_null;
      for (unsigned i = 1; i <= src.size(); ++i) {
        if (favor_diagonal)
          prob_a_i = DiagonalAlignment::UnnormalizedProb(j + 1, i, trg.size(),
              src.size(), diagonal_tension) / az;
        probs[i] = s2t->prob(src[i - 1], f_j) * prob_a_i;
        sum += probs[i];
      }
      if (final_iteration) {
        double max_p = -1;
        int max_index = -1;
        if (use_null) {
          max_index = 0;
          max_p = probs[0];
        }
        for (unsigned i = 1; i <= src.size(); ++i) {
          if (probs[i] > max_p) {
            max_index = i;
            max_p = probs[i];
          }
        }
        if (max_index > 0) {
          if (first_al)
            first_al = false;
          else
            oss << ' ';
          if (is_reverse)
            oss << j << '-' << (max_index - 1);
          else
            oss << (max_index - 1) << '-' << j;
        }
      } else {
        if (use_null) {
          double count = probs[0] / sum;
          c0_ += count;
          s2t->Increment(kNULL, f_j, count);
        }
        for (unsigned i = 1; i <= src.size(); ++i) {
          const double p = probs[i] / sum;
          s2t->Increment(src[i - 1], f_j, p);
          emp_feat_ += DiagonalAlignment::Feature(j, i, trg.size(), src.size()) * p;
        }
      }
      local_likelihood += log(sum);
    }
    likelihood_ += local_likelihood;
    if (final_iteration) {
      if (print_scores) {
        double log_prob = Md::log_poisson(trg.size(), 0.05 + src.size() * mean_srclen_multiplier);
        log_prob += local_likelihood;
        oss << " ||| " << log_prob;
      }
      oss << endl;
      (*outputs)[line_idx] = oss.str();
    }
  }
  *emp_feat += emp_feat_;
  *c0 += c0_;
  *likelihood += likelihood_;
}

inline void AddTranslationOptions(vector<vector<unsigned> >& insert_buffer,
    TTable* s2t) {
  s2t->SetMaxE(insert_buffer.size()-1);
#pragma omp parallel for schedule(dynamic)
  for (unsigned e = 0; e < insert_buffer.size(); ++e) {
    for (unsigned f : insert_buffer[e]) {
      s2t->Insert(e, f);
    }
    insert_buffer[e].clear();
  }
}

void InitialPass(const unsigned kNULL, const bool use_null, TTable* s2t,
    double* n_target_tokens, double* tot_len_ratio,
    vector<pair<pair<short, short>, unsigned>>* size_counts) {
  ifstream in(input.c_str());
  if (!in) {
    cerr << "Can't read " << input << endl;
  }
  unordered_map<pair<short, short>, unsigned, PairHash> size_counts_;
  vector<vector<unsigned>> insert_buffer;
  size_t insert_buffer_items = 0;
  vector<unsigned> src, trg;
  string line;
  bool flag = false;
  int lc = 0;
  cerr << "INITIAL PASS " << endl;
  while (true) {
    getline(in, line);
    if (!in)
      break;
    lc++;
    if (lc % 1000 == 0) { cerr << '.'; flag = true; }
    if (lc %50000 == 0) { cerr << " [" << lc << "]\n" << flush; flag = false; }
    ParseLine(line, &src, &trg);
    if (is_reverse)
      swap(src, trg);
    if (src.size() == 0 || trg.size() == 0) {
      cerr << "Error in line " << lc << "\n" << line << endl;
    }
    *tot_len_ratio += static_cast<double>(trg.size()) / static_cast<double>(src.size());
    *n_target_tokens += trg.size();
    if (use_null) {
      for (const unsigned f : trg) {
        s2t->Insert(kNULL, f);
      }
    }
    for (const unsigned e : src) {
      if (e >= insert_buffer.size()) {
        insert_buffer.resize(e+1);
      }
      for (const unsigned f : trg) {
        insert_buffer[e].push_back(f);
      }
      insert_buffer_items += trg.size();
    }
    if (insert_buffer_items > thread_buffer_size * 100) {
      insert_buffer_items = 0;
      AddTranslationOptions(insert_buffer, s2t);
    }
    ++size_counts_[make_pair<short, short>(trg.size(), src.size())];
  }
  for (const auto& p : size_counts_) {
    size_counts->push_back(p);
  }
  AddTranslationOptions(insert_buffer, s2t);

  mean_srclen_multiplier = (*tot_len_ratio) / lc;
  if (flag) {
    cerr << endl;
  }
  cerr << "expected target length = source length * " << mean_srclen_multiplier << endl;
}

int main(int argc, char** argv) {
  if (!InitCommandLine(argc, argv)) {
    cerr << "Usage: " << argv[0] << " -i file.fr-en\n"
         << " Standard options ([USE] = strongly recommended):\n"
         << "  -i: [REQ] Input parallel corpus\n"
         << "  -v: [USE] Use Dirichlet prior on lexical translation distributions\n"
         << "  -d: [USE] Favor alignment points close to the monotonic diagonoal\n"
         << "  -o: [USE] Optimize how close to the diagonal alignment points should be\n"
         << "  -r: Run alignment in reverse (condition on target and predict source)\n"
         << "  -c: Output conditional probability table\n"
         << " Advanced options:\n"
         << "  -I: number of iterations in EM training (default = 5)\n"
         << "  -q: p_null parameter (default = 0.08)\n"
         << "  -N: No null word\n"
         << "  -a: alpha parameter for optional Dirichlet prior (default = 0.01)\n"
         << "  -T: starting lambda for diagonal distance parameter (default = 4)\n"
         << "  -s: print alignment scores (alignment ||| score, disabled by default)\n";
    return 1;
  }
  const bool use_null = !no_null_word;
  if (variational_bayes && alpha <= 0.0) {
    cerr << "--alpha must be > 0\n";
    return 1;
  }
  const double prob_align_not_null = 1.0 - prob_align_null;
  const unsigned kNULL = d.Convert("<eps>");
  TTable s2t, t2s;
  vector<pair<pair<short, short>, unsigned>> size_counts;
  double tot_len_ratio = 0;
  double n_target_tokens = 0;

  if (force_align) {
    ifstream in(conditional_probability_filename.c_str());
    s2t.DeserializeLogProbsFromText(&in, d);
    ITERATIONS = 0; // don't do any learning
  } else {
    InitialPass(kNULL, use_null, &s2t, &n_target_tokens, &tot_len_ratio, &size_counts);
    s2t.Freeze();
  }

  for (int iter = 0; iter < ITERATIONS; ++iter) {
    const bool final_iteration = (iter == (ITERATIONS - 1));
    cerr << "ITERATION " << (iter + 1) << (final_iteration ? " (FINAL)" : "") << endl;
    ifstream in(input.c_str());
    if (!in) {
      cerr << "Can't read " << input << endl;
      return 1;
    }
    double likelihood = 0;
    const double denom = n_target_tokens;
    int lc = 0;
    bool flag = false;
    string line;
    double c0 = 0;
    double emp_feat = 0;
    vector<string> buffer;
    vector<string> outputs;
    while(true) {
      getline(in, line);
      if (!in) break;
      ++lc;
      if (lc % 1000 == 0) { cerr << '.'; flag = true; }
      if (lc %50000 == 0) { cerr << " [" << lc << "]\n" << flush; flag = false; }
      buffer.push_back(line);

      if (buffer.size() >= thread_buffer_size) {
        UpdateFromPairs(buffer, lc, iter, final_iteration, use_null, kNULL,
            prob_align_not_null, &c0, &emp_feat, &likelihood, &s2t, &outputs);
        if (final_iteration) {
          for (const string& output : outputs) {
            cout << output;
          }
        }
        buffer.clear();
      }
    } // end data loop
    if (buffer.size() > 0) {
      UpdateFromPairs(buffer, lc, iter, final_iteration, use_null, kNULL,
          prob_align_not_null, &c0, &emp_feat, &likelihood, &s2t, &outputs);
      if (final_iteration) {
        for (const string& output : outputs) {
          cout << output;
        }
      }
      buffer.clear();
    }

    // log(e) = 1.0
    double base2_likelihood = likelihood / log(2);

    if (flag) {
      cerr << endl;
    }
    emp_feat /= n_target_tokens;
    cerr << "  log_e likelihood: " << likelihood << endl;
    cerr << "  log_2 likelihood: " << base2_likelihood << endl;
    cerr << "     cross entropy: " << (-base2_likelihood / denom) << endl;
    cerr << "        perplexity: " << pow(2.0, -base2_likelihood / denom) << endl;
    cerr << "      posterior p0: " << c0 / n_target_tokens << endl;
    cerr << " posterior al-feat: " << emp_feat << endl;
    //cerr << "     model tension: " << mod_feat / toks << endl;
    cerr << "       size counts: " << size_counts.size() << endl;
    if (!final_iteration) {
      if (favor_diagonal && optimize_tension && iter > 0) {
        for (int ii = 0; ii < 8; ++ii) {
          double mod_feat = 0;
#pragma omp parallel for reduction(+:mod_feat)
          for(size_t i = 0; i < size_counts.size(); ++i) {
            const pair<short,short>& p = size_counts[i].first;
            for (short j = 1; j <= p.first; ++j)
              mod_feat += size_counts[i].second * DiagonalAlignment::ComputeDLogZ(j, p.first, p.second, diagonal_tension);
          }
          mod_feat /= n_target_tokens;
          cerr << "  " << ii + 1 << "  model al-feat: " << mod_feat << " (tension=" << diagonal_tension << ")\n";
          diagonal_tension += (emp_feat - mod_feat) * 20.0;
          if (diagonal_tension <= 0.1) diagonal_tension = 0.1;
          if (diagonal_tension > 14) diagonal_tension = 14;
        }
        cerr << "     final tension: " << diagonal_tension << endl;
      }
      if (variational_bayes)
        s2t.NormalizeVB(alpha);
      else
        s2t.Normalize();
    }
  }
  if (!force_align && !conditional_probability_filename.empty()) {
    cerr << "conditional probabilities: " << conditional_probability_filename << endl;
    s2t.ExportToFile(conditional_probability_filename.c_str(), d, beam_threshold);
  }
  if (force_align) {
    istream* pin = &cin;
    if (input != "-" && !input.empty())
      pin = new ifstream(input.c_str());
    istream& in = *pin;
    string line;
    vector<unsigned> src, trg;
    int lc = 0;
    double tlp = 0;
    while(getline(in, line)) {
      ++lc;
      ParseLine(line, &src, &trg);
      for (auto s : src) cout << d.Convert(s) << ' ';
      cout << "|||";
      for (auto t : trg) cout << ' ' << d.Convert(t);
      cout << " |||";
      if (is_reverse)
        swap(src, trg);
      if (src.size() == 0 || trg.size() == 0) {
        cerr << "Error in line " << lc << endl;
        return 1;
      }
      double log_prob = Md::log_poisson(trg.size(), 0.05 + src.size() * mean_srclen_multiplier);

      // compute likelihood
      for (unsigned j = 0; j < trg.size(); ++j) {
        unsigned f_j = trg[j];
        double sum = 0;
        int a_j = 0;
        double max_pat = 0;
        double prob_a_i = 1.0 / (src.size() + use_null);  // uniform (model 1)
        if (use_null) {
          if (favor_diagonal) prob_a_i = prob_align_null;
          max_pat = s2t.safe_prob(kNULL, f_j) * prob_a_i;
          sum += max_pat;
        }
        double az = 0;
        if (favor_diagonal)
          az = DiagonalAlignment::ComputeZ(j+1, trg.size(), src.size(), diagonal_tension) / prob_align_not_null;
        for (unsigned i = 1; i <= src.size(); ++i) {
          if (favor_diagonal)
            prob_a_i = DiagonalAlignment::UnnormalizedProb(j + 1, i, trg.size(), src.size(), diagonal_tension) / az;
          double pat = s2t.safe_prob(src[i-1], f_j) * prob_a_i;
          if (pat > max_pat) { max_pat = pat; a_j = i; }
          sum += pat;
        }
        log_prob += log(sum);
        if (true) {
          if (a_j > 0) {
            cout << ' ';
            if (is_reverse)
              cout << j << '-' << (a_j - 1);
            else
              cout << (a_j - 1) << '-' << j;
          }
        }
      }
      tlp += log_prob;
      cout << " ||| " << log_prob << endl << flush;
    } // loop over test set sentences
    cerr << "TOTAL LOG PROB " << tlp << endl;
  }
  return 0;
}
