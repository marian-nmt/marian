
#include <boost/config.hpp>
#include <fstream>
#include <iostream>
#include <regex>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/graph_traits.hpp>

using namespace boost;
typedef adjacency_list<listS,
                       vecS,
                       directedS,
                       no_property,
                       property<edge_weight_t, int>>
    graph_t;
typedef graph_traits<graph_t>::vertex_descriptor vertex_descriptor;
typedef graph_traits<graph_t>::edge_descriptor edge_descriptor;

typedef std::pair<int, int> Edge;

typedef std::pair<int, int> Point;
typedef std::vector<Point> Alignment;

std::vector<std::string> split(const std::string& input,
                               const std::string& regex) {
  std::regex re(regex);
  std::sregex_token_iterator first{input.begin(), input.end(), re, -1}, last;
  return {first, last};
}

float dist(Point a, Point b) {
  return sqrt(pow(b.first - a.first, 2) + pow(b.second - a.second, 2)) - 1;
}

Alignment shortestPath(const Alignment& a) {
  Alignment shortest;

  std::vector<Edge> edges;
  std::vector<int> weights;

  for(int i = 0; i < a.size(); ++i) {
    for(int j = 0; j < a.size(); ++j) {
      if(a[i] != a[j] && a[j].first - a[i].first >= 0
         && a[j].second - a[i].second >= 0) {
        edges.push_back(Edge(i, j));
        weights.push_back(dist(a[i], a[j]));
      }
    }
  }

  graph_t g(edges.begin(), edges.end(), weights.data(), a.size());
  property_map<graph_t, edge_weight_t>::type weightmap = get(edge_weight, g);

  std::vector<vertex_descriptor> p(num_vertices(g));
  std::vector<int> d(num_vertices(g));
  vertex_descriptor s = vertex(0, g);

  dijkstra_shortest_paths(g, s, predecessor_map(&p[0]).distance_map(&d[0]));

  int v = a.size() - 1;
  while(v != 0) {
    shortest.push_back(a[v]);
    v = p[v];
  }
  shortest.push_back(a[v]);
  std::sort(shortest.begin(), shortest.end());

  Alignment shortestU;
  for(auto p : shortest)
    if(shortestU.empty() || shortestU.back().first != p.first)
      shortestU.push_back(p);

  return shortestU;
}

int main(int argc, char** argv) {
  if(argc != 4) {
    std::cerr << "Usage: ./align2steps source target alignment" << std::endl;
    exit(1);
  }

  std::ifstream srcStrm, trgStrm, alnStrm;
  srcStrm.open(argv[1]);
  trgStrm.open(argv[2]);
  alnStrm.open(argv[3]);

  int i = 0;
  std::string source, target, alignment;
  while(std::getline(srcStrm, source) && std::getline(trgStrm, target)
        && std::getline(alnStrm, alignment)) {
    auto srcToks = split(source, R"(\s)");
    auto trgToks = split(target, R"(\s)");
    auto alnToks = split(alignment, R"(\s|-)");

    Alignment alignment;
    for(int i = 0; i < alnToks.size(); i += 2)
      alignment.emplace_back(std::stoi(alnToks[i + 1]), std::stoi(alnToks[i]));

    // add end
    alignment.emplace_back(trgToks.size(), srcToks.size());

    auto shortest = shortestPath(alignment);

    int cTrg = 0, cSrc = 0;
    for(auto& p : shortest) {
      for(int i = cTrg; i < p.first; ++i)
        std::cout << trgToks[i] << " ";

      for(int i = cSrc; i < p.second; ++i)
        std::cout << "<step> ";

      cTrg = p.first;
      cSrc = p.second;
    }

    std::cout << std::endl;
    i++;
    if(i % 10000 == 0)
      std::cerr << i << " ";
  }
  std::cerr << std::endl;

  return 0;
}
