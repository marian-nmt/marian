#pragma once

#include <string>

namespace marian {

std::string citation() {
  return "Marian: Fast Neural Machine Translation in C++\n"
         "\n"
         "Please cite the following paper if you use Marian:\n"
         "\n"
         "@InProceedings{mariannmt,\n"
         "  title     = {Marian: Fast Neural Machine Translation in {C++}},\n"
         "  author    = {Junczys-Dowmunt, Marcin and Grundkiewicz, Roman and\n"
         "               Dwojak, Tomasz and Hoang, Hieu and Heafield, Kenneth and\n"
         "               Neckermann, Tom and Seide, Frank and Germann, Ulrich and\n"
         "               Fikri Aji, Alham and Bogoychev, Nikolay and\n"
         "               Martins, Andr\\'{e} F. T. and Birch, Alexandra},\n"
         "  booktitle = {Proceedings of ACL 2018, System Demonstrations},\n"
         "  pages     = {116--121},\n"
         "  publisher = {Association for Computational Linguistics},\n"
         "  year      = {2018},\n"
         "  month     = {July},\n"
         "  address   = {Melbourne, Australia},\n"
         "  url       = {http://www.aclweb.org/anthology/P18-4020}\n"
         "}\n";
}

// The list of contributors has been compiled semi-automatically from the
// GitHub contributor list in default order. That list can be printed out with
// `git shortlog -s -n`.
std::string authors() {
  return "Marian: Fast Neural Machine Translation in C++\n"
         "\n"
         "An inevitably non-exhaustive list of contributors:\n"
         "\n"
         "Marcin Junczys-Dowmunt <marcinjd@microsoft.com>\n"
         "Roman Grundkiewicz <rgrundki@inf.ed.ac.uk>\n"
         "Frank Seide <fseide@microsoft.com>\n"
         "Hieu Hoang <hieuhoang@gmail.com>\n"
         "Tomasz Dwojak <t.dwojak@amu.edu.pl>\n"
         "Ulrich Germann <ugermann@inf.ed.ac.uk>\n"
         "Alham Fikri Aji <afaji321@gmail.com>\n"
         "Cédric Rousseau <cedrou@gmail.com>\n"
         "Young Jin Kim <youki@microsoft.com>\n"
         "Lane Schwartz <dowobeha@gmail.com>\n"
         "Andre Martins <andre.t.martins@gmail.com>\n"
         "Nikolay Bogoychev <n.bogoych@ed.ac.uk>\n"
         "Kenneth Heafield <kheafiel@ed.ac.uk>\n"
         "Maximiliana Behnke <mbehnke@inf.ed.ac.uk>\n"
         "Tom Neckermann <tomneckermann@gmail.com>\n"
         "Hany Hassan Awadalla <hanyh@microsoft.com>\n"
         "Jim Geovedi <jim@geovedi.com>\n"
         "Catarina Silva <catarina.cruz.csilva@gmail.com>\n"
         "Jon Clark <jonathac@microsoft.com>\n"
         "Rihards Krišlauks <rihards.krislauks@gmail.com>\n"
         "Vishal Chowdhary <vishalc@microsoft.com>\n"
         "Barry Haddow <bhaddow@inf.ed.ac.uk>\n"
         "Dominik Stańczak <stanczakdominik@gmail.com>\n"
         "Michael Hutt <Michael.Hutt@gmail.com>\n"
         "Richard Wei <rxwei@users.noreply.github.com>\n"
         "Wenyong Huang <weyo.huang@gmail.com>\n"
         "alancucki <alancucki+github@gmail.com>\n";
}
}  // namespace marian
