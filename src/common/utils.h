#pragma once

#include <string>
#include <vector>

void Trim(std::string& s);

void Split(const std::string& line,
           std::vector<std::string>& pieces,
           const std::string del = " ");

std::vector<std::string> Split(const std::string& line,
                               const std::string del = " ");

std::string Join(const std::vector<std::string>& words,
                 const std::string& del = " ",
                 bool reverse = false);

std::string Exec(const std::string& cmd);
