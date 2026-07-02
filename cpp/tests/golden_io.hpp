// Reader for the restricted TOML subset used by the golden-vector files under
// tests/golden/ (layout in tests/golden/README.md). The core itself never
// parses text (D-2); this reader exists only inside the test binary, and it
// deliberately supports just the constructs generate.py emits - [[case]]
// tables, quoted-string scalars, and arrays of quoted strings - so there is
// no third-party parser to vendor and validate.
#ifndef STAR_TESTS_GOLDEN_IO_HPP
#define STAR_TESTS_GOLDEN_IO_HPP

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace star_tests {

struct GoldenCase {
  std::map<std::string, std::string> scalars;
  std::map<std::string, std::vector<std::string>> arrays;

  const std::string& scalar(const std::string& key) const {
    const auto it = scalars.find(key);
    if (it == scalars.end()) {
      throw std::runtime_error("golden case missing scalar key: " + key);
    }
    return it->second;
  }

  const std::vector<std::string>& array(const std::string& key) const {
    const auto it = arrays.find(key);
    if (it == arrays.end()) {
      throw std::runtime_error("golden case missing array key: " + key);
    }
    return it->second;
  }
};

inline std::string trim(const std::string& s) {
  const std::size_t begin = s.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return std::string();
  }
  const std::size_t end = s.find_last_not_of(" \t\r\n");
  return s.substr(begin, end - begin + 1);
}

// Strip surrounding double quotes. Golden strings are plain ASCII with no
// embedded quotes or escapes by construction (generate.py emits them), so no
// escape handling is required; a malformed line throws.
inline std::string unquote(const std::string& s) {
  if (s.size() < 2 || s.front() != '"' || s.back() != '"') {
    throw std::runtime_error("golden reader: expected quoted string, got: " +
                             s);
  }
  return s.substr(1, s.size() - 2);
}

inline std::vector<GoldenCase> load_golden_cases(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("golden reader: cannot open " + path);
  }
  std::vector<GoldenCase> cases;
  std::string line;
  std::string array_key;  // non-empty while inside a multi-line array
  while (std::getline(in, line)) {
    const std::string t = trim(line);
    if (t.empty() || t[0] == '#') {
      continue;
    }
    if (!array_key.empty()) {
      if (t == "]") {
        array_key.clear();
        continue;
      }
      std::string item = t;
      if (!item.empty() && item.back() == ',') {
        item.pop_back();
      }
      cases.back().arrays[array_key].push_back(unquote(trim(item)));
      continue;
    }
    if (t == "[[case]]") {
      cases.emplace_back();
      continue;
    }
    const std::size_t eq = t.find('=');
    if (eq == std::string::npos || cases.empty()) {
      throw std::runtime_error("golden reader: unexpected line: " + t);
    }
    const std::string key = trim(t.substr(0, eq));
    const std::string value = trim(t.substr(eq + 1));
    if (value == "[") {
      array_key = key;
      cases.back().arrays[key] = {};
    } else {
      cases.back().scalars[key] = unquote(value);
    }
  }
  if (cases.empty()) {
    throw std::runtime_error("golden reader: no cases in " + path);
  }
  return cases;
}

// "0x..." hex string -> u64. strtoull is locale-independent for hex digits.
inline std::uint64_t parse_hex_u64(const std::string& s) {
  if (s.size() < 3 || s[0] != '0' || (s[1] != 'x' && s[1] != 'X')) {
    throw std::runtime_error("golden reader: expected 0x hex string: " + s);
  }
  char* end = nullptr;
  const std::uint64_t v = std::strtoull(s.c_str(), &end, 16);
  if (end == nullptr || *end != '\0') {
    throw std::runtime_error("golden reader: bad hex string: " + s);
  }
  return v;
}

// Binary64 hex literal (Python float.hex() form, e.g. "-0x1.9p-2") -> double.
// strtod parses C99 hexadecimal floating constants exactly, so the golden
// value survives the text round trip bit-for-bit.
inline double parse_hex_double(const std::string& s) {
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  if (end == nullptr || *end != '\0') {
    throw std::runtime_error("golden reader: bad hex float: " + s);
  }
  return v;
}

}  // namespace star_tests

#endif  // STAR_TESTS_GOLDEN_IO_HPP
