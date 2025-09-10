#pragma once

#include <optional>
#include <string>

namespace ceres_nav {
  
// struct StateID {
//   StateID() {
//     ID = "";
//     timestamp = 0.0;
//   }

//   StateID(const std::string &state_id, const double timestamp_)
//       : ID(state_id), timestamp(timestamp_) {}

//   /** is required to compare keys */
//   bool operator==(const StateID &other) const {
//     return ID == other.ID && timestamp == other.timestamp;
//   }

//   std::string ID;
//   double timestamp;
// };

struct StateID {
  StateID() { ID = ""; }

  // Constructor for static states with no timestamp
  StateID(const std::string &state_id) : ID(state_id) {}

  // Constructor for timestamped states
  StateID(const std::string &state_id, double timestamp_)
      : ID(state_id), timestamp(timestamp_) {}

  // Compare two StateID objects
  bool operator==(const StateID &other) const {
    return ID == other.ID && timestamp == other.timestamp;
  }

  bool isStatic() const { return !timestamp.has_value(); }

  std::string ID;
  std::optional<double> timestamp;
};

} // namespace ceres_nav