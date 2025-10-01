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

  // Comparison operator for map/set
  bool operator<(const StateID &other) const {
    if (ID != other.ID) {
      return ID < other.ID;
    }

    // If IDs are equal, compare timestamps
    // States without timestamps come before states with timestamps
    if (!timestamp.has_value() && other.timestamp.has_value()) {
      return true;
    }
    if (timestamp.has_value() && !other.timestamp.has_value()) {
      return false;
    }
    // If both have timestamps, compare the values
    if (timestamp.has_value() && other.timestamp.has_value()) {
      return timestamp.value() < other.timestamp.value();
    }
    // Both are static (no timestamp) and have same ID
    return false;
  }

  bool isStatic() const { return !timestamp.has_value(); }

  std::string ID;
  std::optional<double> timestamp;
};

} // namespace ceres_nav