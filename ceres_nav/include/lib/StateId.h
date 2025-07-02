#pragma once

#include <string>

struct StateID {
  StateID() {
    ID = "";
    timestamp = 0.0;
  }

  StateID(const std::string &state_id, const double timestamp_)
      : ID(state_id), timestamp(timestamp_) {}

  /** is required to compare keys */
  bool operator==(const StateID &other) const {
    return ID == other.ID && timestamp == other.timestamp;
  }

  std::string ID;
  double timestamp;
};