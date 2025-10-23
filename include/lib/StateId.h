#pragma once

#include <optional>
#include <string>

namespace ceres_nav {

/**
 * @brief Unique identifier for a state.
 *
 * Stores a string key, and an optional timestamp for time-varying states.
 * The timestamp is automatically rounded to a specified precision to avoid
 * floating point issues for retrieving states with a given timestamp.
 */
class StateID {
public:
  // Round to nanosecond precision by default
  inline static double DEFAULT_PRECISION = 1e-9;

  StateID() = default;

  // Construct for time-varying states (poses, velocities, biases, etc.)
  StateID(const std::string &key, double timestamp,
          double precision = DEFAULT_PRECISION)
      : key_(key), timestamp_(roundToPrecision(timestamp, precision)) {}

  // Constructor for time-invariant states (landmarks, calibration,
  // etc.)
  StateID(const std::string &key) : key_(key) {}

  static void setDefaultPrecision(double precision) {
    DEFAULT_PRECISION = precision;
  }

  // Accessors
  const std::string key() const { return key_; }
  std::optional<double> timestamp() const { return timestamp_; }
  bool hasTimestamp() const { return timestamp_.has_value(); }

  // Comparison operators for use in maps and sets
  bool operator<(const StateID &other) const {
    if (key_ != other.key_) {
      return key_ < other.key_;
    }
    if (timestamp_.has_value() != other.timestamp_.has_value()) {
      return !timestamp_.has_value();
    }
    if (timestamp_.has_value()) {
      return timestamp_.value() < other.timestamp_.value();
    }
    return false;
  }

  bool operator==(const StateID &other) const {
    return key_ == other.key_ && timestamp_ == other.timestamp_;
  }

  bool operator!=(const StateID &other) const { return !(*this == other); }

  // String representation for debuggin
  std::string toString() const {
    if (timestamp_.has_value()) {
      return key_ + " at time " + std::to_string(timestamp_.value());
    } else {
      return key_;
    }
  }

private:
  // Round timestamp to a specified precision
  static double roundToPrecision(double value, double precision) {
    return std::round(value / precision) * precision;
  }

  std::string key_;
  std::optional<double> timestamp_;
};
} // namespace ceres_nav