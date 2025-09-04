#pragma once

#include <map>
#include <memory>
#include <unordered_map>

#include "ParameterBlockBase.h"
#include <glog/logging.h>

namespace ceres_nav {
  struct StateID;
}

namespace ceres_nav {

/**
 * @brief Holds a collection of states in time, accessible by a string key and a
 * timestamp.
 */
class StateCollection {
public:
  StateCollection(){};

  /**
   * @brief Adds a state to the collection with a given name and timestamp.
   */
  void addState(const std::string &name, double timestamp,
                std::shared_ptr<ParameterBlockBase> state);

  /**
   * @brief Retrieves a state for a given key and timestamp.
   *
   * @param key The key of the state to retrieve.
   * @param timestamp The timestamp of the state to retrieve.
   * @return A shared pointer to the state, or nullptr if not found.
   */
  std::shared_ptr<ParameterBlockBase> getState(const std::string &key,
                                               double timestamp) const;

  /**
   * @brief Templated version of get state that returns a state of a specific
   * type.
   *
   * @param key The key of the state to retrieve.
   * @param timestamp The timestamp of the state to retrieve.
   * @return A shared pointer to the state of type T, or nullptr if not found or
   * if the downcast fails.
   */
  template <typename T>
  std::shared_ptr<T> getState(const std::string &key, double timestamp) const {
    int64_t timestamp_key = timestampToKey(timestamp);

    auto it1 = states_.find(key);
    // If we've found the key
    if (it1 != states_.end()) {
      auto state_it = it1->second.find(timestamp_key);
      if (state_it != it1->second.end()) {
        // Attempt to downcast to the specific type T
        auto casted_ptr = std::dynamic_pointer_cast<T>(state_it->second);
        if (casted_ptr) {
          return casted_ptr;
        } else {
          // If downcast fails, return nullptr
          LOG(ERROR) << "Failed to downcast state for key: " << key
                     << "at timestamp: " << timestamp;
          return nullptr;
        }
      }
    }
    // Return nullptr if not found or downcast fails
    return nullptr;
  }

  /**
   * @brief Removes a state from the collection for a given key and timestamp.
   *
   * @param key The key of the state to remove.
   * @param timestamp The timestamp of the state to remove.
   */
  void removeState(const std::string &key, double timestamp);

  /**
   * @brief Retrieves a state by its estimate pointer.
   *
   * This is useful for finding a state when you have a pointer to its estimate
   * (i.e., from Ceres.)
   */
  std::shared_ptr<ParameterBlockBase>
  getStateByEstimatePointer(double *ptr) const;

  /**
   * @brief Retrieves the StateID for a state by its estimate pointer.
   * This is also useful for getting an associated StateID when you have a pointer to its estimate.
   * 
   * Returns false if the pointer is not found.
   */
  bool getStateIDByEstimatePointer(double *ptr, StateID &state_id) const;


  // Check if a state exists at a given timestamp
  bool hasState(const std::string &key, double timestamp) const;
  bool hasStateType(const std::string &key) const {
    return states_.find(key) != states_.end();
  } 

  /**
   * @brief Get the number of different state types stored in the collection.
   */
  size_t getNumStateTypes() const { return states_.size(); }

  /**
   * @brief Get the number of states for a given type.
   *
   * @param key The key of the state type to check.
   * @return The number of states for the given type.
   */
  size_t getNumStatesForType(const std::string &key) const {
    auto it = states_.find(key);
    if (it != states_.end()) {
      return it->second.size();
    }
    return 0;
  }

  /**
   * @brief Get the first timestamp for a given key.
   */
  bool getOldestStamp(const std::string &key, double &stamp) const;

  /**
   * @brief Gets the last timestamp for a given key.
  */
  bool getLatestStamp(const std::string &key, double &stamp) const;

  /**
   * @brief Gets all timestamps for a given key.
  */
  bool getTimesForState(const std::string &key,
                        std::vector<double> &stamps) const;

  // Get the oldest and latest states for a given key
  // Returns a pointer to the base class.
  std::shared_ptr<ParameterBlockBase> getOldestState(const std::string &key) const;
  std::shared_ptr<ParameterBlockBase> getLatestState(const std::string &key) const;

  /**
   * @brief Gets the oldest state of a specific type for a given key.
  */
  template <typename T>
  std::shared_ptr<T> getOldestState(const std::string &key) const {
    auto state = getOldestState(key);
    if (state) {
      return std::dynamic_pointer_cast<T>(state);
    }
    return nullptr;
  }

  /**
   * @brief Templated version of getLatestState that returns a state of 
   * specific type.
  */
  template <typename T>
  std::shared_ptr<T> getLatestState(const std::string &key) const {
    auto state = getLatestState(key);
    if (state) {
      return std::dynamic_pointer_cast<T>(state);
    }
    return nullptr;
  }
  
protected:
  static constexpr double default_timestamp_precision = 1e-9;
  double timestamp_precision_ = default_timestamp_precision;

  int64_t timestampToKey(double timestamp) const {
    return static_cast<int64_t>(std::round(timestamp / timestamp_precision_));
  }

  double keyToTimestamp(int64_t key) const {
    return static_cast<double>(key) * timestamp_precision_;
  }

  // The states are stored in a map where the key corresponds to the name of the
  // state, and the value is a map of timestamps to state pointers. This allows
  // for retrieval of states
  std::unordered_map<std::string,
                     std::map<int64_t, std::shared_ptr<ParameterBlockBase>>>
      states_;
};

} // namespace ceres_nav