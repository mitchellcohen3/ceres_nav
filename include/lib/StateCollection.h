#pragma once

#include <map>
#include <memory>
#include <unordered_map>

#include "ParameterBlockBase.h"

namespace ceres_nav {
struct StateID;
}

namespace ceres_nav {

/**
 * @brief Holds a collection of states, allowing users to add, remove, and
 * query states by a StateID.
 */
class StateCollection {
public:
  StateCollection(){};

  /**
   * @brief Adds a parameter block using a StateID.
   */
  bool addState(const StateID &state_id,
                std::shared_ptr<ParameterBlockBase> param_block);

  /**
   * @brief Removes a state from the collection using a StateID.
   */
  bool removeState(const StateID &state_id);

  /**
   * @brief Query single parameter block using a StateID.
   * Returns nullptr if not found.
   */
  std::shared_ptr<ParameterBlockBase> getState(const StateID &state_id) const;

  /**
   * @brief Templated version of getState that returns a state of specific type.
   */

  template <typename T>
  std::shared_ptr<T> getState(const StateID &state_id) const {
    auto state = getState(state_id);
    if (state) {
      return std::dynamic_pointer_cast<T>(state);
    }
    return nullptr;
  }

  /**
   * @brief Check if a state exists in the collection using a StateID.
   */
  bool hasState(const StateID &state_id) const;

  /// Get some information about the timestamps for a given state type
  bool getOldestStamp(const std::string &key, double &timestamp) const;
  bool getLatestStamp(const std::string &key, double &timestamp) const;
  bool getTimesForState(const std::string &key,
                        std::vector<double> &timestamps) const;

  // Gets the oldest and latest states for a given key
  std::shared_ptr<ParameterBlockBase>
  getOldestState(const std::string &key) const;
  std::shared_ptr<ParameterBlockBase>
  getLatestState(const std::string &key) const;

  /**
   * @brief Templated version of getOldestState that returns a state of
   * specific type.
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

  /**
   * @brief Retrieves a state by its estimate pointer.
   *
   * This is useful for finding a state when you have a pointer to its
   estimate
   * (i.e., from Ceres.)
   */
  std::shared_ptr<ParameterBlockBase>
  getStateByEstimatePointer(double *ptr) const;

  /**
   * @brief Retrieves the StateID for a state by its estimate pointer.
   * This is also useful for getting an associated StateID when you have a
   * pointer to its estimate.
   *
   * Returns false if the pointer is not found.
   */
  bool getStateIDByEstimatePointer(double *ptr, StateID &state_id) const;

  /**
   * @brief Clears all states from the collection.
   */
  void clear() {
    time_varying_states_.clear();
    static_states_.clear();
  }

  /**
   * @brief Get the total number of states stored in the collection.
   */
  size_t size() const {
    size_t total_size = static_states_.size();
    for (const auto &pair : time_varying_states_) {
      total_size += pair.second.size();
    }
    return total_size;
  }

  size_t getNumberOfStatesForType(const std::string &key) const {
    auto it = time_varying_states_.find(key);
    if (it != time_varying_states_.end()) {
      return it->second.size();
    }
    return 0;
  }

  size_t staticSize() const { return static_states_.size(); }

protected:
  // Time-varying states stored as a map from string key to a map of timestamp
  // to ParameterBlockBase pointers.
  std::unordered_map<std::string,
                     std::map<double, std::shared_ptr<ParameterBlockBase>>>
      time_varying_states_;

  // Time-invariant states are stored in a separate map from
  // string key to ParameterBlockBase pointers.
  std::unordered_map<std::string, std::shared_ptr<ParameterBlockBase>>
      static_states_;
};

} // namespace ceres_nav