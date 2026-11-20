#include "include/osmi/arrow.hpp"
#include "include/osmi/matrix.hpp"
#include <algorithm>
#include <array>
#include <arrow/csv/options.h>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <string_view>
#include <type_traits>
#include <flat_map>



template<typename T> requires (std::is_arithmetic_v<T>)
struct Feature{
  T lower_ = 0;
  T upper_ = 0;
  std::size_t bins_count_ = 0;
  inline size_t BinValue(T value){
    value = std::clamp(value, lower_, upper_);
    return static_cast<size_t>((value - lower_) * (bins_count_ / (upper_ - lower_)));
  }
};

struct Features{
  Feature<double> imbalance_ { 0.0, 1.0, 10 }; //0.2 bin

} features;

namespace osmi{
  constexpr std::uint64_t micro_ns = 1000;
  constexpr std::uint64_t milli_ns = micro_ns * 1000;
  constexpr std::uint64_t second_ns = milli_ns * 1000;
  constexpr std::uint64_t minute_ns = second_ns * 60;
  constexpr std::uint64_t hour_ns = minute_ns * 60;
  constexpr std::uint64_t day_ns = hour_ns * 24;

  constexpr std::uint64_t stou64(std::string_view sv){
    std::uint64_t result = 0;
    for (std::size_t i = 0; i < sv.size(); ++i){
      if (sv[i] < 48 || sv[i] > 57){ return 0; }
      result = result * 10 + (sv[i] - '0');
    }
    return result;
  }
  constexpr std::uint32_t stou32(std::string_view sv){
    std::uint32_t result = 0;
    for (std::size_t i = 0; i < sv.size(); ++i){
      if (sv[i] < 48 || sv[i] > 57){ return 0; }
      result = result * 10 + (sv[i] - '0');
    }
    return result;
  }
  constexpr bool is_leap_year(const int year){
    return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
  }
  [[nodiscard]] inline const std::uint64_t month_offset_ns(const std::uint16_t month, const std::uint16_t year){
    constexpr std::uint16_t days_before_month[] = {
      0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334
    };
    assert(month >=1 && month <= 12);
    if (is_leap_year(year) && month > 2){
      return days_before_month[month - 1] * day_ns + day_ns; //months after feb +1 day
    }
    return days_before_month[month - 1] * day_ns;
  }
  constexpr std::uint64_t leap_days_through(std::uint32_t year) {
    return year / 4 - year / 100 + year / 400;
  }
  //2025-10-20T00:00:00.000000000Z
  [[nodiscard]] inline const std::uint64_t parse_epoch_ns(std::string_view ts){
    std::uint32_t year = stou32(ts.substr(0, 4));
    std::uint32_t month = stou32(ts.substr(5, 2));
    std::uint32_t day = stou32(ts.substr(8,2));

    std::uint64_t date_ns = (year - 1970) * 365ULL * day_ns +
                            month_offset_ns(month, year) +
                            (day - 1) * day_ns +
                            (leap_days_through(year - 1) - leap_days_through(1969))*day_ns;

    std::uint64_t day_ns = stou32(ts.substr(11, 2))*hour_ns + stou32(ts.substr(14, 2))*minute_ns + stou32(ts.substr(17, 2))*second_ns;
    std::uint64_t ns = stou32(ts.substr(20, 9));

    return date_ns + day_ns + ns;
  }
}

std::array<std::array<double, 2>, 10> state_midchg;
std::array<std::array<std::uint64_t, 2>, 10>  state_frequencies;

std::array<std::array<std::uint64_t, 2*9>,2*10> transition_frequencies;
std::array<std::uint64_t, 2*10> transition_times;


int main(){

  auto convert = arrow::csv::ConvertOptions::Defaults();
  convert.column_types["ts_event"] = arrow::utf8();

  auto table = osmi::Table::from_csv("/home/user/Desktop/tempData/glbx-mdp3-20251020.mbp-1.csv");
  const std::string symbol = "ESZ5";

  table = table.select({"bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00", "ts_event", "symbol"});
  table = table.filter([](const osmi::Row& row){ return row.get<std::string_view>("symbol") == "ESZ5"; });


  osmi::Row last_row = table[0];

  for (size_t r = 1; r < table.num_rows(); ++r){
    const auto& curr_row = table[r];
    //prices

    const std::uint64_t time_curr = osmi::parse_epoch_ns(curr_row.get<std::string_view>("ts_event"));
    const std::uint64_t time_last = osmi::parse_epoch_ns(last_row.get<std::string_view>("ts_event"));

    //100ms snapshots;

    const double bid_px_curr = curr_row.get<double>("bid_px_00");
    const double ask_px_curr = curr_row.get<double>("ask_px_00");
    const double bid_px_last = last_row.get<double>("bid_px_00");
    const double ask_px_last = last_row.get<double>("ask_px_00");
    const int bid_sz_curr = curr_row.get<int>("bid_sz_00");
    const int ask_sz_curr = curr_row.get<int>("ask_sz_00");
    const int bid_sz_last = last_row.get<int>("bid_sz_00");
    const int ask_sz_last = last_row.get<int>("ask_sz_00");


    const double mid_px_curr = (bid_px_curr + ask_px_curr) / 2.0;
    const double mid_px_last = (bid_px_last + ask_px_last) / 2.0;
    const double mid_chg = mid_px_curr - mid_px_last;

    const double imbalance_curr = static_cast<double>(bid_sz_curr) / (bid_sz_curr + ask_sz_curr);
    const double imbalance_last = static_cast<double>(bid_sz_last) / (bid_sz_last + ask_sz_last);

    const double spread_curr = ask_px_curr - bid_px_curr;
    const double spread_last = ask_px_last - bid_px_last;

    const std::uint64_t time_delta = time_curr - time_last;

    if (mid_chg != 0.0){
      //feature values
      const size_t imbalance_bin = features.imbalance_.BinValue(imbalance_last);
      const size_t spread_bin = spread_last > 0.25 ? 1 : 0;

      state_midchg[imbalance_bin][spread_bin] += mid_chg;
      state_frequencies[imbalance_bin][spread_bin] += 1;
      last_row = table[r];
      continue;
    }

    const size_t imbalance_bin_curr = features.imbalance_.BinValue(imbalance_curr);
    const size_t spread_bin_curr = spread_curr > 0.25 ? 1 : 0;

    const size_t imbalance_bin_last = features.imbalance_.BinValue(imbalance_last);
    const size_t spread_bin_last = spread_last > 0.25 ? 1 : 0;

    //2 digit state, imbalance is 1st num and spread 2nd.
    // imbbin *= 2 + sprd bin is index for state value pair
    //
    // from->to
    //
    transition_frequencies[imbalance_bin_last * 2 + spread_bin_last][imbalance_bin_curr * 2 + spread_bin_curr] += 1;
    transition_times[imbalance_bin_last * 2 + spread_bin_last]  += time_delta;

    last_row = table[r];
  }

  std::cout << "--time in state ij--\n";
  for (size_t i = 0; i < 10; ++i){
    for (size_t s = 0; s < 2; ++s){
      size_t rowsum = std::accumulate(transition_frequencies[i * 2 + s].begin(), transition_frequencies[i * 2 + s].end(), std::uint64_t{0});
      if (rowsum == 0){ continue; }
      transition_times[i * 2 + s] /= rowsum;

      std::cout << "state: " << i << s << " time: " << transition_times[i * 2 + s] << '\n';
    }
  }

  std::cout << "--E(dmid)_t+1 in state ij--\n";
  for (size_t i = 0; i < 10; ++i){
    for (size_t s = 0; s < 2; ++s){
      if (state_frequencies[i][s] != 0){
        state_midchg[i][s] /= state_frequencies[i][s];
      }
      std::cout << "state: " << i << s << " midchg: " << state_midchg[i][s] << '\n';
    }
  }

  return 0;
}
