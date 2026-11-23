#ifndef TIME_HPP
#define TIME_HPP
#include <cstddef>
#include <string_view>
#include <cstdint>

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

#endif
