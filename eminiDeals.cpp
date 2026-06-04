#include<iostream>
#include"/home/loki/CLionProjects/osmiArrow/include/osmi/arrow.hpp"

static  std::string BID_PRICE ="bid_px_00";
static  std::string ASK_PRICE = "ask_px_00";
static  std::string MID_CHANGE = "mid_chg";
int main() {
    const std::string file = "/media/loki/New Volume/CME BBO/glbx-mdp3-20250404.mbp-1.csv";
    std::cout << file << "\n";
    const auto loaded = osmi::Table::from_csv(file).filter(
        [](const osmi::Row& r){return r.get<char>("action") == 'T'; },
        [](const osmi::Row& r){return r.get<std::string>("symbol") == "ESM5"; }
        );

    auto sub = loaded.select({"bid_px_00","ask_px_00","action","size"});
    sub.add_column<double>(MID_CHANGE, [&](const osmi::Row& r) {
        auto f = sub.row(r.row_index()+1);

        const double currentMid = (r.get<double>(BID_PRICE) + r.get<double>(ASK_PRICE)) / 2;
        const double futureMid = (f.get<double>(BID_PRICE) + f.get<double>(ASK_PRICE)) / 2;

        return std::abs(futureMid - currentMid);
    });

    sub.to_parquet("/media/loki/New Volume/CME BBO/midChgparquet.parquet", 1024* 1024);


    return 0;
}