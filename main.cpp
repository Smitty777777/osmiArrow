#include "osmi/arrow.hpp"
#include <iostream>
#include <cmath>

std::string HHIpaths[] = {
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-03/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-04/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-05/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-06/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-07/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-08/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-09/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-10/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-11/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-12/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-13/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-14/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-15/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-16/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-17/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-18/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-19/HHIH5_2025-03-28.parquet",


};

static int STEPS = 10;

int imbalanceBin(double imbalance) {
    int imbalanceBin =0;
    imbalanceBin = std::floor(imbalance*10);
    return imbalanceBin ==10 ? 9 : imbalanceBin;
}
std::array<std::array<double,10>,10> multiplyMatrix(std::array<std::array<double,10>,10>& matrix) {
    std::array<std::array<double,10>,10> result{};
    for (int i=0; i<10; ++i) {
        for (int j=0; j<10; ++j) {
            result[i][j] = 0;
            for (int k=0; k<10; ++k) {
                result[i][j] += matrix[i][k] * matrix[k][j];
            }
        }
    }
    return result;
}

int main() {

    std::array<std::array<double,10>,10> frequencyMatrix = {};
    std::array<double,10> midChangeMatrix = {};
    std::array<std::pair<int, double>, 10> adjustmentGrid = {};
    for (std::string file : HHIpaths) {
        std::cout << file << std::endl;
        auto loaded = osmi::Table::from_parquet("/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-04/HHIH5_2025-03-28.parquet");

        auto sub = loaded.select({"BidQuantity_1", "AskQuantity_1", "BidPrice_1", "AskPrice_1"});
        //sub.add_column<double>("doubled_x", [](const osmi::Row& r) { return r.get<double>("x") * 2.0; });
        //sub.add_column<std::string>("label", [](const osmi::Row& r) { return "row#" + std::to_string(r.get<int64_t>("id")); });

        //std::cout << "rows=" << sub.num_rows() << " cols=" << sub.num_columns() << "\n";
        //for (auto&& row : sub) {
        //
        //}
        //from row to col

        for (int64_t i = 0; i < sub.num_rows()-1; ++i) {
            auto r = sub.row(i);

            double bidVol = r.get<double>("BidQuantity_1");
            double askVol = r.get<double>("AskQuantity_1");
            double bidPrice = r.get<double>("BidPrice_1");
            double askPrice = r.get<double>("AskPrice_1");

            auto h = sub.row(i+1);
            double bidVolH = h.get<double>("BidQuantity_1");
            double askVolH = h.get<double>("AskQuantity_1");
            double bidPriceH = h.get<double>("BidPrice_1");
            double askPriceH = h.get<double>("AskPrice_1");

            double midPrice = (bidPrice+askPrice)/2;
            double midPriceH = (bidPriceH+askPriceH)/2;

            double imbalance = (bidVol) / (bidVol+askVol);
            double imbalanceH = (bidVolH) / (bidVolH+askVolH);

            if (imbalanceBin(imbalanceH) != imbalanceBin(imbalance)) {
                frequencyMatrix[imbalanceBin(imbalance)][imbalanceBin(imbalanceH)] +=1;
                midChangeMatrix[imbalanceBin(imbalanceH)] += (midPriceH - midPrice);
            }
        }
    }

    //probabilities
    for (int i = 0; i < 10; ++i) {
        double rowCount = 0;
        for (int j = 0; j < 10; ++j) rowCount += frequencyMatrix[i][j];
        if (rowCount > 0) midChangeMatrix[i] /= rowCount;
    }
    for (int i=0; i<10; ++i) {

        int rowSum = 0;
        for (int j=0; j<10; ++j) {
            rowSum += frequencyMatrix[i][j];
        }
        for (int j=0; j<10; ++j) {
                 frequencyMatrix[i][j] /= rowSum;
        }
    }
    for (int i=0; i<10; ++i) {

        double Gstar = 0;
        auto stoppingTimeMatrix = frequencyMatrix;
        for (int j=0; j<STEPS; ++j) {
            for (int k=0; k<10; ++k) {
               Gstar+= (midChangeMatrix[k] *stoppingTimeMatrix[i][k]);
            }
            stoppingTimeMatrix = multiplyMatrix(stoppingTimeMatrix);
        }
        adjustmentGrid[i].first = i;
        adjustmentGrid[i].second = Gstar;
    }

    for (std::pair<int, double> a : adjustmentGrid) {
        std::cout << a.first << " " << a.second << std::endl;
    }
    return 0;
}
