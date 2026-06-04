#include "osmi/arrow.hpp"
#include <iostream>
#include <cmath>
#include <format>
#include <fstream>
#include <numeric>
#include <mkl.h>

std::string HHIpaths[] = {
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-03/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-04/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-05/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-06/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-07/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-10/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-11/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-12/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-13/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-14/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-17/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-18/HHIH5_2025-03-28.parquet",
    "/media/jewels/storage/sharing_data/HK_data/Depth data/2025-03-19/HHIH5_2025-03-28.parquet",
};

static int STEPS = 10;
constexpr static int BINS  = 20;
constexpr static int NFEATURES = 3;
constexpr std::string GlobBidPrice = "BidPrice_1";
constexpr std::string GlobBidQty   = "BidQuantity_1";
constexpr std::string GlobAskPrice = "AskPrice_1";
constexpr std::string GlobAskQty   = "AskQuantity_1";


void addFeatures(osmi::Table& dataTable) {

    dataTable.add_column<double>("Spread", [](const osmi::Row& r) {
        return r.get<double>(GlobAskPrice) - r.get<double>(GlobBidPrice);
    });
    dataTable.add_column<double>("Imbalance", [](const osmi::Row& r) {
        double bid = r.get<double>(GlobBidQty);
        double ask = r.get<double>(GlobAskQty);
        return bid / (bid + ask);
    });
    dataTable.add_column<double>("Imbalance5", [](const osmi::Row& r) {
        double bid = 0.0, ask = 0.0;
        for (int i = 1; i <= 5; ++i) {
            bid += r.get<double>(std::format("BidQuantity_{}", i));
            ask += r.get<double>(std::format("AskQuantity_{}", i));
        }
        double denom = bid + ask;
        return denom > 0.0 ? bid / denom : 0.5;
    });
}


struct PCAResult {
    std::array<double, NFEATURES> mean{};
    std::array<double, NFEATURES> stddev{};
    std::array<double, NFEATURES> pc1{};
};

std::vector<std::array<double, NFEATURES>> collectFeatures(const std::vector<std::string>& paths)
{
    std::vector<std::array<double, NFEATURES>> rows;
    rows.reserve(2'000'000);

    for (const auto& file : paths) {
        auto loaded = osmi::Table::from_parquet(file);
        auto sub = loaded.select({GlobBidPrice, GlobBidQty, GlobAskPrice, GlobAskQty,
                                  "BidQuantity_2","BidQuantity_3","BidQuantity_4","BidQuantity_5",
                                  "AskQuantity_2","AskQuantity_3","AskQuantity_4","AskQuantity_5"});
        addFeatures(sub);

        for (int64_t i = 0; i < sub.num_rows(); ++i) {
            auto r = sub.row(i);
            rows.push_back({
                r.get<double>("Spread"),
                r.get<double>("Imbalance"),
                r.get<double>("Imbalance5")
            });
        }
    }
    return rows;
}

PCAResult computePCA(const std::vector<std::array<double, NFEATURES>>& data) {
    PCAResult result;
    const int64_t N = static_cast<int64_t>(data.size());

    result.mean.fill(0.0);
    for (const auto& row : data)
        for (int j = 0; j < NFEATURES; ++j)
            result.mean[j] += row[j];
    for (int j = 0; j < NFEATURES; ++j)
        result.mean[j] /= N;


    result.stddev.fill(0.0);
    for (const auto& row : data)
        for (int j = 0; j < NFEATURES; ++j) {
            double d = row[j] - result.mean[j];
            result.stddev[j] += d * d;
        }
    for (int j = 0; j < NFEATURES; ++j)
        result.stddev[j] = std::sqrt(result.stddev[j] / (N - 1));

    std::array<double, NFEATURES * NFEATURES> cov{};
    for (const auto& row : data) {
        std::array<double, NFEATURES> z;
        for (int j = 0; j < NFEATURES; ++j)
            z[j] = (row[j] - result.mean[j]) / result.stddev[j];
        for (int r = 0; r < NFEATURES; ++r)
            for (int c = 0; c < NFEATURES; ++c)
                cov[c * NFEATURES + r] += z[r] * z[c];
    }
    for (auto& v : cov) v /= (N - 1);

    std::array<double, NFEATURES> eigenvalues{};
    int info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', NFEATURES, cov.data(), NFEATURES, eigenvalues.data());

    if (info != 0) {throw std::runtime_error(std::format("LAPACKE_dsyev failed: {}", info));}

    for (int j = 0; j < NFEATURES; ++j) {result.pc1[j] = cov[(NFEATURES - 1) * NFEATURES + j];}

    double totalVar = 0.0;
    for (double ev : eigenvalues) totalVar += ev;

    for (int i = 0; i < NFEATURES; ++i) {std::cout << std::format("  PC{}: {:.2f}%\n",i + 1,100.0 * eigenvalues[NFEATURES - 1 - i] / totalVar);}

    std::cout << std::format("  Spread={:.4f}  Imbalance={:.4f}  Imbalance5={:.4f}\n", result.pc1[0], result.pc1[1], result.pc1[2]);

    return result;
}


inline double projectPC1(const osmi::Row& r, const PCAResult& pca) {
    std::array<double, NFEATURES> raw = {
        r.get<double>("Spread"),
        r.get<double>("Imbalance"),
        r.get<double>("Imbalance5")
    };
    double score = 0.0;
    for (int j = 0; j < NFEATURES; ++j)
        score += pca.pc1[j] * (raw[j] - pca.mean[j]) / pca.stddev[j];
    return score;
}


struct PC1Range { double lo, hi; };

PC1Range computePC1Range(const std::vector<std::array<double, NFEATURES>>& data, const PCAResult& pca)
{
    double lo = std::numeric_limits<double>::max();
    double hi = std::numeric_limits<double>::lowest();
    for (const auto& row : data) {
        double score = 0.0;
        for (int j = 0; j < NFEATURES; ++j)
            score += pca.pc1[j] * (row[j] - pca.mean[j]) / pca.stddev[j];
        lo = std::min(lo, score);
        hi = std::max(hi, score);
    }
    return {lo, hi};
}

inline int pc1Bin(double score, const PC1Range& range) {
    double t = (score - range.lo) / (range.hi - range.lo);
    t = std::clamp(t, 0.0, 1.0);
    int bin = static_cast<int>(t * BINS);
    return bin >= BINS ? BINS - 1 : bin;
}


std::array<std::array<double,BINS>,BINS> multiplyMatrix(const std::array<std::array<double,BINS>,BINS>& m)
{
    std::array<std::array<double,BINS>,BINS> result{};
    for (int i = 0; i < BINS; ++i)
        for (int k = 0; k < BINS; ++k) {
            if (m[i][k] == 0.0) continue;
            for (int j = 0; j < BINS; ++j)
                result[i][j] += m[i][k] * m[k][j];
        }
    return result;
}

int main() {

    std::vector<std::string> paths(std::begin(HHIpaths), std::end(HHIpaths));

    auto featureData = collectFeatures(paths);
    auto pca         = computePCA(featureData);
    auto pc1Range    = computePC1Range(featureData, pca);
    featureData.clear();   //freemem
    featureData.shrink_to_fit();

    std::cout << std::format("PC1 range: [{:.4f}, {:.4f}]\n", pc1Range.lo, pc1Range.hi);

    std::cout << "Pass 2: building transition matrix...\n";
    std::array<std::array<double,BINS>,BINS> frequencyMatrix{};
    std::array<double,BINS>                  midChangeMatrix{};
    std::array<std::pair<int,double>,BINS>   adjustmentGrid{};

    for (const auto& file : paths) {
        std::cout << file << "\n";
        auto loaded = osmi::Table::from_parquet(file);
        auto sub = loaded.select({GlobBidPrice, GlobBidQty, GlobAskPrice, GlobAskQty,
                                  "BidQuantity_2","BidQuantity_3","BidQuantity_4","BidQuantity_5",
                                  "AskQuantity_2","AskQuantity_3","AskQuantity_4","AskQuantity_5"});
        addFeatures(sub);

        for (int64_t i = 0; i < sub.num_rows() - 1; ++i) {
            auto r  = sub.row(i);
            auto rH = sub.row(i + 1);

            const double bidPrice  = r.get<double>(GlobBidPrice);
            const double askPrice  = r.get<double>(GlobAskPrice);
            const double bidPriceH = rH.get<double>(GlobBidPrice);
            const double askPriceH = rH.get<double>(GlobAskPrice);

            const double midPrice  = (bidPrice  + askPrice)  / 2.0;
            const double midPriceH = (bidPriceH + askPriceH) / 2.0;

            const int binCur  = pc1Bin(projectPC1(r,  pca), pc1Range);
            const int binNext = pc1Bin(projectPC1(rH, pca), pc1Range);

            if (binCur != binNext) {
                frequencyMatrix[binCur][binNext] += 1.0;
                midChangeMatrix[binNext]         += (midPriceH - midPrice);
            }
        }
    }

    for (int i = 0; i < BINS; ++i) {
        double rowCount = 0.0;
        for (int j = 0; j < BINS; ++j) rowCount += frequencyMatrix[i][j];
        if (rowCount > 0.0) midChangeMatrix[i] /= rowCount;
    }
    for (int i = 0; i < BINS; ++i) {
        double rowSum = 0.0;
        for (int j = 0; j < BINS; ++j) rowSum += frequencyMatrix[i][j];
        if (rowSum > 0.0)
            for (int j = 0; j < BINS; ++j)
                frequencyMatrix[i][j] /= rowSum;
    }

    for (int i = 0; i < BINS; ++i) {
        double Gstar = 0.0;
        auto stoppingMatrix = frequencyMatrix;
        for (int step = 0; step < STEPS; ++step) {
            for (int k = 0; k < BINS; ++k)
                Gstar += midChangeMatrix[k] * stoppingMatrix[i][k];
            stoppingMatrix = multiplyMatrix(stoppingMatrix);
        }
        adjustmentGrid[i] = {i, Gstar};
    }
    //csv
    std::ofstream csv("/home/loki/Documents/GstarHHI.csv");

    if (csv.is_open()) {
        csv << "bin, Gstar" <<std::endl;
        for (const auto& [bin, gstar] : adjustmentGrid)
            csv << std::format("bin{}, {}\n", bin, gstar);
        csv.close();
    }
    return 0;
}