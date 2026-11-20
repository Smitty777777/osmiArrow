#pragma once
#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/csv/writer.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>
#include <arrow/compute/api.h>

namespace osmi {

inline void _ck(const arrow::Status& s) { if (!s.ok()) throw std::runtime_error("osmi: " + s.ToString()); }
template <class T> T _ck(arrow::Result<T> r) { if (!r.ok()) throw std::runtime_error("osmi: " + r.status().ToString()); return std::move(r).ValueOrDie(); }

struct Value {
    std::variant<std::monostate, bool, int64_t, double, std::string> v;
    bool is_null() const { return std::holds_alternative<std::monostate>(v); }
    template <class T> const T& as() const { return std::get<T>(v); }
};

inline Value _val(const arrow::Array& a, int64_t i) {
    if (a.IsNull(i)) return {};
    switch (a.type_id()) {
        case arrow::Type::BOOL: return {static_cast<const arrow::BooleanArray&>(a).Value(i)};
        case arrow::Type::INT8: return {(int64_t)static_cast<const arrow::Int8Array&>(a).Value(i)};
        case arrow::Type::INT16: return {(int64_t)static_cast<const arrow::Int16Array&>(a).Value(i)};
        case arrow::Type::INT32: return {(int64_t)static_cast<const arrow::Int32Array&>(a).Value(i)};
        case arrow::Type::INT64: return {static_cast<const arrow::Int64Array&>(a).Value(i)};
        case arrow::Type::UINT8: return {(int64_t)static_cast<const arrow::UInt8Array&>(a).Value(i)};
        case arrow::Type::UINT16: return {(int64_t)static_cast<const arrow::UInt16Array&>(a).Value(i)};
        case arrow::Type::UINT32: return {(int64_t)static_cast<const arrow::UInt32Array&>(a).Value(i)};
        case arrow::Type::UINT64: return {(int64_t)static_cast<const arrow::UInt64Array&>(a).Value(i)};
        case arrow::Type::FLOAT: return {(double)static_cast<const arrow::FloatArray&>(a).Value(i)};
        case arrow::Type::DOUBLE: return {static_cast<const arrow::DoubleArray&>(a).Value(i)};
        case arrow::Type::STRING: return {std::string(static_cast<const arrow::StringArray&>(a).GetView(i))};
        case arrow::Type::LARGE_STRING: return {std::string(static_cast<const arrow::LargeStringArray&>(a).GetView(i))};
        default: throw std::runtime_error("osmi: unsupported type " + a.type()->ToString());
    }
}

template <class T> T _typed(const arrow::Array& a, int64_t i) {
    if constexpr (std::is_same_v<T, bool>) return static_cast<const arrow::BooleanArray&>(a).Value(i);
    else if constexpr (std::is_same_v<T, std::string> || std::is_same_v<T, std::string_view>) {
        if (a.type_id() == arrow::Type::STRING) return T(static_cast<const arrow::StringArray&>(a).GetView(i));
        if (a.type_id() == arrow::Type::LARGE_STRING) return T(static_cast<const arrow::LargeStringArray&>(a).GetView(i));
        throw std::runtime_error("osmi: not a string column");
    } else {
        switch (a.type_id()) {
            case arrow::Type::INT8: return (T)static_cast<const arrow::Int8Array&>(a).Value(i);
            case arrow::Type::INT16: return (T)static_cast<const arrow::Int16Array&>(a).Value(i);
            case arrow::Type::INT32: return (T)static_cast<const arrow::Int32Array&>(a).Value(i);
            case arrow::Type::INT64: return (T)static_cast<const arrow::Int64Array&>(a).Value(i);
            case arrow::Type::UINT8: return (T)static_cast<const arrow::UInt8Array&>(a).Value(i);
            case arrow::Type::UINT16: return (T)static_cast<const arrow::UInt16Array&>(a).Value(i);
            case arrow::Type::UINT32: return (T)static_cast<const arrow::UInt32Array&>(a).Value(i);
            case arrow::Type::UINT64: return (T)static_cast<const arrow::UInt64Array&>(a).Value(i);
            case arrow::Type::FLOAT: return (T)static_cast<const arrow::FloatArray&>(a).Value(i);
            case arrow::Type::DOUBLE: return (T)static_cast<const arrow::DoubleArray&>(a).Value(i);
            default: throw std::runtime_error("osmi: type mismatch for " + a.type()->ToString());
        }
    }
}

class Row {
public:
    Row(const arrow::RecordBatch* b, int64_t i, const std::unordered_map<std::string, int>* m, int64_t g = -1) : b_(b), i_(i), m_(m), g_(g < 0 ? i : g) {}
    int num_columns() const { return b_->num_columns(); }
    int64_t row_index() const { return g_; }
    bool is_null(int c) const { return b_->column(c)->IsNull(i_); }
    bool is_null(std::string_view n) const { return is_null(_idx(n)); }
    Value operator[](int c) const { return _val(*b_->column(c), i_); }
    Value operator[](std::string_view n) const { return (*this)[_idx(n)]; }
    template <class T> T get(int c) const { return _typed<T>(*b_->column(c), i_); }
    template <class T> T get(std::string_view n) const { return get<T>(_idx(n)); }
    std::string name(int c) const { return b_->schema()->field(c)->name(); }
private:
    int _idx(std::string_view n) const { auto it = m_->find(std::string(n)); if (it == m_->end()) throw std::runtime_error("osmi: no column " + std::string(n)); return it->second; }
    const arrow::RecordBatch* b_;
    int64_t i_;
    const std::unordered_map<std::string, int>* m_;
    int64_t g_;
};

template <class T> struct _builder_for;
template <> struct _builder_for<bool> { using type = arrow::BooleanBuilder; };
template <> struct _builder_for<int8_t> { using type = arrow::Int8Builder; };
template <> struct _builder_for<int16_t> { using type = arrow::Int16Builder; };
template <> struct _builder_for<int32_t> { using type = arrow::Int32Builder; };
template <> struct _builder_for<int64_t> { using type = arrow::Int64Builder; };
template <> struct _builder_for<uint8_t> { using type = arrow::UInt8Builder; };
template <> struct _builder_for<uint16_t> { using type = arrow::UInt16Builder; };
template <> struct _builder_for<uint32_t> { using type = arrow::UInt32Builder; };
template <> struct _builder_for<uint64_t> { using type = arrow::UInt64Builder; };
template <> struct _builder_for<float> { using type = arrow::FloatBuilder; };
template <> struct _builder_for<double> { using type = arrow::DoubleBuilder; };
template <> struct _builder_for<std::string> { using type = arrow::StringBuilder; };
template <> struct _builder_for<const char*> { using type = arrow::StringBuilder; };

class Table {
public:
    Table() = default;
    explicit Table(std::shared_ptr<arrow::Table> t) { _set(std::move(t)); }

    static Table from_parquet(const std::string& path) {
        auto in = _ck(arrow::io::ReadableFile::Open(path));
        auto rd = _ck(parquet::arrow::OpenFile(in, arrow::default_memory_pool()));
        std::shared_ptr<arrow::Table> t; _ck(rd->ReadTable(&t));
        return Table(t);
    }
    static Table from_csv(const std::string& path) {
        auto in = _ck(arrow::io::ReadableFile::Open(path));
        auto rd = _ck(arrow::csv::TableReader::Make(arrow::io::default_io_context(), in,
            arrow::csv::ReadOptions::Defaults(), arrow::csv::ParseOptions::Defaults(), arrow::csv::ConvertOptions::Defaults()));
        return Table(_ck(rd->Read()));
    }

    Table select(const std::vector<std::string>& cols) const {
        std::vector<int> idx; idx.reserve(cols.size());
        for (auto& c : cols) { int i = t_->schema()->GetFieldIndex(c); if (i < 0) throw std::runtime_error("osmi: no column " + c); idx.push_back(i); }
        return Table(_ck(t_->SelectColumns(idx)));
    }

    template <class T, class Fn> Table& add_column(const std::string& name, Fn&& fn) {
        typename _builder_for<T>::type b;
        for (auto&& bb : batches_) for (int64_t i = 0; i < bb->num_rows(); ++i) {
            Row r(bb.get(), i, &m_);
            _ck(b.Append(fn(r)));
        }
        std::shared_ptr<arrow::Array> arr; _ck(b.Finish(&arr));
        auto field = arrow::field(name, arr->type());
        _set(_ck(t_->AddColumn(t_->num_columns(), field, std::make_shared<arrow::ChunkedArray>(arr))));
        return *this;
    }
    template <class... Preds> Table filter(Preds&&... preds) const {
        arrow::BooleanBuilder mask;
        _ck(mask.Reserve(t_->num_rows()));
        for (auto&& bb : batches_)
            for (int64_t i = 0; i < bb->num_rows(); ++i) {
                Row r(bb.get(), i, &m_);
                mask.UnsafeAppend((... && static_cast<bool>(preds(r))));   //empty pack = true (keep all)
            }
        std::shared_ptr<arrow::Array> mask_arr; _ck(mask.Finish(&mask_arr));
        auto out = _ck(arrow::compute::Filter(t_, mask_arr));
        return Table(out.table());
    }

    void to_parquet(const std::string& path, int64_t chunk_size = 1024 * 1024) const {
        auto out = _ck(arrow::io::FileOutputStream::Open(path));
        _ck(parquet::arrow::WriteTable(*t_, arrow::default_memory_pool(), out, chunk_size));
    }
    void to_csv(const std::string& path) const {
        auto out = _ck(arrow::io::FileOutputStream::Open(path));
        _ck(arrow::csv::WriteCSV(*t_, arrow::csv::WriteOptions::Defaults(), out.get()));
    }

    int64_t num_rows() const { return t_->num_rows(); }
    int num_columns() const { return t_->num_columns(); }
    std::vector<std::string> column_names() const { return t_->ColumnNames(); }
    const std::shared_ptr<arrow::Table>& arrow_table() const { return t_; }

    Row row(int64_t i) const {
        if (i < 0 || i >= t_->num_rows()) throw std::out_of_range("osmi: row " + std::to_string(i));
        auto it = std::upper_bound(starts_.begin(), starts_.end(), i);
        size_t bi = (size_t)std::distance(starts_.begin(), it) - 1;
        return Row(batches_[bi].get(), i - starts_[bi], &m_, i);
    }
    Row operator[](int64_t i) const { return row(i); }

    class iterator {
    public:
        iterator(const Table* t, size_t bi, int64_t ri, int64_t g) : t_(t), bi_(bi), ri_(ri), g_(g) {}
        Row operator*() const { return Row(t_->batches_[bi_].get(), ri_, &t_->m_, g_); }
        iterator& operator++() {
            ++ri_; ++g_;
            while (bi_ < t_->batches_.size() && ri_ >= t_->batches_[bi_]->num_rows()) { ++bi_; ri_ = 0; }
            return *this;
        }
        bool operator==(const iterator& o) const { return bi_ == o.bi_ && ri_ == o.ri_; }
        bool operator!=(const iterator& o) const { return !(*this == o); }
    private:
        const Table* t_; size_t bi_; int64_t ri_; int64_t g_;
    };
    iterator begin() const {
        size_t bi = 0;
        while (bi < batches_.size() && batches_[bi]->num_rows() == 0) ++bi;
        return iterator(this, bi, 0, 0);
    }
    iterator end() const { return iterator(this, batches_.size(), 0, t_ ? t_->num_rows() : 0); }

private:
    void _set(std::shared_ptr<arrow::Table> t) {
        t_ = std::move(t);
        m_.clear();
        for (int i = 0; i < t_->num_columns(); ++i) m_[t_->field(i)->name()] = i;
        batches_.clear();
        arrow::TableBatchReader rd(*t_);
        std::shared_ptr<arrow::RecordBatch> b;
        while (true) { _ck(rd.ReadNext(&b)); if (!b) break; batches_.push_back(b); }
        starts_.clear(); starts_.reserve(batches_.size() + 1);
        int64_t s = 0;
        for (auto& bb : batches_) { starts_.push_back(s); s += bb->num_rows(); }
        starts_.push_back(s);
    }
    std::shared_ptr<arrow::Table> t_;
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches_;
    std::vector<int64_t> starts_;
    std::unordered_map<std::string, int> m_;
};

class _ParquetRBR : public arrow::RecordBatchReader {
public:
    _ParquetRBR(std::unique_ptr<parquet::arrow::FileReader> f, std::shared_ptr<arrow::RecordBatchReader> r) : f_(std::move(f)), r_(std::move(r)) {}
    [[nodiscard]]std::shared_ptr<arrow::Schema> schema() const override { return r_->schema(); }
    arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* b) override { return r_->ReadNext(b); }
private:
    std::unique_ptr<parquet::arrow::FileReader> f_;
    std::shared_ptr<arrow::RecordBatchReader> r_;
};

class LazyTable {
public:
    using Factory = std::function<std::shared_ptr<arrow::RecordBatchReader>()>;

    static LazyTable from_parquet(const std::string& path) {
        return LazyTable([path] {
            auto in = _ck(arrow::io::ReadableFile::Open(path));
            auto fr = _ck(parquet::arrow::OpenFile(in, arrow::default_memory_pool()));
            std::shared_ptr<arrow::RecordBatchReader> rb = std::shared_ptr<arrow::RecordBatchReader>(_ck(fr->GetRecordBatchReader()).release());
            return std::static_pointer_cast<arrow::RecordBatchReader>(std::make_shared<_ParquetRBR>(std::move(fr), std::move(rb)));
        });
    }
    static LazyTable from_csv(const std::string& path) {
        return LazyTable([path] {
            auto in = _ck(arrow::io::ReadableFile::Open(path));
            auto rd = _ck(arrow::csv::StreamingReader::Make(arrow::io::default_io_context(), in,
                arrow::csv::ReadOptions::Defaults(), arrow::csv::ParseOptions::Defaults(), arrow::csv::ConvertOptions::Defaults()));
            return std::static_pointer_cast<arrow::RecordBatchReader>(rd);
        });
    }

    [[nodiscard]]LazyTable select(std::vector<std::string>& cols) const {
        auto f = f_;
        return LazyTable([f, cols] {
            auto inner = f();
            auto sch = inner->schema();
            std::vector<int> idx; idx.reserve(cols.size());
            for (auto& c : cols) { int i = sch->GetFieldIndex(c); if (i < 0) throw std::runtime_error("osmi: no column " + c); idx.push_back(i); }
            class Proj : public arrow::RecordBatchReader {
            public:
                Proj(std::shared_ptr<arrow::RecordBatchReader> inner, std::vector<int> idx) : in_(std::move(inner)), idx_(std::move(idx)) {
                    auto s = in_->schema();
                    arrow::FieldVector fv;
                    for (int i : idx_) fv.push_back(s->field(i));
                    sc_ = arrow::schema(fv);
                }
                [[nodiscard]] std::shared_ptr<arrow::Schema> schema() const override { return sc_; }
                arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* out) override {
                    std::shared_ptr<arrow::RecordBatch> b; ARROW_RETURN_NOT_OK(in_->ReadNext(&b));
                    if (!b) { *out = nullptr; return arrow::Status::OK(); }
                    return b->SelectColumns(idx_).Value(out);
                }
            private:
                std::shared_ptr<arrow::RecordBatchReader> in_;
                std::vector<int> idx_;
                std::shared_ptr<arrow::Schema> sc_;
            };
            return std::static_pointer_cast<arrow::RecordBatchReader>(std::make_shared<Proj>(inner, idx));
        });
    }

    [[nodiscard]]Table collect() const {
        auto rd = f_();
        return Table(_ck(arrow::Table::FromRecordBatchReader(rd.get())));
    }

    class iterator {
    public:
        iterator() = default;
        explicit iterator(std::shared_ptr<arrow::RecordBatchReader> r) : r_(std::move(r)) {
            m_ = std::make_shared<std::unordered_map<std::string, int>>();
            auto s = r_->schema();
            for (int i = 0; i < s->num_fields(); ++i) (*m_)[s->field(i)->name()] = i;
            _next();
            while (b_ && i_ >= b_->num_rows()) _next();
        }
        Row operator*() const { return Row(b_.get(), i_, m_.get()); }
        iterator& operator++() { ++i_; while (b_ && i_ >= b_->num_rows()) _next(); return *this; }
        bool operator!=(const iterator& o) const { return (bool)b_ != (bool)o.b_; }
        bool operator==(const iterator& o) const { return !(*this != o); }
    private:
        void _next() { _ck(r_->ReadNext(&b_)); i_ = 0; }
        std::shared_ptr<arrow::RecordBatchReader> r_;
        std::shared_ptr<arrow::RecordBatch> b_;
        int64_t i_ = 0;
        std::shared_ptr<std::unordered_map<std::string, int>> m_;
    };
    [[nodiscard]]iterator begin() const { return iterator(f_()); }
    [[nodiscard]] iterator end() const { return iterator(); }

private:
    explicit LazyTable(Factory f) : f_(std::move(f)) {}
    Factory f_;
};

inline std::ostream& operator<<(std::ostream& os, const Value& val) {
    std::visit([&](auto&& x) {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, std::monostate>) os << "null";
        else if constexpr (std::is_same_v<T, bool>) os << (x ? "true" : "false");
        else os << x;
    }, val.v);
    return os;
}

}
