#include "server-trace.h"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <utility>

namespace {

static std::string trace_timestamp_now() {
    const auto now = std::chrono::system_clock::now();
    const auto tt = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S")
        << '.'
        << std::setw(3) << std::setfill('0') << ms.count();
    return oss.str();
}

static uint64_t fnv1a_64(const std::string & input) {
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char ch : input) {
        hash ^= ch;
        hash *= 1099511628211ULL;
    }
    return hash;
}

static std::string fnv1a_hex(const std::string & input) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << fnv1a_64(input);
    return oss.str();
}

static std::string lower_ascii(std::string value) {
    for (char & ch : value) {
        ch = (char) std::tolower((unsigned char) ch);
    }
    return value;
}

static std::string zero_pad_u64(uint64_t value) {
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(12) << value;
    return oss.str();
}

static json sanitize_headers(const std::map<std::string, std::string> & headers) {
    json out = json::object();
    for (const auto & [key, value] : headers) {
        std::string lowered = lower_ascii(key);
        if (lowered == "authorization" || lowered == "x-api-key") {
            out[key] = "[redacted]";
        } else {
            out[key] = value;
        }
    }
    return out;
}

static json summarize_body(const std::string & body, int32_t max_body_bytes) {
    json out = json::object();
    out["body_bytes"] = body.size();
    out["body_hash_fnv1a64"] = fnv1a_hex(body);

    if (max_body_bytes < 0 || (int32_t) body.size() <= max_body_bytes) {
        auto parsed = json::parse(body, nullptr, false);
        if (!parsed.is_discarded()) {
            out["body_json"] = std::move(parsed);
        } else {
            out["body_text"] = body;
        }
        return out;
    }

    out["body_truncated"] = true;
    const std::string excerpt = body.substr(0, (size_t) max_body_bytes);
    auto parsed = json::parse(excerpt, nullptr, false);
    if (!parsed.is_discarded()) {
        out["body_json_excerpt"] = std::move(parsed);
    } else {
        out["body_text_excerpt"] = excerpt;
    }
    return out;
}

}  // namespace

server_http_trace::server_http_trace(const std::string & trace_dir, int32_t max_body_bytes)
    : trace_dir(trace_dir),
      max_body_bytes(max_body_bytes) {
    std::filesystem::create_directories(trace_dir);
}

server_http_trace::~server_http_trace() {
    if (!enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(file_mutex);
    finalize_active_requests_locked();
}

bool server_http_trace::enabled() const {
    return !trace_dir.empty();
}

bool server_http_trace::should_trace(const std::string & method, const std::string & path) const {
    if (!enabled() || method != "POST") {
        return false;
    }
    return path == "/v1/chat/completions" || path == "/v1/responses";
}

std::string server_http_trace::log_request_start(
        const std::string & method,
        const std::string & path,
        const std::string & query_string,
        const std::map<std::string, std::string> & headers,
        const std::string & body) {
    const uint64_t id_num = ++next_request_id;
    const std::string request_id = string_format("req-%" PRIu64, id_num);
    const uint64_t trace_seq = ++next_record_seq;
    const std::string base_name = zero_pad_u64(trace_seq) + "_" + request_id + ".jsonl";
    const auto temp_path = std::filesystem::path(trace_dir) / (base_name + ".tmp");
    const auto final_path = std::filesystem::path(trace_dir) / base_name;

    json record = {
        {"type", "request_start"},
        {"ts", trace_timestamp_now()},
        {"request_id", request_id},
        {"method", method},
        {"path", path},
        {"query_string", query_string},
        {"headers", sanitize_headers(headers)},
    };
    const json body_summary = summarize_body(body, max_body_bytes);
    for (const auto & item : body_summary.items()) {
        record[item.key()] = item.value();
    }
    std::lock_guard<std::mutex> lock(file_mutex);
    if (!append_record(temp_path, record, trace_seq)) {
        next_record_seq--;
        return request_id;
    }
    active_requests.emplace(request_id, request_trace_file{temp_path, final_path});
    return request_id;
}

void server_http_trace::log_response_finish(
        const std::string & request_id,
        int status,
        const std::string & content_type,
        const std::map<std::string, std::string> & headers,
        const std::string & body,
        int64_t duration_ms,
        bool stream,
        const std::string & error) {
    std::lock_guard<std::mutex> lock(file_mutex);
    const auto it = active_requests.find(request_id);
    if (it == active_requests.end()) {
        LOG_ERR("missing HTTP trace request state for %s\n", request_id.c_str());
        return;
    }

    json record = {
        {"type", "response_finish"},
        {"ts", trace_timestamp_now()},
        {"request_id", request_id},
        {"status", status},
        {"content_type", content_type},
        {"headers", sanitize_headers(headers)},
        {"duration_ms", duration_ms},
        {"stream", stream},
    };
    if (!error.empty()) {
        record["error"] = error;
    }
    if (stream) {
        record["body_bytes"] = body.size();
        record["body_hash_fnv1a64"] = fnv1a_hex(body);
    } else {
        const json body_summary = summarize_body(body, max_body_bytes);
        for (const auto & item : body_summary.items()) {
            record[item.key()] = item.value();
        }
    }
    const uint64_t trace_seq = ++next_record_seq;
    if (!append_record(it->second.temp_path, record, trace_seq)) {
        next_record_seq--;
        return;
    }
    std::error_code ec;
    std::filesystem::rename(it->second.temp_path, it->second.final_path, ec);
    if (ec) {
        LOG_ERR("failed to finalize HTTP trace file %s -> %s: %s\n",
                it->second.temp_path.string().c_str(),
                it->second.final_path.string().c_str(),
                ec.message().c_str());
        return;
    }
    active_requests.erase(it);
}

void server_http_trace::log_stream_event(
        const std::string & request_id,
        uint64_t sequence,
        const std::string & chunk) {
    std::lock_guard<std::mutex> lock(file_mutex);
    const auto it = active_requests.find(request_id);
    if (it == active_requests.end()) {
        LOG_ERR("missing HTTP trace request state for %s\n", request_id.c_str());
        return;
    }

    json record = {
        {"type", "stream_event"},
        {"ts", trace_timestamp_now()},
        {"request_id", request_id},
        {"sequence", sequence},
    };
    const json body_summary = summarize_body(chunk, max_body_bytes);
    for (const auto & item : body_summary.items()) {
        record[item.key()] = item.value();
    }
    const uint64_t trace_seq = ++next_record_seq;
    if (!append_record(it->second.temp_path, record, trace_seq)) {
        next_record_seq--;
    }
}

bool server_http_trace::append_record(const std::filesystem::path & path, const json & record, uint64_t trace_seq) {
    json enriched = record;
    enriched["trace_seq"] = trace_seq;
    std::ofstream out(path, std::ios::app);
    if (!out) {
        LOG_ERR("failed to open HTTP trace file: %s\n", path.string().c_str());
        return false;
    }
    out << enriched.dump() << '\n';
    if (!out) {
        LOG_ERR("failed to write HTTP trace file: %s\n", path.string().c_str());
        return false;
    }
    return true;
}

void server_http_trace::finalize_active_requests_locked() {
    for (auto it = active_requests.begin(); it != active_requests.end(); ) {
        json record = {
            {"type", "response_finish"},
            {"ts", trace_timestamp_now()},
            {"request_id", it->first},
            {"status", 499},
            {"content_type", "application/json"},
            {"headers", json::object()},
            {"duration_ms", 0},
            {"stream", true},
            {"error", "trace aborted before response finished"},
            {"aborted", true},
            {"body_bytes", 0},
            {"body_hash_fnv1a64", fnv1a_hex("")},
        };
        const uint64_t trace_seq = ++next_record_seq;
        append_record(it->second.temp_path, record, trace_seq);
        std::error_code ec;
        std::filesystem::rename(it->second.temp_path, it->second.final_path, ec);
        if (ec) {
            LOG_ERR("failed to finalize aborted HTTP trace file %s -> %s: %s\n",
                    it->second.temp_path.string().c_str(),
                    it->second.final_path.string().c_str(),
                    ec.message().c_str());
        }
        it = active_requests.erase(it);
    }
}

std::shared_ptr<server_http_trace> server_http_trace_create(const common_params & params) {
    if (params.http_trace_dir.empty()) {
        return nullptr;
    }
    return std::make_shared<server_http_trace>(params.http_trace_dir, params.http_trace_max_bytes);
}
