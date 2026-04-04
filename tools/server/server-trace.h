#pragma once

#include "server-common.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>

class server_http_trace {
public:
    server_http_trace(const std::string & trace_dir, int32_t max_body_bytes);
    ~server_http_trace();

    bool enabled() const;
    bool should_trace(const std::string & method, const std::string & path) const;

    std::string log_request_start(
            const std::string & method,
            const std::string & path,
            const std::string & query_string,
            const std::map<std::string, std::string> & headers,
            const std::string & body);

    void log_response_finish(
            const std::string & request_id,
            int status,
            const std::string & content_type,
            const std::map<std::string, std::string> & headers,
            const std::string & body,
            int64_t duration_ms,
            bool stream,
            const std::string & error = "");

    void log_stream_event(
            const std::string & request_id,
            uint64_t sequence,
            const std::string & chunk);

private:
    struct request_trace_file {
        std::filesystem::path temp_path;
        std::filesystem::path final_path;
    };

    std::string trace_dir;
    int32_t max_body_bytes;
    std::atomic<uint64_t> next_request_id = 0;
    std::atomic<uint64_t> next_record_seq = 0;
    std::mutex file_mutex;
    std::map<std::string, request_trace_file> active_requests;

    bool append_record(const std::filesystem::path & path, const json & record, uint64_t trace_seq);
    void finalize_active_requests_locked();
};

std::shared_ptr<server_http_trace> server_http_trace_create(const common_params & params);
