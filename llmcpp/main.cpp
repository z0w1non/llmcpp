#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/program_options.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/nowide/args.hpp>
#include <boost/nowide/fstream.hpp>
#include <boost/nowide/iostream.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/stacktrace.hpp>
#include <boost/exception/all.hpp>
#include <boost/algorithm/string.hpp> 
#include <boost/date_time.hpp>
#include <boost/date_time/time_facet.hpp>
#include <system_error>
#include <string>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <regex>
#include <numeric>
#include <filesystem>
#include <random>
#include <vector>
#include <map>
#include <deque>
#include <memory>
#include <stdexcept>
#include <optional>

class runtime_exception
    : public boost::exception
{
public:
    explicit runtime_exception();
};

class runtime_exception;
class io_exception : public runtime_exception {};
class file_open_exception : public io_exception {};
class socket_exception : public io_exception {};
class text_generation_exception : public runtime_exception {};
class syntax_exception : public runtime_exception {};
class macro_exception : public runtime_exception {};
class command_line_syntax_exception : public runtime_exception {};
class array_index_out_of_bounds_exception : public runtime_exception {};

namespace error_info
{
    using stacktrace = boost::error_info<struct tag_stacktrace, boost::stacktrace::stacktrace>;
    using description = boost::error_info<struct tag_description, std::string>;
    using path = boost::error_info<struct tag_file_path, std::filesystem::path>;

    namespace http
    {
        namespace response
        {
            using result_int = boost::error_info<struct tag_result_int, unsigned int>;
            using reason = boost::error_info<struct tag_result_int, std::string>;
            using body = boost::error_info<struct tag_body, std::string>;
        }
    }

    namespace macro
    {
        using name = boost::error_info<struct tag_name, std::string>;
        using arguments = boost::error_info<struct tag_arguments, std::string>;
    }
}

runtime_exception::runtime_exception()
{
    *this << error_info::stacktrace{ boost::stacktrace::stacktrace() };
}


struct completions_parameters
{
    std::string model;
    int best_of{};
    bool echo{};
    double frequency_penalty{};
    //std::map<int, double> logit_bias{};
    double logprobs{};
    int max_tokens{};
    int n{};
    double presence_penalty{};
    std::vector<std::string> stop;
    bool stream{};
    std::string suffix;
    double temperature{};
    double top_p{};
    int seed{};
    std::string user;
    std::string preset;
    double dynatemp_low{};
    double dynatemp_high{};
    double dynatemp_exponent{};
    double smoothing_factor{};
    double smoothing_curve{};
    double min_p{};
    int top_k{};
    double typical_p{};
    double xtc_threshold{};
    double xtc_probability{};
    double epsilon_cutoff{};
    double eta_cutoff{};
    double tfs{};
    double top_a{};
    double top_n_sigma{};
    double dry_multiplier{};
    int dry_allowed_length{};
    double dry_base{};
    double repetition_penalty{};
    double encoder_repetition_penalty{};
    int no_repeat_ngram_size{};
    int repetition_penalty_range{};
    double penalty_alpha{};
    double guidance_scale{};
    int mirostat_mode{};
    double mirostat_tau{};
    double mirostat_eta{};
    int prompt_lookup_num_tokens{};
    int max_tokens_second{};
    bool do_sample{};
    bool dynamic_temperature{};
    bool temperature_last{};
    bool auto_max_new_tokens{};
    bool ban_eos_token{};
    bool add_bos_token{};
    bool skip_special_tokens{};
    bool static_cache{};
    int truncation_length{};
    std::vector<std::string> sampler_priority;
    std::string custom_token_bans;
    std::string negative_prompt;
    std::string dry_sequence_breakers;
    std::string grammar_string;
};

using macros = std::map<std::string, std::string>;

struct token_count_string
{
    std::string str{};
    int tokens{};
};

struct key_tag {};
struct lru_tag {};

using lru_cache = boost::multi_index::multi_index_container<
    token_count_string,
    boost::multi_index::indexed_by<
    boost::multi_index::ordered_unique<
    boost::multi_index::tag<key_tag>,
    boost::multi_index::member<token_count_string, std::string, &token_count_string::str>
    >,
    boost::multi_index::sequenced<
    boost::multi_index::tag<lru_tag>
    >
    >
>;

struct item
{
    std::string head;
    std::vector<std::string> descriptions;
};

struct config
{
    std::string host;
    std::string port;
    std::string api_key;

    std::string mode{};

    // for novel mode
    std::string plot_file{};

    std::string log_level;
    std::string log_file;

    std::string base_path;
    std::string system_prompts_file;
    std::string examples_file;
    std::string history_file;
    std::string output_file;
    std::string example_separator;
    std::vector<std::string> phases;
    std::string generation_prefix;
    std::string retry_generation_prefix;

    int seed{};
    bool verbose{};
    int number_iterations{};
    int min_completion_tokens{};
    int max_new_tokens{};
    int max_completion_iterations{};
    int max_total_context_tokens{};

    completions_parameters params;
    macros macros;
    mutable lru_cache lru_cache;
    std::vector<std::string> predefined_macros;
};

struct llm_response
{
    std::string text;
    std::string finish_reason;
    int prompt_tokens{};
    int completion_tokens{};
    int total_tokens{};
};

struct prompts
{
    std::vector<std::string> system_prompts;
    std::vector<std::string> examples;
    std::deque<std::string> history;

    std::string to_string(const config& config) const;
};

std::string trim(const std::string& str)
{
    const std::regex leading_spaces{ R"(^\s+)", std::regex_constants::ECMAScript };
    const std::regex trailing_spaces{ R"(\s+$)", std::regex_constants::ECMAScript };
    std::string trimmed_string = std::regex_replace(str, leading_spaces, "");
    trimmed_string = std::regex_replace(trimmed_string, trailing_spaces, "");
    return trimmed_string;
}

template <typename Container>
void split_string_by_new_line(const std::string& str, Container& container)
{
    size_t start_position = 0;
    size_t end_position = 0;

    while ((end_position = str.find('\n', start_position)) != std::string::npos)
    {
        container.push_back(str.substr(start_position, end_position - start_position + 1));
        start_position = end_position + 1;
    }

    if (start_position < str.length())
    {
        container.push_back(str.substr(start_position));
    }
}

void create_parent_directories(const std::filesystem::path& path)
{
    if (path.empty() || !path.has_parent_path())
    {
        return;
    }

    std::filesystem::create_directories(path.parent_path());
}

template <typename Container>
void read_file_to_container(Container& container, const std::filesystem::path& file)
{
    container.clear();
    if (std::filesystem::exists(file) && std::filesystem::is_regular_file(file))
    {
        boost::nowide::ifstream ifs{ file };
        if (!ifs.is_open())
        {
            throw file_open_exception{} << error_info::path{ file };
        }
        const std::string file_content{ (std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>() };
        split_string_by_new_line(file_content, container);
    }
}

void read_file_to_string(std::string& result, const std::filesystem::path& file)
{
    result.clear();
    if (!std::filesystem::exists(file) || !std::filesystem::is_regular_file(file))
    {
        throw file_open_exception{} << error_info::path{ file };
    }
    boost::nowide::ifstream ifs{ file };
    if (!ifs.is_open())
    {
        throw file_open_exception{} << error_info::path{ file };
    }
    const std::string file_content{ (std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>() };
    result = file_content;
}

int generate_random_seed()
{
    static std::random_device seed_gen;
    static std::default_random_engine random_engine(seed_gen());
    static std::uniform_int_distribution<> distribution(0, std::numeric_limits<int>::max());
    return distribution(random_engine);
}

std::string include_predefiend_macro(const std::string& right)
{
    std::string result;
    std::filesystem::path macro_file{ right };
    if (!std::filesystem::exists(macro_file))
    {
        macro_file.replace_extension(".txt");
    }
    read_file_to_string(result, macro_file);
    return result;
}

std::string datetime_predefiend_macro(const std::string&)
{
    boost::posix_time::ptime local_time = boost::posix_time::second_clock::local_time();
    boost::posix_time::time_facet* facet = new boost::posix_time::time_facet("%Y%m%d%H%M%S");
    std::ostringstream oss;
    oss.imbue(std::locale(oss.getloc(), facet));
    oss << local_time;
    return oss.str();
}

std::optional<std::string> expand_predefined_macro(
    const std::string& name,
    const std::string& arguments
)
{
    const static std::map<std::string, std::function<std::string(const std::string&)>> predefiend_macro_impls
    {
        {"include", include_predefiend_macro},
        {"datetime", datetime_predefiend_macro}
    };

    auto found = std::find_if(predefiend_macro_impls.begin(), predefiend_macro_impls.end(), [&name](const auto& pair) { return boost::iequals(name, pair.first); });
    if (found != predefiend_macro_impls.end())
    {
        BOOST_LOG_TRIVIAL(trace) << "found predefined macro. name: \"" << name << "\" arguments: \"" << arguments << "\"";

        try
        {
            try
            {
                return found->second(arguments);
            }
            catch (const runtime_exception& exception)
            {
                BOOST_LOG_TRIVIAL(warning) << boost::diagnostic_information(exception);
                throw macro_exception{} << error_info::macro::name{ name } << error_info::macro::arguments{ arguments };
            }
        }
        catch (const macro_exception& exception)
        {
            BOOST_LOG_TRIVIAL(warning) << boost::diagnostic_information(exception);
        }
    }

    return std::nullopt;
}

std::string expand_macro(const std::string& str, const macros& macros, int depth = 0)
{
    constexpr int max_recursive_count = 32;
    if (depth > max_recursive_count)
    {
        return {};
    }

    std::string result;
    const std::regex macro{ R"(\{\{([^}]+)\}\})", std::regex_constants::ECMAScript };
    std::string::size_type last_position = 0;

    for (std::sregex_iterator iter{ str.begin(), str.end(), macro }, end; iter != end; ++iter)
    {
        const std::smatch& match = *iter;
        const std::string macro_string = match[1].str();
        std::string expanded_string;

        result += str.substr(last_position, match.position() - last_position);

        std::string name;
        std::string arguments;
        auto colon_position = macro_string.find(':');
        if (colon_position != std::string::npos)
        {
            name = macro_string.substr(0, colon_position);
            arguments = macro_string.substr(colon_position + 1);
        }
        else
        {
            name = macro_string;
        }

        std::optional<std::string> predefined_macro_result{ expand_predefined_macro(name, arguments) };
        if (predefined_macro_result)
        {
            expanded_string = *predefined_macro_result;
            result += expanded_string;
            BOOST_LOG_TRIVIAL(trace) << "macro expanded: \"{{" << macro_string << "}}\" -> \"" << expanded_string << "\"";
        }
        else
        {
            auto found = std::find_if(macros.begin(), macros.end(), [&macro_string](const auto& pair) { return boost::iequals(macro_string, pair.first); });

            if (found != macros.end())
            {
                expanded_string = found->second;
                if ("{{" + macro_string + "}}" != expanded_string)
                {
                    expanded_string = expand_macro(expanded_string, macros, depth + 1);
                }
                result += expanded_string;
                BOOST_LOG_TRIVIAL(trace) << "macro expanded: \"{{" << macro_string << "}}\" -> \"" << expanded_string << "\"";
            }
        }

        last_position = match.position() + match.length();
    }

    result += str.substr(last_position);

    return result;
}

std::filesystem::path string_to_path_by_config(const std::string& path, const config& config)
{
    const std::filesystem::path file_path{ expand_macro(path, config.macros) };
    if (file_path.is_relative())
    {
        const std::filesystem::path base_path{ expand_macro(config.base_path, config.macros) };
        return base_path / file_path;

    }
    return file_path;
}

std::vector<item> parse_item_list(const std::string& str)
{
    std::vector<item> result;

    const std::regex item_regex{ R"(^(?:[-*+]|[0-9a-zA-Z]+[.\)]) (.+))", std::regex_constants::ECMAScript };
    const std::regex sub_item_regex{ R"(^(?:[ \t]+)(?:[-*+]|[0-9a-zA-Z]+[.\)]) (.+))", std::regex_constants::ECMAScript };

    std::istringstream iss{ str };
    std::string line;
    bool is_prev_line_item{};
    while (std::getline(iss, line))
    {
        if (line.empty())
        {
            continue;
        }
        if (std::smatch match; std::regex_match(line, match, item_regex))
        {
            const std::string trimmed{ trim(match[1].str()) };
            if (!trimmed.empty())
            {
                result.push_back({ trimmed });
                is_prev_line_item = true;
            }
        }
        else if (std::smatch match; is_prev_line_item && std::regex_match(line, match, sub_item_regex))
        {
            const std::string trimmed{ trim(match[1].str()) };
            if (!trimmed.empty())
            {
                result.back().descriptions.push_back(trimmed);
            }
        }
        else
        {
            is_prev_line_item = false;
        }
    }

    return result;
}

void write_item_list(const config& config, const std::string& task)
{
    std::vector<item> items = parse_item_list(task);

    for (const item& item : items)
    {
        std::filesystem::path item_file_path{ string_to_path_by_config(item.head, config) };
        item_file_path += ".txt";
        if (std::filesystem::exists(item_file_path))
        {
            throw file_open_exception{} << error_info::description{ "File already exists." } << error_info::path{ item_file_path };
        }
        else
        {
            create_parent_directories(item_file_path);
            boost::nowide::ofstream ofs{ item_file_path };
            if (!ofs.is_open())
            {
                throw file_open_exception{} << error_info::path{ item_file_path };
            }
            for (const std::string& description : item.descriptions)
            {
                ofs << description;
            }
        }
    }
}

llm_response send_oobabooga_completions_request(
    const config& config,
    const std::string& prompt,
    const completions_parameters& params
)
{
    namespace beast = boost::beast;
    namespace http = beast::http;
    namespace net = boost::asio;
    using tcp = net::ip::tcp;
    namespace pt = boost::property_tree;

    llm_response result;

    try
    {
        net::io_context ioc;
        tcp::resolver resolver(ioc);
        beast::tcp_stream tcp_stream(ioc);

        const auto results = resolver.resolve(config.host, config.port);
        tcp_stream.connect(results);

        pt::ptree request_body_json;
        request_body_json.put("prompt", prompt);

        if (!params.model.empty())
        {
            request_body_json.put("model", params.model);
        }

        request_body_json.put("best_of", params.best_of);
        request_body_json.put("echo", params.echo);
        request_body_json.put("frequency_penalty", params.frequency_penalty);
        //request_body_json.put("logit_bias", params.logit_bias);
        request_body_json.put("logprobs", params.logprobs);
        request_body_json.put("max_tokens", params.max_tokens);
        request_body_json.put("n", params.n);
        request_body_json.put("presence_penalty", params.presence_penalty);

        if (!params.stop.empty())
        {
            pt::ptree stop_array;
            for (const std::string& str : params.stop)
            {
                stop_array.push_back(std::make_pair("", pt::ptree(str)));
            }
            request_body_json.add_child("stop", stop_array);
        }

        request_body_json.put("stream", params.stream);

        if (!params.suffix.empty())
        {
            request_body_json.put("suffix", params.suffix);
        }

        request_body_json.put("temperature", params.temperature);
        request_body_json.put("top_p", params.top_p);

        if (params.seed != -1)
        {
            request_body_json.put("seed", params.seed);
        }

        request_body_json.put("max_tokens", params.max_tokens);

        if (!params.user.empty())
        {
            request_body_json.put("user", params.user);
        }

        if (!params.preset.empty())
        {
            request_body_json.put("preset", params.preset);
        }

        request_body_json.put("dynatemp_low", params.dynatemp_low);
        request_body_json.put("dynatemp_high", params.dynatemp_high);
        request_body_json.put("dynatemp_exponent", params.dynatemp_exponent);
        request_body_json.put("smoothing_factor", params.smoothing_factor);
        request_body_json.put("smoothing_curve", params.smoothing_curve);
        request_body_json.put("min_p", params.min_p);
        request_body_json.put("top_k", params.top_k);
        request_body_json.put("typical_p", params.typical_p);
        request_body_json.put("xtc_threshold", params.xtc_threshold);
        request_body_json.put("xtc_probability", params.xtc_probability);
        request_body_json.put("epsilon_cutoff", params.epsilon_cutoff);
        request_body_json.put("eta_cutoff", params.eta_cutoff);
        request_body_json.put("tfs", params.tfs);
        request_body_json.put("top_a", params.top_a);
        request_body_json.put("top_n_sigma", params.top_n_sigma);
        request_body_json.put("dry_multiplier", params.dry_multiplier);
        request_body_json.put("dry_allowed_length", params.dry_allowed_length);
        request_body_json.put("dry_base", params.dry_base);
        request_body_json.put("repetition_penalty", params.repetition_penalty);
        request_body_json.put("encoder_repetition_penalty", params.encoder_repetition_penalty);
        request_body_json.put("no_repeat_ngram_size", params.no_repeat_ngram_size);
        request_body_json.put("repetition_penalty_range", params.repetition_penalty_range);
        request_body_json.put("penalty_alpha", params.penalty_alpha);
        request_body_json.put("guidance_scale", params.guidance_scale);
        request_body_json.put("mirostat_mode", params.mirostat_mode);
        request_body_json.put("mirostat_tau", params.mirostat_tau);
        request_body_json.put("mirostat_eta", params.mirostat_eta);
        request_body_json.put("prompt_lookup_num_tokens", params.prompt_lookup_num_tokens);
        request_body_json.put("max_tokens_second", params.max_tokens_second);
        request_body_json.put("do_sample", params.do_sample);
        request_body_json.put("dynamic_temperature", params.max_tokens_second);
        request_body_json.put("temperature_last", params.temperature_last);
        request_body_json.put("auto_max_new_tokens", params.auto_max_new_tokens);
        request_body_json.put("ban_eos_token", params.ban_eos_token);
        request_body_json.put("add_bos_token", params.add_bos_token);
        request_body_json.put("skip_special_tokens", params.skip_special_tokens);
        request_body_json.put("static_cache", params.static_cache);
        request_body_json.put("truncation_length", params.truncation_length);

        if (!params.sampler_priority.empty())
        {
            pt::ptree sampler_priority_array;
            for (const std::string& str : params.sampler_priority)
            {
                sampler_priority_array.push_back(std::make_pair("", pt::ptree(str)));
            }
            request_body_json.add_child("sampler_priority", sampler_priority_array);
        }

        request_body_json.put("custom_token_bans", params.custom_token_bans);
        request_body_json.put("negative_prompt", params.negative_prompt);
        request_body_json.put("dry_sequence_breakers", params.dry_sequence_breakers);
        request_body_json.put("grammar_string", params.grammar_string);

        std::stringstream ss_request_body;
        pt::write_json(ss_request_body, request_body_json, false);

        http::request<http::string_body> request{ http::verb::post, "/v1/completions", 11 }; // HTTP/1.1
        request.set(http::field::host, config.host);
        request.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
        request.set(http::field::content_type, "application/json; charset=UTF-8");
        request.body() = ss_request_body.str();
        request.prepare_payload();

        if (!config.api_key.empty())
        {
            request.set(http::field::authorization, ("Bearer ") + config.api_key);
        }

        http::write(tcp_stream, request);

        beast::flat_buffer buffer;
        http::response<http::string_body> response;
        http::read(tcp_stream, buffer, response);

        tcp_stream.socket().shutdown(tcp::socket::shutdown_both);

        if (response.result() == http::status::ok)
        {
            pt::ptree response_json;
            std::stringstream ss_response_body(response.body());
            pt::read_json(ss_response_body, response_json);

            try
            {
                pt::ptree choices{ response_json.get_child("choices") };
                result.text = choices.front().second.get<std::string>("text");
                result.finish_reason = choices.front().second.get<std::string>("finish_reason", "");
                result.prompt_tokens = choices.front().second.get<int>("prompt_tokens", 0);
                result.completion_tokens = choices.front().second.get<int>("completion_tokens", 0);
                result.total_tokens = choices.front().second.get<int>("total_tokens", 0);
            }
            catch (const pt::ptree_bad_path& exception)
            {
                // Could not parse response text.
                throw socket_exception{} << error_info::description{ std::string{ "Error parsing response: " } + exception.what() };
            }
        }
        else
        {
            // Error: HTTP request failed.
            throw socket_exception{}
                << error_info::description{ "HTTP error" }
                << error_info::http::response::result_int{ response.result_int() }
                << error_info::http::response::reason{ response.reason() }
                << error_info::http::response::body{ response.body() }
            ;
        }

    }
    catch (const beast::system_error& exception)
    {
        throw socket_exception{} << error_info::description{ std::string{ "Network communication error: " + exception.code().message() } };
    }
    catch (const std::exception& exception)
    {
        throw socket_exception{} << error_info::description{ std::string{ "Unexcepted error: " } + exception.what() };
    }

    return result;
}

int send_oobabooga_token_count_request(const config& config, const std::string& prompt)
{
    namespace beast = boost::beast;
    namespace http = beast::http;
    namespace net = boost::asio;
    using tcp = net::ip::tcp;
    namespace pt = boost::property_tree;

    try
    {
        net::io_context ioc;
        tcp::resolver resolver(ioc);
        beast::tcp_stream stream(ioc);

        auto const results = resolver.resolve(config.host, config.port);
        stream.connect(results);

        pt::ptree request_body_json;
        request_body_json.put("text", prompt);

        std::stringstream ss_request_body;
        pt::write_json(ss_request_body, request_body_json, false);

        http::request<http::string_body> request{ http::verb::post, "/v1/internal/token-count", 11 }; // HTTP/1.1

        request.set(http::field::host, config.host);
        request.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
        request.set(http::field::content_type, "application/json; charset=UTF-8");
        request.body() = ss_request_body.str();
        request.prepare_payload();

        http::write(stream, request);

        beast::flat_buffer buffer;
        http::response<http::string_body> response;
        http::read(stream, buffer, response);

        stream.socket().shutdown(tcp::socket::shutdown_both);

        if (response.result() == http::status::ok)
        {
            pt::ptree response_json;
            std::stringstream ss_response_body(response.body());
            pt::read_json(ss_response_body, response_json);

            try
            {
                return response_json.get<int>("length");
            }
            catch (const pt::ptree_bad_path& e)
            {
                throw syntax_exception{} << error_info::description(e.what());
            }
        }
        else
        {
            throw socket_exception{}
                << error_info::description{ "HTTP Error getting token count" }
                << error_info::http::response::result_int{ response.result_int() }
                << error_info::http::response::reason{ response.reason() }
            << error_info::http::response::body{ response.body() };
            return -1;
        }
    }
    catch (const beast::system_error& exception)
    {
        throw socket_exception{} << error_info::description{ std::string{ "Network communication error: " + exception.code().message() } };
    }
    catch (const std::exception& exception)
    {
        throw socket_exception{} << error_info::description{ std::string{ "Unexcepted error: " } + exception.what() };
    }
}

int get_tokens_from_cache(const config& config, const std::string& str)
{
    constexpr std::size_t capacity = 1000;
    int tokens{};

    auto iter = config.lru_cache.get<key_tag>().find(str);
    if (iter != config.lru_cache.get<key_tag>().end())
    {
        tokens = iter->tokens;
        config.lru_cache.get<lru_tag>().relocate(
            config.lru_cache.get<lru_tag>().end(),
            config.lru_cache.get<lru_tag>().iterator_to(*iter));
    }
    else
    {
        tokens = send_oobabooga_token_count_request(config, str);
        config.lru_cache.insert({ str, tokens });
    }

    if (config.lru_cache.size() > capacity)
    {
        config.lru_cache.get<lru_tag>().pop_front();
    }

    return tokens;
}

void write_cache(const config& config)
{
    namespace pt = boost::property_tree;

    pt::ptree cache;
    for (const token_count_string& element : config.lru_cache.get<lru_tag>())
    {
        pt::ptree node;
        node.put("string", element.str);
        node.put("tokens", element.tokens);
        cache.push_back(std::make_pair("", node));
    }
    pt::ptree json;
    json.add_child("cache", cache);

    std::filesystem::path cache_path{ string_to_path_by_config("cache.json", config) };
    create_parent_directories(cache_path);
    boost::nowide::ofstream ofs{ cache_path };
    pt::write_json(ofs, json, false);
}

void read_cache(const config& config)
{
    namespace pt = boost::property_tree;

    pt::ptree json;
    std::filesystem::path cache_path{ string_to_path_by_config("cache.json", config) };

    if (!std::filesystem::exists(cache_path))
    {
        return;
    }

    try
    {
        boost::nowide::ifstream ifs{ cache_path };
        pt::read_json(ifs, json);
        lru_cache lru_cache;
        for (const pt::ptree::value_type& node : json.get_child("cache"))
        {
            const std::string str{ node.second.get<std::string>("string") };
            const int tokens{ node.second.get<int>("tokens") };
            lru_cache.insert({ str, tokens });
        }
        config.lru_cache = lru_cache;
    }
    catch (const boost::exception& exception)
    {
        BOOST_LOG_TRIVIAL(error) << boost::diagnostic_information(exception);
        throw syntax_exception{};
    }
}

std::string generate_and_complete_text(
    const config& config,
    const std::string& prompts,
    const std::string& prefix
)
{
    std::string initial_prompts = prompts;
    const std::size_t initial_prompts_size = initial_prompts.size();
    initial_prompts += expand_macro(prefix, config.macros);
    const int initial_tokens = send_oobabooga_token_count_request(config, initial_prompts);

    BOOST_LOG_TRIVIAL(info) << "Prompt created.\n```\n" << initial_prompts << "\n```";

    std::string current_text = initial_prompts;
    int current_tokens = initial_tokens;
    for (int completion_iterations = 0; completion_iterations < config.max_completion_iterations; ++completion_iterations)
    {
        BOOST_LOG_TRIVIAL(trace) << "completion_iterations: " << completion_iterations;

        if (current_tokens - initial_tokens >= config.min_completion_tokens)
        {
            break;
        }

        int remaining_context = config.max_total_context_tokens - current_tokens;
        if (remaining_context <= 0)
        {
            BOOST_LOG_TRIVIAL(warning) << "Context window full. Cannot generate more tokens.";
            break;
        }

        int tokens_to_generate = std::min(config.params.max_tokens, remaining_context);
        if (tokens_to_generate <= 0)
        {
            BOOST_LOG_TRIVIAL(warning) << "No tokens left to generate. Aborting.";
            break;
        }

        completions_parameters temp_params = config.params;
        temp_params.max_tokens = tokens_to_generate;
        llm_response response = send_oobabooga_completions_request(
            config, current_text, temp_params
        );

        if (response.text.empty())
        {
            break;
        }

        current_text += response.text;
        current_tokens = send_oobabooga_token_count_request(config, current_text);

        if (response.finish_reason == "stop")
        {
            break;
        }
    }

    return current_text.substr(initial_prompts_size);
}

std::string unescape_string(const std::string& str)
{
    std::stringstream ss;
    bool in_escape = false;

    for (char c : str)
    {
        if (in_escape)
        {
            switch (c)
            {
            case '\"': ss << '\"'; break;
            case '\'': ss << '\''; break;
            case '\\': ss << '\\'; break;
            case 'a':  ss << '\a'; break;
            case 'b':  ss << '\b'; break;
            case 'f':  ss << '\f'; break;
            case 'n':  ss << '\n'; break;
            case 'r':  ss << '\r'; break;
            case 't':  ss << '\t'; break;
            default:
                ss << '\\' << c;
                break;
            }
            in_escape = false;
        }
        else
        {
            if (c == '\\')
            {
                in_escape = true;
            }
            else
            {
                ss << c;
            }
        }
    }

    if (in_escape)
    {
        ss << '\\';
    }

    return ss.str();
}

void parse_predefined_macros(const std::vector<std::string>& predefined_macros, macros& macros)
{
    for (const std::string& key_value_pair : predefined_macros)
    {
        const std::size_t separator_position = key_value_pair.find('=');
        if (separator_position != std::string::npos)
        {
            std::string key = key_value_pair.substr(0, separator_position);
            std::string value = key_value_pair.substr(separator_position + 1);
            macros[key] = value;
        }
        else
        {
            BOOST_LOG_TRIVIAL(warning) << "Invalid define format: " << key_value_pair << ". Expected key=value.";
        }
    }
}

void init_logging_with_nowide_cout()
{
    boost::shared_ptr<boost::log::sinks::text_ostream_backend> backend{ boost::make_shared<boost::log::sinks::text_ostream_backend>() };
    backend->add_stream(boost::shared_ptr<std::ostream>{ &boost::nowide::cout, boost::null_deleter{} });
    backend->auto_flush(true);
    boost::shared_ptr<boost::log::sinks::synchronous_sink<boost::log::sinks::text_ostream_backend>> sink{
        boost::make_shared<boost::log::sinks::synchronous_sink<boost::log::sinks::text_ostream_backend>>(backend)
    };
    sink->set_formatter(
        boost::log::expressions::stream
        << "[" << boost::log::trivial::severity << "] "
        << boost::log::expressions::smessage
    );
    boost::log::core::get()->add_sink(sink);
}

void init_logging_with_nowide_file_log(const std::filesystem::path& log)
{
    boost::shared_ptr<boost::log::sinks::text_ostream_backend> backend{ boost::make_shared<boost::log::sinks::text_ostream_backend>() };
    create_parent_directories(log);
    boost::shared_ptr<boost::nowide::ofstream> ofs{ boost::make_shared<boost::nowide::ofstream>(log, std::ios::app) };
    if (!ofs->is_open())
    {
        throw file_open_exception{} << error_info::path{ log };
    }
    backend->add_stream(ofs);
    backend->auto_flush(true);

    boost::shared_ptr<boost::log::sinks::synchronous_sink<boost::log::sinks::text_ostream_backend>> sink{
        boost::make_shared<boost::log::sinks::synchronous_sink<boost::log::sinks::text_ostream_backend>>(backend)
    };

    sink->set_formatter(
        boost::log::expressions::stream
        << boost::log::expressions::format_date_time<boost::posix_time::ptime>("TimeStamp", "%Y-%m-%d %H:%M:%S")
        << " [" << boost::log::trivial::severity << "] "
        << boost::log::expressions::smessage
    );

    boost::log::core::get()->add_sink(sink);

    boost::log::core::get()->add_global_attribute("TimeStamp", boost::log::attributes::local_clock());
}

void init_logging(const config& config)
{
    boost::log::trivial::severity_level level = boost::log::trivial::info;
    if (config.log_level == "trace")
    {
        level = boost::log::trivial::trace;
    }
    else if (config.log_level == "debug")
    {
        level = boost::log::trivial::debug;
    }
    else if (config.log_level == "info")
    {
        level = boost::log::trivial::info;
    }
    else if (config.log_level == "warning")
    {
        level = boost::log::trivial::warning;
    }
    else if (config.log_level == "error")
    {
        level = boost::log::trivial::error;
    }
    else if (config.log_level == "fatal")
    {
        level = boost::log::trivial::fatal;
    }
    else
    {
        BOOST_LOG_TRIVIAL(warning) << "Unkown log level: \"" << config.log_level << "\"";
    }

    if (config.verbose)
    {
        init_logging_with_nowide_cout();
    }

    if (!config.log_file.empty())
    {
        const std::filesystem::path log_file_path{ string_to_path_by_config(config.log_file, config) };
        init_logging_with_nowide_file_log(log_file_path);
    }

    boost::log::core::get()->set_filter(boost::log::trivial::severity >= level);
}

void init_chat_mode(config& config)
{
    if (config.phases.empty())
    {
        config.phases = { "{{user}}", "{{char}}" };
    }
    if (config.generation_prefix.empty())
    {
        config.generation_prefix = "\\n{{phase}}: ";
    }
}

void set_phases_macro(
    const std::vector<std::string>& phases,
    std::size_t phase_index,
    std::map<std::string, std::string>& macros
)
{
    if (phase_index >= phases.size())
    {
        throw array_index_out_of_bounds_exception{};
    }

    if (phase_index > 0)
    {
        macros["prev_phase"] = phases[phase_index - 1];
    }
    else
    {
        macros.erase("prev_phase");
    }

    macros["phase"] = phases[phase_index];

    if (phase_index < phases.size() - 1)
    {
        macros["next_phase"] = phases[phase_index + 1];
    }
    else
    {
        macros.erase("next_phase");
    }
}

void set_paragraphs_to_phases(
    const std::vector<item>& paragraphs,
    std::vector<std::string>& phases
)
{
    for (const item& paragraph : paragraphs)
    {
        std::string temp{ paragraph.head };
        for (const std::string& description : paragraph.descriptions)
        {
            temp += "\n";
            temp += description;
        }
        phases.push_back(temp);
    }
}

void init_novel_mode(config& config)
{
    if (!config.plot_file.empty())
    {
        config.phases.clear();
        std::filesystem::path plot_file_path{ string_to_path_by_config(config.plot_file, config) };
        std::string content;
        read_file_to_string(content, plot_file_path);
        std::vector<item> paragraphs{ parse_item_list(content) };
        set_paragraphs_to_phases(paragraphs, config.phases);
    }
}

int parse_commandline(
    int argc,
    char** argv,
    config& config
)
{
    namespace po = boost::program_options;

    try
    {
        config.params.stop = { "\\n\\n", ":", "***" };
        config.params.sampler_priority =
        {
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "dry",
            "temperature",
            "dynamic_temperature",
            "quadratic_sampling",
            "top_n_sigma",
            "top_k",
            "top_p",
            "typical_p",
            "epsilon_cutoff",
            "eta_cutoff",
            "tfs",
            "top_a",
            "min_p",
            "mirostat",
            "xtc",
            "encoder_repetition_penalty",
            "no_repeat_ngram"
        };
        config.params.dry_sequence_breakers = "(\"\\n\", \":\", \"\\\"\", \"*\")";

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "produce help message")
            ("host", po::value<std::string>(&config.host)->default_value("localhost", "host"))
            ("port", po::value<std::string>(&config.port)->default_value("5000", "port"))
            ("api-key", po::value<std::string>(&config.api_key)->default_value("", "API key"))
            ("mode", po::value<std::string>(&config.mode)->default_value("chat"), "Specify mode chat or novel. novel mode means \"--phases \"{{user}}\" \"{{char}}\" --generation-prefix \"\\n{{phase}} :\"\" as default. novel mode means \"--phases \"\" --generation-prefix \"\"\" as default.")
            ("plot-file", po::value<std::string>(&config.plot_file)->default_value(""), "plot file")
            ("log-level", po::value<std::string>(&config.log_level)->default_value("info", "log level (trace|debug|info|warning|error|fatal)"))
            ("log-file", po::value<std::string>(&config.log_file)->default_value("log.txt", "log file path"))
            ("base-path", po::value<std::string>(&config.base_path)->default_value(".", "base path"))
            ("system-prompts-file", po::value<std::string>(&config.system_prompts_file)->default_value("system_prompts.txt", "system prompt file path"))
            ("examples-file", po::value<std::string>(&config.examples_file)->default_value("examples.txt", "exmaples file path"))
            ("history-file", po::value<std::string>(&config.history_file)->default_value("history.txt", "history file path"))
            ("output-file", po::value<std::string>(&config.output_file)->default_value("history.txt", "output file path"))
            ("example-separator", po::value<std::string>(&config.example_separator)->default_value("***", "separator to be inserted before and after examples"))
            ("phases", po::value<std::vector<std::string>>(&config.phases)->multitoken(), "phases name list")
            ("generation-prefix", po::value<std::string>(&config.generation_prefix)->default_value("", "generation prefix"))
            ("retry-generation-prefix", po::value<std::string>(&config.retry_generation_prefix)->default_value(""), "prefix to be used after a failed text generation")
            ("verbose,v", po::bool_switch(&config.verbose)->default_value(false), "enable verbose output")
            ("number-iterations,N", po::value<int>(&config.number_iterations)->default_value(1), "number of iterations (-1 means infinity)")
            ("min-completion-tokens", po::value<int>(&config.min_completion_tokens)->default_value(256), "min completion tokens")
            ("max-completion-iterations", po::value<int>(&config.max_completion_iterations)->default_value(5), "max completion iterations")
            ("max-total-context-tokens", po::value<int>(&config.max_total_context_tokens)->default_value(4096), "max total context tokens")
            ("define,D", po::value<std::vector<std::string>>(&config.predefined_macros), "define macro by key-value pair")
            ("model", po::value<std::string>(&config.params.model)->default_value("", "model"))
            ("num-best-of", po::value<int>(&config.params.best_of)->default_value(1), "best_of")
            ("echo", po::bool_switch(&config.params.echo)->default_value(false), "echo")
            ("frequency-penalty", po::value<double>(&config.params.frequency_penalty)->default_value(0.0), "frequency penalty")
            //std::map<int, double> logit_bias;
            ("logprobs", po::value<double>(&config.params.logprobs)->default_value(0.0), "presence penalty")
            ("max-tokens", po::value<int>(&config.params.max_tokens)->default_value(512), "max tokens")
            ("n", po::value<int>(&config.params.n)->default_value(1), "number of responses generated for the same prompt")
            ("presence-penalty", po::value<double>(&config.params.presence_penalty)->default_value(0.0), "presence penalty")
            ("stop", po::value<std::vector<std::string>>(&config.params.stop)->multitoken(), "stop sequences")
            ("stream", po::bool_switch(&config.params.stream)->default_value(false), "stream")
            ("suffix", po::value<std::string>(&config.params.suffix)->default_value("", "suffix"))
            ("temperature", po::value<double>(&config.params.temperature)->default_value(1.0), "temperature")
            ("top-p", po::value<double>(&config.params.top_p)->default_value(1.0), "top p")
            ("seed", po::value<int>(&config.seed)->default_value(-1), "seed value")
            ("dynatemp-low", po::value<double>(&config.params.dynatemp_low)->default_value(0.75), "dynatemp low")
            ("dynatemp-high", po::value<double>(&config.params.dynatemp_high)->default_value(1.25), "dynatemp high")
            ("dynatemp-exponent", po::value<double>(&config.params.dynatemp_exponent)->default_value(1.0), "dynatemp exponent")
            ("smoothing-factor", po::value<double>(&config.params.smoothing_factor)->default_value(0.0), "smoothing factor")
            ("smoothing-curve", po::value<double>(&config.params.smoothing_curve)->default_value(1.0), "smoothing curve")
            ("min-p", po::value<double>(&config.params.min_p)->default_value(0.1), "min p")
            ("top-k", po::value<int>(&config.params.top_k)->default_value(0), "top k")
            ("typical-p", po::value<double>(&config.params.typical_p)->default_value(1.0), "typical p")
            ("xtc-threshold", po::value<double>(&config.params.xtc_threshold)->default_value(0.1), "Exclude Top Choices (XTC) threshold")
            ("xtc-probability", po::value<double>(&config.params.xtc_probability)->default_value(0.0), "Exclude Top Choices (XTC) probability")
            ("epsilon-cutoff", po::value<double>(&config.params.epsilon_cutoff)->default_value(0), "epsilon cutoff")
            ("eta-cutoff", po::value<double>(&config.params.eta_cutoff)->default_value(0), "eta cutoff")
            ("tfs", po::value<double>(&config.params.tfs)->default_value(1.0), "tfs")
            ("top-a", po::value<double>(&config.params.top_a)->default_value(0.0), "top a")
            ("top-n-sigma", po::value<double>(&config.params.top_n_sigma)->default_value(1.0), "top n sigma")
            ("dry-multiplier", po::value<double>(&config.params.dry_multiplier)->default_value(0.0), "DRY multiplier")
            ("dry-allowed-length", po::value<int>(&config.params.dry_allowed_length)->default_value(2), "DRY allowed length")
            ("dry-base", po::value<double>(&config.params.dry_base)->default_value(1.75), "DRY base")
            ("repetition-penalty", po::value<double>(&config.params.repetition_penalty)->default_value(1.2), "repetition penalty")
            ("encoder-repetition-penalty", po::value<double>(&config.params.encoder_repetition_penalty)->default_value(1.0), "encoder repetition penalty")
            ("no-repeat-ngram-size", po::value<int>(&config.params.no_repeat_ngram_size)->default_value(0), "no repeat ngram size")
            ("repetition-penalty-range", po::value<int>(&config.params.repetition_penalty_range)->default_value(0), "repetition penalty range")
            ("penalty-alpha", po::value<double>(&config.params.penalty_alpha)->default_value(0.9), "penalty alpha")
            ("guidance-scale", po::value<double>(&config.params.guidance_scale)->default_value(1.0), "guidance scale")
            ("mirostat-mode", po::value<int>(&config.params.mirostat_mode)->default_value(0), "mirostat mode")
            ("mirostat-tau", po::value<double>(&config.params.mirostat_tau)->default_value(5), "mirostat tau")
            ("mirostat-eta", po::value<double>(&config.params.mirostat_eta)->default_value(0.1), "mirostat eta")
            ("prompt-lookup-num-tokens", po::value<int>(&config.params.prompt_lookup_num_tokens)->default_value(0), "prompt lookup num tokens")
            ("max-tokens-second", po::value<int>(&config.params.max_tokens_second)->default_value(0), "max tokens second")
            ("do_sample", po::bool_switch(&config.params.do_sample)->default_value(true), "do sample")
            ("dynamic_temperature", po::bool_switch(&config.params.dynamic_temperature)->default_value(false), "dynamic temperature")
            ("temperature_last", po::bool_switch(&config.params.temperature_last)->default_value(false), "temperature last")
            ("auto_max_new_tokens", po::bool_switch(&config.params.auto_max_new_tokens)->default_value(false), "auto max_new tokens")
            ("ban_eos_token", po::bool_switch(&config.params.ban_eos_token)->default_value(false), "ban eos token")
            ("add_bos_token", po::bool_switch(&config.params.add_bos_token)->default_value(true), "add Beginning of Sequence Token (BOS) token")
            ("skip_special_tokens", po::bool_switch(&config.params.skip_special_tokens)->default_value(true), "skip special tokens (bos_token, eos_token, unk_token, pad_token, etc.)")
            ("static_cache", po::bool_switch(&config.params.static_cache)->default_value(false), "static 1")
            ("truncation_length", po::value<int>(&config.params.truncation_length)->default_value(0), "truncation length")
            ("sampler-priority", po::value<std::vector<std::string>>(&config.params.sampler_priority)->multitoken(), "sampler priority")
            ("custom-token-bans", po::value<std::string>(&config.params.custom_token_bans)->default_value("", "custom token bans"))
            ("negative-prompt", po::value<std::string>(&config.params.negative_prompt)->default_value("", "negative prompt"))
            ("dry-sequence-breakers", po::value<std::string>(&config.params.dry_sequence_breakers)->default_value("", "dry sequence breakers"))
            ("grammar-string", po::value<std::string>(&config.params.grammar_string)->default_value("", "grammar-string"))
            ;

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help"))
        {
            boost::nowide::cout << desc << std::endl;
            return 1;
        }

        init_logging(config);

        if (config.mode == "chat")
        {
            init_chat_mode(config);
        }
        else if (config.mode == "novel")
        {
            init_novel_mode(config);
        }
        else
        {
            BOOST_LOG_TRIVIAL(error) << "mode options must be chat or novel.";
            return 1;
        }

        if (config.phases.empty())
        {
            config.phases = { "" };
        }

        std::transform(config.params.stop.begin(), config.params.stop.end(), config.params.stop.begin(), unescape_string);
        std::transform(config.predefined_macros.begin(), config.predefined_macros.end(), config.predefined_macros.begin(), unescape_string);
        config.params.dry_sequence_breakers = unescape_string(config.params.dry_sequence_breakers);
        config.generation_prefix = unescape_string(config.generation_prefix);
        config.retry_generation_prefix = unescape_string(config.retry_generation_prefix);

        parse_predefined_macros(config.predefined_macros, config.macros);
    }
    catch (const po::error& e)
    {
        throw command_line_syntax_exception{} << error_info::description{ std::string{ "boost::program_options::error: " } + e.what() };
    }

    return 0;
}

std::string prompts::to_string(const config& config) const
{
    std::string result;

    auto try_append = [&config](
        auto first,
        auto last,
        std::string& result,
        int max_tokens,
        int& written_tokens,
        bool reverse,
        const std::string separator = {}
        )
        {
            std::vector<std::string> temp;
            for (; first != last; ++first)
            {
                const std::string macro_expanded_string{ expand_macro(*first, config.macros) };
                const int next_tokens = get_tokens_from_cache(config, macro_expanded_string);
                if (written_tokens + next_tokens > max_tokens)
                {
                    break;
                }
                temp.push_back(macro_expanded_string);
                written_tokens += next_tokens;
            }
            if (reverse)
            {
                std::reverse(temp.begin(), temp.end());
            }
            for (const std::string& line : temp)
            {
                result += line;
            }
        };

    int remaining_tokens = config.max_total_context_tokens - config.params.max_tokens;

    std::string system_prompts_string;
    int system_prompts_tokens{};
    {
        try_append(system_prompts.begin(), system_prompts.end(), system_prompts_string, remaining_tokens, system_prompts_tokens, false);
    }

    if (remaining_tokens >= system_prompts_tokens)
    {
        result += system_prompts_string;
        remaining_tokens -= system_prompts_tokens;
    }

    std::string history_string;
    {
        int written_tokens = 0;
        try_append(history.rbegin(), history.rend(), history_string, remaining_tokens, written_tokens, true);
        remaining_tokens -= written_tokens;
    }

    std::string examples_string;
    {
        if (!config.example_separator.empty())
        {
            remaining_tokens -= (static_cast<int>(config.example_separator.size()) + 2) * 2;
        }
        int written_tokens = 0;
        try_append(examples.begin(), examples.end(), examples_string, remaining_tokens, written_tokens, false);
        if (written_tokens > 0)
        {
            if (!config.example_separator.empty())
            {
                std::string temp = "\n";
                temp += config.example_separator;
                temp += "\n";
                temp += examples_string;
                temp += "\n";
                temp += config.example_separator;
                temp += "\n";
                std::swap(examples_string, temp);
            }
            remaining_tokens -= written_tokens;
        }
    }

    result += examples_string;
    result += history_string;

    return result;
}

void read_prompts(const config& config, prompts& prompts)
{
    read_file_to_container(prompts.system_prompts, string_to_path_by_config(config.system_prompts_file, config));
    read_file_to_container(prompts.examples, string_to_path_by_config(config.examples_file, config));
    read_file_to_container(prompts.history, string_to_path_by_config(config.history_file, config));
}

void write_response(const config& config, const std::string& response)
{
    const std::filesystem::path output_file_path{ string_to_path_by_config(config.output_file, config) };
    create_parent_directories(output_file_path);
    boost::nowide::ofstream ofs{ output_file_path, std::ios_base::app };
    if (!ofs.is_open())
    {
        throw file_open_exception{} << error_info::path{ output_file_path };
    }
    ofs << response;
}

void generate_and_output(const config& config, prompts& prompts, const std::string& generation_prefix)
{
    std::string prompts_string = prompts.to_string(config);
    prompts_string = expand_macro(prompts_string, config.macros);

    const std::string response{ generate_and_complete_text(config, prompts_string, generation_prefix) };
    BOOST_LOG_TRIVIAL(info) << "Text generated.\n```\n" << response << "\n```\n";

    write_response(config, response);
}

void set_seed(config& config)
{
    if (config.seed == -1)
    {
        config.params.seed = generate_random_seed();
    }
    else
    {
        config.params.seed = config.seed;
    }
}

void iterate(config& config)
{
    read_cache(config);

    int iteration_count = 0;
    while (config.number_iterations == -1 || iteration_count < config.number_iterations)
    {
        prompts prompts;
        read_prompts(config, prompts);

        set_seed(config);

        config.macros["N"] = std::to_string(iteration_count + 1);

        for (std::size_t phase_index = 0; phase_index < config.phases.size(); ++phase_index)
        {
            set_phases_macro(config.phases, phase_index, config.macros);

            try
            {
                generate_and_output(config, prompts, config.generation_prefix);
            }
            catch (const text_generation_exception& exception)
            {
                BOOST_LOG_TRIVIAL(warning) << boost::diagnostic_information(exception);
                if (!config.retry_generation_prefix.empty())
                {
                    BOOST_LOG_TRIVIAL(info) << "Start to retry text generation with retry-generation-prefix.";
                    generate_and_output(config, prompts, config.retry_generation_prefix);
                }
            }
        }

        write_cache(config);

        iteration_count += 1;
    }
}

int exception_safe_main(int argc, char** argv)
{
    try
    {
        config config;

        if (parse_commandline(argc, argv, config))
        {
            return 0;
        }

        iterate(config);
    }
    catch (const boost::exception& exception)
    {
        BOOST_LOG_TRIVIAL(error) << boost::diagnostic_information(exception);
        return -1;
    }
    catch (const std::exception& exception)
    {
        BOOST_LOG_TRIVIAL(error) << exception.what();
        return -1;
    }

    return 0;
}

int main(int argc, char** argv)
{
    boost::nowide::args a(argc, argv);
    return exception_safe_main(argc, argv);
}