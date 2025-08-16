#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
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
#include <boost/archive/iterators/binary_from_base64.hpp>
#include <boost/archive/iterators/transform_width.hpp>
#include <boost/url.hpp>
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
#include "picojson.h"

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
class image_generation_exception : public runtime_exception {};
class syntax_exception : public runtime_exception {};
class json_parse_exception : public runtime_exception {};
class macro_exception : public runtime_exception {};
class command_line_syntax_exception : public runtime_exception {};
class array_index_out_of_bounds_exception : public runtime_exception {};

namespace error_info
{
    using stacktrace = boost::error_info<struct tag_stacktrace, boost::stacktrace::stacktrace>;
    using description = boost::error_info<struct tag_description, std::string>;
    using wrapped_std_exception = boost::error_info<struct tag_wrapped_std_exception, std::exception>;
    using wrapped_boost_exception = boost::error_info<struct tag_wrapped_boost_exception, boost::exception>;
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

    namespace beast
    {
        using error_code = boost::error_info<struct tag_error_code, boost::beast::error_code>;
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

struct tg_prompt_parameters
{
    std::string system_prompts_file;
    std::string examples_file;
    std::string history_file;
    std::string output_file;
    std::string example_separator;
    std::string generation_prefix;
    bool skip_generation_prefix{};
    std::string retry_generation_prefix;
    std::string paragraphs_file;
};

struct tg_completions_parameters
{
    std::string host;
    std::string port;
    std::string api_key;

    std::string completions_target;
    std::string token_count_target;

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

struct adetailer_parametesrs
{
    bool ad_enable{};
    bool skip_img2img{};

    struct arg
    {
        std::string ad_model;;
        std::string ad_model_classes;
        bool ad_tab_enable{};
        std::string ad_prompt;
        std::string ad_negative_prompt;
        double ad_confidence{};
        std::string ad_mask_filter_method;
        int ad_mask_k{};
        double ad_mask_min_ratio;
        double ad_mask_max_ratio;
        int ad_dilate_erode{};
        int ad_x_offset{};
        int ad_y_offset{};
        std::string ad_mask_merge_invert;
        int ad_mask_blur{};
        double ad_denoising_strength{};
        bool ad_inpaint_only_masked{};;
        int ad_inpaint_only_masked_padding{};
        bool ad_use_inpaint_width_height{};
        int ad_inpaint_width{};
        int ad_inpaint_height{};
        bool ad_use_steps{};
        int ad_steps{};
        bool ad_use_cfg_scale{};
        double ad_cfg_scale{};
        bool ad_use_checkpoint{};
        std::string ad_checkpoint;
        bool ad_use_vae{};
        std::string ad_vae;
        bool ad_use_sampler{};
        std::string ad_sampler;
        std::string ad_scheduler;
        bool ad_use_noise_multiplier{};
        double ad_noise_multiplier{};
        bool ad_use_clip_skip{};
        int ad_clip_skip{};
        bool ad_restore_face{};
        std::string ad_controlnet_model;
        std::string ad_controlnet_module;
        std::string ad_controlnet_weight{};
        double ad_controlnet_guidance_start{};
        double ad_controlnet_guidance_end{};
    };

    arg args1;
};

struct alwayson_scripts
{
    adetailer_parametesrs adetailer_parametesrs;
};

struct sd_txt2img_parameters
{
    std::string host;
    std::string port;
    std::string target;

    std::string prompt_file;
    std::string negative_prompt_file;
    std::string output_file;

    std::string prompt;
    std::string negative_prompt;
    std::vector<std::string> styles;
    int seed{};
    int subseed{};
    double subseed_strength{};
    int seed_resize_from_h{};
    int seed_resize_from_w{};
    std::string sampler_name;
    std::string scheduler;
    int batch_size{};
    int n_iter{};
    int steps{};
    double cfg_scale{};
    int width{};
    int height{};
    bool restore_faces{};
    bool tiling{};
    bool do_not_save_samples{};
    bool do_not_save_grid{};
    int eta{};
    double denoising_strength{};
    int s_min_uncond{};
    int s_churn{};
    int s_tmax{};
    int s_tmin{};
    int s_noise{};
    std::string override_settings;
    bool override_settings_restore_afterwards{};
    std::string refiner_checkpoint;
    double refiner_switch_at{};
    bool disable_extra_networks{};
    std::string firstpass_image;
    std::string comments;
    bool enable_hr{};
    int firstphase_width{};
    int firstphase_height{};
    double hr_scale{};
    std::string hr_upscaler;
    int hr_second_pass_steps{};
    int hr_resize_x{};
    int hr_resize_y{};
    std::string hr_checkpoint_name;
    //std::string hr_sampler_name;
    //std::string hr_scheduler;
    //std::string hr_prompt;
    //std::string hr_negative_prompt;
    std::string force_task_id;
    std::string sampler_index;
    std::string script_name;
    std::vector<std::string> script_args;
    bool send_images{};
    bool save_images{};
    alwayson_scripts alwayson_scripts;
    std::string infotext;

    bool abg_remover_enable{};
};

struct sb_generation_parameters
{
    std::string host;
    std::string port;
    std::string target;
    std::string text_file;
    std::string output_file;

    std::string text;
    std::string model_name;
    int model_id{};
    std::string speaker_name;
    int speaker_id{};
    double sdp_ratio{};
    double noise{};
    double noisew{};
    double length{};
    std::string language;
    bool auto_split{};
    double split_interval{};
    std::string assist_text;
    double assist_text_weight{};
    std::string style;
    double style_weight{};
    std::string reference_audio_path{};
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
    std::string mode{};
    std::string base_path;
    std::string log_level;
    std::string log_file;
    bool verbose{};
    int number_iterations{};
    std::vector<std::string> predefined_macros;
    std::vector<std::string> phases;

    int seed{};
    int min_completion_tokens{};
    int max_completion_iterations{};

    tg_prompt_parameters tg_prompt_params;
    tg_completions_parameters tg_completions_params;
    sd_txt2img_parameters sd_txt2img_params;
    sb_generation_parameters sb_generation_params;
    mutable lru_cache lru_cache;
    macros macros;
    mutable std::optional<std::string> opt_stdin;
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

template<typename Value>
const Value& throwable_get(const picojson::value& value);

template<typename Value>
const Value& throwable_at(const picojson::array& array, std::size_t index);

template<typename Value>
const Value& throwable_find(const picojson::object& object, const std::string& key);

std::string base64_decode(const std::string& encoded_string);
std::string trim(const std::string& str);

template <typename Container>
void split_string_by_new_line(const std::string& str, Container& container);

void create_parent_directories(const std::filesystem::path& path);

template <typename Container>
void read_file_to_container(const std::filesystem::path& file, Container& container);

void read_file_to_string(const std::filesystem::path& file, std::string& result);

int generate_random_seed();

std::string include_predefiend_macro(const config& config, const std::string& right);

std::string datetime_predefiend_macro(const config& config, const std::string&);

std::string stdin_predefiend_macro(const config& config, const std::string&);

std::optional<std::string> expand_predefined_macro(
    const config& config,
    const std::string& name,
    const std::string& arguments
);

std::string expand_macro(const std::string& str, const config& macros, int depth = 0);
std::filesystem::path string_to_path_by_config(const std::string& path, const config& config);
void send_automatic1111_txt2img_request(
    const config& config,
    const std::string& prompt,
    const std::string& negative_prompt,
    const std::filesystem::path& path
);

void send_style_bert_voice_request(
    const config& config,
    const std::string& text
);

std::vector<item> parse_item_list(const std::string& str);

void write_item_list(const config& config, const std::string& task);

llm_response send_oobabooga_completions_request(
    const config& config,
    const std::string& prompt,
    const tg_completions_parameters& params
);

int send_oobabooga_token_count_request(const config& config, const std::string& prompt);
void write_cache(const config& config);
void read_cache(const config& config);

std::string generate_and_complete_text(
    const config& config,
    const std::string& prompts,
    const std::string& prefix
);

std::string unescape_string(const std::string& str);

void parse_predefined_macros(const std::vector<std::string>& predefined_macros, macros& macros);
void init_logging_with_nowide_cout();
void init_logging_with_nowide_file_log(const std::filesystem::path& log);
void init_logging(const config& config);
void init_chat_mode(config& config);

void set_phases_macro(
    const std::vector<std::string>& phases,
    std::size_t phase_index,
    std::map<std::string, std::string>& macros
);

void set_paragraphs_to_phases(
    const std::vector<item>& paragraphs,
    std::vector<std::string>& phases
);

void init_tg_mode(config& config);

int parse_commandline(
    int argc,
    char** argv,
    config& config
);

void read_prompts(const config& config, prompts& prompts);
void write_response(const config& config, const std::string& response);
void generate_and_output(const config& config, prompts& prompts, const std::string& generation_prefix);
void set_seed(config& config);
void iterate(config& config);
int exception_safe_main(int argc, char** argv);

template<typename Value>
const Value& throwable_get(const picojson::value& value)
{
    if (!value.is<Value>())
    {
        throw json_parse_exception{};
    }
    return value.get<Value>();
}

template<typename Value>
const Value& throwable_at(const picojson::array& array, std::size_t index)
{
    if (index >= array.size())
    {
        throw json_parse_exception{};
    }
    const picojson::value& element = array[index];
    if (!element.is<Value>())
    {
        throw json_parse_exception{};
    }
    return element.get<Value>();
}

template<typename Value>
const Value& throwable_find(const picojson::object& object, const std::string& key)
{
    auto iter = object.find(key);
    if (iter == object.end() || !iter->second.is<Value>())
    {
        throw json_parse_exception{};
    }
    return iter->second.get<Value>();
}

std::string base64_decode(const std::string& encoded_string)
{
    using iterator = boost::archive::iterators::transform_width<boost::archive::iterators::binary_from_base64<std::string::const_iterator>, 8, 6>;
    return std::string{ iterator{ encoded_string.begin() }, iterator{ encoded_string.end() } };
}

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
void read_file_to_container(const std::filesystem::path& file, Container& container)
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

void read_file_to_string(const std::filesystem::path& file, std::string& result)
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

std::string include_predefiend_macro(const config& config, const std::string& right)
{
    std::string result;
    std::filesystem::path macro_file{ right };
    if (!std::filesystem::exists(macro_file))
    {
        macro_file.replace_extension(".txt");
    }
    read_file_to_string(macro_file, result);
    return result;
}

std::string datetime_predefiend_macro(const config& config, const std::string&)
{
    boost::posix_time::ptime local_time = boost::posix_time::second_clock::local_time();
    boost::posix_time::time_facet* facet = new boost::posix_time::time_facet("%Y%m%d%H%M%S");
    std::ostringstream oss;
    oss.imbue(std::locale(oss.getloc(), facet));
    oss << local_time;
    return oss.str();
}

std::string stdin_predefiend_macro(const config& config, const std::string&)
{
    if (config.opt_stdin)
    {
        return *config.opt_stdin;
    }
    config.opt_stdin = std::string{ std::istreambuf_iterator<char>{ boost::nowide::cin }, std::istreambuf_iterator<char>{} };
    return *config.opt_stdin;
}

std::optional<std::string> expand_predefined_macro(
    const config& cfg,
    const std::string& name,
    const std::string& arguments
)
{
    const static std::map<std::string, std::function<std::string(const config&, const std::string&)>> predefiend_macro_impls
    {
        {"include", include_predefiend_macro},
        {"datetime", datetime_predefiend_macro},
        {"stdin", stdin_predefiend_macro}
    };

    auto found = std::find_if(predefiend_macro_impls.begin(), predefiend_macro_impls.end(), [&name](const auto& pair) { return boost::iequals(name, pair.first); });
    if (found != predefiend_macro_impls.end())
    {
        BOOST_LOG_TRIVIAL(trace) << "found predefined macro. name: \"" << name << "\" arguments: \"" << arguments << "\"";

        try
        {
            try
            {
                return found->second(cfg, arguments);
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

std::string expand_macro(const std::string& str, const config& config, int depth)
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

        std::optional<std::string> predefined_macro_result{ expand_predefined_macro(config, name, arguments) };
        if (predefined_macro_result)
        {
            expanded_string = *predefined_macro_result;
            result += expanded_string;
            BOOST_LOG_TRIVIAL(trace) << "macro expanded: \"{{" << macro_string << "}}\" -> \"" << expanded_string << "\"";
        }
        else
        {
            auto found = std::find_if(config.macros.begin(), config.macros.end(), [&macro_string](const auto& pair) { return boost::iequals(macro_string, pair.first); });

            if (found != config.macros.end())
            {
                expanded_string = found->second;
                if ("{{" + macro_string + "}}" != expanded_string)
                {
                    expanded_string = expand_macro(expanded_string, config, depth + 1);
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
    const std::filesystem::path file_path{ expand_macro(path, config) };
    if (file_path.is_relative())
    {
        const std::filesystem::path base_path{ expand_macro(config.base_path, config) };
        return base_path / file_path;

    }
    return file_path;
}

void send_automatic1111_txt2img_request(
    const config& config,
    const std::string& prompt,
    const std::string& negative_prompt,
    const std::filesystem::path& path
)
{
    namespace beast = boost::beast;
    namespace http = beast::http;
    namespace net = boost::asio;
    using tcp = net::ip::tcp;

    try
    {
        net::io_context ioc;
        tcp::resolver resolver{ ioc };
        beast::tcp_stream tcp_stream{ ioc };

        auto const results = resolver.resolve(config.sd_txt2img_params.host, config.sd_txt2img_params.port);
        tcp_stream.connect(results);

        picojson::object request_body_json;

        request_body_json.insert(std::make_pair("prompt", picojson::value{ prompt }));
        request_body_json.insert(std::make_pair("negative_prompt", picojson::value{ negative_prompt }));

        //request_body_json.insert(std::make_pair("styles", picojson::value{ config.sd_txt2img_params.styles }));
        request_body_json.insert(std::make_pair("seed", picojson::value{ static_cast<double>(config.sd_txt2img_params.seed) }));
        request_body_json.insert(std::make_pair("subseed", picojson::value{ static_cast<double>(config.sd_txt2img_params.subseed) }));
        request_body_json.insert(std::make_pair("subseed_strength", picojson::value{ config.sd_txt2img_params.subseed_strength }));
        request_body_json.insert(std::make_pair("seed_resize_from_h", picojson::value{ static_cast<double>(config.sd_txt2img_params.seed_resize_from_h) }));
        request_body_json.insert(std::make_pair("seed_resize_from_w", picojson::value{ static_cast<double>(config.sd_txt2img_params.seed_resize_from_w) }));
        request_body_json.insert(std::make_pair("sampler_name", picojson::value{ config.sd_txt2img_params.sampler_name }));
        request_body_json.insert(std::make_pair("scheduler", picojson::value{ config.sd_txt2img_params.scheduler }));
        request_body_json.insert(std::make_pair("batch_size", picojson::value{ static_cast<double>(config.sd_txt2img_params.batch_size) }));
        request_body_json.insert(std::make_pair("n_iter", picojson::value{ static_cast<double>(config.sd_txt2img_params.n_iter) }));
        request_body_json.insert(std::make_pair("steps", picojson::value{ static_cast<double>(config.sd_txt2img_params.steps) }));
        request_body_json.insert(std::make_pair("cfg_scale", picojson::value{ config.sd_txt2img_params.cfg_scale }));
        request_body_json.insert(std::make_pair("width", picojson::value{ static_cast<double>(config.sd_txt2img_params.width) }));
        request_body_json.insert(std::make_pair("height", picojson::value{ static_cast<double>(config.sd_txt2img_params.height) }));
        request_body_json.insert(std::make_pair("restore_faces", picojson::value{ config.sd_txt2img_params.restore_faces }));
        request_body_json.insert(std::make_pair("tiling", picojson::value{ config.sd_txt2img_params.tiling }));
        request_body_json.insert(std::make_pair("do_not_save_samples", picojson::value{ config.sd_txt2img_params.do_not_save_samples }));
        request_body_json.insert(std::make_pair("do_not_save_grid", picojson::value{ config.sd_txt2img_params.do_not_save_grid }));
        request_body_json.insert(std::make_pair("eta", picojson::value{ static_cast<double>(config.sd_txt2img_params.eta) }));
        request_body_json.insert(std::make_pair("denoising_strength", picojson::value{ config.sd_txt2img_params.denoising_strength }));
        request_body_json.insert(std::make_pair("s_min_uncond", picojson::value{ static_cast<double>(config.sd_txt2img_params.s_min_uncond) }));
        request_body_json.insert(std::make_pair("s_churn", picojson::value{ static_cast<double>(config.sd_txt2img_params.s_churn) }));
        request_body_json.insert(std::make_pair("s_tmax", picojson::value{ static_cast<double>(config.sd_txt2img_params.s_tmax) }));
        request_body_json.insert(std::make_pair("s_tmin", picojson::value{ static_cast<double>(config.sd_txt2img_params.s_tmin) }));
        request_body_json.insert(std::make_pair("s_noise", picojson::value{ static_cast<double>(config.sd_txt2img_params.s_noise) }));
        request_body_json.insert(std::make_pair("override_settings", picojson::value{ config.sd_txt2img_params.override_settings }));
        request_body_json.insert(std::make_pair("override_settings_restore_afterwards", picojson::value{ config.sd_txt2img_params.override_settings_restore_afterwards }));
        request_body_json.insert(std::make_pair("refiner_checkpoint", picojson::value{ config.sd_txt2img_params.refiner_checkpoint }));
        request_body_json.insert(std::make_pair("refiner_switch_at", picojson::value{ config.sd_txt2img_params.refiner_switch_at }));
        request_body_json.insert(std::make_pair("disable_extra_networks", picojson::value{ config.sd_txt2img_params.disable_extra_networks }));
        request_body_json.insert(std::make_pair("firstpass_image", picojson::value{ config.sd_txt2img_params.firstpass_image }));
        request_body_json.insert(std::make_pair("comments", picojson::value{ config.sd_txt2img_params.comments }));
        request_body_json.insert(std::make_pair("enable_hr", picojson::value{ config.sd_txt2img_params.enable_hr }));
        request_body_json.insert(std::make_pair("firstphase_width", picojson::value{ static_cast<double>(config.sd_txt2img_params.firstphase_width) }));
        request_body_json.insert(std::make_pair("firstphase_height", picojson::value{ static_cast<double>(config.sd_txt2img_params.firstphase_height) }));
        request_body_json.insert(std::make_pair("hr_scale", picojson::value{ config.sd_txt2img_params.hr_scale }));
        request_body_json.insert(std::make_pair("hr_upscaler", picojson::value{ config.sd_txt2img_params.hr_upscaler }));
        request_body_json.insert(std::make_pair("hr_second_pass_steps", picojson::value{ static_cast<double>(config.sd_txt2img_params.hr_second_pass_steps) }));
        request_body_json.insert(std::make_pair("hr_resize_x", picojson::value{ static_cast<double>(config.sd_txt2img_params.hr_resize_x) }));
        request_body_json.insert(std::make_pair("hr_resize_y", picojson::value{ static_cast<double>(config.sd_txt2img_params.hr_resize_y) }));
        request_body_json.insert(std::make_pair("hr_checkpoint_name", picojson::value{ config.sd_txt2img_params.hr_checkpoint_name }));
        request_body_json.insert(std::make_pair("hr_prompt", picojson::value{ config.sd_txt2img_params.prompt }));
        request_body_json.insert(std::make_pair("hr_negative_prompt", picojson::value{ config.sd_txt2img_params.negative_prompt }));
        request_body_json.insert(std::make_pair("force_task_id", picojson::value{ config.sd_txt2img_params.force_task_id }));

        if (!config.sd_txt2img_params.sampler_index.empty() && config.sd_txt2img_params.sampler_name.empty())
        {
            request_body_json.insert(std::make_pair("sampler_index", picojson::value{ config.sd_txt2img_params.sampler_index }));
        }

        if (config.sd_txt2img_params.abg_remover_enable)
        {
            request_body_json.insert(std::make_pair("script_name", picojson::value{ "abg remover" }));
            picojson::array args_array;
            args_array.push_back(picojson::value{ false });
            args_array.push_back(picojson::value{ false });
            args_array.push_back(picojson::value{ false });
            args_array.push_back(picojson::value{ "#000000" });
            args_array.push_back(picojson::value{ false });
            request_body_json.insert(std::make_pair("script_args", args_array));
        }

        request_body_json.insert(std::make_pair("send_images", picojson::value{ config.sd_txt2img_params.send_images }));
        request_body_json.insert(std::make_pair("save_images", picojson::value{ config.sd_txt2img_params.save_images }));

        picojson::object alwayson_scripts;
        if (config.sd_txt2img_params.alwayson_scripts.adetailer_parametesrs.ad_enable)
        {
            picojson::object adetailer;
            picojson::array args_array;
            picojson::object args;
            picojson::object object;
            object.insert(std::make_pair("ad_model", picojson::value{ config.sd_txt2img_params.alwayson_scripts.adetailer_parametesrs.args1.ad_model }));
            if (!config.sd_txt2img_params.alwayson_scripts.adetailer_parametesrs.args1.ad_prompt.empty())
            {
                const std::string ad_prompt{ expand_macro(config.sd_txt2img_params.alwayson_scripts.adetailer_parametesrs.args1.ad_prompt, config) };
                object.insert(std::make_pair("ad_prompt", picojson::value{ ad_prompt }));
            }
            if (!config.sd_txt2img_params.alwayson_scripts.adetailer_parametesrs.args1.ad_negative_prompt.empty())
            {
                const std::string ad_negative_prompt{ expand_macro(config.sd_txt2img_params.alwayson_scripts.adetailer_parametesrs.args1.ad_negative_prompt, config) };
                object.insert(std::make_pair("ad_negative_prompt", picojson::value{ ad_negative_prompt }));
            }
            args_array.push_back(picojson::value{ true });
            args_array.push_back(picojson::value{ false });
            args_array.push_back(picojson::value{ object });
            adetailer.insert(std::make_pair("args", picojson::value{ args_array }));
            alwayson_scripts.insert(std::make_pair("ADetailer", picojson::value{ adetailer }));
        }
        //{
        //    picojson::object sampler;
        //    picojson::array args_array;
        //    args_array.push_back(picojson::value{ static_cast<double>(config.sd_txt2img_params.steps) });
        //    args_array.push_back(picojson::value{ config.sd_txt2img_params.sampler_name });
        //    args_array.push_back(picojson::value{ config.sd_txt2img_params.scheduler });
        //    sampler.insert(std::make_pair("args", picojson::value{ args_array }));
        //    alwayson_scripts.insert(std::make_pair("Sampler", picojson::value{ sampler }));
        //}
        //{
        //    picojson::object seed;
        //    picojson::array args_array;
        //    args_array.push_back(picojson::value{ static_cast<double>(config.sd_txt2img_params.seed) });
        //    args_array.push_back(picojson::value{ false });
        //    args_array.push_back(picojson::value{ static_cast<double>(config.sd_txt2img_params.subseed) });
        //    args_array.push_back(picojson::value{ static_cast<double>(0) });
        //    args_array.push_back(picojson::value{ static_cast<double>(0) });
        //    args_array.push_back(picojson::value{ static_cast<double>(0) });
        //    seed.insert(std::make_pair("args", picojson::value{ args_array }));
        //    alwayson_scripts.insert(std::make_pair("Seed", picojson::value{ seed }));
        //}
        request_body_json.insert(std::make_pair("alwayson_scripts", picojson::value{ alwayson_scripts }));

        request_body_json.insert(std::make_pair("infotext", picojson::value{ config.sd_txt2img_params.infotext }));

        const std::string request_body{ picojson::value{ request_body_json }.serialize() };
        BOOST_LOG_TRIVIAL(info) << "Send JSON\n```\n" << request_body << "\n```";

        http::request<http::string_body> request{ http::verb::post, config.sd_txt2img_params.target, 11 }; // HTTP/1.1
        request.set(http::field::host, config.sd_txt2img_params.host);
        request.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
        request.set(http::field::content_type, "application/json; charset=UTF-8");
        request.body() = request_body;
        request.prepare_payload();

        http::write(tcp_stream, request);

        beast::flat_buffer buffer;
        http::response<http::string_body> response;
        http::read(tcp_stream, buffer, response);

        std::stringstream ss_response;
        ss_response << response.body();
        picojson::value response_json;
        picojson::parse(response_json, ss_response);

        const picojson::object& object{ throwable_get<picojson::object>(response_json) };
        const picojson::array& images{ throwable_find<picojson::array>(object, "images") };
        const std::string base64_image_data{ throwable_at<std::string>(images, 0) };

        if (base64_image_data.empty())
        {
            throw image_generation_exception{} << error_info::description{ "No image data found in the response." };
        }

        std::string decoded_image{ base64_decode(base64_image_data) };

        {
            boost::nowide::ofstream ofs{ path, std::ios::binary };
            if (!ofs.is_open())
            {
                throw file_open_exception{} << error_info::path{ path };
            }
            ofs.write(decoded_image.data(), decoded_image.size());
        }

        beast::error_code error_code;
        tcp_stream.socket().shutdown(tcp::socket::shutdown_both, error_code);
        if (error_code && error_code != beast::errc::not_connected)
        {
            throw image_generation_exception{} << error_info::beast::error_code{ error_code };
        }
    }
    catch (const std::exception& exception)
    {
        throw file_open_exception{} << error_info::description{ exception.what() };
    }
}

void send_style_bert_voice_request(
    const config& config,
    const std::string& text
)
{
    namespace beast = boost::beast;
    namespace http = beast::http;
    namespace net = boost::asio;
    using tcp = net::ip::tcp;

    try
    {
        net::io_context ioc;
        tcp::resolver resolver{ ioc };
        beast::tcp_stream tcp_stream{ ioc };

        auto const results = resolver.resolve(config.sb_generation_params.host, config.sb_generation_params.port);
        tcp_stream.connect(results);

        boost::url target{ config.sb_generation_params.target };
        target.params().set("text", text);
        //target.params().set("encoding", "utf-8");

        if (!config.sb_generation_params.model_name.empty())
        {
            target.params().set("model_name", config.sb_generation_params.model_name);
        }
        else
        {
            target.params().set("model_id", std::to_string(config.sb_generation_params.model_id));
        }

        if (!config.sb_generation_params.speaker_name.empty())
        {
            target.params().set("speaker_name", config.sb_generation_params.speaker_name);
        }
        else
        {
            target.params().set("speaker_id", std::to_string(config.sb_generation_params.speaker_id));
        }

        target.params().set("sdp_ratio", std::to_string(config.sb_generation_params.sdp_ratio));
        target.params().set("noise", std::to_string(config.sb_generation_params.noise));
        target.params().set("noisew", std::to_string(config.sb_generation_params.noisew));
        target.params().set("length", std::to_string(config.sb_generation_params.length));
        target.params().set("language", config.sb_generation_params.language);
        target.params().set("auto_split", config.sb_generation_params.auto_split ? "true" : "false");
        target.params().set("split_interval", std::to_string(config.sb_generation_params.split_interval));

        if (!config.sb_generation_params.assist_text.empty())
        {
            target.params().set("assist_text", config.sb_generation_params.assist_text);
            target.params().set("assist_text_weight", std::to_string(config.sb_generation_params.assist_text_weight));
        }

        if (!config.sb_generation_params.style.empty())
        {
            target.params().set("style", config.sb_generation_params.style);
            target.params().set("style_weight", std::to_string(config.sb_generation_params.style_weight));
        }

        if (!config.sb_generation_params.reference_audio_path.empty())
        {
            target.params().set("reference_audio_path", config.sb_generation_params.reference_audio_path);
        }

        BOOST_LOG_TRIVIAL(info) << "Send target\n```\n" << target.c_str() << "\n```";

        http::request<http::string_body> request{ http::verb::get, target, 11 }; // HTTP/1.1
        request.set(http::field::host, config.sb_generation_params.host);
        request.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
        request.set(http::field::content_type, "application/json; charset=UTF-8");
        request.prepare_payload();

        http::write(tcp_stream, request);

        beast::flat_buffer buffer;
        http::response<http::string_body> response;
        http::read(tcp_stream, buffer, response);
        if (response.result_int() != 200)
        {
            throw socket_exception{} << error_info::http::response::result_int{ response.result_int() };
        }

        std::stringstream ss_response;
        ss_response << response.body();

        {
            const std::string macro_expanded_string{ expand_macro(config.sb_generation_params.output_file, config) };
            const std::filesystem::path output_file_path{ string_to_path_by_config(macro_expanded_string, config) };
            boost::nowide::ofstream ofs{ output_file_path, std::ios::binary };
            if (!ofs.is_open())
            {
                throw file_open_exception{} << error_info::path{ output_file_path };
            }
            ofs.write(ss_response.str().data(), ss_response.str().size());
        }

        beast::error_code error_code;
        tcp_stream.socket().shutdown(tcp::socket::shutdown_both, error_code);
        if (error_code && error_code != beast::errc::not_connected)
        {
            throw image_generation_exception{} << error_info::beast::error_code{ error_code };
        }
    }
    catch (const std::exception& exception)
    {
        throw file_open_exception{} << error_info::description{ exception.what() };
    }
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
    const tg_completions_parameters& params
)
{
    namespace beast = boost::beast;
    namespace http = beast::http;
    namespace net = boost::asio;
    using tcp = net::ip::tcp;

    llm_response result;

    try
    {
        net::io_context ioc;
        tcp::resolver resolver{ ioc };
        beast::tcp_stream tcp_stream{ ioc };

        const auto results = resolver.resolve(config.tg_completions_params.host, config.tg_completions_params.port);
        tcp_stream.connect(results);

        picojson::object request_body_json;
        request_body_json.insert(std::make_pair("prompt", picojson::value{ prompt }));

        if (!params.model.empty())
        {
            request_body_json.insert(std::make_pair("model", picojson::value{ params.model }));
        }

        request_body_json.insert(std::make_pair("best_of", picojson::value{ static_cast<double>(params.best_of) }));
        request_body_json.insert(std::make_pair("echo", picojson::value{ params.echo }));
        request_body_json.insert(std::make_pair("frequency_penalty", picojson::value{ params.frequency_penalty }));
        //request_body_json.insert(std::make_pair("logit_bias", picojson::value{ params.logit_bias }));
        request_body_json.insert(std::make_pair("logprobs", picojson::value{ params.logprobs }));
        request_body_json.insert(std::make_pair("max_tokens", picojson::value{ static_cast<double>(params.max_tokens) }));
        request_body_json.insert(std::make_pair("n", picojson::value{ static_cast<double>(params.n) }));
        request_body_json.insert(std::make_pair("presence_penalty", picojson::value{ params.presence_penalty }));

        if (!params.stop.empty())
        {
            picojson::array stop_array;
            for (const std::string& str : params.stop)
            {
                stop_array.push_back(picojson::value{ str });
            }
            request_body_json.insert(std::make_pair("stop", picojson::value{ stop_array }));
        }

        request_body_json.insert(std::make_pair("stream", picojson::value{ params.stream }));

        if (!params.suffix.empty())
        {
            request_body_json.insert(std::make_pair("suffix", picojson::value{ params.suffix }));
        }

        request_body_json.insert(std::make_pair("temperature", picojson::value{ params.temperature }));
        request_body_json.insert(std::make_pair("top_p", picojson::value{ params.top_p }));

        if (params.seed != -1)
        {
            request_body_json.insert(std::make_pair("seed", picojson::value{ static_cast<double>(params.seed) }));
        }

        request_body_json.insert(std::make_pair("max_tokens", picojson::value{ static_cast<double>(params.max_tokens) }));

        if (!params.user.empty())
        {
            request_body_json.insert(std::make_pair("user", picojson::value{ params.user }));
        }

        if (!params.preset.empty())
        {
            request_body_json.insert(std::make_pair("preset", picojson::value{ params.preset }));
        }

        request_body_json.insert(std::make_pair("dynatemp_low", picojson::value{ params.dynatemp_low }));
        request_body_json.insert(std::make_pair("dynatemp_high", picojson::value{ params.dynatemp_high }));
        request_body_json.insert(std::make_pair("dynatemp_exponent", picojson::value{ params.dynatemp_exponent }));
        request_body_json.insert(std::make_pair("smoothing_factor", picojson::value{ params.smoothing_factor }));
        request_body_json.insert(std::make_pair("smoothing_curve", picojson::value{ params.smoothing_curve }));
        request_body_json.insert(std::make_pair("min_p", picojson::value{ params.min_p }));
        request_body_json.insert(std::make_pair("top_k", picojson::value{ static_cast<double>(params.top_k) }));
        request_body_json.insert(std::make_pair("typical_p", picojson::value{ params.typical_p }));
        request_body_json.insert(std::make_pair("xtc_threshold", picojson::value{ params.xtc_threshold }));
        request_body_json.insert(std::make_pair("xtc_probability", picojson::value{ params.xtc_probability }));
        request_body_json.insert(std::make_pair("epsilon_cutoff", picojson::value{ params.epsilon_cutoff }));
        request_body_json.insert(std::make_pair("eta_cutoff", picojson::value{ params.eta_cutoff }));
        request_body_json.insert(std::make_pair("tfs", picojson::value{ params.tfs }));
        request_body_json.insert(std::make_pair("top_a", picojson::value{ params.top_a }));
        request_body_json.insert(std::make_pair("top_n_sigma", picojson::value{ params.top_n_sigma }));
        request_body_json.insert(std::make_pair("dry_multiplier", picojson::value{ params.dry_multiplier }));
        request_body_json.insert(std::make_pair("dry_allowed_length", picojson::value{ static_cast<double>(params.dry_allowed_length) }));
        request_body_json.insert(std::make_pair("dry_base", picojson::value{ params.dry_base }));
        request_body_json.insert(std::make_pair("repetition_penalty", picojson::value{ params.repetition_penalty }));
        request_body_json.insert(std::make_pair("encoder_repetition_penalty", picojson::value{ params.encoder_repetition_penalty }));
        request_body_json.insert(std::make_pair("no_repeat_ngram_size", picojson::value{ static_cast<double>(params.no_repeat_ngram_size) }));
        request_body_json.insert(std::make_pair("repetition_penalty_range", picojson::value{ static_cast<double>(params.repetition_penalty_range) }));
        request_body_json.insert(std::make_pair("penalty_alpha", picojson::value{ params.penalty_alpha }));
        request_body_json.insert(std::make_pair("guidance_scale", picojson::value{ params.guidance_scale }));
        request_body_json.insert(std::make_pair("mirostat_mode", picojson::value{ static_cast<double>(params.mirostat_mode) }));
        request_body_json.insert(std::make_pair("mirostat_tau", picojson::value{ params.mirostat_tau }));
        request_body_json.insert(std::make_pair("mirostat_eta", picojson::value{ params.mirostat_eta }));
        request_body_json.insert(std::make_pair("prompt_lookup_num_tokens", picojson::value{ static_cast<double>(params.prompt_lookup_num_tokens) }));
        request_body_json.insert(std::make_pair("max_tokens_second", picojson::value{ static_cast<double>(params.max_tokens_second) }));
        request_body_json.insert(std::make_pair("do_sample", picojson::value{ params.do_sample }));
        request_body_json.insert(std::make_pair("dynamic_temperature", picojson::value{ static_cast<double>(params.max_tokens_second) }));
        request_body_json.insert(std::make_pair("temperature_last", picojson::value{ params.temperature_last }));
        request_body_json.insert(std::make_pair("auto_max_new_tokens", picojson::value{ params.auto_max_new_tokens }));
        request_body_json.insert(std::make_pair("ban_eos_token", picojson::value{ params.ban_eos_token }));
        request_body_json.insert(std::make_pair("add_bos_token", picojson::value{ params.add_bos_token }));
        request_body_json.insert(std::make_pair("skip_special_tokens", picojson::value{ params.skip_special_tokens }));
        request_body_json.insert(std::make_pair("static_cache", picojson::value{ params.static_cache }));
        request_body_json.insert(std::make_pair("truncation_length", picojson::value{ static_cast<double>(params.truncation_length) }));

        if (!params.sampler_priority.empty())
        {
            picojson::array sampler_priority_array;
            for (const std::string& str : params.sampler_priority)
            {
                sampler_priority_array.push_back(picojson::value{ str });
            }
            request_body_json.insert(std::make_pair("sampler_priority", picojson::value{ sampler_priority_array }));
        }

        request_body_json.insert(std::make_pair("custom_token_bans", picojson::value{ params.custom_token_bans }));
        request_body_json.insert(std::make_pair("negative_prompt", picojson::value{ params.negative_prompt }));
        request_body_json.insert(std::make_pair("dry_sequence_breakers", picojson::value{ params.dry_sequence_breakers }));
        request_body_json.insert(std::make_pair("grammar_string", picojson::value{ params.grammar_string }));

        const std::string request_body{ picojson::value{ request_body_json }.serialize() };
        BOOST_LOG_TRIVIAL(info) << "Send JSON\n```\n" << request_body << "\n```";

        http::request<http::string_body> request{ http::verb::post, config.tg_completions_params.completions_target, 11 }; // HTTP/1.1
        request.set(http::field::host, config.tg_completions_params.host);
        request.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
        request.set(http::field::content_type, "application/json; charset=UTF-8");
        request.body() = request_body;
        request.prepare_payload();

        if (!config.tg_completions_params.api_key.empty())
        {
            request.set(http::field::authorization, ("Bearer ") + config.tg_completions_params.api_key);
        }

        http::write(tcp_stream, request);

        beast::flat_buffer buffer;
        http::response<http::string_body> response;
        http::read(tcp_stream, buffer, response);

        beast::error_code error_code;
        tcp_stream.socket().shutdown(tcp::socket::shutdown_both, error_code);
        if (error_code && error_code != beast::errc::not_connected)
        {
            throw text_generation_exception{} << error_info::beast::error_code{ error_code };
        }

        if (response.result() == http::status::ok)
        {
            picojson::value response_json;
            std::stringstream ss_response_body(response.body());
            picojson::parse(response_json, ss_response_body);
            const picojson::object& object{ throwable_get<picojson::object>(response_json) };
            const picojson::array& choices{ throwable_find<picojson::array>(object, "choices") };
            const picojson::object& choice{ throwable_at<picojson::object>(choices, 0) };
            result.text = throwable_find<std::string>(choice, "text");
            const picojson::object& usage{ throwable_find<picojson::object>(object, "usage") };
            result.prompt_tokens = static_cast<int>(throwable_find<double>(usage, "prompt_tokens"));
            result.completion_tokens = static_cast<int>(throwable_find<double>(usage, "completion_tokens"));
            result.total_tokens = static_cast<int>(throwable_find<double>(usage, "total_tokens"));
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

    try
    {
        net::io_context ioc;
        tcp::resolver resolver{ ioc };
        beast::tcp_stream tcp_stream{ ioc };

        auto const results = resolver.resolve(config.tg_completions_params.host, config.tg_completions_params.port);
        tcp_stream.connect(results);

        picojson::object request_body_json;
        request_body_json.insert(std::make_pair("text", picojson::value{ prompt }));

        const std::string request_body{ picojson::value{ request_body_json }.serialize() };
        BOOST_LOG_TRIVIAL(info) << "Send JSON\n```\n" << request_body << "\n```";

        http::request<http::string_body> request{ http::verb::post, config.tg_completions_params.token_count_target, 11 }; // HTTP/1.1
        request.set(http::field::host, config.tg_completions_params.host);
        request.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
        request.set(http::field::content_type, "application/json; charset=UTF-8");
        request.body() = request_body;
        request.prepare_payload();

        http::write(tcp_stream, request);

        beast::flat_buffer buffer;
        http::response<http::string_body> response;
        http::read(tcp_stream, buffer, response);

        beast::error_code error_code;
        tcp_stream.socket().shutdown(tcp::socket::shutdown_both, error_code);
        if (error_code && error_code != beast::errc::not_connected)
        {
            throw text_generation_exception{} << error_info::beast::error_code{ error_code };
        }

        if (response.result() == http::status::ok)
        {
            picojson::value response_json;
            std::stringstream ss_response_body(response.body());
            picojson::parse(response_json, ss_response_body);

            const picojson::object& object{ throwable_get<picojson::object>(response_json) };
            return static_cast<int>(throwable_find<double>(object, "length"));

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
    if (config.mode != "tg")
    {
        return;
    }

    picojson::array cache;
    for (const token_count_string& element : config.lru_cache.get<lru_tag>())
    {
        picojson::object node;
        node.insert(std::make_pair("string", picojson::value{ element.str }));
        node.insert(std::make_pair("tokens", picojson::value{ static_cast<double>(element.tokens) }));
        cache.push_back(picojson::value{ node });
    }
    picojson::object json;
    json.insert(std::make_pair("cache", picojson::value{ cache }));
    const std::string serialized = picojson::value{ json }.serialize();

    std::filesystem::path cache_path{ string_to_path_by_config("cache.json", config) };
    create_parent_directories(cache_path);
    boost::nowide::ofstream ofs{ cache_path };
    ofs << serialized;
}

void read_cache(const config& config)
{
    if (config.mode != "tg")
    {
        return;
    }

    picojson::value json;
    std::filesystem::path cache_path{ string_to_path_by_config("cache.json", config) };

    if (!std::filesystem::exists(cache_path))
    {
        return;
    }

    try
    {
        boost::nowide::ifstream ifs{ cache_path };
        picojson::parse(json, ifs);
        lru_cache lru_cache;
        const picojson::object& object{ throwable_get<picojson::object>(json) };
        const picojson::array& caches{ throwable_find<picojson::array>(object, "cache") };
        for (const auto& cache : caches)
        {
            const picojson::object& cache_object{ throwable_get<picojson::object>(cache) };
            const std::string str{ throwable_find<std::string>(cache_object, "string") };
            const int tokens{ static_cast<int>(throwable_find<double>(cache_object, "tokens")) };
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
    std::size_t initial_prompts_size;
    if (!config.tg_prompt_params.skip_generation_prefix)
    {
        initial_prompts_size = initial_prompts.size();
    }
    initial_prompts += expand_macro(prefix, config);
    if (config.tg_prompt_params.skip_generation_prefix)
    {
        initial_prompts_size = initial_prompts.size();
    }
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

        int remaining_context = config.tg_completions_params.truncation_length - current_tokens;
        if (remaining_context <= 0)
        {
            BOOST_LOG_TRIVIAL(warning) << "Context window full. Cannot generate more tokens.";
            break;
        }

        int tokens_to_generate = std::min(config.tg_completions_params.max_tokens, remaining_context);
        if (tokens_to_generate <= 0)
        {
            BOOST_LOG_TRIVIAL(warning) << "No tokens left to generate. Aborting.";
            break;
        }

        tg_completions_parameters temp_params = config.tg_completions_params;
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
    if (config.tg_prompt_params.generation_prefix.empty())
    {
        config.tg_prompt_params.generation_prefix = "\\n{{phase}}: ";
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

void init_tg_mode(config& config)
{
    if (!config.tg_prompt_params.paragraphs_file.empty())
    {
        config.phases.clear();
        std::filesystem::path plot_file_path{ string_to_path_by_config(config.tg_prompt_params.paragraphs_file, config) };
        std::string content;
        read_file_to_string(plot_file_path, content);
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
        config.tg_completions_params.stop = { "\\n\\n", ":", "***" };
        config.tg_completions_params.sampler_priority =
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
        config.tg_completions_params.dry_sequence_breakers = "(\"\\n\", \":\", \"\\\"\", \"*\")";

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "produce help message")
            ("mode", po::value<std::string>(&config.mode)->default_value("tg"), "Specify mode tg | sd | sb")
            ("base-path", po::value<std::string>(&config.base_path)->default_value("."), "base path")
            ("log-level", po::value<std::string>(&config.log_level)->default_value("info"), "log level (trace|debug|info|warning|error|fatal)")
            ("log-file", po::value<std::string>(&config.log_file)->default_value("log.txt"), "log file path")
            ("verbose,v", po::bool_switch(&config.verbose)->default_value(false), "enable verbose output")
            ("number-iterations,N", po::value<int>(&config.number_iterations)->default_value(1), "number of iterations (-1 means infinity)")
            ("define,D", po::value<std::vector<std::string>>(&config.predefined_macros), "define macro by key-value pair")
            ("phases", po::value<std::vector<std::string>>(&config.phases)->multitoken(), "phases name list")

            ("tg-system-prompts-file", po::value<std::string>(&config.tg_prompt_params.system_prompts_file)->default_value("system_prompts.txt"), "TG system prompt file path")
            ("tg-examples-file", po::value<std::string>(&config.tg_prompt_params.examples_file)->default_value("examples.txt"), "TG exmaples file path")
            ("tg-history-file", po::value<std::string>(&config.tg_prompt_params.history_file)->default_value("history.txt"), "TG history file path")
            ("tg-output-file", po::value<std::string>(&config.tg_prompt_params.output_file)->default_value("history.txt"), "TG output file path")
            ("tg-example-separator", po::value<std::string>(&config.tg_prompt_params.example_separator)->default_value("***"), "TG separator to be inserted before and after examples")
            ("tg-generation-prefix", po::value<std::string>(&config.tg_prompt_params.generation_prefix)->default_value(""), "TG generation prefix")
            ("tg-skip-generation-prefix", po::bool_switch(&config.tg_prompt_params.skip_generation_prefix)->default_value(false), "TG skip generation prefix")
            ("tg-retry-generation-prefix", po::value<std::string>(&config.tg_prompt_params.retry_generation_prefix)->default_value(""), "TG prefix to be used after a failed text generation")
            ("tg-paragraphs-file", po::value<std::string>(&config.tg_prompt_params.paragraphs_file)->default_value(""), "TG paragraphs file")
            ("tg-host", po::value<std::string>(&config.tg_completions_params.host)->default_value("localhost"), "TG host")
            ("tg-port", po::value<std::string>(&config.tg_completions_params.port)->default_value("5000"), "TG port")
            ("tg-api-key", po::value<std::string>(&config.tg_completions_params.api_key)->default_value(""), "TG API key")
            ("tg-completions-target", po::value<std::string>(&config.tg_completions_params.completions_target)->default_value("/v1/completions"), "TG completions target")
            ("tg-token-count-target", po::value<std::string>(&config.tg_completions_params.token_count_target)->default_value("/v1/internal/token-count"), "TG token count target")
            ("tg-min-completion-tokens", po::value<int>(&config.min_completion_tokens)->default_value(256), "TG min completion tokens")
            ("tg-max-completion-iterations", po::value<int>(&config.max_completion_iterations)->default_value(5), "TG max completion iterations")
            ("tg-model", po::value<std::string>(&config.tg_completions_params.model)->default_value("", "TG model"))
            ("tg-num-best-of", po::value<int>(&config.tg_completions_params.best_of)->default_value(1), "TG best of")
            ("tg-echo", po::bool_switch(&config.tg_completions_params.echo)->default_value(false), "TG echo")
            ("tg-frequency-penalty", po::value<double>(&config.tg_completions_params.frequency_penalty)->default_value(0.0), "TG frequency penalty")
            //std::map<int, double> logit_bias;
            ("tg-logprobs", po::value<double>(&config.tg_completions_params.logprobs)->default_value(0.0), "TG presence penalty")
            ("tg-max-tokens", po::value<int>(&config.tg_completions_params.max_tokens)->default_value(512), "TG max tokens")
            ("tg-n", po::value<int>(&config.tg_completions_params.n)->default_value(1), "TG number of responses generated for the same prompt")
            ("tg-presence-penalty", po::value<double>(&config.tg_completions_params.presence_penalty)->default_value(0.0), "TG presence penalty")
            ("tg-stop", po::value<std::vector<std::string>>(&config.tg_completions_params.stop)->multitoken(), "TG stop sequences")
            ("tg-stream", po::bool_switch(&config.tg_completions_params.stream)->default_value(false), "TG stream")
            ("tg-suffix", po::value<std::string>(&config.tg_completions_params.suffix)->default_value(""), "TG suffix")
            ("tg-temperature", po::value<double>(&config.tg_completions_params.temperature)->default_value(1.0), "TG temperature")
            ("tg-top-p", po::value<double>(&config.tg_completions_params.top_p)->default_value(1.0), "TG top p")
            ("tg-seed", po::value<int>(&config.seed)->default_value(-1), "TG seed value")
            ("tg-dynatemp-low", po::value<double>(&config.tg_completions_params.dynatemp_low)->default_value(0.75, "0.75"), "TG dynatemp low")
            ("tg-dynatemp-high", po::value<double>(&config.tg_completions_params.dynatemp_high)->default_value(1.25, "1.25"), "TG dynatemp high")
            ("tg-dynatemp-exponent", po::value<double>(&config.tg_completions_params.dynatemp_exponent)->default_value(1.0), "TG dynatemp exponent")
            ("tg-smoothing-factor", po::value<double>(&config.tg_completions_params.smoothing_factor)->default_value(0.0), "TG smoothing factor")
            ("tg-smoothing-curve", po::value<double>(&config.tg_completions_params.smoothing_curve)->default_value(1.0), "TG smoothing curve")
            ("tg-min-p", po::value<double>(&config.tg_completions_params.min_p)->default_value(0.1, "0.1"), "TG min p")
            ("tg-top-k", po::value<int>(&config.tg_completions_params.top_k)->default_value(0), "TG top k")
            ("tg-typical-p", po::value<double>(&config.tg_completions_params.typical_p)->default_value(1.0), "TG typical p")
            ("tg-xtc-threshold", po::value<double>(&config.tg_completions_params.xtc_threshold)->default_value(0.1, "0.1"), "TG Exclude Top Choices (XTC) threshold")
            ("tg-xtc-probability", po::value<double>(&config.tg_completions_params.xtc_probability)->default_value(0.0), "TG Exclude Top Choices (XTC) probability")
            ("tg-epsilon-cutoff", po::value<double>(&config.tg_completions_params.epsilon_cutoff)->default_value(0), "TG epsilon cutoff")
            ("tg-eta-cutoff", po::value<double>(&config.tg_completions_params.eta_cutoff)->default_value(0), "TG eta cutoff")
            ("tg-tfs", po::value<double>(&config.tg_completions_params.tfs)->default_value(1.0), "TG tfs")
            ("tg-top-a", po::value<double>(&config.tg_completions_params.top_a)->default_value(0.0), "TG top a")
            ("tg-top-n-sigma", po::value<double>(&config.tg_completions_params.top_n_sigma)->default_value(1.0), "TG top n sigma")
            ("tg-dry-multiplier", po::value<double>(&config.tg_completions_params.dry_multiplier)->default_value(0.0), "TG DRY multiplier")
            ("tg-dry-allowed-length", po::value<int>(&config.tg_completions_params.dry_allowed_length)->default_value(2), "TG DRY allowed length")
            ("tg-dry-base", po::value<double>(&config.tg_completions_params.dry_base)->default_value(1.75), "TG DRY base")
            ("tg-repetition-penalty", po::value<double>(&config.tg_completions_params.repetition_penalty)->default_value(1.2), "TG repetition penalty")
            ("tg-encoder-repetition-penalty", po::value<double>(&config.tg_completions_params.encoder_repetition_penalty)->default_value(1.0), "TG encoder repetition penalty")
            ("tg-no-repeat-ngram-size", po::value<int>(&config.tg_completions_params.no_repeat_ngram_size)->default_value(0), "TG no repeat ngram size")
            ("tg-repetition-penalty-range", po::value<int>(&config.tg_completions_params.repetition_penalty_range)->default_value(0), "TG repetition penalty range")
            ("tg-penalty-alpha", po::value<double>(&config.tg_completions_params.penalty_alpha)->default_value(0.9, "0.9"), "TG penalty alpha")
            ("tg-guidance-scale", po::value<double>(&config.tg_completions_params.guidance_scale)->default_value(1.0), "TG guidance scale")
            ("tg-mirostat-mode", po::value<int>(&config.tg_completions_params.mirostat_mode)->default_value(0), "TG mirostat mode")
            ("tg-mirostat-tau", po::value<double>(&config.tg_completions_params.mirostat_tau)->default_value(5), "TG mirostat tau")
            ("tg-mirostat-eta", po::value<double>(&config.tg_completions_params.mirostat_eta)->default_value(0.1, "0.1"), "TG mirostat eta")
            ("tg-prompt-lookup-num-tokens", po::value<int>(&config.tg_completions_params.prompt_lookup_num_tokens)->default_value(0), "TG prompt lookup num tokens")
            ("tg-max-tokens-second", po::value<int>(&config.tg_completions_params.max_tokens_second)->default_value(0), "TG max tokens second")
            ("tg-do-sample", po::bool_switch(&config.tg_completions_params.do_sample)->default_value(true), "TG do sample")
            ("tg-dynamic-temperature", po::bool_switch(&config.tg_completions_params.dynamic_temperature)->default_value(false), "TG dynamic temperature")
            ("tg-temperature-last", po::bool_switch(&config.tg_completions_params.temperature_last)->default_value(false), "TG temperature last")
            ("tg-auto-max-new-tokens", po::bool_switch(&config.tg_completions_params.auto_max_new_tokens)->default_value(false), "TG auto max_new tokens")
            ("tg-ban-eos-token", po::bool_switch(&config.tg_completions_params.ban_eos_token)->default_value(false), "TG ban eos token")
            ("tg-add-bos-token", po::bool_switch(&config.tg_completions_params.add_bos_token)->default_value(true), "TG add Beginning of Sequence Token (BOS) token")
            ("tg-skip-special-tokens", po::bool_switch(&config.tg_completions_params.skip_special_tokens)->default_value(true), "TG skip special tokens (bos_token, eos_token, unk_token, pad_token, etc.)")
            ("tg-static-cache", po::bool_switch(&config.tg_completions_params.static_cache)->default_value(false), "TG static cache")
            ("tg-truncation-length", po::value<int>(&config.tg_completions_params.truncation_length)->default_value(4096), "TG truncation length")
            ("tg-sampler-priority", po::value<std::vector<std::string>>(&config.tg_completions_params.sampler_priority)->multitoken(), "TG sampler priority")
            ("tg-custom-token-bans", po::value<std::string>(&config.tg_completions_params.custom_token_bans)->default_value(""), "TG custom token bans")
            ("tg-negative-prompt", po::value<std::string>(&config.tg_completions_params.negative_prompt)->default_value(""), "TG negative prompt")
            ("tg-dry-sequence-breakers", po::value<std::string>(&config.tg_completions_params.dry_sequence_breakers)->default_value(""), "TG dry sequence breakers")
            ("tg-grammar-string", po::value<std::string>(&config.tg_completions_params.grammar_string)->default_value(""), "TG grammar-string")

            ("sd-host", po::value<std::string>(&config.sd_txt2img_params.host)->default_value("localhost"), "SD host")
            ("sd-port", po::value<std::string>(&config.sd_txt2img_params.port)->default_value("7860"), "SD port")
            ("sd-target", po::value<std::string>(&config.sd_txt2img_params.target)->default_value("/sdapi/v1/txt2img"), "SD txt2img target")
            ("sd-prompt-file", po::value<std::string>(&config.sd_txt2img_params.prompt_file)->default_value("prompt.txt"), "SD prompt file")
            ("sd-negative-prompt-file", po::value<std::string>(&config.sd_txt2img_params.negative_prompt_file)->default_value("negative_prompt.txt"), "SD negative prompt file")
            ("sd-output-file", po::value<std::string>(&config.sd_txt2img_params.output_file)->default_value("{{datetime}}.png"), "SD output PNG file")
            ("sd-prompt", po::value<std::string>(&config.sd_txt2img_params.prompt)->default_value(""), "SD prompt")
            ("sd-negative-prompt", po::value<std::string>(&config.sd_txt2img_params.negative_prompt)->default_value(""), "SD negative prompt")
            ("sd-styles", po::value<std::vector<std::string>>(&config.sd_txt2img_params.styles), "SD styles")
            ("sd-seed", po::value<int>(&config.sd_txt2img_params.seed)->default_value(-1), "SD seed")
            ("sd-subseed", po::value<int>(&config.sd_txt2img_params.subseed)->default_value(-1), "SD subseed")
            ("sd-subseed-strength", po::value<double>(&config.sd_txt2img_params.subseed_strength)->default_value(0), "SD subseed strength")
            ("sd-seed-resize-from-h", po::value<int>(&config.sd_txt2img_params.seed_resize_from_h)->default_value(-1), "SD seed resize from height")
            ("sd-seed-resize-from-w", po::value<int>(&config.sd_txt2img_params.seed_resize_from_w)->default_value(-1), "SD seed resize from width")
            ("sd-sampler-name", po::value<std::string>(&config.sd_txt2img_params.sampler_name)->default_value("Euler a"), "SD sampler name")
            ("sd-scheduler", po::value<std::string>(&config.sd_txt2img_params.scheduler)->default_value("Automatic"), "SD scheduler")
            ("sd-batch_size", po::value<int>(&config.sd_txt2img_params.batch_size)->default_value(1), "SD batch size")
            ("sd-n-iter", po::value<int>(&config.sd_txt2img_params.n_iter)->default_value(1), "SD n iter")
            ("sd-steps", po::value<int>(&config.sd_txt2img_params.steps)->default_value(30), "SD steps")
            ("sd-cfg-scale", po::value<double>(&config.sd_txt2img_params.cfg_scale)->default_value(7), "SD cfg scale")
            ("sd-width", po::value<int>(&config.sd_txt2img_params.width)->default_value(1024), "SD image width")
            ("sd-height", po::value<int>(&config.sd_txt2img_params.height)->default_value(1024), "SD image height")
            ("sd-restore-faces", po::bool_switch(&config.sd_txt2img_params.restore_faces)->default_value(false), "SD restore faces")
            ("sd-tiling", po::bool_switch(&config.sd_txt2img_params.tiling)->default_value(false), "SD tiling")
            ("sd-do-not-save-samples", po::bool_switch(&config.sd_txt2img_params.do_not_save_samples)->default_value(false), "SD do not save samples")
            ("sd-do-not-save-grid", po::bool_switch(&config.sd_txt2img_params.do_not_save_grid)->default_value(false), "SD do not save grid")
            ("sd-eta", po::value<int>(&config.sd_txt2img_params.eta)->default_value(0), "SD eta")
            ("sd-denoising-strength", po::value<double>(&config.sd_txt2img_params.denoising_strength)->default_value(0.7, "0.7"), "SD denoising strength")
            ("sd-s-min-uncond", po::value<int>(&config.sd_txt2img_params.s_min_uncond)->default_value(0), "SD s min uncond")
            ("sd-s-churn", po::value<int>(&config.sd_txt2img_params.s_churn)->default_value(0), "SD s churn")
            ("sd-s-tmax", po::value<int>(&config.sd_txt2img_params.s_tmax)->default_value(0), "SD s tmax")
            ("sd-s-tmin", po::value<int>(&config.sd_txt2img_params.s_tmin)->default_value(0), "SD s tmin")
            ("sd-s-noise", po::value<int>(&config.sd_txt2img_params.s_noise)->default_value(1), "SD s noise")
            ("sd-override-settings", po::value<std::string>(&config.sd_txt2img_params.override_settings)->default_value(""), "SD override settings")
            ("sd-override-settings-restore-afterwards", po::bool_switch(&config.sd_txt2img_params.override_settings_restore_afterwards)->default_value(true), "SD override settings restore afterwards")
            ("sd-refiner-checkpoint", po::value<std::string>(&config.sd_txt2img_params.refiner_checkpoint)->default_value(""), "SD refiner checkpoint")
            ("sd-refiner-switch-at", po::value<double>(&config.sd_txt2img_params.refiner_switch_at)->default_value(0.8, "0.8"), "SD refiner switch at")
            ("sd-disable-extra-networks", po::bool_switch(&config.sd_txt2img_params.disable_extra_networks)->default_value(false), "SD disable extra networks")
            ("sd-firstpass-image", po::value<std::string>(&config.sd_txt2img_params.firstpass_image)->default_value(""), "SD firstpass image")
            ("sd-comments", po::value<std::string>(&config.sd_txt2img_params.comments)->default_value(""), "SD comments")
            ("sd-enable-hr", po::bool_switch(&config.sd_txt2img_params.enable_hr)->default_value(false), "SD enable hr")
            ("sd-firstphase-width", po::value<int>(&config.sd_txt2img_params.firstphase_width)->default_value(0), "SD firstphase width")
            ("sd-firstphase-height", po::value<int>(&config.sd_txt2img_params.firstphase_height)->default_value(0), "SD firstphase height")
            ("sd-hr-scale", po::value<double>(&config.sd_txt2img_params.hr_scale)->default_value(0), "SD hr scale")
            ("sd-hr-upscaler", po::value<std::string>(&config.sd_txt2img_params.hr_upscaler)->default_value("SwinIR_4x"), "SD hr upscaler")
            ("sd-hr-second-pass-steps", po::value<int>(&config.sd_txt2img_params.hr_second_pass_steps)->default_value(20), "SD hr second pass steps")
            ("sd-hr-resize-x", po::value<int>(&config.sd_txt2img_params.hr_resize_x)->default_value(0), "SD hr resize x")
            ("sd-hr-resize-y", po::value<int>(&config.sd_txt2img_params.hr_resize_y)->default_value(0), "SD hr resize y")
            ("sd-hr-checkpoint-name", po::value<std::string>(&config.sd_txt2img_params.hr_checkpoint_name)->default_value(""), "SD hr checkpoint name")
            //("sd-hr-prompt", po::value<std::string>(&config.sd_txt2img_params.hr_prompt)->default_value(""), "SD hr prompt")
            //("sd-hr-negative-prompt", po::value<std::string>(&config.sd_txt2img_params.hr_negative_prompt)->default_value(""), "SD hr negative prompt")
            ("sd-force-task-id", po::value<std::string>(&config.sd_txt2img_params.force_task_id)->default_value(""), "SD force task id")
            ("sd-sampler-index", po::value<std::string>(&config.sd_txt2img_params.sampler_index)->default_value(""), "SD sampler index")
            ("sd-script-name", po::value<std::string>(&config.sd_txt2img_params.script_name)->default_value(""), "SD script name")
            ("sd-script-args", po::value<std::vector<std::string>>(&config.sd_txt2img_params.script_args), "SD script_args")
            ("sd-send-images", po::bool_switch(&config.sd_txt2img_params.send_images)->default_value(true), "SD send images")
            ("sd-save-images", po::bool_switch(&config.sd_txt2img_params.save_images)->default_value(false), "SD save images")
            ("sd-ad-enable", po::bool_switch(&config.sd_txt2img_params.alwayson_scripts.adetailer_parametesrs.ad_enable)->default_value(false), "SD ADetailer enable")
            ("sd-ad-model", po::value<std::string>(&config.sd_txt2img_params.alwayson_scripts.adetailer_parametesrs.args1.ad_model)->default_value("face_yolov8n.pt"), "SD ADetailer model")
            ("sd-ad-prompt", po::value<std::string>(&config.sd_txt2img_params.alwayson_scripts.adetailer_parametesrs.args1.ad_prompt)->default_value(""), "SD ADetailer prompt")
            ("sd-ad-negative-prompt", po::value<std::string>(&config.sd_txt2img_params.alwayson_scripts.adetailer_parametesrs.args1.ad_prompt)->default_value(""), "SD ADetailer negative prompt")
            ("sd-infotext", po::value<std::string>(&config.sd_txt2img_params.infotext)->default_value(""), "SD infotext")
            ("sd-abg-remover-enable", po::bool_switch(&config.sd_txt2img_params.abg_remover_enable)->default_value(false), "SD ABG Remover enable")

            ("sb-host", po::value<std::string>(&config.sb_generation_params.host)->default_value("localhost"), "SB host")
            ("sb-port", po::value<std::string>(&config.sb_generation_params.port)->default_value("5001"), "SB port")
            ("sb-target", po::value<std::string>(&config.sb_generation_params.target)->default_value("/voice"), "SB voide target")
            ("sb-text-file", po::value<std::string>(&config.sb_generation_params.text_file)->default_value("text.txt"), "SB text file")
            ("sb-output-file", po::value<std::string>(&config.sb_generation_params.output_file)->default_value("{{datetime}}.wav"), "SB output WAV")
            ("sb-text", po::value<std::string>(&config.sb_generation_params.text)->default_value(""), "SB text")
            ("sb-model-name", po::value<std::string>(&config.sb_generation_params.model_name)->default_value(""), "SB model name")
            ("sb-model-id", po::value<int>(&config.sb_generation_params.model_id)->default_value(0), "SB model id")
            ("sb-speaker-name", po::value<std::string>(&config.sb_generation_params.speaker_name)->default_value(""), "SB speaker name")
            ("sb-speaker-id", po::value<int>(&config.sb_generation_params.speaker_id)->default_value(0), "SB speaker id")
            ("sb-sdp-ratio", po::value<double>(&config.sb_generation_params.sdp_ratio)->default_value(0.2, "0.2"), "SB sdp ratio")
            ("sb-noise", po::value<double>(&config.sb_generation_params.noise)->default_value(0.6, "0.6"), "SB noise")
            ("sb-noisew", po::value<double>(&config.sb_generation_params.noisew)->default_value(0.8, "0.8"), "SB noisew")
            ("sb-length", po::value<double>(&config.sb_generation_params.length)->default_value(1), "SB length")
            ("sb-language", po::value<std::string>(&config.sb_generation_params.language)->default_value(""), "SB language")
            ("sb-auto-split", po::bool_switch(&config.sb_generation_params.auto_split)->default_value(true), "SB auto split")
            ("sb-split-interval", po::value<double>(&config.sb_generation_params.split_interval)->default_value(0.5, "0.5"), "SB split interval")
            ("sb-assist-text", po::value<std::string>(&config.sb_generation_params.assist_text)->default_value(""), "SB assist text")
            ("sb-assist-text-weight", po::value<double>(&config.sb_generation_params.assist_text_weight)->default_value(1), "SB assist text weight")
            ("sb-style", po::value<std::string>(&config.sb_generation_params.style)->default_value(""), "SB style")
            ("sb-style-weight", po::value<double>(&config.sb_generation_params.style_weight)->default_value(1), "SB style weight")
            ("sb-reference-audio-path", po::value<std::string>(&config.sb_generation_params.reference_audio_path)->default_value(""), "SB reference audio path")
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

        if (config.mode == "tg")
        {
            init_tg_mode(config);
        }
        else if (config.mode == "sd")
        {
            ;
        }
        else if (config.mode == "sb")
        {
            ;
        }
        else
        {
            BOOST_LOG_TRIVIAL(error) << "mode options must be tg | sd | sb.";
            return 1;
        }

        if (config.phases.empty())
        {
            config.phases = { "" };
        }

        std::transform(config.tg_completions_params.stop.begin(), config.tg_completions_params.stop.end(), config.tg_completions_params.stop.begin(), unescape_string);
        std::transform(config.predefined_macros.begin(), config.predefined_macros.end(), config.predefined_macros.begin(), unescape_string);
        config.tg_completions_params.dry_sequence_breakers = unescape_string(config.tg_completions_params.dry_sequence_breakers);
        config.tg_prompt_params.generation_prefix = unescape_string(config.tg_prompt_params.generation_prefix);
        config.tg_prompt_params.retry_generation_prefix = unescape_string(config.tg_prompt_params.retry_generation_prefix);

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
                const std::string macro_expanded_string{ expand_macro(*first, config) };
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

    int remaining_tokens = config.tg_completions_params.truncation_length - config.tg_completions_params.max_tokens;

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
        if (!config.tg_prompt_params.example_separator.empty())
        {
            remaining_tokens -= (static_cast<int>(config.tg_prompt_params.example_separator.size()) + 2) * 2;
        }
        int written_tokens = 0;
        try_append(examples.begin(), examples.end(), examples_string, remaining_tokens, written_tokens, false);
        if (written_tokens > 0)
        {
            if (!config.tg_prompt_params.example_separator.empty())
            {
                std::string temp = "\n";
                temp += config.tg_prompt_params.example_separator;
                temp += "\n";
                temp += examples_string;
                temp += "\n";
                temp += config.tg_prompt_params.example_separator;
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
    if (config.mode == "tg")
    {
        const std::filesystem::path system_prompts_path{ string_to_path_by_config(config.tg_prompt_params.system_prompts_file, config) };
        read_file_to_container(system_prompts_path, prompts.system_prompts);

        const std::filesystem::path examples_path{ string_to_path_by_config(config.tg_prompt_params.examples_file, config) };
        if (std::filesystem::exists(examples_path) && std::filesystem::is_regular_file(examples_path))
        {
            read_file_to_container(examples_path, prompts.examples);
        }

        const std::filesystem::path history_path{ string_to_path_by_config(config.tg_prompt_params.history_file, config) };
        if (std::filesystem::exists(history_path) && std::filesystem::is_regular_file(history_path))
        {
            read_file_to_container(history_path, prompts.history);
        }
    }
}

void write_response(const config& config, const std::string& response)
{
    const std::filesystem::path output_file_path{ string_to_path_by_config(config.tg_prompt_params.output_file, config) };
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
    if (config.mode == "tg")
    {
        std::string prompts_string = prompts.to_string(config);
        prompts_string = expand_macro(prompts_string, config);

        try
        {
            const std::string response{ generate_and_complete_text(config, prompts_string, generation_prefix) };
            BOOST_LOG_TRIVIAL(info) << "Text generated.\n```\n" << response << "\n```\n";
            write_response(config, response);
        }
        catch (const text_generation_exception& exception)
        {
            BOOST_LOG_TRIVIAL(warning) << boost::diagnostic_information(exception);
            if (!config.tg_prompt_params.retry_generation_prefix.empty())
            {
                BOOST_LOG_TRIVIAL(info) << "Start to retry text generation with retry-generation-prefix.";
                const std::string response{ generate_and_complete_text(config, prompts_string, config.tg_prompt_params.retry_generation_prefix) };
                BOOST_LOG_TRIVIAL(info) << "Text generated.\n```\n" << response << "\n```\n";
                write_response(config, response);
            }
        }
    }
    else if (config.mode == "sd")
    {
        const std::filesystem::path output_file_path{ string_to_path_by_config(config.sd_txt2img_params.output_file, config) };
        create_parent_directories(output_file_path);

        std::string prompt_string;
        if (!config.sd_txt2img_params.prompt.empty())
        {
            prompt_string = expand_macro(config.sd_txt2img_params.prompt, config);
        }
        else
        {
            const std::filesystem::path prompt_file_path{ string_to_path_by_config(config.sd_txt2img_params.prompt_file, config) };
            read_file_to_string(prompt_file_path, prompt_string);
            prompt_string = expand_macro(prompt_string, config);
        }

        std::string negative_prompt_string;
        if (!config.sd_txt2img_params.prompt.empty())
        {
            negative_prompt_string = expand_macro(config.sd_txt2img_params.negative_prompt, config);
        }
        else
        {
            const std::filesystem::path negative_prompt_file_path{ string_to_path_by_config(config.sd_txt2img_params.negative_prompt_file, config) };
            read_file_to_string(negative_prompt_file_path, negative_prompt_string);
            negative_prompt_string = expand_macro(negative_prompt_string, config);
        }

        send_automatic1111_txt2img_request(config, prompt_string, negative_prompt_string, output_file_path);
    }
    else if (config.mode == "sb")
    {
        std::string text_string;
        if (!config.sb_generation_params.text.empty())
        {
            text_string = expand_macro(config.sb_generation_params.text, config);
        }
        else
        {
            const std::filesystem::path text_file_path{ string_to_path_by_config(config.sb_generation_params.text_file, config) };
            read_file_to_string(text_file_path, text_string);
            text_string = expand_macro(text_string, config);
        }
        send_style_bert_voice_request(config, text_string);
    }
}

void set_seed(config& config)
{
    if (config.seed == -1)
    {
        config.tg_completions_params.seed = generate_random_seed();
        config.sd_txt2img_params.seed = generate_random_seed();
    }
    else
    {
        config.tg_completions_params.seed = config.seed;
        config.sd_txt2img_params.seed = config.seed;
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
            generate_and_output(config, prompts, config.tg_prompt_params.generation_prefix);
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