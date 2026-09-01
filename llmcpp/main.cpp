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
#include <unordered_map>
#include <deque>
#include <memory>
#include <stdexcept>
#include <optional>
#include <thread>
#include <type_traits>
#include <cstdint>
#include <variant>

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
#include <boost/nowide/cstdlib.hpp>
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
#include <boost/process/v2/process.hpp>
#include <boost/process/v2/environment.hpp>
#include <boost/process/v2/execute.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/noncopyable.hpp>
#include <boost/range/algorithm.hpp>

#include "picojson.h"

#if defined(_WIN32)
#include <boost/process/v2/windows/creation_flags.hpp>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef STRICT
#define STRICT
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <tlhelp32.h>
#undef IN
#undef OUT
#undef NEAR
#undef FAR
#endif

#if defined(_WIN32) || defined(_WIN64)
#include <io.h>
#else
#include <unistd.h>
#endif

class runtime_exception
    : public boost::exception
    , public std::exception
{
public:
    runtime_exception();
};

class runtime_exception;
class io_exception : public runtime_exception {};
class file_open_exception : public io_exception {};
class socket_exception : public runtime_exception {};
class text_generation_exception : public runtime_exception {};
class image_generation_exception : public runtime_exception {};
class comfy_ui_generation_exception : public runtime_exception {};
class syntax_exception : public runtime_exception {};
class json_parse_exception : public runtime_exception {};
class macro_exception : public runtime_exception {};
class command_line_syntax_exception : public runtime_exception {};
class array_index_out_of_bounds_exception : public runtime_exception {};
class dns_resolve_exception : public runtime_exception {};
class connect_exception : public runtime_exception {};
class http_send_exception : public runtime_exception {};
class http_receive_exception : public runtime_exception {};
class http_status_exception : public runtime_exception {};
class png_exception : public runtime_exception {};

namespace error_info
{
    using stacktrace = boost::error_info<struct tag_stacktrace, boost::stacktrace::stacktrace>;
    using description = boost::error_info<struct tag_description, std::string>;
    using wrapped_std_exception = boost::error_info<struct tag_wrapped_std_exception, std::exception>;
    using wrapped_boost_exception = boost::error_info<struct tag_wrapped_boost_exception, boost::exception>;
    using path = boost::error_info<struct tag_file_path, std::filesystem::path>;

    namespace asio
    {
        using error_code = boost::error_info<struct tag_error_code, boost::beast::error_code>;
    }

    namespace http
    {
        namespace response
        {
            using status = boost::error_info<struct tag_status_int, boost::beast::http::status>;
            using reason = boost::error_info<struct tag_result_int, std::string>;
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

struct config;

struct text_generation_parameters
{
    virtual ~text_generation_parameters() {}
    virtual std::string get_request_body_for_text_completions(std::string_view prompt, int max_tokens) const = 0;
    virtual std::string parse_response_for_text_completions(const boost::beast::http::response<boost::beast::http::string_body>& response) const = 0;
    virtual std::string get_request_body_for_token_count(std::string_view prompt) const = 0;
    virtual int parse_response_for_token_count(const boost::beast::http::response<boost::beast::http::string_body>& response) const = 0;
    virtual int get_max_tokens() const = 0;
    virtual int get_truncation_length() const = 0;
};

struct llm_prompt_parameters
{
    std::string prompt;
    std::string prompt_file;
    std::string output_file;
    std::string generation_prefix;
    std::string generation_suffix;
    std::string paragraphs_file;

    std::string host;
    std::string port;
    std::string api_key;
    std::string completions_target;
    std::string token_count_target;

    int min_completion_tokens{};
    int max_completion_iterations{};

    std::string reasoning_prefix;
    std::string reasoning_suffix;

    bool code_block_extract{};

    text_generation_parameters* backend{};
};

struct tg_completions_parameters
    : text_generation_parameters
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

    std::string get_request_body_for_text_completions(std::string_view prompt, int max_tokens) const override;
    std::string parse_response_for_text_completions(const boost::beast::http::response<boost::beast::http::string_body>& response) const override;
    std::string get_request_body_for_token_count(std::string_view prompt) const override;
    int parse_response_for_token_count(const boost::beast::http::response<boost::beast::http::string_body>& response) const override;

    int get_max_tokens() const override
    {
        return max_tokens;
    }

    int get_truncation_length() const override
    {
        return truncation_length;
    }
};

struct kc_generation_parameters
    : text_generation_parameters
{
    int max_context_length{};
    int max_length{};
    std::string prompt;
    double rep_pen{};
    int rep_pen_range{};
    std::vector<int> sampler_order;
    int sampler_seed{};
    std::vector<std::string> stop_sequence;
    double temperature{};
    double tfs{};
    double top_a{};
    double top_k{};
    double top_p{};
    double min_p{};
    double typical{};
    bool use_default_badwordsids{};
    double dynatemp_range{};
    double smoothing_factor{};
    double dynatemp_exponent{};
    int mirostat{};
    double mirostat_tau{};
    double mirostat_eta{};
    std::string genkey;
    std::string grammar;
    bool grammar_retain_state{};
    std::string memory;
    std::vector<std::string> images;
    bool trim_stop{};
    bool render_special{};
    bool bypass_eos{};
    std::vector<std::string> banned_tokens;
    //std::vector<std::pair<std::string, double>> logit_bias;
    double dry_multiplier{};
    double dry_base{};
    int dry_allowed_length{};
    int dry_penalty_last_n{};
    std::vector<std::string> dry_sequence_breakers;
    double xtc_threshold{};
    double xtc_probability{};
    double nsigma{};
    bool logprobs{};
    bool replace_instruct_placeholders{};

    std::string get_request_body_for_text_completions(std::string_view prompt, int max_tokens) const override;
    std::string parse_response_for_text_completions(const boost::beast::http::response<boost::beast::http::string_body>& response) const override;
    std::string get_request_body_for_token_count(std::string_view prompt) const override;
    int parse_response_for_token_count(const boost::beast::http::response<boost::beast::http::string_body>& response) const override;

    int get_max_tokens() const override
    {
        return max_length;;
    }

    int get_truncation_length() const override
    {
        return max_context_length;
    }
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
        double ad_mask_min_ratio{};
        double ad_mask_max_ratio{};
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

struct cu_generation_parameters
{
    std::string host;
    std::string port;
    std::string prompt_target;
    std::string upload_image_target;

    std::string prompt;
    std::string prompt_file;
    std::string output_directory;
    std::vector<std::string> upload_images;
    bool preserve_subdirectories{};
};

class context
    : private boost::noncopyable
{
public:
    using variable_map_type = std::unordered_map<std::string, std::string>;

    context();
    context make_pushed() const;
    void set(std::string_view key, std::string_view value);
    std::optional<std::string> get(std::string_view key) const;

private:
    context(const context& ctx);
    variable_map_type variable_map;
    const context* base{};
};

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
    std::string mode;
    std::string base_path;
    std::string log_level;
    std::string log_file;
    std::string config_file;
    bool verbose{};
    unsigned int expires_after{};
    int number_iterations{};
    std::vector<std::string> user_defined_variables;
    std::vector<std::string> phases;

    int seed{};

    bool create_process{};
    bool terminate_process{};

    std::string server_executable_file;
    std::string server_arguments;
    std::string server_host;
    std::string server_port;
    int server_max_retries;
    int server_wait_ms;

    llm_prompt_parameters llm;
    tg_completions_parameters tg;
    kc_generation_parameters kc;
    sd_txt2img_parameters sd;
    sb_generation_parameters sb;
    cu_generation_parameters cu;

    mutable lru_cache lru_cache;
    context context;
};

std::string truncate_prompt_by_config(std::string_view prompt, const config& config);

template<typename Value>
const Value& throwable_get(const picojson::value& value);

template<typename Value>
const Value& throwable_at(const picojson::array& array, std::size_t index);

template<typename Value>
const Value& throwable_find(const picojson::object& object, std::string_view key);

std::string base64_decode(std::string_view encoded_string);

std::string trim(std::string_view str);

void truncate_by_tokens(std::string_view string, int max_tokens, const config& config, bool reverse, std::string& result, int& tokens);

void truncate_prompt(std::string_view string, const config& config, bool reverse, std::string& result, int& remaining_tokens);

void create_parent_directories(const std::filesystem::path& path);

std::string read_file_to_string(const std::filesystem::path& file, std::ios::openmode openmode = {});

template<typename Integer>
Integer random(Integer min = std::numeric_limits<Integer>::min(), Integer max = std::numeric_limits<Integer>::max());

std::string complement_extension(std::string_view filepath, std::string_view extension);

std::filesystem::path string_to_path_by_config(std::string_view path, const config& config);

boost::beast::http::response<boost::beast::http::string_body> send_http_get(
    std::string_view host,
    std::string_view port,
    std::string_view target,
    unsigned int expires_after
);

std::string make_automatic1111_png_parameters(const sd_txt2img_parameters& parameters, std::string_view prompt, std::string_view negative_prompt);

std::string send_automatic1111_txt2img_request(
    const config& config,
    std::string_view prompt,
    std::string_view negative_prompt
);

std::string send_style_bert_voice_request(
    const config& config,
    std::string_view text
);

std::string generate_boundary();

std::string upload_image_to_comfy_ui(
    const config& config,
    const std::filesystem::path& image_path,
    bool overwrite = true
);

void send_comfy_ui_prompt(
    const config& config,
    std::string_view workflow
);

std::vector<item> parse_item_list(std::string_view str);

void write_item_list(const config& config, std::string_view task);

int send_token_count_request(const config& config, std::string_view prompt);
int get_tokens_from_cache(const config& config, std::string_view str);
void write_cache(const config& config);
void read_cache(const config& config);

std::string generate_text(
    const config& config,
    std::string_view prompt,
    std::string_view prefix,
    const context& ctx
);

std::string unescape_string(std::string_view str);

std::string json_escape_string(std::string_view str);

void unescape_parameters(config& config);

void parse_user_defined_variables(const std::vector<std::string>& predefined_macros, context& context);

void init_logging_with_nowide_cout();
void init_logging_with_nowide_file_log(const std::filesystem::path& log);
void init_logging(const config& config);
void init_chat_mode(config& config);

void set_phase_variables(
    const std::vector<std::string>& phases,
    std::size_t phase_index,
    const context& context
);

void set_static_builtin_variables(
    config& config
);

void set_dynamic_builtin_variables(
    config& config
);

void set_paragraphs_to_phases(
    const std::vector<item>& paragraphs,
    std::vector<std::string>& phases
);

void init_llm_mode(config& config);

std::string sanitize_as_filename(std::string_view name);

std::map<std::string, std::string> extract_code_block_from_markdown(std::string_view markdown_content);

bool wait_for_port(const std::string& host, const std::string& port, unsigned int max_retries, unsigned int wait_ms);

void create_process_async(std::string_view excutable_file, const std::vector<std::string>& arguments);

std::size_t terminate_process_by_path(const std::filesystem::path& executable_file_path);

std::vector<std::string> parse_command_line_args(std::string_view args);

int parse_command_line(
    int argc,
    char** argv,
    config& config
);

std::string remove_reasoning(std::string_view response, std::string_view prefix, std::string_view suffix);

void write_file(const config& config, std::string_view response, std::string_view filepath, std::ios_base::openmode mode);

void write_code_block(const config& config, std::string_view markdown);

void generate_text_and_write(const config& config, std::string_view prompt, const context& ctx);

std::string prompt_from_string_or_file_path(
    std::string_view string,
    std::string_view file_path,
    const config& config
);

void generate_and_output(const config& config);

void set_seed(config& config);

void process_create_or_terminate(const config& config);

void iterate(config& config);

int exception_safe_main(int argc, char** argv);

context::context()
{
}

context::context(const context& ctx)
    : base{ &ctx }
{
}

context context::make_pushed() const
{
    return context{ *this };
}

void context::set(std::string_view key, std::string_view value)
{
    variable_map[std::string{ key }] = value;
}

std::optional<std::string> context::get(std::string_view key) const
{
    const std::string key_string{ key };
    const context* current{ this };

    while (current != nullptr)
    {
        const context::variable_map_type::const_iterator map_iterator{ current->variable_map.find(key_string) };
        if (map_iterator != current->variable_map.end())
        {
            return map_iterator->second;
        }
        current = current->base;
    }
    return std::nullopt;
}

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
    const picojson::value& element{ array[index] };
    if (!element.is<Value>())
    {
        throw json_parse_exception{};
    }
    return element.get<Value>();
}

template<typename Value>
const Value& throwable_find(const picojson::object& object, std::string_view key)
{
    picojson::object::const_iterator iter{ object.find(std::string{ key }) };
    if (iter == object.end() || !iter->second.is<Value>())
    {
        throw json_parse_exception{};
    }
    return iter->second.get<Value>();
}

std::string base64_decode(std::string_view encoded_string)
{
    using iterator = boost::archive::iterators::transform_width<boost::archive::iterators::binary_from_base64<std::string_view::const_iterator>, 8, 6>;
    return std::string{ iterator{ encoded_string.begin() }, iterator{ encoded_string.end() } };
}

std::string trim(std::string_view str)
{
    const std::regex leading_spaces{ R"(^\s+)", std::regex_constants::ECMAScript };
    const std::regex trailing_spaces{ R"(\s+$)", std::regex_constants::ECMAScript };
    std::string trimmed_string{ str };
    trimmed_string = std::regex_replace(trimmed_string, leading_spaces, {});
    trimmed_string = std::regex_replace(trimmed_string, trailing_spaces, {});
    return trimmed_string;
}

template<typename Integer>
Integer random(Integer min, Integer max)
{
    static std::random_device seed_gen;
    static std::default_random_engine random_engine{ seed_gen() };
    static std::uniform_int_distribution<Integer> distribution{ min, max };
    return distribution(random_engine);
}

namespace parser
{
    struct macro_call;

    struct variable
    {
        std::string name;
    };

    using argument = std::variant<int, std::string, variable, boost::recursive_wrapper<macro_call>>;

    struct macro_call
    {
        std::string name;
        std::vector<argument> arguments;
    };

    using expression = std::variant<variable, macro_call>;
    using node = std::variant<std::string, expression>;

    struct escaped_symbols
        : boost::spirit::qi::symbols<char, char>
    {
        escaped_symbols()
        {
            add
            ("\"", '\"')
                ("\'", '\'')
                ("\\", '\\')
                ("a", '\a')
                ("b", '\b')
                ("f", '\f')
                ("n", '\n')
                ("r", '\r')
                ("t", '\t')
                ;
        }
    } escaped_char;

    template<typename Iterator>
    struct document_grammar
        : boost::spirit::qi::grammar<Iterator, std::vector<node>()>
    {
        document_grammar()
            : document_grammar::base_type(document)
        {
            namespace qi = boost::spirit::qi;

            using qi::int_;
            using qi::lexeme;
            using qi::char_;

            document = *node;
            node = placeholder | plain_text;
            plain_text = lexeme[+(char_ - "{{")];
            placeholder = "{{" >> qi::skip(qi::space)[expr] >> "}}";

            expr = macro | variable;
            name = lexeme[+(char_("a-zA-Z0-9_"))];
            string_literal = lexeme['"' >> *(("\\" >> escaped_char) | (char_ - '"' - '\\')) >> '"'];
            variable = name;
            macro = name >> arg_list;
            arg_list = '(' >> -(arg % ',') >> ')';
            arg = macro | variable | int_ | string_literal;
        }

        boost::spirit::qi::rule<Iterator, std::vector<node>()> document;
        boost::spirit::qi::rule<Iterator, node()> node;
        boost::spirit::qi::rule<Iterator, std::string()> plain_text;
        boost::spirit::qi::rule<Iterator, expression()> placeholder;

        boost::spirit::qi::rule<Iterator, expression(), boost::spirit::qi::space_type> expr;
        boost::spirit::qi::rule<Iterator, std::string(), boost::spirit::qi::space_type> name;
        boost::spirit::qi::rule<Iterator, std::string(), boost::spirit::qi::space_type> string_literal;
        boost::spirit::qi::rule<Iterator, variable(), boost::spirit::qi::space_type> variable;
        boost::spirit::qi::rule<Iterator, macro_call(), boost::spirit::qi::space_type> macro;
        boost::spirit::qi::rule<Iterator, std::vector<argument>(), boost::spirit::qi::space_type> arg_list;
        boost::spirit::qi::rule<Iterator, argument(), boost::spirit::qi::space_type> arg;
    };

    using grammar = document_grammar<std::string_view::const_iterator>;

    std::string evaluate_expression(const expression& expr, const config& config, context& ctx);
    std::string evaluate_argument(const argument& arg, const config& config, context& ctx);
    std::string evaluate_expression(const expression& expr, const config& config, context& ctx);
    std::string evaluate_node(const std::vector<node>& ast, const config& config, const grammar& grammar, context& ctx);
    std::string evaluate_document(std::string_view document, const config& config, const grammar& grammar, context& ctx);
    std::string evaluate_document_recursive(std::string input, const config& config, unsigned int max_depth, context& ctx);
}

BOOST_FUSION_ADAPT_STRUCT(
    parser::variable,
    (std::string, name)
);

BOOST_FUSION_ADAPT_STRUCT(
    parser::macro_call,
    (std::string, name)
    (std::vector<parser::argument>, arguments)
);

namespace builtin
{
    std::string include(const std::vector<std::string>& arguments, const config& config, context& ctx);
    std::string head(const std::vector<std::string>& arguments, const config& config, context& ctx);
    std::string tail(const std::vector<std::string>& arguments, const config& config, context& ctx);
    std::string head_tail(const std::vector<std::string>& arguments, const config& config, context& ctx);
    std::string json_literal(const std::vector<std::string>& arguments, const config& config, context& ctx);
    std::string env(const std::vector<std::string>& arguments, const config& config, context& ctx);
    std::string generated(const std::vector<std::string>& arguments, const config& config, context& ctx);
    std::string let(const std::vector<std::string>& arguments, const config& config, context& ctx);
    std::string random(const std::vector<std::string>& arguments, const config& config, context& ctx);
    std::string choice(const std::vector<std::string>& arguments, const config& config, context& ctx);

    static const std::unordered_map<std::string, std::function<std::string(const std::vector<std::string>&, const config&, context&)>> macros
    {
        {"include", include},
        {"head", head},
        {"tail", tail},
        {"head_tail", head_tail},
        {"json_literal", json_literal},
        {"env", env},
        {"generated", generated},
        {"let", let},
        {"random", random},
        {"choice", choice}
    };

    std::string date();
    std::string time();
    std::string datetime();
    std::string stdin_(const config& config);
}

std::string expand_macro(std::string_view input, const config& config, const context& ctx);

std::string parser::evaluate_argument(const argument& arg, const config& config, context& ctx)
{
    auto visitor = [&](auto&& value) -> std::string
        {
            using decayed_type = std::decay_t<decltype(value)>;

            if constexpr (std::is_same_v<decayed_type, int>)
            {
                return std::to_string(value);
            }
            else if constexpr (std::is_same_v<decayed_type, std::string>)
            {
                return value;
            }
            else if constexpr (std::is_same_v<decayed_type, variable>)
            {
                if (const std::optional<std::string> variable_value{ config.context.get(value.name) }; variable_value)
                {
                    return *variable_value;
                }
                return "{{" + value.name + "}}";
            }
            else if constexpr (std::is_same_v<decayed_type, boost::recursive_wrapper<macro_call>>)
            {
                return evaluate_expression(value.get(), config, ctx);
            }
        };
    return std::visit(visitor, arg);
}

std::string parser::evaluate_expression(const expression& expr, const config& config, context& ctx)
{
    auto visitor = [&](auto&& value) -> std::string
        {
            using decayed_type = std::decay_t<decltype(value)>;

            if constexpr (std::is_same_v<decayed_type, variable>)
            {
                if (const std::optional<std::string> variable_value{ ctx.get(value.name) }; variable_value)
                {
                    BOOST_LOG_TRIVIAL(trace) << "Variable found (" << value.name << "=" << *variable_value << ")";
                    return *variable_value;
                }

                BOOST_LOG_TRIVIAL(warning) << "Variable not found (" << value.name << ")";

                return std::string{};
            }
            else if constexpr (std::is_same_v<decayed_type, macro_call>)
            {
                std::vector<std::string> evaluated_args;
                for (const argument& arg : value.arguments)
                {
                    evaluated_args.push_back(evaluate_argument(arg, config, ctx));
                }

                if (auto iter{ builtin::macros.find(value.name) }; iter != builtin::macros.end())
                {
                    try
                    {
                        const std::string evaluated{ iter->second(evaluated_args, config, ctx) };
                        BOOST_LOG_TRIVIAL(trace) << "Macro evaluated (" << value.name << " => " << evaluated << ")";
                        return evaluated;
                    }
                    catch (const runtime_exception&)
                    {
                        BOOST_LOG_TRIVIAL(warning) << "Evaluation failed (" << value.name << ")";
                    }

                    return std::string{};
                }

                BOOST_LOG_TRIVIAL(warning) << "Macro not found (" << value.name << ")";
                return std::string{};
            }
        };
    return std::visit(visitor, expr);
}

std::string parser::evaluate_node(const std::vector<node>& ast, const config& config, const grammar& grammar, context& ctx)
{
    std::string result;

    auto visitor = [&](auto&& value)
        {
            using decayed_type = std::decay_t<decltype(value)>;

            if constexpr (std::is_same_v<decayed_type, std::string>)
            {
                result += value;
            }
            else if constexpr (std::is_same_v<decayed_type, expression>)
            {
                result += evaluate_expression(value, config, ctx);
            }
        };

    for (const node& node : ast)
    {
        std::visit(visitor, node);
    }

    return result;
}

std::string parser::evaluate_document(std::string_view document, const config& config, const grammar& grammar, context& ctx)
{
    namespace qi = boost::spirit::qi;

    std::vector<node> ast;

    grammar::iterator_type iter{ document.begin() };
    grammar::iterator_type end{ document.end() };

    if (qi::parse(iter, end, grammar, ast) && iter == end)
    {
        return evaluate_node(ast, config, grammar, ctx);
    }
    else
    {
        std::ostringstream description;
        description << "Parse failed at: " << std::string{ iter, end };
        throw macro_exception{} << error_info::description{ description.str() };
    }
}

std::string parser::evaluate_document_recursive(std::string input, const config& config, unsigned int max_depth, context& ctx)
{
    grammar grammar;

    unsigned int depth{};

    while (depth < max_depth)
    {
        if (input.find("{{") == std::string_view::npos)
        {
            return input;
        }

        std::string evaluated{ evaluate_document(input, config, grammar, ctx) };

        if (evaluated == input)
        {
            break;
        }

        input = std::move(evaluated);
        depth += 1;
    }

    if (depth >= max_depth)
    {
        throw macro_exception{} << error_info::description{ "Maximum recursion depth reached." };
    }

    return input;
}

std::string builtin::include(const std::vector<std::string>& arguments, const config& config, context& ctx)
{
    if (arguments.size() < 1)
    {
        throw macro_exception{};
    }

    const std::filesystem::path file_path{ string_to_path_by_config(complement_extension(arguments[0], ".txt"), config) };
    return read_file_to_string(file_path);
}

std::string head_tail_impl(const std::vector<std::string>& arguments, const config& config, context& ctx, bool reverse)
{
    if (arguments.size() < 2)
    {
        throw macro_exception{};
    }

    const std::string_view filename{ arguments[0] };
    int max_tokens{};
    try
    {
        max_tokens = boost::lexical_cast<unsigned int>(arguments[1]);

    }
    catch (const boost::bad_lexical_cast&)
    {
        throw macro_exception{};
    }

    const std::filesystem::path file_path{ string_to_path_by_config(complement_extension(filename, ".txt"), config) };
    const std::string file_content{ read_file_to_string(file_path) };
    const std::string expaned_file_content{ expand_macro(file_content, config, ctx) };

    std::string result;
    int tokens{};
    truncate_by_tokens(expaned_file_content, max_tokens, config, reverse, result, tokens);

    return result;
}

std::string builtin::head(const std::vector<std::string>& arguments, const config& config, context& ctx)
{
    return head_tail_impl(arguments, config, ctx, false);
}

std::string builtin::tail(const std::vector<std::string>& arguments, const config& config, context& ctx)
{
    return head_tail_impl(arguments, config, ctx, true);
}

std::string builtin::head_tail(const std::vector<std::string>& arguments, const config& config, context& ctx)
{
    if (arguments.size() < 3)
    {
        throw macro_exception{};
    }

    const std::string_view filename{ arguments[0] };
    int head_max_tokens{};
    int tail_max_tokens{};
    try
    {
        head_max_tokens = boost::lexical_cast<unsigned int>(arguments[1]);
        tail_max_tokens = boost::lexical_cast<unsigned int>(arguments[2]);
    }
    catch (const boost::bad_lexical_cast&)
    {
        throw macro_exception{};
    }

    const std::string_view ellipsis{ "..." };
    const int ellipsis_tokens{ get_tokens_from_cache(config, ellipsis) };

    const std::filesystem::path file_path{ string_to_path_by_config(complement_extension(filename, ".txt"), config) };
    const std::string file_content{ read_file_to_string(file_path) };
    const std::string expaned_file_content{ expand_macro(file_content, config, ctx) };
    const int total_tokens{ get_tokens_from_cache(config, expaned_file_content) };

    if (head_max_tokens + ellipsis_tokens + tail_max_tokens >= total_tokens)
    {
        return expaned_file_content;
    }

    std::string result;
    int tokens{};
    truncate_by_tokens(expaned_file_content, head_max_tokens, config, false, result, tokens);
    result.append(ellipsis);
    truncate_by_tokens(expaned_file_content, tail_max_tokens, config, true, result, tokens);

    return result;
}

std::string builtin::json_literal(const std::vector<std::string>& arguments, const config& config, context& ctx)
{
    if (arguments.size() < 1)
    {
        throw macro_exception{};
    }

    return json_escape_string(arguments[0]);
}

std::string builtin::env(const std::vector<std::string>& arguments, const config& config, context& ctx)
{
    if (arguments.size() < 1)
    {
        throw macro_exception{};
    }

    const char* env{ boost::nowide::getenv(arguments[0].c_str()) };

    if (!env)
    {
        return std::string{};
    }

    return std::string{ env };
}

std::string builtin::generated(const std::vector<std::string>& arguments, const config& config, context& ctx)
{
    if (arguments.size() < 1)
    {
        throw macro_exception{};
    }

    const std::filesystem::path file_path{ string_to_path_by_config(complement_extension(arguments[0], ".txt"), config) };
    const std::string prompt{ read_file_to_string(file_path) };
    const std::string prefix{ arguments.size() >= 2 ? arguments[1] : std::string{} };

    std::string result;
    {
        context pushed{ ctx.make_pushed() };
        result = generate_text(config, prompt, prefix, pushed);
    }
    return result;
}

std::string builtin::let(const std::vector<std::string>& arguments, const config& config, context& ctx)
{
    if (arguments.size() < 2)
    {
        throw macro_exception{};
    }

    const std::string_view key{ arguments[0] };
    const std::string_view value{ arguments[1] };

    ctx.set(key, value);
    BOOST_LOG_TRIVIAL(info) << "Variable set " << key << " = " << value;

    return std::string{};
}

std::string builtin::random(const std::vector<std::string>& arguments, const config& config, context& ctx)
{
    const std::int64_t min{ arguments.size() >= 1 ? boost::lexical_cast<std::int64_t>(arguments[0]) : 0 };
    const std::int64_t max{ arguments.size() >= 2 ? boost::lexical_cast<std::int64_t>(arguments[1]) : static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max()) };
    return std::to_string(::random<std::int64_t>(min, max));
}

std::string builtin::choice(const std::vector<std::string>& arguments, const config& config, context& ctx)
{
    if (arguments.empty())
    {
        return std::string{};
    }

    return arguments[::random<std::size_t>(0, arguments.size() - 1)];
}

std::string builtin::date()
{
    const boost::posix_time::ptime local_time{ boost::posix_time::second_clock::local_time() };
    const boost::posix_time::time_facet* facet{ new boost::posix_time::time_facet("%Y%m%d") };
    std::ostringstream oss;
    oss.imbue(std::locale(oss.getloc(), facet));
    oss << local_time;
    return oss.str();
}

std::string builtin::time()
{
    const boost::posix_time::ptime local_time{ boost::posix_time::second_clock::local_time() };
    const boost::posix_time::time_facet* facet{ new boost::posix_time::time_facet("%H%M%S") };
    std::ostringstream oss;
    oss.imbue(std::locale(oss.getloc(), facet));
    oss << local_time;
    return oss.str();
}

std::string builtin::datetime()
{
    const boost::posix_time::ptime local_time{ boost::posix_time::second_clock::local_time() };
    const boost::posix_time::time_facet* facet{ new boost::posix_time::time_facet("%Y%m%d%H%M%S") };
    std::ostringstream oss;
    oss.imbue(std::locale(oss.getloc(), facet));
    oss << local_time;
    return oss.str();
}

std::string builtin::stdin_(const config& config)
{
#if defined(_WIN32) || defined(_WIN64)
    bool is_terminal = (_isatty(0) != 0);
#else
    bool is_terminal = (isatty() != 0);
#endif

    if (is_terminal)
    {
        return std::string{};
    }

    return std::string{ std::istreambuf_iterator<char>{ boost::nowide::cin }, std::istreambuf_iterator<char>{} };
}

std::string expand_macro(std::string_view input, const config& config, const context& ctx)
{
    constexpr unsigned int max_depth{ 32 };
    context pushed{ ctx.make_pushed() };
    return parser::evaluate_document_recursive(std::string{ input }, config, max_depth, pushed);
}

template<typename T>
struct promote_integral_to_double
{
    using type = std::conditional_t<
        std::is_integral_v<T> && !std::is_same_v<std::remove_cv_t<T>, bool>,
        double,
        T
    >;
};

template<typename T>
using promote_integral_to_double_t = typename promote_integral_to_double<T>::type;

namespace detail
{
    template<typename Value>
    struct add_pair_into_json_impl
    {
        void operator ()(picojson::object& object, std::string_view key, const Value& value)
        {
            object.insert(std::pair<std::string, picojson::value>{ key, picojson::value{ static_cast<promote_integral_to_double_t<Value>>(value) } });
        }
    };

    template<>
    struct add_pair_into_json_impl<picojson::value>
    {
        void operator ()(picojson::object& object, std::string_view key, const picojson::value& value)
        {
            object.insert(std::pair<std::string, picojson::value>{ key, value });
        }
    };

    template<>
    struct add_pair_into_json_impl<std::string_view>
    {
        void operator ()(picojson::object& object, std::string_view key, std::string_view value)
        {
            if (!value.empty())
            {
                object.insert(std::pair<std::string, picojson::value>{ key, picojson::value{ std::string{ value } } });
            }
        }
    };

    template<>
    struct add_pair_into_json_impl<char*>
    {
        void operator ()(picojson::object& object, std::string_view key, const char* value)
        {
            add_pair_into_json_impl<std::string_view>{}(object, key, value);
        }
    };

    template<>
    struct add_pair_into_json_impl<std::string>
    {
        void operator ()(picojson::object& object, std::string_view key, const std::string& value)
        {
            add_pair_into_json_impl<std::string_view>{}(object, key, value);
        }
    };
}

template<typename Value>
void add_pair_into_json(picojson::object& object, std::string_view key, const Value& value)
{
    detail::add_pair_into_json_impl<std::decay_t<Value>>{}(object, key, value);
}

template<typename Value>
void add_pair_into_json_from_vector(picojson::object& object, std::string_view key, const std::vector<Value>& value)
{
    if (!value.empty())
    {
        picojson::array json_array;
        for (const auto& element : value)
        {
            json_array.push_back(picojson::value{ static_cast<promote_integral_to_double_t<Value>>(element) });
        }
        object.insert(std::pair<std::string, picojson::value>{ key, json_array });
    }
}

void truncate_by_tokens(std::string_view string, int max_tokens, const config& config, bool reverse, std::string& result, int& tokens)
{
    result = {};
    tokens = {};

    std::vector<std::string> lines;
    boost::split(lines, string, boost::is_any_of("\n"));
    std::vector<std::string> temp;

    auto truncate = [&](auto first, auto last)
        {
            for (; first != last; ++first)
            {
                const int next_tokens{ get_tokens_from_cache(config, *first) };
                if (tokens + next_tokens > max_tokens)
                {
                    break;
                }
                temp.push_back(*first);
                tokens += next_tokens;
            }
        };

    if (reverse)
    {
        truncate(lines.rbegin(), lines.rend());
        std::reverse(temp.begin(), temp.end());
    }
    else
    {
        truncate(lines.begin(), lines.end());
    }

    for (const std::string& line : temp)
    {
        result.append(line);
    }
}

void truncate_prompt(std::string_view string, const config& config, bool reverse, std::string& result, int& remaining_tokens)
{
    std::string truncated;
    int tokens{};
    truncate_by_tokens(string, remaining_tokens, config, reverse, truncated, tokens);
    result += string;
    remaining_tokens -= tokens;
}

void create_parent_directories(const std::filesystem::path& path)
{
    if (path.empty() || !path.has_parent_path())
    {
        return;
    }

    std::filesystem::create_directories(path.parent_path());
}

std::string read_file_to_string(const std::filesystem::path& file, std::ios::openmode openmode)
{
    std::string result;
    if (!std::filesystem::exists(file) || !std::filesystem::is_regular_file(file))
    {
        throw file_open_exception{} << error_info::path{ file };
    }
    boost::nowide::ifstream ifs{ file, openmode };
    if (!ifs.is_open())
    {
        throw file_open_exception{} << error_info::path{ file };
    }
    const std::string file_content{ (std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>() };
    result = file_content;
    return result;
}

std::string complement_extension(std::string_view filepath, std::string_view extension)
{
    std::filesystem::path temp{ filepath };
    if (!temp.has_extension())
    {
        temp.replace_extension(extension);
    }
    return temp.string();
}

std::filesystem::path string_to_path_by_config(std::string_view path, const config& config)
{
    const std::filesystem::path file_path{ expand_macro(path, config, config.context) };
    if (file_path.is_relative())
    {
        const std::filesystem::path base_path{ expand_macro(config.base_path, config, config.context) };
        return base_path / file_path;

    }
    return file_path;
}

// unused
namespace tEXt
{
    using crc_table_type = std::array<uint32_t, 256>;

    constexpr crc_table_type generate_crc_table()
    {
        crc_table_type result{};
        for (std::uint32_t i{}; i <= 0xFF; ++i)
        {
            std::uint32_t value{ i };
            for (std::size_t k{}; k < 8; k++)
            {
                value = (value & 1) ? (0xEDB88320L ^ (value >> 1)) : (value >> 1);
            }
            result[i] = value;
        }
        return result;
    }

    constexpr const crc_table_type crc_table{ generate_crc_table() };

    std::uint32_t calculate_crc32(const std::uint8_t* data, size_t length)
    {
        std::uint32_t c{ 0xFFFFFFFFL };
        for (std::size_t i{}; i < length; ++i)
        {
            c = crc_table[(c ^ data[i]) & 0xFF] ^ (c >> 8);
        }
        return c ^ 0xFFFFFFFFL;
    }

    void append_uint32_be(std::uint32_t val, std::vector<unsigned char>& buffer)
    {
        buffer.push_back((val >> 24) & 0xFF);
        buffer.push_back((val >> 16) & 0xFF);
        buffer.push_back((val >> 8) & 0xFF);
        buffer.push_back(val & 0xFF);
    }

    std::vector<unsigned char> create_tEXt_chunk(std::string_view key, std::string_view text)
    {
        const std::size_t total_size{ 4 + 4 + key.size() + 1 + text.size() + 4 };
        std::vector<unsigned char> chunk;
        chunk.reserve(total_size);

        std::vector<unsigned char> data;
        data.insert(data.end(), key.begin(), key.end());
        data.push_back(0);
        data.insert(data.end(), text.begin(), text.end());

        append_uint32_be(static_cast<std::uint32_t>(data.size()), chunk);

        const std::size_t crc_start{ chunk.size() };
        chunk.push_back('t');
        chunk.push_back('E');
        chunk.push_back('X');
        chunk.push_back('t');
        chunk.insert(chunk.end(), data.begin(), data.end());

        const std::uint32_t crc{ calculate_crc32(&chunk[crc_start], chunk.size() - crc_start) };
        append_uint32_be(crc, chunk);

        return chunk;
    }

    struct png_context
    {
        std::vector<unsigned char> result_bytes;
        std::vector<unsigned char> metadata_chunk;
        bool metadata_inserted{};
    };

    std::string insert_metadata(std::string_view image, std::string_view key, std::string_view metadata)
    {
        constexpr std::size_t ihdr_end_offset{ 8 + 25 };
        if (image.size() < ihdr_end_offset)
        {
            throw png_exception{};
        }

        const std::vector<unsigned char> text_chunk{ create_tEXt_chunk(key, metadata) };

        std::string result;
        result.reserve(image.size() + text_chunk.size());

        result.append(image.substr(0, ihdr_end_offset));
        result.append(reinterpret_cast<const char*>(text_chunk.data()), text_chunk.size());
        result.append(image.substr(ihdr_end_offset));

        return result;
    }
}

template<typename BoostException>
void if_error_throw(const boost::beast::error_code& error_code)
{
    if (error_code)
    {
        throw BoostException{} << error_info::beast::error_code{ error_code };
    }
}

boost::beast::http::response<boost::beast::http::string_body> send_http_get(
    std::string_view host,
    std::string_view port,
    std::string_view target,
    unsigned int expires_after)
{
    namespace beast = boost::beast;
    namespace http = beast::http;
    namespace net = boost::asio;
    using tcp = net::ip::tcp;

    beast::error_code error_code;

    net::io_context ioc;
    tcp::resolver resolver{ ioc };
    beast::tcp_stream tcp_stream{ ioc };

    const tcp::resolver::results_type results{ resolver.resolve(host, port, error_code) };
    if_error_throw<dns_resolve_exception>(error_code);

    tcp_stream.expires_after(std::chrono::seconds{ expires_after });
    tcp_stream.connect(results, error_code);
    if_error_throw<connect_exception>(error_code);

    http::request<http::empty_body> req{ http::verb::get, target, 11 };
    req.set(http::field::host, host);
    req.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);

    http::write(tcp_stream, req, error_code);
    if_error_throw<http_send_exception>(error_code);

    beast::flat_buffer buffer;
    http::response_parser<http::string_body> parser;
    parser.body_limit(boost::none);
    http::read(tcp_stream, buffer, parser, error_code);
    if_error_throw<http_receive_exception>(error_code);

    tcp_stream.socket().shutdown(tcp::socket::shutdown_both);

    return parser.release();
};

// unused
std::string make_automatic1111_png_parameters(const sd_txt2img_parameters& parameters, std::string_view prompt, std::string_view negative_prompt)
{
    std::ostringstream oss;
    oss
        << prompt << std::endl
        << "Negative prompt: " << negative_prompt << std::endl
        << "Steps: " << parameters.steps << ", "
        << "Sampler: " << parameters.sampler_name << ", "
        << "CFG scale: " << parameters.cfg_scale << ", "
        << "Seed: " << parameters.seed << ", "
        << "Size: " << parameters.width << "x" << parameters.height << ", "
        //<< "Model hash: "
        << "Denoising strength: " << parameters.denoising_strength << ", "
        << "Hires upscale: " << parameters.hr_scale << ", "
        << "Hires steps: " << parameters.hr_second_pass_steps << ", "
        << "Hires upscaler: " << parameters.hr_upscaler
        << std::flush;
    return oss.str();
}

std::string send_automatic1111_txt2img_request(
    const config& config,
    std::string_view prompt,
    std::string_view negative_prompt
)
{
    namespace beast = boost::beast;
    namespace http = beast::http;
    namespace net = boost::asio;
    using tcp = net::ip::tcp;

    beast::error_code error_code;

    net::io_context ioc;
    tcp::resolver resolver{ ioc };
    beast::tcp_stream tcp_stream{ ioc };

    const tcp::resolver::results_type results{ resolver.resolve(config.sd.host, config.sd.port) };
    tcp_stream.expires_after(std::chrono::seconds{ config.expires_after });
    tcp_stream.connect(results, error_code);
    if_error_throw<connect_exception>(error_code);

    picojson::object json;

    add_pair_into_json(json, "prompt", prompt);
    add_pair_into_json(json, "negative_prompt", negative_prompt);
    //add_pair_into_json(json, "styles", config.sd_txt2img_params.styles);
    add_pair_into_json(json, "seed", config.sd.seed);
    add_pair_into_json(json, "subseed", config.sd.subseed);
    add_pair_into_json(json, "subseed_strength", config.sd.subseed_strength);
    add_pair_into_json(json, "seed_resize_from_h", config.sd.seed_resize_from_h);
    add_pair_into_json(json, "seed_resize_from_w", config.sd.seed_resize_from_w);
    add_pair_into_json(json, "sampler_name", config.sd.sampler_name);
    add_pair_into_json(json, "scheduler", config.sd.scheduler);
    add_pair_into_json(json, "batch_size", config.sd.batch_size);
    add_pair_into_json(json, "n_iter", config.sd.n_iter);
    add_pair_into_json(json, "steps", config.sd.steps);
    add_pair_into_json(json, "cfg_scale", config.sd.cfg_scale);
    add_pair_into_json(json, "width", config.sd.width);
    add_pair_into_json(json, "height", config.sd.height);
    add_pair_into_json(json, "restore_faces", config.sd.restore_faces);
    add_pair_into_json(json, "tiling", config.sd.tiling);
    add_pair_into_json(json, "do_not_save_samples", config.sd.do_not_save_samples);
    add_pair_into_json(json, "do_not_save_grid", config.sd.do_not_save_grid);
    add_pair_into_json(json, "eta", config.sd.eta);
    add_pair_into_json(json, "denoising_strength", config.sd.denoising_strength);
    add_pair_into_json(json, "s_min_uncond", config.sd.s_min_uncond);
    add_pair_into_json(json, "s_churn", config.sd.s_churn);
    add_pair_into_json(json, "s_tmax", config.sd.s_tmax);
    add_pair_into_json(json, "s_tmin", config.sd.s_tmin);
    add_pair_into_json(json, "s_noise", config.sd.s_noise);
    add_pair_into_json(json, "override_settings", config.sd.override_settings);
    add_pair_into_json(json, "override_settings_restore_afterwards", config.sd.override_settings_restore_afterwards);
    add_pair_into_json(json, "refiner_checkpoint", config.sd.refiner_checkpoint);
    add_pair_into_json(json, "refiner_switch_at", config.sd.refiner_switch_at);
    add_pair_into_json(json, "disable_extra_networks", config.sd.disable_extra_networks);
    add_pair_into_json(json, "firstpass_image", config.sd.firstpass_image);
    add_pair_into_json(json, "comments", config.sd.comments);
    add_pair_into_json(json, "enable_hr", config.sd.enable_hr);
    add_pair_into_json(json, "firstphase_width", config.sd.firstphase_width);
    add_pair_into_json(json, "firstphase_height", config.sd.firstphase_height);
    add_pair_into_json(json, "hr_scale", config.sd.hr_scale);
    add_pair_into_json(json, "hr_upscaler", config.sd.hr_upscaler);
    add_pair_into_json(json, "hr_second_pass_steps", config.sd.hr_second_pass_steps);
    add_pair_into_json(json, "hr_resize_x", config.sd.hr_resize_x);
    add_pair_into_json(json, "hr_resize_y", config.sd.hr_resize_y);
    add_pair_into_json(json, "hr_checkpoint_name", config.sd.hr_checkpoint_name);
    //add_pair_into_json(json, "hr_prompt", prompt);
    //add_pair_into_json(json, "hr_negative_prompt", negative_prompt);
    add_pair_into_json(json, "force_task_id", config.sd.force_task_id);

    if (!config.sd.sampler_index.empty() && config.sd.sampler_name.empty())
    {
        add_pair_into_json(json, "sampler_index", config.sd.sampler_index);
    }

    if (config.sd.abg_remover_enable)
    {
        add_pair_into_json(json, "script_name", "abg remover");
        picojson::array args_array
        {
            picojson::value{ false },
            picojson::value{ false },
            picojson::value{ false },
            picojson::value{ "#000000" },
            picojson::value{ false }
        };
        add_pair_into_json(json, "script_args", args_array);
    }

    add_pair_into_json(json, "send_images", config.sd.send_images);
    add_pair_into_json(json, "save_images", config.sd.save_images);

    picojson::object alwayson_scripts;
    if (config.sd.alwayson_scripts.adetailer_parametesrs.ad_enable)
    {
        picojson::object adetailer;
        picojson::array args_array;
        picojson::object args;
        picojson::object object;
        add_pair_into_json(object, "ad_model", config.sd.alwayson_scripts.adetailer_parametesrs.args1.ad_model);
        if (!config.sd.alwayson_scripts.adetailer_parametesrs.args1.ad_prompt.empty())
        {
            add_pair_into_json(object, "ad_prompt", config.sd.alwayson_scripts.adetailer_parametesrs.args1.ad_prompt);
        }
        if (!config.sd.alwayson_scripts.adetailer_parametesrs.args1.ad_negative_prompt.empty())
        {
            add_pair_into_json(object, "ad_negative_prompt", config.sd.alwayson_scripts.adetailer_parametesrs.args1.ad_negative_prompt);
        }
        args_array.push_back(picojson::value{ true });
        args_array.push_back(picojson::value{ false });
        args_array.push_back(picojson::value{ object });
        add_pair_into_json(adetailer, "args", args_array);
        add_pair_into_json(alwayson_scripts, "ADetailer", adetailer);
    }
    //{
    //    picojson::object sampler;
    //    picojson::array args_array;
    //    args_array.push_back(picojson::value{ static_cast<double>(config.sd_txt2img_params.steps) });
    //    args_array.push_back(picojson::value{ config.sd_txt2img_params.sampler_name });
    //    args_array.push_back(picojson::value{ config.sd_txt2img_params.scheduler });
    //    add_pair_into_json(sampler, "args", args_array);
    //    add_pair_into_json(alwayson_scripts, "Sampler", sampler);
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
    //    add_pair_into_json(seed, "args", args_array);
    //    add_pair_into_json(alwayson_scripts, "Seed", seed);
    //}
    add_pair_into_json(json, "alwayson_scripts", alwayson_scripts);

    if (!config.sd.infotext.empty())
    {
        add_pair_into_json(json, "infotext", config.sd.infotext);
    }

    const std::string request_body{ picojson::value{ json }.serialize() };
    BOOST_LOG_TRIVIAL(info) << "Send JSON\n```\n" << request_body << "\n```";

    http::request<http::string_body> request{ http::verb::post, config.sd.target, 11 };
    request.set(http::field::host, config.sd.host);
    request.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
    request.set(http::field::content_type, "application/json; charset=UTF-8");
    request.body() = request_body;
    request.prepare_payload();

    http::write(tcp_stream, request, error_code);
    if_error_throw<http_send_exception>(error_code);

    beast::flat_buffer buffer;
    http::response<http::string_body> response;
    http::read(tcp_stream, buffer, response, error_code);
    if_error_throw<http_receive_exception>(error_code);

    tcp_stream.socket().shutdown(tcp::socket::shutdown_both);

    picojson::value response_json;
    picojson::parse(response_json, response.body());

    const picojson::object& object{ throwable_get<picojson::object>(response_json) };
    const picojson::array& images{ throwable_find<picojson::array>(object, "images") };
    const std::string base64_image_data{ throwable_at<std::string>(images, 0) };

    if (base64_image_data.empty())
    {
        throw image_generation_exception{} << error_info::description{ "No image data found in the response." };
    }

    const std::string decoded_image{ base64_decode(base64_image_data) };

    return decoded_image;
}

std::string send_style_bert_voice_request(
    const config& config,
    std::string_view text
)
{
    namespace beast = boost::beast;
    namespace http = beast::http;
    namespace net = boost::asio;
    using tcp = net::ip::tcp;

    beast::error_code error_code;

    net::io_context ioc;
    tcp::resolver resolver{ ioc };
    beast::tcp_stream tcp_stream{ ioc };

    const tcp::resolver::results_type results{ resolver.resolve(config.sb.host, config.sb.port) };
    tcp_stream.expires_after(std::chrono::seconds{ config.expires_after });
    tcp_stream.connect(results, error_code);
    if_error_throw<connect_exception>(error_code);

    boost::url target{ config.sb.target };
    target.params().set("text", text);
    //target.params().set("encoding", "utf-8");

    if (!config.sb.model_name.empty())
    {
        target.params().set("model_name", config.sb.model_name);
    }
    else
    {
        target.params().set("model_id", std::to_string(config.sb.model_id));
    }

    if (!config.sb.speaker_name.empty())
    {
        target.params().set("speaker_name", config.sb.speaker_name);
    }
    else
    {
        target.params().set("speaker_id", std::to_string(config.sb.speaker_id));
    }

    target.params().set("sdp_ratio", std::to_string(config.sb.sdp_ratio));
    target.params().set("noise", std::to_string(config.sb.noise));
    target.params().set("noisew", std::to_string(config.sb.noisew));
    target.params().set("length", std::to_string(config.sb.length));
    target.params().set("language", config.sb.language);
    target.params().set("auto_split", config.sb.auto_split ? "true" : "false");
    target.params().set("split_interval", std::to_string(config.sb.split_interval));

    if (!config.sb.assist_text.empty())
    {
        target.params().set("assist_text", config.sb.assist_text);
        target.params().set("assist_text_weight", std::to_string(config.sb.assist_text_weight));
    }

    if (!config.sb.style.empty())
    {
        target.params().set("style", config.sb.style);
        target.params().set("style_weight", std::to_string(config.sb.style_weight));
    }

    if (!config.sb.reference_audio_path.empty())
    {
        target.params().set("reference_audio_path", config.sb.reference_audio_path);
    }

    BOOST_LOG_TRIVIAL(info) << "Send target\n```\n" << target.c_str() << "\n```";

    http::request<http::string_body> request{ http::verb::get, target, 11 };
    request.set(http::field::host, config.sb.host);
    request.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
    request.set(http::field::content_type, "application/json; charset=UTF-8");
    request.prepare_payload();

    http::write(tcp_stream, request, error_code);
    if_error_throw<http_send_exception>(error_code);

    beast::flat_buffer buffer;
    http::response_parser<http::string_body> parser;
    parser.body_limit(boost::none);

    http::read(tcp_stream, buffer, parser, error_code);
    if_error_throw<http_receive_exception>(error_code);

    http::response<http::string_body> response{ parser.release() };

    tcp_stream.socket().shutdown(tcp::socket::shutdown_both);

    if (response.result() != http::status::ok)
    {
        throw http_status_exception{}
            << error_info::http::response::status{ response.result() }
            << error_info::http::response::reason{ std::to_string(response.result_int()) }
        ;
    }

    return response.body();
}

std::string generate_boundary()
{
    std::ostringstream oss;
    oss << "----UniqueBoundary_" << std::hex << std::setfill('0');
    oss << std::setw(sizeof(std::uint64_t) * 2) << random<std::uint64_t>()
        << std::setw(sizeof(std::uint64_t) * 2) << random<std::uint64_t>();
    return oss.str();
}

std::string upload_image_to_comfy_ui(
    const config& config,
    const std::filesystem::path& image_path,
    bool overwrite
)
{
    namespace beast = boost::beast;
    namespace http = beast::http;
    namespace net = boost::asio;
    using tcp = net::ip::tcp;

    beast::error_code error_code;

    const std::string image_data{ read_file_to_string(image_path, std::ios::binary) };
    const std::string boundary{ generate_boundary() };
    const std::string filename{ image_path.filename().string() };

    std::ostringstream body;
    body
        << "--" << boundary << "\r\n"
        << "Content-Disposition: form-data; name=\"image\"; filename=\"" << filename << "\"\r\n"
        << "Content-Type: image/png\r\n\r\n"
        << image_data + "\r\n";

    if (overwrite)
    {
        body
            << "--" << boundary << "\r\n"
            << "Content-Disposition: form-data; name=\"overwrite\"\r\n\r\n"
            << "true\r\n";
    }
    body << "--" << boundary << "--\r\n";

    net::io_context ioc;
    tcp::resolver resolver{ ioc };
    beast::tcp_stream tcp_stream{ ioc };

    const tcp::resolver::results_type results{ resolver.resolve(config.cu.host, config.cu.port) };
    tcp_stream.expires_after(std::chrono::seconds{ config.expires_after });
    tcp_stream.connect(results, error_code);
    if_error_throw<connect_exception>(error_code);

    http::request<http::string_body> request{ http::verb::post, config.cu.upload_image_target, 11 };
    request.set(http::field::host, config.cu.host);
    request.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
    request.set(http::field::content_type, "multipart/form-data; boundary=" + std::string{ boundary });
    request.body() = body.str();
    request.prepare_payload();

    http::write(tcp_stream, request, error_code);
    if_error_throw<http_send_exception>(error_code);

    beast::flat_buffer buffer;
    http::response_parser<http::string_body> parser;
    parser.body_limit(boost::none);

    http::read(tcp_stream, buffer, parser, error_code);
    if_error_throw<http_receive_exception>(error_code);

    http::response<http::string_body> response{ parser.release() };

    tcp_stream.socket().shutdown(tcp::socket::shutdown_both);

    if (response.result() != http::status::ok)
    {
        throw comfy_ui_generation_exception{} << error_info::description{ "Failed to upload image: " + response.body() };
    }

    picojson::value response_json;
    picojson::parse(response_json, response.body());
    const picojson::object& response_object{ response_json.get<picojson::object>() };

    return response_object.at("name").get<std::string>();
}

void upload_images_to_comfy_ui(
    const config& config,
    context& context
)
{
    for (const std::string& key_value_pair : config.cu.upload_images)
    {
        const std::size_t separator_position{ key_value_pair.find('=') };
        if (separator_position != std::string::npos)
        {
            const std::string variable_name{ key_value_pair.substr(0, separator_position) };
            const std::string local_relative_path{ key_value_pair.substr(separator_position + 1) };
            if (!variable_name.empty())
            {
                const std::filesystem::path local_path{ string_to_path_by_config(local_relative_path, config) };
                const std::string server_path{ upload_image_to_comfy_ui(config, local_path) };
                context.set(variable_name, server_path);
                BOOST_LOG_TRIVIAL(info) << "Successfully uploaded. (" << variable_name << "=" << server_path << ")";
            }
        }
        else
        {
            BOOST_LOG_TRIVIAL(warning) << "Invalid upload images format: " << key_value_pair << ". Expected variable_name=local_path.";
        }
    }
}

void send_comfy_ui_prompt(
    const config& config,
    std::string_view prompt
)
{
    namespace beast = boost::beast;
    namespace http = beast::http;
    namespace net = boost::asio;
    using tcp = net::ip::tcp;

    beast::error_code error_code;

    net::io_context ioc;
    tcp::resolver resolver{ ioc };
    beast::tcp_stream tcp_stream{ ioc };

    struct generated_file_info
    {
        std::string filename;
        std::string subfolder;
        std::string type;
    };

    const tcp::resolver::results_type results{ resolver.resolve(config.cu.host, config.cu.port) };
    tcp_stream.expires_after(std::chrono::seconds{ config.expires_after });
    tcp_stream.connect(results, error_code);
    if_error_throw<connect_exception>(error_code);

    picojson::object json;
    picojson::value prompt_json;
    picojson::parse(prompt_json, std::string{ prompt });
    add_pair_into_json(json, "prompt", prompt_json);

    const std::string request_body{ picojson::value{ json }.serialize() };
    BOOST_LOG_TRIVIAL(info) << "Send JSON\n```\n" << request_body << "\n```";

    http::request<http::string_body> request{ http::verb::post, config.cu.prompt_target, 11 };
    request.set(http::field::host, config.cu.host);
    request.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
    request.set(http::field::content_type, "application/json; charset=UTF-8");
    request.body() = request_body;
    request.prepare_payload();

    http::write(tcp_stream, request, error_code);
    if_error_throw<http_send_exception>(error_code);

    beast::flat_buffer buffer;
    http::response<http::string_body> response;
    http::read(tcp_stream, buffer, response, error_code);
    if_error_throw<http_receive_exception>(error_code);

    picojson::value response_json;
    picojson::parse(response_json, response.body());

    BOOST_LOG_TRIVIAL(info) << "Response: " << response.body();

    const picojson::object& response_object{ throwable_get<picojson::object>(response_json) };
    const std::string prompt_id{ throwable_find<std::string>(response_object, "prompt_id") };
    BOOST_LOG_TRIVIAL(info) << "Queued successfully. Prompt ID: " << prompt_id;

    std::vector<generated_file_info> target_files;
    bool is_finished{};

    while (!is_finished)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        http::response<http::string_body> history_response{
            send_http_get(
                config.cu.host,
                config.cu.port,
                "/history/" + prompt_id,
                config.expires_after
            )
        };

        picojson::value history_json;
        picojson::parse(history_json, history_response.body());

        if (!history_json.is<picojson::object>())
        {
            continue;
        }
        const picojson::object& history_object{ throwable_get<picojson::object>(history_json) };

        try
        {
            const picojson::object& prompt_response_obj{ throwable_find<picojson::object>(history_object, prompt_id) };

            try
            {
                const picojson::object& status_object{ throwable_find<picojson::object>(prompt_response_obj, "status") };
                const std::string status_str{ throwable_find<std::string>(status_object, "status_str") };
                if (status_str == "error")
                {
                    throw comfy_ui_generation_exception{} << error_info::description{ "ComfyUI generation failed on server." };
                }
            }
            catch (const json_parse_exception&) {
                ;
            }

            target_files.clear();

            const picojson::object& outputs_object{ throwable_find<picojson::object>(prompt_response_obj, "outputs") };
            for (const std::pair<const std::string, picojson::value>& node_pair : outputs_object)
            {
                if (!node_pair.second.is<picojson::object>())
                {
                    continue;
                }

                const picojson::object& node_object{ throwable_get<picojson::object>(node_pair.second) };

                for (const std::pair<const std::string, picojson::value>& prop_pair : node_object)
                {
                    if (!prop_pair.second.is<picojson::array>())
                    {
                        continue;
                    }

                    const picojson::array& file_list{ throwable_get<picojson::array>(prop_pair.second) };
                    for (std::size_t i{}; i < file_list.size(); ++i)
                    {
                        try
                        {
                            const picojson::object& file_object{ throwable_at<picojson::object>(file_list, i) };

                            target_files.emplace_back(
                                throwable_find<std::string>(file_object, "filename"),
                                throwable_find<std::string>(file_object, "subfolder"),
                                throwable_find<std::string>(file_object, "type")
                            );
                        }
                        catch (const json_parse_exception&)
                        {
                            continue;
                        }
                    }
                }
            }

            if (!target_files.empty())
            {
                is_finished = true;
            }
        }
        catch (const json_parse_exception&)
        {
            continue;
        }
    }

    BOOST_LOG_TRIVIAL(info) << "Generation complete.";

    for (const generated_file_info& file_info : target_files)
    {
        std::filesystem::path relative_file_path{ config.cu.output_directory };
        if (config.cu.preserve_subdirectories)
        {
            relative_file_path /= file_info.subfolder;
        }
        relative_file_path /= file_info.filename;

        const std::string view_target
            = "/view?filename=" + file_info.filename
            + "&subfolder=" + file_info.subfolder
            + "&type=" + file_info.type;

        const http::response<http::string_body> view_response{ send_http_get(
            config.cu.host,
            config.cu.port,
            view_target,
            config.expires_after
        ) };

        write_file(config, view_response.body(), relative_file_path.string(), std::ios_base::binary);
    }

    tcp_stream.socket().shutdown(tcp::socket::shutdown_both);
}

std::vector<item> parse_item_list(std::string_view str)
{
    std::vector<item> result;

    const std::regex item_regex{ R"(^(?:[-*+]|[0-9a-zA-Z]+[.\)]) (.+))", std::regex_constants::ECMAScript };
    const std::regex sub_item_regex{ R"(^(?:[ \t]+)(?:[-*+]|[0-9a-zA-Z]+[.\)]) (.+))", std::regex_constants::ECMAScript };

    std::istringstream iss{ std::string{ str } };
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

void write_item_list(const config& config, std::string_view task)
{
    const std::vector<item> items{ parse_item_list(task) };

    for (const item& item : items)
    {
        std::string descriptions;
        for (const std::string& description : item.descriptions)
        {
            descriptions.append(description);
        }
        write_file(config, descriptions, complement_extension(item.head, ".txt"), std::ios_base::binary);
    }
}

std::string send_completions_request(
    const config& config,
    std::string_view prompt,
    const text_generation_parameters& params,
    int max_tokens
)
{
    namespace beast = boost::beast;
    namespace http = beast::http;
    namespace net = boost::asio;
    using tcp = net::ip::tcp;

    beast::error_code error_code;

    net::io_context ioc;
    tcp::resolver resolver{ ioc };
    beast::tcp_stream tcp_stream{ ioc };

    const tcp::resolver::results_type results{ resolver.resolve(config.llm.host, config.llm.port) };
    tcp_stream.expires_after(std::chrono::seconds{ config.expires_after });
    tcp_stream.connect(results, error_code);
    if_error_throw<connect_exception>(error_code);

    const std::string request_body{ params.get_request_body_for_text_completions(prompt, max_tokens) };
    BOOST_LOG_TRIVIAL(info) << "Send JSON\n```\n" << request_body << "\n```";

    http::request<http::string_body> request{ http::verb::post, config.llm.completions_target, 11 };
    request.set(http::field::host, config.llm.host);
    request.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
    request.set(http::field::content_type, "application/json; charset=UTF-8");
    request.body() = request_body;
    request.prepare_payload();

    if (!config.llm.api_key.empty())
    {
        request.set(http::field::authorization, ("Bearer ") + config.llm.api_key);
    }

    http::write(tcp_stream, request, error_code);
    if_error_throw<http_send_exception>(error_code);

    beast::flat_buffer buffer;
    http::response<http::string_body> response;
    http::read(tcp_stream, buffer, response, error_code);
    if_error_throw<http_receive_exception>(error_code);

    tcp_stream.socket().shutdown(tcp::socket::shutdown_both);

    if (response.result() != http::status::ok)
    {
        throw http_status_exception{}
            << error_info::http::response::status{ response.result() }
            << error_info::http::response::reason{ std::to_string(response.result_int()) }
        ;
    }

    return params.parse_response_for_text_completions(response);
}

std::string tg_completions_parameters::get_request_body_for_text_completions(std::string_view prompt, int max_tokens) const
{
    picojson::object json;
    add_pair_into_json(json, "prompt", prompt);
    add_pair_into_json(json, "model", model);
    add_pair_into_json(json, "best_of", best_of);
    add_pair_into_json(json, "echo", echo);
    add_pair_into_json(json, "frequency_penalty", frequency_penalty);
    //add_pair_into_json(json, "logit_bias", logit_bias);
    add_pair_into_json(json, "logprobs", logprobs);
    add_pair_into_json(json, "max_tokens", max_tokens);
    add_pair_into_json(json, "n", n);
    add_pair_into_json(json, "presence_penalty", presence_penalty);
    add_pair_into_json_from_vector(json, "stop", stop);
    add_pair_into_json(json, "stream", stream);
    add_pair_into_json(json, "suffix", suffix);
    add_pair_into_json(json, "temperature", temperature);
    add_pair_into_json(json, "top_p", top_p);

    if (seed != -1)
    {
        add_pair_into_json(json, "seed", seed);
    }

    add_pair_into_json(json, "user", user);
    add_pair_into_json(json, "preset", preset);
    add_pair_into_json(json, "dynatemp_low", dynatemp_low);
    add_pair_into_json(json, "dynatemp_high", dynatemp_high);
    add_pair_into_json(json, "dynatemp_exponent", dynatemp_exponent);
    add_pair_into_json(json, "smoothing_factor", smoothing_factor);
    add_pair_into_json(json, "smoothing_curve", smoothing_curve);
    add_pair_into_json(json, "min_p", min_p);
    add_pair_into_json(json, "top_k", top_k);
    add_pair_into_json(json, "typical_p", typical_p);
    add_pair_into_json(json, "xtc_threshold", xtc_threshold);
    add_pair_into_json(json, "xtc_probability", xtc_probability);
    add_pair_into_json(json, "epsilon_cutoff", epsilon_cutoff);
    add_pair_into_json(json, "eta_cutoff", eta_cutoff);
    add_pair_into_json(json, "tfs", tfs);
    add_pair_into_json(json, "top_a", top_a);
    add_pair_into_json(json, "top_n_sigma", top_n_sigma);
    add_pair_into_json(json, "dry_multiplier", dry_multiplier);
    add_pair_into_json(json, "dry_allowed_length", dry_allowed_length);
    add_pair_into_json(json, "dry_base", dry_base);
    add_pair_into_json(json, "repetition_penalty", repetition_penalty);
    add_pair_into_json(json, "encoder_repetition_penalty", encoder_repetition_penalty);
    add_pair_into_json(json, "no_repeat_ngram_size", no_repeat_ngram_size);
    add_pair_into_json(json, "repetition_penalty_range", repetition_penalty_range);
    add_pair_into_json(json, "penalty_alpha", penalty_alpha);
    add_pair_into_json(json, "guidance_scale", guidance_scale);
    add_pair_into_json(json, "mirostat_mode", mirostat_mode);
    add_pair_into_json(json, "mirostat_tau", mirostat_tau);
    add_pair_into_json(json, "mirostat_eta", mirostat_eta);
    add_pair_into_json(json, "prompt_lookup_num_tokens", prompt_lookup_num_tokens);
    add_pair_into_json(json, "max_tokens_second", max_tokens_second);
    add_pair_into_json(json, "do_sample", do_sample);
    add_pair_into_json(json, "dynamic_temperature", max_tokens_second);
    add_pair_into_json(json, "temperature_last", temperature_last);
    add_pair_into_json(json, "auto_max_new_tokens", auto_max_new_tokens);
    add_pair_into_json(json, "ban_eos_token", ban_eos_token);
    add_pair_into_json(json, "add_bos_token", add_bos_token);
    add_pair_into_json(json, "skip_special_tokens", skip_special_tokens);
    add_pair_into_json(json, "static_cache", static_cache);
    add_pair_into_json(json, "truncation_length", truncation_length);
    add_pair_into_json_from_vector(json, "sampler_priority", sampler_priority);
    add_pair_into_json(json, "custom_token_bans", custom_token_bans);
    add_pair_into_json(json, "negative_prompt", negative_prompt);
    add_pair_into_json(json, "dry_sequence_breakers", dry_sequence_breakers);
    add_pair_into_json(json, "grammar_string", grammar_string);

    return picojson::value{ json }.serialize();
}

std::string tg_completions_parameters::parse_response_for_text_completions(const boost::beast::http::response<boost::beast::http::string_body>& response) const
{
    picojson::value response_json;
    picojson::parse(response_json, response.body());
    const picojson::object& object{ throwable_get<picojson::object>(response_json) };
    const picojson::array& choices{ throwable_find<picojson::array>(object, "choices") };
    const picojson::object& choice{ throwable_at<picojson::object>(choices, 0) };
    return throwable_find<std::string>(choice, "text");
}

std::string tg_completions_parameters::get_request_body_for_token_count(std::string_view prompt) const
{
    picojson::object json;
    add_pair_into_json(json, "text", prompt);
    return picojson::value{ json }.serialize();
}

int tg_completions_parameters::parse_response_for_token_count(const boost::beast::http::response<boost::beast::http::string_body>& response) const
{
    picojson::value response_json;
    BOOST_LOG_TRIVIAL(trace) << "Recieve JSON\n```\n" << response.body() << "\n```";
    picojson::parse(response_json, response.body());
    const picojson::object& object{ throwable_get<picojson::object>(response_json) };
    return static_cast<int>(throwable_find<double>(object, "length"));
}

std::string kc_generation_parameters::get_request_body_for_text_completions(std::string_view prompt, int max_tokens) const
{
    picojson::object json;

    add_pair_into_json(json, "max_context_length", max_context_length);
    add_pair_into_json(json, "max_length", max_tokens);
    add_pair_into_json(json, "prompt", std::string{ prompt });
    add_pair_into_json(json, "rep_pen", rep_pen);
    add_pair_into_json(json, "rep_pen_range", rep_pen_range);
    add_pair_into_json_from_vector(json, "sampler_order", sampler_order);

    if (sampler_seed != -1)
    {
        add_pair_into_json(json, "sampler_seed", sampler_seed);
    }

    add_pair_into_json_from_vector(json, "stop_sequence", stop_sequence);
    add_pair_into_json(json, "temperature", temperature);
    add_pair_into_json(json, "tfs", tfs);
    add_pair_into_json(json, "top_a", top_a);
    add_pair_into_json(json, "top_k", top_k);
    add_pair_into_json(json, "top_p", top_p);
    add_pair_into_json(json, "min_p", min_p);
    add_pair_into_json(json, "typical", typical);
    add_pair_into_json(json, "use_default_badwordsids", use_default_badwordsids);
    add_pair_into_json(json, "dynatemp_range", dynatemp_range);
    add_pair_into_json(json, "smoothing_factor", smoothing_factor);
    add_pair_into_json(json, "dynatemp_exponent", dynatemp_exponent);
    add_pair_into_json(json, "mirostat", mirostat);
    add_pair_into_json(json, "mirostat_tau", mirostat_tau);
    add_pair_into_json(json, "mirostat_eta", mirostat_eta);
    add_pair_into_json(json, "genkey", genkey);
    add_pair_into_json(json, "grammar", grammar);
    add_pair_into_json(json, "grammar_retain_state", grammar_retain_state);
    add_pair_into_json(json, "memory", memory);
    add_pair_into_json_from_vector(json, "images", images);
    add_pair_into_json(json, "trim_stop", trim_stop);
    add_pair_into_json(json, "render_special", render_special);
    add_pair_into_json(json, "bypass_eos", bypass_eos);
    add_pair_into_json_from_vector(json, "banned_tokens", banned_tokens);
    add_pair_into_json(json, "dry_multiplier", dry_multiplier);
    add_pair_into_json(json, "dry_base", dry_base);
    add_pair_into_json(json, "dry_allowed_length", dry_allowed_length);
    add_pair_into_json(json, "dry_penalty_last_n", dry_penalty_last_n);
    add_pair_into_json_from_vector(json, "dry_sequence_breakers", dry_sequence_breakers);
    add_pair_into_json(json, "xtc_probability", xtc_probability);
    add_pair_into_json(json, "nsigma", nsigma);
    add_pair_into_json(json, "logprobs", logprobs);
    add_pair_into_json(json, "replace_instruct_placeholders", replace_instruct_placeholders);

    return picojson::value{ json }.serialize();
}

std::string kc_generation_parameters::parse_response_for_text_completions(const boost::beast::http::response<boost::beast::http::string_body>& response) const
{
    picojson::value response_json;
    picojson::parse(response_json, response.body());
    const picojson::object& object{ throwable_get<picojson::object>(response_json) };
    const picojson::array& results{ throwable_find<picojson::array>(object, "results") };
    const picojson::object& result{ throwable_at<picojson::object>(results, 0) };
    return throwable_find<std::string>(result, "text");
}

std::string kc_generation_parameters::get_request_body_for_token_count(std::string_view prompt) const
{
    picojson::object json;
    add_pair_into_json(json, "prompt", prompt);
    return picojson::value{ json }.serialize();
}

int kc_generation_parameters::parse_response_for_token_count(const boost::beast::http::response<boost::beast::http::string_body>& response) const
{
    picojson::value response_json;
    BOOST_LOG_TRIVIAL(trace) << "Recieve JSON\n```\n" << response.body() << "\n```";
    picojson::parse(response_json, response.body());
    const picojson::object& object{ throwable_get<picojson::object>(response_json) };
    return static_cast<int>(throwable_find<double>(object, "value"));
}

int send_token_count_request(const config& config, std::string_view prompt)
{
    namespace beast = boost::beast;
    namespace http = beast::http;
    namespace net = boost::asio;
    using tcp = net::ip::tcp;

    beast::error_code error_code;

    net::io_context ioc;
    tcp::resolver resolver{ ioc };
    beast::tcp_stream tcp_stream{ ioc };

    const tcp::resolver::results_type results{ resolver.resolve(config.llm.host, config.llm.port) };
    tcp_stream.expires_after(std::chrono::seconds{ config.expires_after });
    tcp_stream.connect(results, error_code);
    if_error_throw<connect_exception>(error_code);

    const std::string request_body{ config.llm.backend->get_request_body_for_token_count(prompt) };
    BOOST_LOG_TRIVIAL(trace) << "Send JSON\n```\n" << request_body << "\n```";

    http::request<http::string_body> request{ http::verb::post, config.llm.token_count_target, 11 };
    request.set(http::field::host, config.llm.host);
    request.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
    request.set(http::field::content_type, "application/json; charset=UTF-8");
    request.body() = request_body;
    request.prepare_payload();

    http::write(tcp_stream, request, error_code);
    if_error_throw<http_send_exception>(error_code);

    beast::flat_buffer buffer;
    http::response_parser<http::string_body> parser;
    parser.body_limit(boost::none);

    http::read(tcp_stream, buffer, parser, error_code);
    if_error_throw<http_send_exception>(error_code);

    http::response<http::string_body> response{ parser.release() };

    tcp_stream.socket().shutdown(tcp::socket::shutdown_both);

    if (response.result() != http::status::ok)
    {
        throw http_status_exception{}
            << error_info::http::response::status{ response.result() }
            << error_info::http::response::reason{ std::to_string(response.result_int()) }
        ;
    }

    return config.llm.backend->parse_response_for_token_count(response);
}

int get_tokens_from_cache(const config& config, std::string_view str)
{
    constexpr std::size_t capacity{ 1000 };
    int tokens{};

    lru_cache::const_iterator iter{ config.lru_cache.get<key_tag>().find(std::string{ str }) };
    if (iter != config.lru_cache.get<key_tag>().end())
    {
        tokens = iter->tokens;
        config.lru_cache.get<lru_tag>().relocate(
            config.lru_cache.get<lru_tag>().end(),
            config.lru_cache.get<lru_tag>().iterator_to(*iter));
    }
    else
    {
        tokens = send_token_count_request(config, str);
        config.lru_cache.insert({ std::string{ str }, tokens });
    }

    if (config.lru_cache.size() > capacity)
    {
        config.lru_cache.get<lru_tag>().pop_front();
    }

    return tokens;
}

void write_cache(const config& config)
{
    if (config.mode != "tg"/* && config.mode != "kc"*/)
    {
        return;
    }

    picojson::array cache;
    for (const token_count_string& element : config.lru_cache.get<lru_tag>())
    {
        picojson::object node;
        add_pair_into_json(node, "string", element.str);
        add_pair_into_json(node, "tokens", element.tokens);
        cache.push_back(picojson::value{ node });
    }
    picojson::object json;
    add_pair_into_json(json, "cache", cache);
    const std::string serialized{ picojson::value{ json }.serialize() };

    const std::filesystem::path cache_path{ string_to_path_by_config("cache.json", config) };
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
        for (const picojson::value& cache : caches)
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

std::string generate_text(
    const config& config,
    std::string_view prompt,
    std::string_view prefix,
    const context& ctx
)
{
    std::string expanded_prompt{ expand_macro(prompt, config, ctx) };
    const std::string expanded_prefix{ expand_macro(prefix, config, ctx) };
    const std::size_t initial_prompt_size{ expanded_prompt.size() };
    expanded_prompt += expanded_prefix;

    const int initial_tokens{ send_token_count_request(config, expanded_prompt) };

    BOOST_LOG_TRIVIAL(info) << "Prompt created.\n```\n" << expanded_prompt << "\n```";

    std::string current_prompt{ expanded_prompt };
    int current_tokens = initial_tokens;
    for (int completion_iterations{}; completion_iterations < config.llm.max_completion_iterations; ++completion_iterations)
    {
        BOOST_LOG_TRIVIAL(trace) << "completion_iterations: " << completion_iterations;

        if (current_tokens - initial_tokens >= config.llm.min_completion_tokens)
        {
            break;
        }

        const int remaining_context{ config.llm.backend->get_truncation_length() - current_tokens };
        if (remaining_context <= 0)
        {
            BOOST_LOG_TRIVIAL(warning) << "Context window full. Cannot generate more tokens.";
            break;
        }

        int tokens_to_generate = std::min(config.llm.backend->get_max_tokens(), remaining_context);
        if (tokens_to_generate <= 0)
        {
            BOOST_LOG_TRIVIAL(warning) << "No tokens left to generate. Aborting.";
            break;
        }

        const int max_tokens{ tokens_to_generate };
        const std::string response{ send_completions_request(
            config, current_prompt, *config.llm.backend, max_tokens
        ) };

        if (response.empty())
        {
            break;
        }

        current_prompt += response;
        current_tokens = send_token_count_request(config, current_prompt);
    }

    return current_prompt.substr(initial_prompt_size);
}

std::string unescape_string(std::string_view str)
{
    std::string result;
    result.reserve(str.size());

    bool in_escape{};

    for (const char c : str)
    {
        if (in_escape)
        {
            switch (c)
            {
            case '\"': result += '\"'; break;
            case '\'': result += '\''; break;
            case '\\': result += '\\'; break;
            case 'a':  result += '\a'; break;
            case 'b':  result += '\b'; break;
            case 'f':  result += '\f'; break;
            case 'n':  result += '\n'; break;
            case 'r':  result += '\r'; break;
            case 't':  result += '\t'; break;
            default:
                result += '\\';
                result += c;
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
                result += c;
            }
        }
    }

    if (in_escape)
    {
        result += '\\';
    }

    return result;
}

std::string json_escape_string(std::string_view str)
{
    constexpr std::size_t escape_overhead_denominator{ 8 };
    std::string result;
    result.reserve(str.size() + str.size() / escape_overhead_denominator);

    for (const char c : str)
    {
        switch (c)
        {
        case '"':   result.append("\\\""); break;
        case '\\':  result.append("\\\\"); break;
        case '\b':  result.append("\\b"); break;
        case '\f':  result.append("\\f"); break;
        case '\n':  result.append("\\n"); break;
        case '\r':  result.append("\\r"); break;
        case '\t':  result.append("\\t"); break;
        default:
            if (static_cast<unsigned char>(c) < 0x20)
            {
                char buf[7];
                std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                result.append(buf, std::size(buf) - 1);
            }
            else
            {
                result.push_back(c);
            }
            break;
        }
    }

    return result;
}

void unescape_parameters(config& config)
{
    boost::transform(config.user_defined_variables, config.user_defined_variables.begin(), unescape_string);
    boost::transform(config.phases, config.phases.begin(), unescape_string);
    config.llm.prompt = unescape_string(config.llm.prompt);
    config.llm.generation_prefix = unescape_string(config.llm.generation_prefix);
    config.llm.generation_suffix = unescape_string(config.llm.generation_suffix);
    config.llm.reasoning_prefix = unescape_string(config.llm.reasoning_prefix);
    config.llm.reasoning_suffix = unescape_string(config.llm.reasoning_suffix);
    boost::transform(config.tg.stop, config.tg.stop.begin(), unescape_string);
    config.tg.dry_sequence_breakers = unescape_string(config.tg.dry_sequence_breakers);
    boost::transform(config.kc.stop_sequence, config.kc.stop_sequence.begin(), unescape_string);
    boost::transform(config.kc.banned_tokens, config.kc.banned_tokens.begin(), unescape_string);
    boost::transform(config.kc.dry_sequence_breakers, config.kc.dry_sequence_breakers.begin(), unescape_string);
    config.sd.prompt = unescape_string(config.sd.prompt);
    config.sd.negative_prompt = unescape_string(config.sd.negative_prompt);
    config.sb.text = unescape_string(config.sb.text);
    config.cu.prompt = unescape_string(config.cu.prompt);
}

void parse_user_defined_variables(const std::vector<std::string>& user_defined_variables, context& context)
{
    for (const std::string& key_value_pair : user_defined_variables)
    {
        const std::size_t separator_position{ key_value_pair.find('=') };
        if (separator_position != std::string::npos)
        {
            const std::string key{ key_value_pair.substr(0, separator_position) };
            const std::string value{ key_value_pair.substr(separator_position + 1) };
            if (!key.empty())
            {
                context.set(key, value);
                BOOST_LOG_TRIVIAL(info) << "Variable set " << key << " = " << value;
            }
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
    if (config.llm.generation_prefix.empty())
    {
        config.llm.generation_prefix = "\\n{{phase}}: ";
    }
}

void set_phase_variables(
    const std::vector<std::string>& phases,
    std::size_t phase_index,
    context& context
)
{
    if (phase_index >= phases.size())
    {
        throw array_index_out_of_bounds_exception{};
    }

    if (phase_index > 0)
    {
        context.set("prev_phase", phases[phase_index - 1]);
    }

    context.set("phase", phases[phase_index]);

    if (phase_index < phases.size() - 1)
    {
        context.set("next_phase", phases[phase_index + 1]);
    }
}

void set_static_builtin_variables(
    config& config
)
{
    config.context.set("stdin", builtin::stdin_(config));
}

void set_dynamic_builtin_variables(
    config& config
)
{
    config.context.set("date", builtin::date());
    config.context.set("time", builtin::time());
    config.context.set("datetime", builtin::datetime());
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

void init_llm_mode(config& config)
{
    if (!config.llm.paragraphs_file.empty())
    {
        config.phases.clear();
        const std::filesystem::path plot_file_path{ string_to_path_by_config(config.llm.paragraphs_file, config) };
        const std::string content{ read_file_to_string(plot_file_path) };
        std::vector<item> paragraphs{ parse_item_list(content) };
        set_paragraphs_to_phases(paragraphs, config.phases);
    }

    if (config.mode == "tg")
    {
        config.llm.backend = &config.tg;
        if (config.llm.completions_target.empty())
        {
            config.llm.completions_target = "/v1/completions";
        }
        if (config.llm.token_count_target.empty())
        {
            config.llm.token_count_target = "/v1/internal/token-count";
        }
    }
    else if (config.mode == "kc")
    {
        config.llm.backend = &config.kc;
        if (config.llm.completions_target.empty())
        {
            config.llm.completions_target = "/api/v1/generate";
        }
        if (config.llm.token_count_target.empty())
        {
            config.llm.token_count_target = "/api/extra/tokencount";
        }
    }
}

std::string sanitize_as_filename(std::string_view name)
{
    const std::regex illegal_chars(R"([:*?"<>|#])");
    std::string sanitized{ name.begin(), name.end() };
    std::replace(sanitized.begin(), sanitized.end(), ' ', '_');
    sanitized = std::regex_replace(sanitized, illegal_chars, "");
    boost::algorithm::trim(sanitized);
    return sanitized;
}

std::map<std::string, std::string> extract_code_block_from_markdown(std::string_view markdown_content)
{
    std::map<std::string, std::string> result;
    const std::regex code_block_regex{ R"(```(\S+)\s*\n([\s\S]*?)```)" };

    for (std::cregex_iterator iter{ markdown_content.data(), markdown_content.data() + markdown_content.size(), code_block_regex }; iter != std::cregex_iterator{}; ++iter)
    {
        const std::cmatch match{ *iter };
        const std::string name{ sanitize_as_filename(match[1].str()) };
        const std::string code{ match[2].str() };
        result[name] = code;
    }

    return result;
}

bool wait_for_port(const std::string& host, const std::string& port, unsigned int max_retries, unsigned int wait_ms)
{
    boost::system::error_code error_code;

    boost::asio::io_context ctx;
    boost::asio::ip::tcp::resolver resolver{ ctx };
    const boost::asio::ip::tcp::resolver::results_type results{ resolver.resolve(host, port, error_code) };
    if (error_code || results.empty())
    {
        throw dns_resolve_exception{} << error_info::asio::error_code{ error_code };
    }

    boost::asio::ip::tcp::endpoint endpoint{ *results.begin() };
    for (unsigned int retries{}; retries < max_retries; ++retries)
    {
        boost::asio::ip::tcp::socket socket{ ctx };
        socket.connect(endpoint, error_code);
        if (!error_code)
        {
            socket.close();
            return true;
        }

        BOOST_LOG_TRIVIAL(trace)
            << "[Waiting " << (retries + 1) << "/" << max_retries << "] "
            << host << ":" << port << " (" << error_code.message() << ")";

        std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
    }

    return false;
}

void create_process_async(std::string_view excutable_file, const std::vector<std::string>& arguments)
{
    namespace process = boost::process::v2;
    boost::asio::io_context ctx;
    //auto exe = process::environment::find_executable(boost::filesystem::path{ excutable });
    //if (exe.empty())
    //{
    //    BOOST_LOG_TRIVIAL(warning) << "exe not found.";
    //    return;
    //}
    process::process proc{ ctx, excutable_file, arguments, process::windows::create_new_console };
    proc.detach();
}

std::size_t terminate_process_by_path(const std::filesystem::path& executable_file_path)
{
    std::size_t terminated_count{};

#if defined(_WIN32)
    const HANDLE snapshot{ CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) };
    if (snapshot == INVALID_HANDLE_VALUE)
    {
        return 0;
    }

    const DWORD current_pid{ GetCurrentProcessId() };

    PROCESSENTRY32W entry;
    entry.dwSize = sizeof(PROCESSENTRY32W);

    if (Process32FirstW(snapshot, &entry))
    {
        do
        {
            if (entry.th32ProcessID == current_pid)
            {
                continue;
            }

            const HANDLE process{ OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE, FALSE, entry.th32ProcessID) };
            if (process != nullptr)
            {
                wchar_t current_path_buffer[MAX_PATH]{};
                DWORD size{ MAX_PATH };

                if (QueryFullProcessImageNameW(process, 0, current_path_buffer, &size))
                {
                    const std::filesystem::path current_path{ current_path_buffer };
                    const std::wstring current_target{
                        executable_file_path.has_parent_path()
                        ? current_path.wstring()
                        : current_path.filename().wstring()
                    };

                    if (boost::algorithm::iequals(current_target, executable_file_path.wstring()))
                    {
                        if (TerminateProcess(process, 1))
                        {
                            terminated_count += 1;
                        }
                    }
                }

                CloseHandle(process);
            }

        } while (Process32NextW(snapshot, &entry));
    }

    CloseHandle(snapshot);
#endif

    return terminated_count;
}

std::vector<std::string> parse_command_line_args(std::string_view args)
{
    boost::escaped_list_separator<char> separator{ '\0', ' ', '"' };
    boost::tokenizer<
        boost::escaped_list_separator<char>,
        std::string_view::const_iterator,
        std::string
    > tokenizer{ args, separator };

    std::vector<std::string> result;
    for (const std::string& token : tokenizer)
    {
        if (!token.empty())
        {
            result.push_back(token);
        }
    }
    return result;
}

int parse_command_line(
    int argc,
    char** argv,
    config& config
)
{
    namespace po = boost::program_options;

    try
    {
        config.tg.stop = { "\\n\\n", ":", "***" };
        config.tg.sampler_priority =
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
        config.tg.dry_sequence_breakers = "(\"\\n\", \":\", \"\\\"\", \"*\")";

        po::options_description allowed_options("Allowed options");
        allowed_options.add_options()
            ("help,h", "produce help message")
            ("mode", po::value<std::string>(&config.mode)->default_value("tg"), "Specify mode tg | kc | sd | sb")
            ("base-path", po::value<std::string>(&config.base_path)->default_value("."), "base path")
            ("log-level", po::value<std::string>(&config.log_level)->default_value("info"), "log level (trace|debug|info|warning|error|fatal)")
            ("log-file", po::value<std::string>(&config.log_file)->default_value("log.txt"), "log file path")
            ("config-file,c", po::value<std::string>(&config.config_file)->default_value("config.ini"), "config file path")
            ("verbose,v", po::bool_switch(&config.verbose)->default_value(false), "enable verbose output")
            ("expires-after", po::value<unsigned int>(&config.expires_after)->default_value(30), "connection timeout")
            ("number-iterations,N", po::value<int>(&config.number_iterations)->default_value(1), "number of iterations (-1 means infinity)")
            ("define,D", po::value<std::vector<std::string>>(&config.user_defined_variables)->multitoken(), "define variables (key=value)")
            ("phases", po::value<std::vector<std::string>>(&config.phases)->multitoken(), "phases name list")
            ("seed", po::value<int>(&config.seed)->default_value(-1), "seed value")

            ("create-process", po::bool_switch(&config.create_process)->default_value(false), "create process switch")
            ("terminate-process", po::bool_switch(&config.terminate_process)->default_value(false), "terminate process switch")
            ("server-executable-file", po::value<std::string>(&config.server_executable_file)->default_value(""), "server executable file")
            ("server-arguments", po::value<std::string>(&config.server_arguments), "server arguments")
            ("server-host", po::value<std::string>(&config.server_host)->default_value("localhost"), "server ip")
            ("server-port", po::value<std::string>(&config.server_port)->default_value("5000"), "server port")
            ("server-max-retries", po::value<int>(&config.server_max_retries)->default_value(60), "server max retries")
            ("server-wait-ms", po::value<int>(&config.server_wait_ms)->default_value(1000), "server wait ms")

            ("llm-prompt", po::value<std::string>(&config.llm.prompt)->default_value(""), "LLM prompt")
            ("llm-prompt-file", po::value<std::string>(&config.llm.prompt_file)->default_value("prompt.txt"), "LLM prompt file path")
            ("llm-output-file", po::value<std::string>(&config.llm.output_file)->default_value("output.txt"), "LLM output file path")
            ("llm-generation-prefix", po::value<std::string>(&config.llm.generation_prefix)->default_value(""), "LLM generation prefix")
            ("llm-generation-suffix", po::value<std::string>(&config.llm.generation_suffix)->default_value(""), "LLM generation suffix")
            ("llm-paragraphs-file", po::value<std::string>(&config.llm.paragraphs_file)->default_value(""), "LLM paragraphs file")
            ("llm-host", po::value<std::string>(&config.llm.host)->default_value("localhost"), "LLM host")
            ("llm-port", po::value<std::string>(&config.llm.port)->default_value("5000"), "LLM port")
            ("llm-api-key", po::value<std::string>(&config.llm.api_key)->default_value(""), "LLM API key")
            ("llm-completions-target", po::value<std::string>(&config.llm.completions_target)->default_value(""), "LLM completions target")
            ("llm-token-count-target", po::value<std::string>(&config.llm.token_count_target)->default_value(""), "LLM token count target")
            ("llm-min-completion-tokens", po::value<int>(&config.llm.min_completion_tokens)->default_value(256), "LLM min completion tokens")
            ("llm-max-completion-iterations", po::value<int>(&config.llm.max_completion_iterations)->default_value(5), "LLM max completion iterations")
            ("llm-reasoning-prefix", po::value<std::string>(&config.llm.reasoning_prefix)->default_value(""), "LLM reasoning prefix")
            ("llm-reasoning-suffix", po::value<std::string>(&config.llm.reasoning_suffix)->default_value(""), "LLM reasoning suffix")
            ("llm-code-block-extract", po::bool_switch(&config.llm.code_block_extract)->default_value(false), "code block extract switch")

            ("tg-model", po::value<std::string>(&config.tg.model)->default_value("", "TG model"))
            ("tg-num-best-of", po::value<int>(&config.tg.best_of)->default_value(1), "TG best of")
            ("tg-echo", po::bool_switch(&config.tg.echo)->default_value(false), "TG echo")
            ("tg-frequency-penalty", po::value<double>(&config.tg.frequency_penalty)->default_value(0.0), "TG frequency penalty")
            //std::map<int, double> logit_bias;
            ("tg-logprobs", po::value<double>(&config.tg.logprobs)->default_value(0.0), "TG presence penalty")
            ("tg-max-tokens", po::value<int>(&config.tg.max_tokens)->default_value(512), "TG max tokens")
            ("tg-n", po::value<int>(&config.tg.n)->default_value(1), "TG number of responses generated for the same prompt")
            ("tg-presence-penalty", po::value<double>(&config.tg.presence_penalty)->default_value(0.0), "TG presence penalty")
            ("tg-stop", po::value<std::vector<std::string>>(&config.tg.stop)->multitoken(), "TG stop sequences")
            ("tg-stream", po::bool_switch(&config.tg.stream)->default_value(false), "TG stream")
            ("tg-suffix", po::value<std::string>(&config.tg.suffix)->default_value(""), "TG suffix")
            ("tg-temperature", po::value<double>(&config.tg.temperature)->default_value(1.0), "TG temperature")
            ("tg-top-p", po::value<double>(&config.tg.top_p)->default_value(1.0), "TG top p")
            ("tg-dynatemp-low", po::value<double>(&config.tg.dynatemp_low)->default_value(0.75, "0.75"), "TG dynatemp low")
            ("tg-dynatemp-high", po::value<double>(&config.tg.dynatemp_high)->default_value(1.25, "1.25"), "TG dynatemp high")
            ("tg-dynatemp-exponent", po::value<double>(&config.tg.dynatemp_exponent)->default_value(1.0), "TG dynatemp exponent")
            ("tg-smoothing-factor", po::value<double>(&config.tg.smoothing_factor)->default_value(0.0), "TG smoothing factor")
            ("tg-smoothing-curve", po::value<double>(&config.tg.smoothing_curve)->default_value(1.0), "TG smoothing curve")
            ("tg-min-p", po::value<double>(&config.tg.min_p)->default_value(0.1, "0.1"), "TG min p")
            ("tg-top-k", po::value<int>(&config.tg.top_k)->default_value(0), "TG top k")
            ("tg-typical-p", po::value<double>(&config.tg.typical_p)->default_value(1.0), "TG typical p")
            ("tg-xtc-threshold", po::value<double>(&config.tg.xtc_threshold)->default_value(0.1, "0.1"), "TG Exclude Top Choices (XTC) threshold")
            ("tg-xtc-probability", po::value<double>(&config.tg.xtc_probability)->default_value(0.0), "TG Exclude Top Choices (XTC) probability")
            ("tg-epsilon-cutoff", po::value<double>(&config.tg.epsilon_cutoff)->default_value(0), "TG epsilon cutoff")
            ("tg-eta-cutoff", po::value<double>(&config.tg.eta_cutoff)->default_value(0), "TG eta cutoff")
            ("tg-tfs", po::value<double>(&config.tg.tfs)->default_value(1.0), "TG tfs")
            ("tg-top-a", po::value<double>(&config.tg.top_a)->default_value(0.0), "TG top a")
            ("tg-top-n-sigma", po::value<double>(&config.tg.top_n_sigma)->default_value(1.0), "TG top n sigma")
            ("tg-dry-multiplier", po::value<double>(&config.tg.dry_multiplier)->default_value(0.0), "TG DRY multiplier")
            ("tg-dry-allowed-length", po::value<int>(&config.tg.dry_allowed_length)->default_value(2), "TG DRY allowed length")
            ("tg-dry-base", po::value<double>(&config.tg.dry_base)->default_value(1.75), "TG DRY base")
            ("tg-repetition-penalty", po::value<double>(&config.tg.repetition_penalty)->default_value(1.2), "TG repetition penalty")
            ("tg-encoder-repetition-penalty", po::value<double>(&config.tg.encoder_repetition_penalty)->default_value(1.0), "TG encoder repetition penalty")
            ("tg-no-repeat-ngram-size", po::value<int>(&config.tg.no_repeat_ngram_size)->default_value(0), "TG no repeat ngram size")
            ("tg-repetition-penalty-range", po::value<int>(&config.tg.repetition_penalty_range)->default_value(0), "TG repetition penalty range")
            ("tg-penalty-alpha", po::value<double>(&config.tg.penalty_alpha)->default_value(0.9, "0.9"), "TG penalty alpha")
            ("tg-guidance-scale", po::value<double>(&config.tg.guidance_scale)->default_value(1.0), "TG guidance scale")
            ("tg-mirostat-mode", po::value<int>(&config.tg.mirostat_mode)->default_value(0), "TG mirostat mode")
            ("tg-mirostat-tau", po::value<double>(&config.tg.mirostat_tau)->default_value(5), "TG mirostat tau")
            ("tg-mirostat-eta", po::value<double>(&config.tg.mirostat_eta)->default_value(0.1, "0.1"), "TG mirostat eta")
            ("tg-prompt-lookup-num-tokens", po::value<int>(&config.tg.prompt_lookup_num_tokens)->default_value(0), "TG prompt lookup num tokens")
            ("tg-max-tokens-second", po::value<int>(&config.tg.max_tokens_second)->default_value(0), "TG max tokens second")
            ("tg-do-sample", po::bool_switch(&config.tg.do_sample)->default_value(true), "TG do sample")
            ("tg-dynamic-temperature", po::bool_switch(&config.tg.dynamic_temperature)->default_value(false), "TG dynamic temperature")
            ("tg-temperature-last", po::bool_switch(&config.tg.temperature_last)->default_value(false), "TG temperature last")
            ("tg-auto-max-new-tokens", po::bool_switch(&config.tg.auto_max_new_tokens)->default_value(false), "TG auto max_new tokens")
            ("tg-ban-eos-token", po::bool_switch(&config.tg.ban_eos_token)->default_value(false), "TG ban eos token")
            ("tg-add-bos-token", po::bool_switch(&config.tg.add_bos_token)->default_value(true), "TG add Beginning of Sequence Token (BOS) token")
            ("tg-skip-special-tokens", po::bool_switch(&config.tg.skip_special_tokens)->default_value(true), "TG skip special tokens (bos_token, eos_token, unk_token, pad_token, etc.)")
            ("tg-static-cache", po::bool_switch(&config.tg.static_cache)->default_value(false), "TG static cache")
            ("tg-truncation-length", po::value<int>(&config.tg.truncation_length)->default_value(4096), "TG truncation length")
            ("tg-sampler-priority", po::value<std::vector<std::string>>(&config.tg.sampler_priority)->multitoken(), "TG sampler priority")
            ("tg-custom-token-bans", po::value<std::string>(&config.tg.custom_token_bans)->default_value(""), "TG custom token bans")
            ("tg-negative-prompt", po::value<std::string>(&config.tg.negative_prompt)->default_value(""), "TG negative prompt")
            ("tg-dry-sequence-breakers", po::value<std::string>(&config.tg.dry_sequence_breakers)->default_value(""), "TG dry sequence breakers")
            ("tg-grammar-string", po::value<std::string>(&config.tg.grammar_string)->default_value(""), "TG grammar-string")

            ("kc-max-context-length", po::value<int>(&config.kc.max_context_length)->default_value(4096), "Maximum number of tokens to send to the model. (minimum: 1)")
            ("kc-max-length", po::value<int>(&config.kc.max_length)->default_value(512), "Number of tokens to generate. (minimum: 1)")
            ("kc-rep-pen", po::value<double>(&config.kc.rep_pen)->default_value(1.0), "Base repetition penalty value. (minimum: 1.0)")
            ("kc-rep-pen-range", po::value<int>(&config.kc.rep_pen_range)->default_value(0), "Repetition penalty range. (minimum: 0)")
            ("kc-sampler-order", po::value<std::vector<int>>(&config.kc.sampler_order)->multitoken(), "Sampler order to be used. If N is the length of this array, then N must be greater than or equal to 6 and the array must be a permutation of the first N non-negative integers.")
            ("kc-sampler-seed", po::value<int>(&config.kc.sampler_seed)->default_value(1), "RNG seed to use for sampling. If not specified, the global RNG will be used. (minimum: 1, maximum: 999999)")
            ("kc-stop-sequence", po::value<std::vector<std::string>>(&config.kc.stop_sequence)->multitoken(), "An array of string sequences where the API will stop generating further tokens. The returned text WILL contain the stop sequence if trim_stop is false.")
            ("kc-temperature", po::value<double>(&config.kc.temperature)->default_value(1.0), "Temperature value.")
            ("kc-tfs", po::value<double>(&config.kc.tfs)->default_value(1.0), "Tail free sampling value. (minimum: 0.0, maximum: 1.0)")
            ("kc-top-a", po::value<double>(&config.kc.top_a)->default_value(1.0), "Top-a sampling value. (minimum: 0.0)")
            ("kc-top-k", po::value<double>(&config.kc.top_k)->default_value(0.0), "Top-k sampling value. (minimum: 0.0)")
            ("kc-top-p", po::value<double>(&config.kc.top_p)->default_value(1.0), "Top-p sampling value. (minimum: 0.0, maximum: 1.0)")
            ("kc-min-p", po::value<double>(&config.kc.min_p)->default_value(0.1), "Min-p sampling value. (minimum: 0.0, maximum: 1.0)")
            ("kc-typical", po::value<double>(&config.kc.typical)->default_value(1.0), "Typical sampling value. (minimum: 0.0, maximum: 1.0)")
            ("kc-use-default-badwordsids", po::bool_switch(&config.kc.use_default_badwordsids)->default_value(false), "If true, prevents the EOS token from being generated (Ban EOS).")
            ("kc-dynatemp_range", po::value<double>(&config.kc.dynatemp_range)->default_value(0.0), "If not equal to 0, uses dynamic temperature. Dynamic temperature range will be between Temp+Range and Temp-Range. If equal to 0 , uses static temperature. (default: 0, minimum: -5.0, maximum: 5.0)")
            ("kc-smoothing-factor", po::value<double>(&config.kc.smoothing_factor)->default_value(0.0), "Modifies temperature behavior. If greater than 0 uses smoothing factor. (default: 0.0, minimum: 0.0)")
            ("kc-dynatemp-exponent", po::value<double>(&config.kc.dynatemp_exponent)->default_value(1.0), "Exponent used in dynatemp. (default: 0.0)")
            ("kc-mirostat", po::value<int>(&config.kc.mirostat)->default_value(0), "KoboldCpp ONLY. Sets the mirostat mode, 0=disabled, 1=mirostat_v1, 2=mirostat_v2. (minimum: 0, maximum: 2)")
            ("kc-mirostat-tau", po::value<double>(&config.kc.mirostat_tau)->default_value(0.0), "KoboldCpp ONLY. Mirostat tau value. (minimum: 0.0)")
            ("kc-mirostat-eta", po::value<double>(&config.kc.mirostat_eta)->default_value(0.0), "KoboldCpp ONLY. Mirostat eta value. (minimum: 0.0)")
            ("kc-genkey", po::value<std::string>(&config.kc.genkey)->default_value(""), "KoboldCpp ONLY. A unique genkey set by the user. When checking a polled-streaming request, use this key to be able to fetch pending text even if multiuser is enabled.")
            ("kc-grammar", po::value<std::string>(&config.kc.grammar)->default_value(""), "KoboldCpp ONLY. A string containing the GBNF grammar to use.")
            ("kc-grammar-retain-state", po::bool_switch(&config.kc.grammar_retain_state)->default_value(false), "KoboldCpp ONLY. If true, retains the previous generation's grammar state, otherwise it is reset on new generation.")
            ("kc-memory", po::value<std::string>(&config.kc.memory)->default_value(""), "KoboldCpp ONLY. If set, forcefully appends this string to the beginning of any submitted prompt text. If resulting context exceeds the limit, forcefully overwrites text from the beginning of the main prompt until it can fit. Useful to guarantee full memory insertion even when you cannot determine exact token count.")
            ("kc-images", po::value<std::vector<std::string>>(&config.kc.images)->multitoken(), "KoboldCpp ONLY. If set, takes an array of base64 encoded strings, each one representing an image to be processed.")
            ("kc-trim-stop", po::bool_switch(&config.kc.trim_stop)->default_value(true), "KoboldCpp ONLY. If true, also removes detected stop_sequences from the output and truncates all text after them. If false, output will also include stop sequence and potentially a few additional characters.")
            ("kc-render-special", po::bool_switch(&config.kc.render_special)->default_value(false), "KoboldCpp ONLY. If true, prints special tokens as text for GGUF models")
            ("kc-bypass-eos", po::bool_switch(&config.kc.trim_stop)->default_value(false), "KoboldCpp ONLY. If true, allows EOS token to be generated, but does not stop generation. Not recommended unless you know what you are doing.")
            ("kc-banned-tokens", po::value<std::vector<std::string>>(&config.kc.banned_tokens)->multitoken(), "An array of string sequences, each entry represents a word or phrase prevented from being generated, either modifying model vocab or by backtracking and regenerating when they appear.")
            ("kc-dry-multiplier", po::value<double>(&config.kc.dry_multiplier)->default_value(0.0), "KoboldCpp ONLY. DRY multiplier value, 0 to disable. (minimum: 0)")
            ("kc-dry-base", po::value<double>(&config.kc.dry_base)->default_value(1.75), "KoboldCpp ONLY. DRY base value. (minimum: 0)")
            ("kc-dry-allowed-length", po::value<int>(&config.kc.dry_allowed_length)->default_value(2), "KoboldCpp ONLY. DRY allowed length value. (minimum: 0)")
            ("kc-dry-penalty-last-n", po::value<int>(&config.kc.dry_penalty_last_n)->default_value(0), "KoboldCpp ONLY. DRY last n tokens penalized value. (minimum: 0)")
            ("kc-dry-sequence-breakers", po::value<std::vector<std::string>>(&config.kc.dry_sequence_breakers)->multitoken(), "An array of string sequence breakers for DRY.")
            ("kc-xtc-threshold", po::value<double>(&config.kc.xtc_threshold)->default_value(0.1), "KoboldCpp ONLY. XTC threshold. (minimum: 0)")
            ("kc-xtc-probability", po::value<double>(&config.kc.xtc_probability)->default_value(0.0), "KoboldCpp ONLY. XTC probability. Set to above 0 to enable XTC. (minimum: 0)")
            ("kc-nsigma", po::value<double>(&config.kc.nsigma)->default_value(0.0), "KoboldCpp ONLY. Top N-Sigma value. Set to above 0 to enable nsigma. (minimum: 0)")
            ("kc-logprobs", po::bool_switch(&config.kc.logprobs)->default_value(false), "If true, return up to 5 top logprobs for generated tokens. Incurs performance overhead.")
            ("kc-replace-instruct-placeholders", po::bool_switch(&config.kc.use_default_badwordsids)->default_value(false), "If true, replaces instruct placeholders {{[INPUT]}} and {{[OUTPUT]}} with backend selected instruct tags.")

            ("sd-host", po::value<std::string>(&config.sd.host)->default_value("localhost"), "SD host")
            ("sd-port", po::value<std::string>(&config.sd.port)->default_value("7860"), "SD port")
            ("sd-target", po::value<std::string>(&config.sd.target)->default_value("/sdapi/v1/txt2img"), "SD txt2img target")
            ("sd-prompt-file", po::value<std::string>(&config.sd.prompt_file)->default_value("prompt.txt"), "SD prompt file")
            ("sd-negative-prompt-file", po::value<std::string>(&config.sd.negative_prompt_file)->default_value("negative_prompt.txt"), "SD negative prompt file")
            ("sd-output-file", po::value<std::string>(&config.sd.output_file)->default_value("{{datetime}}.png"), "SD output PNG file")
            ("sd-prompt", po::value<std::string>(&config.sd.prompt)->default_value(""), "SD prompt")
            ("sd-negative-prompt", po::value<std::string>(&config.sd.negative_prompt)->default_value(""), "SD negative prompt")
            ("sd-styles", po::value<std::vector<std::string>>(&config.sd.styles), "SD styles")
            ("sd-seed", po::value<int>(&config.sd.seed)->default_value(-1), "SD seed")
            ("sd-subseed", po::value<int>(&config.sd.subseed)->default_value(-1), "SD subseed")
            ("sd-subseed-strength", po::value<double>(&config.sd.subseed_strength)->default_value(0), "SD subseed strength")
            ("sd-seed-resize-from-h", po::value<int>(&config.sd.seed_resize_from_h)->default_value(-1), "SD seed resize from height")
            ("sd-seed-resize-from-w", po::value<int>(&config.sd.seed_resize_from_w)->default_value(-1), "SD seed resize from width")
            ("sd-sampler-name", po::value<std::string>(&config.sd.sampler_name)->default_value("Euler a"), "SD sampler name")
            ("sd-scheduler", po::value<std::string>(&config.sd.scheduler)->default_value("Automatic"), "SD scheduler")
            ("sd-batch_size", po::value<int>(&config.sd.batch_size)->default_value(1), "SD batch size")
            ("sd-n-iter", po::value<int>(&config.sd.n_iter)->default_value(1), "SD n iter")
            ("sd-steps", po::value<int>(&config.sd.steps)->default_value(30), "SD steps")
            ("sd-cfg-scale", po::value<double>(&config.sd.cfg_scale)->default_value(7), "SD cfg scale")
            ("sd-width", po::value<int>(&config.sd.width)->default_value(1024), "SD image width")
            ("sd-height", po::value<int>(&config.sd.height)->default_value(1024), "SD image height")
            ("sd-restore-faces", po::bool_switch(&config.sd.restore_faces)->default_value(false), "SD restore faces")
            ("sd-tiling", po::bool_switch(&config.sd.tiling)->default_value(false), "SD tiling")
            ("sd-do-not-save-samples", po::bool_switch(&config.sd.do_not_save_samples)->default_value(false), "SD do not save samples")
            ("sd-do-not-save-grid", po::bool_switch(&config.sd.do_not_save_grid)->default_value(false), "SD do not save grid")
            ("sd-eta", po::value<int>(&config.sd.eta)->default_value(0), "SD eta")
            ("sd-denoising-strength", po::value<double>(&config.sd.denoising_strength)->default_value(0.7, "0.7"), "SD denoising strength")
            ("sd-s-min-uncond", po::value<int>(&config.sd.s_min_uncond)->default_value(0), "SD s min uncond")
            ("sd-s-churn", po::value<int>(&config.sd.s_churn)->default_value(0), "SD s churn")
            ("sd-s-tmax", po::value<int>(&config.sd.s_tmax)->default_value(0), "SD s tmax")
            ("sd-s-tmin", po::value<int>(&config.sd.s_tmin)->default_value(0), "SD s tmin")
            ("sd-s-noise", po::value<int>(&config.sd.s_noise)->default_value(1), "SD s noise")
            ("sd-override-settings", po::value<std::string>(&config.sd.override_settings)->default_value(""), "SD override settings")
            ("sd-override-settings-restore-afterwards", po::bool_switch(&config.sd.override_settings_restore_afterwards)->default_value(true), "SD override settings restore afterwards")
            ("sd-refiner-checkpoint", po::value<std::string>(&config.sd.refiner_checkpoint)->default_value(""), "SD refiner checkpoint")
            ("sd-refiner-switch-at", po::value<double>(&config.sd.refiner_switch_at)->default_value(0.8, "0.8"), "SD refiner switch at")
            ("sd-disable-extra-networks", po::bool_switch(&config.sd.disable_extra_networks)->default_value(false), "SD disable extra networks")
            ("sd-firstpass-image", po::value<std::string>(&config.sd.firstpass_image)->default_value(""), "SD firstpass image")
            ("sd-comments", po::value<std::string>(&config.sd.comments)->default_value(""), "SD comments")
            ("sd-enable-hr", po::bool_switch(&config.sd.enable_hr)->default_value(false), "SD enable hr")
            ("sd-firstphase-width", po::value<int>(&config.sd.firstphase_width)->default_value(0), "SD firstphase width")
            ("sd-firstphase-height", po::value<int>(&config.sd.firstphase_height)->default_value(0), "SD firstphase height")
            ("sd-hr-scale", po::value<double>(&config.sd.hr_scale)->default_value(0), "SD hr scale")
            ("sd-hr-upscaler", po::value<std::string>(&config.sd.hr_upscaler)->default_value("SwinIR_4x"), "SD hr upscaler")
            ("sd-hr-second-pass-steps", po::value<int>(&config.sd.hr_second_pass_steps)->default_value(20), "SD hr second pass steps")
            ("sd-hr-resize-x", po::value<int>(&config.sd.hr_resize_x)->default_value(0), "SD hr resize x")
            ("sd-hr-resize-y", po::value<int>(&config.sd.hr_resize_y)->default_value(0), "SD hr resize y")
            ("sd-hr-checkpoint-name", po::value<std::string>(&config.sd.hr_checkpoint_name)->default_value(""), "SD hr checkpoint name")
            //("sd-hr-prompt", po::value<std::string>(&config.sd_txt2img_params.hr_prompt)->default_value(""), "SD hr prompt")
            //("sd-hr-negative-prompt", po::value<std::string>(&config.sd_txt2img_params.hr_negative_prompt)->default_value(""), "SD hr negative prompt")
            ("sd-force-task-id", po::value<std::string>(&config.sd.force_task_id)->default_value(""), "SD force task id")
            ("sd-sampler-index", po::value<std::string>(&config.sd.sampler_index)->default_value(""), "SD sampler index")
            ("sd-script-name", po::value<std::string>(&config.sd.script_name)->default_value(""), "SD script name")
            ("sd-script-args", po::value<std::vector<std::string>>(&config.sd.script_args), "SD script_args")
            ("sd-send-images", po::bool_switch(&config.sd.send_images)->default_value(true), "SD send images")
            ("sd-save-images", po::bool_switch(&config.sd.save_images)->default_value(false), "SD save images")
            ("sd-ad-enable", po::bool_switch(&config.sd.alwayson_scripts.adetailer_parametesrs.ad_enable)->default_value(false), "SD ADetailer enable")
            ("sd-ad-model", po::value<std::string>(&config.sd.alwayson_scripts.adetailer_parametesrs.args1.ad_model)->default_value("face_yolov8n.pt"), "SD ADetailer model")
            ("sd-ad-prompt", po::value<std::string>(&config.sd.alwayson_scripts.adetailer_parametesrs.args1.ad_prompt)->default_value(""), "SD ADetailer prompt")
            ("sd-ad-negative-prompt", po::value<std::string>(&config.sd.alwayson_scripts.adetailer_parametesrs.args1.ad_negative_prompt)->default_value(""), "SD ADetailer negative prompt")
            ("sd-infotext", po::value<std::string>(&config.sd.infotext)->default_value(""), "SD infotext")
            ("sd-abg-remover-enable", po::bool_switch(&config.sd.abg_remover_enable)->default_value(false), "SD ABG Remover enable")

            ("sb-host", po::value<std::string>(&config.sb.host)->default_value("localhost"), "SB host")
            ("sb-port", po::value<std::string>(&config.sb.port)->default_value("5001"), "SB port")
            ("sb-target", po::value<std::string>(&config.sb.target)->default_value("/voice"), "SB voide target")
            ("sb-text-file", po::value<std::string>(&config.sb.text_file)->default_value("text.txt"), "SB text file")
            ("sb-output-file", po::value<std::string>(&config.sb.output_file)->default_value("{{datetime}}.wav"), "SB output WAV")
            ("sb-text", po::value<std::string>(&config.sb.text)->default_value(""), "SB text")
            ("sb-model-name", po::value<std::string>(&config.sb.model_name)->default_value(""), "SB model name")
            ("sb-model-id", po::value<int>(&config.sb.model_id)->default_value(0), "SB model id")
            ("sb-speaker-name", po::value<std::string>(&config.sb.speaker_name)->default_value(""), "SB speaker name")
            ("sb-speaker-id", po::value<int>(&config.sb.speaker_id)->default_value(0), "SB speaker id")
            ("sb-sdp-ratio", po::value<double>(&config.sb.sdp_ratio)->default_value(0.2, "0.2"), "SB sdp ratio")
            ("sb-noise", po::value<double>(&config.sb.noise)->default_value(0.6, "0.6"), "SB noise")
            ("sb-noisew", po::value<double>(&config.sb.noisew)->default_value(0.8, "0.8"), "SB noisew")
            ("sb-length", po::value<double>(&config.sb.length)->default_value(1), "SB length")
            ("sb-language", po::value<std::string>(&config.sb.language)->default_value(""), "SB language")
            ("sb-auto-split", po::bool_switch(&config.sb.auto_split)->default_value(true), "SB auto split")
            ("sb-split-interval", po::value<double>(&config.sb.split_interval)->default_value(0.5, "0.5"), "SB split interval")
            ("sb-assist-text", po::value<std::string>(&config.sb.assist_text)->default_value(""), "SB assist text")
            ("sb-assist-text-weight", po::value<double>(&config.sb.assist_text_weight)->default_value(1), "SB assist text weight")
            ("sb-style", po::value<std::string>(&config.sb.style)->default_value(""), "SB style")
            ("sb-style-weight", po::value<double>(&config.sb.style_weight)->default_value(1), "SB style weight")
            ("sb-reference-audio-path", po::value<std::string>(&config.sb.reference_audio_path)->default_value(""), "SB reference audio path")

            ("cu-host", po::value<std::string>(&config.cu.host)->default_value("localhost"), "Comfy UI host")
            ("cu-port", po::value<std::string>(&config.cu.port)->default_value("8188"), "Comfy UI port")
            ("cu-prompt-target", po::value<std::string>(&config.cu.prompt_target)->default_value("/prompt"), "Comfy UI prompt target")
            ("cu-upload-image-target", po::value<std::string>(&config.cu.upload_image_target)->default_value("/upload/image"), "Comfy UI upload image target")
            ("cu-prompt", po::value<std::string>(&config.cu.prompt)->default_value(""), "Comfy UI prompt")
            ("cu-prompt-file", po::value<std::string>(&config.cu.prompt_file)->default_value("prompt.json"), "Comfy UI prompt file")
            ("cu-output-directory", po::value<std::string>(&config.cu.output_directory)->default_value("output"), "Comfy UI output directory")
            ("cu-upload-images", po::value<std::vector<std::string>>(&config.cu.upload_images)->multitoken(), "Comfy UI upload images (macro_name=local_path)")
            ("cu-preserve-subdirectories", po::bool_switch(&config.cu.preserve_subdirectories)->default_value(false), "Comfy UI preserve server side sub-directories")
            ;

        po::options_description config_file_options;
        config_file_options.add(allowed_options);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, allowed_options), vm, true);
        po::notify(vm);

        if (vm.count("config-file"))
        {
            const std::filesystem::path config_file_path{ string_to_path_by_config(config.config_file, config) };
            if (std::filesystem::exists(config_file_path) && std::filesystem::is_regular_file(config_file_path))
            {
                boost::nowide::ifstream ifs{ config_file_path };
                if (!ifs.is_open())
                {
                    throw file_open_exception{} << error_info::path{ config_file_path };
                }
                po::store(po::parse_config_file(ifs, allowed_options), vm, true);
                po::notify(vm);
            }
        }

        if (vm.find("help") != vm.end())
        {
            boost::nowide::cout << allowed_options << std::endl;
            return 1;
        }

        init_logging(config);

        if (config.mode == "tg" || config.mode == "kc")
        {
            init_llm_mode(config);
        }
        else if (config.mode == "sd")
        {
            ;
        }
        else if (config.mode == "sb")
        {
            ;
        }
        else if (config.mode == "cu")
        {
            ;
        }
        else
        {
            BOOST_LOG_TRIVIAL(error) << "mode options must be (tg | kc | sd | sb | cu).";
            return 1;
        }

        if (config.phases.empty())
        {
            config.phases = { "" };
        }

        unescape_parameters(config);
        parse_user_defined_variables(config.user_defined_variables, config.context);
    }
    catch (const po::error& e)
    {
        throw command_line_syntax_exception{} << error_info::description{ std::string{ "boost::program_options::error: " } + e.what() };
    }

    return 0;
}

std::string truncate_prompt_by_config(std::string_view prompt, const config& config)
{
    std::string result;
    int remaining_tokens{ config.tg.truncation_length - config.tg.max_tokens };
    truncate_prompt(prompt, config, false, result, remaining_tokens);
    return result;
}

std::string remove_reasoning(std::string_view response, std::string_view prefix, std::string_view suffix)
{
    std::string result{ response };

    if (prefix.empty() || suffix.empty())
    {
        return result;
    }

    std::string::size_type first{};
    while ((first = result.find(prefix, first)) != std::string::npos)
    {
        const std::string::size_type last{ result.find(suffix, first + prefix.length()) };
        if (last != std::string::npos)
        {
            const std::string::size_type remove_length{ (last + suffix.length()) - first };
            BOOST_LOG_TRIVIAL(info) << "Reasoning removed.\n```\n" << result.substr(first, remove_length) << "\n```\n";
            result.erase(first, remove_length);
        }
        else
        {
            break;
        }
    }

    if (result != response)
    {
        BOOST_LOG_TRIVIAL(info) << "Reasoning removed.\n```\n" << result << "\n```\n";
    }

    return result;
}

void write_file(const config& config, std::string_view response, std::string_view filepath, std::ios_base::openmode mode)
{
    const std::filesystem::path file_path{ string_to_path_by_config(filepath, config) };
    create_parent_directories(file_path);
    boost::nowide::ofstream ofs{ file_path, mode };
    if (!ofs.is_open())
    {
        throw file_open_exception{} << error_info::path{ file_path };
    }
    ofs << response;

    const std::string_view file_type{ (mode & std::ios_base::binary) ? "binary" : "text" };
    BOOST_LOG_TRIVIAL(info) << "Write " << file_type << " to " << file_path;
}

void write_code_block(const config& config, std::string_view markdown)
{
    if (config.llm.code_block_extract)
    {
        const std::map<std::string, std::string> blocks{ extract_code_block_from_markdown(markdown) };
        for (const auto& [name, code] : blocks)
        {
            if (name == "stdout")
            {
                boost::nowide::cout << code << std::flush;
            }
            else
            {
                write_file(config, code, complement_extension(name, ".txt"), 0);
            }
        }
    }
}

void generate_text_and_write(const config& config, std::string_view prompt, const context& ctx)
{
    const std::string truncated_prompt{ truncate_prompt_by_config(prompt, config) };

    std::string response{ generate_text(config, truncated_prompt, config.llm.generation_prefix, ctx) };
    response = remove_reasoning(response, config.llm.reasoning_prefix, config.llm.reasoning_suffix);
    response += config.llm.generation_suffix;

    write_file(config, response, config.llm.output_file, std::ios_base::app);

    if (!config.verbose)
    {
        boost::nowide::cout << response << std::flush;
    }

    write_code_block(config, response);
}

std::string prompt_from_string_or_file_path(
    std::string_view string,
    std::string_view file_path,
    const config& config
)
{
    return string.empty() ? read_file_to_string(string_to_path_by_config(file_path, config)) : std::string{ string };
}

void generate_and_output(const config& config)
{
    if (config.mode == "tg" || config.mode == "kc")
    {
        const std::string prompt{ prompt_from_string_or_file_path(config.llm.prompt, config.llm.prompt_file, config) };
        generate_text_and_write(config, prompt, config.context);
    }
    else if (config.mode == "sd")
    {
        const std::string prompt_string{ expand_macro(prompt_from_string_or_file_path(config.sd.prompt, config.sd.prompt_file, config), config, config.context) };
        const std::string negative_prompt_string{ expand_macro(prompt_from_string_or_file_path(config.sd.negative_prompt, config.sd.negative_prompt_file, config), config, config.context) };
        const std::string image{ send_automatic1111_txt2img_request(config, prompt_string, negative_prompt_string) };
        write_file(config, image, config.sd.output_file, std::ios_base::binary);
    }
    else if (config.mode == "sb")
    {
        const std::string text{ expand_macro(prompt_from_string_or_file_path(config.sb.text, config.sb.text_file, config), config, config.context) };
        const std::string voice{ send_style_bert_voice_request(config, text) };
        write_file(config, voice, config.sb.output_file, std::ios_base::binary);
    }
    else if (config.mode == "cu")
    {
        const std::string prompt{ expand_macro(prompt_from_string_or_file_path(config.cu.prompt, config.cu.prompt_file, config), config, config.context) };
        send_comfy_ui_prompt(config, prompt);
    }
}

void set_seed(config& config)
{
    if (config.seed == -1)
    {
        config.tg.seed = random<int>(0);
        config.kc.sampler_seed = random<int>(0, 999999);
        config.sd.seed = random<int>(0);
    }
    else
    {
        config.tg.seed = config.seed;
        config.kc.sampler_seed = config.seed;
        config.sd.seed = config.seed;
    }
}

void process_create_or_terminate(const config& config)
{
    if (config.create_process)
    {
        if (!config.server_executable_file.empty())
        {
            const std::vector<std::string> arguments{ parse_command_line_args(config.server_arguments) };
            create_process_async(config.server_executable_file, arguments);
            if (!wait_for_port(config.server_host, config.server_port, config.server_max_retries, config.server_wait_ms))
            {
                BOOST_LOG_TRIVIAL(warning) << "Connection timed out waiting for server response.";
            }
        }
    }
    else if (config.terminate_process)
    {
        if (!config.server_executable_file.empty())
        {
            if (terminate_process_by_path(config.server_executable_file) == 0)
            {
                BOOST_LOG_TRIVIAL(warning) << "Failed to terminate process by executable file path (" << config.server_executable_file << ").";
            }
        }
    }
}

void iterate(config& config)
{
    read_cache(config);

    int iteration_count{};
    while (config.number_iterations == -1 || iteration_count < config.number_iterations)
    {
        set_seed(config);

        set_dynamic_builtin_variables(config);
        config.context.set("N", std::to_string(iteration_count + 1));

        for (std::size_t phase_index{}; phase_index < config.phases.size(); ++phase_index)
        {
            set_phase_variables(config.phases, phase_index, config.context);
            generate_and_output(config);
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

        if (parse_command_line(argc, argv, config))
        {
            return 0;
        }

        if (config.create_process || config.terminate_process)
        {
            process_create_or_terminate(config);
            return 0;
        }

        set_static_builtin_variables(config);

        if (config.mode == "cu")
        {
            upload_images_to_comfy_ui(config, config.context);
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
    catch (...)
    {
        BOOST_LOG_TRIVIAL(error) << "Unknown exception caught.";
        return -1;
    }

    return 0;
}

int main(int argc, char** argv)
{
    boost::nowide::args a(argc, argv);
    return exception_safe_main(argc, argv);
}