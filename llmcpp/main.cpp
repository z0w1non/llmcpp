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

// is_cin_from_pipe
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

struct llm_prompt_parameters
{
    std::string system_prompts_file;
    std::string output_file;
    std::string generation_prefix;
    std::string generation_suffix;
    bool skip_generation_prefix{};
    std::string retry_generation_prefix;
    std::string paragraphs_file;

    std::string host;
    std::string port;
    std::string api_key;
    std::string completions_target;
    std::string token_count_target;

    std::string reasoning_prefix;
    std::string reasoning_suffix;

    bool code_block_extract{};
};

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
    bool trim_stop;
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
    std::string target;
    std::string prompt_file;
    std::string output_directory;
    std::vector<std::string> upload_images;
    bool preserve_subdirectories{};
};

//using macros = std::map<std::string, std::string>;

struct context
{
    std::unordered_map<std::string, std::string> variables;
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
    int min_completion_tokens{};
    int max_completion_iterations{};

    bool create_process{};
    bool terminate_process{};

    std::string server_executable_file;
    std::string server_arguments;
    std::string server_host;
    std::string server_port;
    int server_max_retries;
    int server_wait_ms;

    llm_prompt_parameters llm_prompt_params;
    tg_completions_parameters tg_completions_params;
    kc_generation_parameters kc_generation_params;
    text_generation_parameters* llm_backend_params{};
    sd_txt2img_parameters sd_txt2img_params;
    sb_generation_parameters sb_generation_params;
    cu_generation_parameters cu_generation_params;
    mutable lru_cache lru_cache;
    context context;
};

struct prompts
{
    std::vector<std::string> system_prompts;

    std::string to_string(const config& config) const;
};

template<typename Value>
const Value& throwable_get(const picojson::value& value);

template<typename Value>
const Value& throwable_at(const picojson::array& array, std::size_t index);

template<typename Value>
const Value& throwable_find(const picojson::object& object, std::string_view key);

std::string base64_decode(std::string_view encoded_string);

std::string trim(std::string_view str);

void truncate_by_tokens(std::string_view string, int max_tokens, const config& config, bool reverse, std::string& result, int& tokens);

void try_append(std::string_view string, const config& config, bool reverse, std::string& result, int& remaining_tokens);

template<typename Container>
std::string concatenate(const Container& strings);

template <typename Container>
void split_string_by_new_line(std::string_view str, Container& container);

void create_parent_directories(const std::filesystem::path& path);

template <typename Container>
void read_file_to_container(const std::filesystem::path& file, Container& container, std::ios::openmode openmode = {});

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

void send_automatic1111_txt2img_request(
    const config& config,
    std::string_view prompt,
    std::string_view negative_prompt,
    const std::filesystem::path& path
);

void send_style_bert_voice_request(
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

std::string generate_and_complete_text(
    const config& config,
    std::string_view prompts,
    std::string_view prefix
);

std::string unescape_string(std::string_view str);

std::string json_escape_string(std::string_view str);

void parse_user_defined_variables(const std::vector<std::string>& predefined_macros, context& context);

void init_logging_with_nowide_cout();
void init_logging_with_nowide_file_log(const std::filesystem::path& log);
void init_logging(const config& config);
void init_chat_mode(config& config);

void set_phase_variables(
    const std::vector<std::string>& phases,
    std::size_t phase_index,
    std::unordered_map<std::string, std::string>& variables
);

void set_builtin_variables(
    config& config
);

void set_builtin_variables_each_iteration(
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

void read_prompts(const config& config, prompts& prompts);

std::string remove_reasoning(std::string_view response, std::string_view prefix, std::string_view suffix);

void write_response(const config& config, std::string_view response, std::string_view filepath, std::ios_base::openmode mode);

void llm_write_code_block(const config& config, std::string_view markdown);

void llm_append_mode(const config& config, prompts& prompts);

std::string prompt_from_string_or_file_path(
    std::string_view string,
    std::string_view file_path,
    const config& config
);

void generate_and_output(const config& config, prompts& prompts);

void set_seed(config& config);

void process_create_or_terminate(const config& config);

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

    std::string evaluate_expression(const expression& expr, const config& config);
    std::string evaluate_argument(const argument& arg, const config& config);
    std::string evaluate_expression(const expression& expr, const config& config);
    std::string evaluate_node(const std::vector<node>& ast, const config& config, const grammar& grammar);
    std::string evaluate_document(std::string_view document, const config& config, grammar& grammar);
    std::string evaluate_document_recursive(std::string input, const config& config, unsigned int max_depth);
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
    std::string include(const std::vector<std::string>& arguments, const config& config);
    std::string tail(const std::vector<std::string>& arguments, const config& config);
    std::string include_json_literal(const std::vector<std::string>& arguments, const config& config);
    std::string tail_json_literal(const std::vector<std::string>& arguments, const config& config);
    std::string env(const std::vector<std::string>& arguments, const config& config);

    const static std::unordered_map<std::string, std::function<std::string(const std::vector<std::string>&, const config&)>> macros
    {
        {"include", include},
        {"tail", tail},
        {"include_json_literal", include_json_literal},
        {"tail_json_literal", tail_json_literal},
        {"env", env},
    };

    std::string date();
    std::string time();
    std::string datetime();
    std::string stdin_(const config& config);
}

std::string expand_macro(std::string_view input, const config& config);

std::string parser::evaluate_argument(const argument& arg, const config& config)
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
                if (auto iter{ config.context.variables.find(value.name) }; iter != config.context.variables.end())
                {
                    return iter->second;
                }
                return "{{" + value.name + "}}";
            }
            else if constexpr (std::is_same_v<decayed_type, boost::recursive_wrapper<macro_call>>)
            {
                return evaluate_expression(value.get(), config);
            }
        };
    return std::visit(visitor, arg);
}

std::string parser::evaluate_expression(const expression& expr, const config& config)
{
    auto visitor = [&](auto&& value) -> std::string
        {
            using decayed_type = std::decay_t<decltype(value)>;

            if constexpr (std::is_same_v<decayed_type, variable>)
            {
                if (auto iter{ config.context.variables.find(value.name) }; iter != config.context.variables.end())
                {
                    BOOST_LOG_TRIVIAL(trace) << "Variable found (" << value.name << "=" << iter->second << ")";
                    return iter->second;
                }

                BOOST_LOG_TRIVIAL(warning) << "Variable not found (" << value.name << ")";

                return std::string{};
            }
            else if constexpr (std::is_same_v<decayed_type, macro_call>)
            {
                std::vector<std::string> evaluated_args;
                for (const argument& arg : value.arguments)
                {
                    evaluated_args.push_back(evaluate_argument(arg, config));
                }

                if (auto iter{ builtin::macros.find(value.name) }; iter != builtin::macros.end())
                {
                    try
                    {
                        const std::string evaluated{ iter->second(evaluated_args, config) };
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

std::string parser::evaluate_node(const std::vector<node>& ast, const config& config, const grammar& grammar)
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
                result += evaluate_expression(value, config);
            }
        };

    for (const node& node : ast)
    {
        std::visit(visitor, node);
    }

    return result;
}

std::string parser::evaluate_document(std::string_view document, const config& config, grammar& grammar)
{
    namespace qi = boost::spirit::qi;

    std::vector<node> ast;

    grammar::iterator_type iter{ document.begin() };
    grammar::iterator_type end{ document.end() };

    if (qi::parse(iter, end, grammar, ast) && iter == end)
    {
        return evaluate_node(ast, config, grammar);
    }
    else
    {
        std::ostringstream description;
        description << "Parse failed at: " << std::string{ iter, end };
        throw macro_exception{} << error_info::description{ description.str() };
    }
}

std::string parser::evaluate_document_recursive(std::string input, const config& config, unsigned int max_depth)
{
    grammar grammar;

    unsigned int depth{};

    while (depth < max_depth)
    {
        if (input.find("{{") == std::string_view::npos)
        {
            return input;
        }

        std::string evaluated{ evaluate_document(input, config, grammar) };

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

std::string builtin::include(const std::vector<std::string>& arguments, const config& config)
{
    if (arguments.size() < 1)
    {
        throw macro_exception{};
    }

    const std::filesystem::path file_path{ string_to_path_by_config(complement_extension(arguments[0], ".txt"), config) };
    return read_file_to_string(file_path);
}

std::string builtin::tail(const std::vector<std::string>& arguments, const config& config)
{
    if (arguments.size() < 2)
    {
        throw macro_exception{};
    }

    std::string result;

    int max_tokens{};
    try
    {
        max_tokens = boost::lexical_cast<unsigned int>(arguments[1]);

    }
    catch (const boost::bad_lexical_cast&)
    {
        throw macro_exception{};
    }

    const std::filesystem::path file_path{ string_to_path_by_config(complement_extension(arguments[0], ".txt"), config) };
    if (!std::filesystem::exists(file_path))
    {
        return result;
    }

    const std::string file_content{ read_file_to_string(file_path) };
    const std::string expaned_file_content{ expand_macro(file_content, config) };

    int tokens{};
    truncate_by_tokens(expaned_file_content, max_tokens, config, true, result, tokens);

    return result;
}

std::string builtin::include_json_literal(const std::vector<std::string>& arguments, const config& config)
{
    return json_escape_string(include(arguments, config));
}

std::string builtin::tail_json_literal(const std::vector<std::string>& arguments, const config& config)
{
    return json_escape_string(tail(arguments, config));
}

std::string builtin::env(const std::vector<std::string>& arguments, const config& config)
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

std::string expand_macro(std::string_view input, const config& config)
{
    constexpr unsigned int max_depth{ 32 };
    return parser::evaluate_document_recursive(std::string{ input }, config, max_depth);
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
        result += line;
    }
}

void try_append(std::string_view string, const config& config, bool reverse, std::string& result, int& remaining_tokens)
{
    std::string truncated;
    int tokens{};
    truncate_by_tokens(string, remaining_tokens, config, reverse, truncated, tokens);
    result += string;
    remaining_tokens -= tokens;
}

template<typename Container>
std::string concatenate(const Container& strings)
{
    std::string result;
    for (const std::string& s : strings)
    {
        result += s;
    }
    return result;
}

template <typename Container>
void split_string_by_new_line(std::string_view str, Container& container)
{
    size_t start_position{};
    size_t end_position{};

    while ((end_position = str.find('\n', start_position)) != std::string::npos)
    {
        container.push_back(typename Container::value_type{ str.substr(start_position, end_position - start_position + 1) });
        start_position = end_position + 1;
    }

    if (start_position < str.length())
    {
        container.push_back(typename Container::value_type{ str.substr(start_position) });
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
void read_file_to_container(const std::filesystem::path& file, Container& container, std::ios::openmode openmode)
{
    container.clear();
    if (std::filesystem::exists(file) && std::filesystem::is_regular_file(file))
    {
        boost::nowide::ifstream ifs{ file, openmode };
        if (!ifs.is_open())
        {
            throw file_open_exception{} << error_info::path{ file };
        }
        const std::string file_content{ (std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>() };
        split_string_by_new_line(file_content, container);
    }
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

template<typename Integer>
Integer random(Integer min, Integer max)
{
    static std::random_device seed_gen;
    static std::default_random_engine random_engine(seed_gen());
    static std::uniform_int_distribution<Integer> distribution(min, max);
    return distribution(random_engine);
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
    const std::filesystem::path file_path{ expand_macro(path, config) };
    if (file_path.is_relative())
    {
        const std::filesystem::path base_path{ expand_macro(config.base_path, config) };
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

    const auto results = resolver.resolve(host, port, error_code);
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

void send_automatic1111_txt2img_request(
    const config& config,
    std::string_view prompt,
    std::string_view negative_prompt,
    const std::filesystem::path& path
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

    const auto results = resolver.resolve(config.sd_txt2img_params.host, config.sd_txt2img_params.port);
    tcp_stream.expires_after(std::chrono::seconds{ config.expires_after });
    tcp_stream.connect(results, error_code);
    if_error_throw<connect_exception>(error_code);

    picojson::object request_body_json;

    add_pair_into_json(request_body_json, "prompt", prompt);
    add_pair_into_json(request_body_json, "negative_prompt", negative_prompt);
    //add_pair_into_json(request_body_json, "styles", config.sd_txt2img_params.styles);
    add_pair_into_json(request_body_json, "seed", config.sd_txt2img_params.seed);
    add_pair_into_json(request_body_json, "subseed", config.sd_txt2img_params.subseed);
    add_pair_into_json(request_body_json, "subseed_strength", config.sd_txt2img_params.subseed_strength);
    add_pair_into_json(request_body_json, "seed_resize_from_h", config.sd_txt2img_params.seed_resize_from_h);
    add_pair_into_json(request_body_json, "seed_resize_from_w", config.sd_txt2img_params.seed_resize_from_w);
    add_pair_into_json(request_body_json, "sampler_name", config.sd_txt2img_params.sampler_name);
    add_pair_into_json(request_body_json, "scheduler", config.sd_txt2img_params.scheduler);
    add_pair_into_json(request_body_json, "batch_size", config.sd_txt2img_params.batch_size);
    add_pair_into_json(request_body_json, "n_iter", config.sd_txt2img_params.n_iter);
    add_pair_into_json(request_body_json, "steps", config.sd_txt2img_params.steps);
    add_pair_into_json(request_body_json, "cfg_scale", config.sd_txt2img_params.cfg_scale);
    add_pair_into_json(request_body_json, "width", config.sd_txt2img_params.width);
    add_pair_into_json(request_body_json, "height", config.sd_txt2img_params.height);
    add_pair_into_json(request_body_json, "restore_faces", config.sd_txt2img_params.restore_faces);
    add_pair_into_json(request_body_json, "tiling", config.sd_txt2img_params.tiling);
    add_pair_into_json(request_body_json, "do_not_save_samples", config.sd_txt2img_params.do_not_save_samples);
    add_pair_into_json(request_body_json, "do_not_save_grid", config.sd_txt2img_params.do_not_save_grid);
    add_pair_into_json(request_body_json, "eta", config.sd_txt2img_params.eta);
    add_pair_into_json(request_body_json, "denoising_strength", config.sd_txt2img_params.denoising_strength);
    add_pair_into_json(request_body_json, "s_min_uncond", config.sd_txt2img_params.s_min_uncond);
    add_pair_into_json(request_body_json, "s_churn", config.sd_txt2img_params.s_churn);
    add_pair_into_json(request_body_json, "s_tmax", config.sd_txt2img_params.s_tmax);
    add_pair_into_json(request_body_json, "s_tmin", config.sd_txt2img_params.s_tmin);
    add_pair_into_json(request_body_json, "s_noise", config.sd_txt2img_params.s_noise);
    add_pair_into_json(request_body_json, "override_settings", config.sd_txt2img_params.override_settings);
    add_pair_into_json(request_body_json, "override_settings_restore_afterwards", config.sd_txt2img_params.override_settings_restore_afterwards);
    add_pair_into_json(request_body_json, "refiner_checkpoint", config.sd_txt2img_params.refiner_checkpoint);
    add_pair_into_json(request_body_json, "refiner_switch_at", config.sd_txt2img_params.refiner_switch_at);
    add_pair_into_json(request_body_json, "disable_extra_networks", config.sd_txt2img_params.disable_extra_networks);
    add_pair_into_json(request_body_json, "firstpass_image", config.sd_txt2img_params.firstpass_image);
    add_pair_into_json(request_body_json, "comments", config.sd_txt2img_params.comments);
    add_pair_into_json(request_body_json, "enable_hr", config.sd_txt2img_params.enable_hr);
    add_pair_into_json(request_body_json, "firstphase_width", config.sd_txt2img_params.firstphase_width);
    add_pair_into_json(request_body_json, "firstphase_height", config.sd_txt2img_params.firstphase_height);
    add_pair_into_json(request_body_json, "hr_scale", config.sd_txt2img_params.hr_scale);
    add_pair_into_json(request_body_json, "hr_upscaler", config.sd_txt2img_params.hr_upscaler);
    add_pair_into_json(request_body_json, "hr_second_pass_steps", config.sd_txt2img_params.hr_second_pass_steps);
    add_pair_into_json(request_body_json, "hr_resize_x", config.sd_txt2img_params.hr_resize_x);
    add_pair_into_json(request_body_json, "hr_resize_y", config.sd_txt2img_params.hr_resize_y);
    add_pair_into_json(request_body_json, "hr_checkpoint_name", config.sd_txt2img_params.hr_checkpoint_name);
    //add_pair_into_json(request_body_json, "hr_prompt", prompt);
    //add_pair_into_json(request_body_json, "hr_negative_prompt", negative_prompt);
    add_pair_into_json(request_body_json, "force_task_id", config.sd_txt2img_params.force_task_id);

    if (!config.sd_txt2img_params.sampler_index.empty() && config.sd_txt2img_params.sampler_name.empty())
    {
        add_pair_into_json(request_body_json, "sampler_index", config.sd_txt2img_params.sampler_index);
    }

    if (config.sd_txt2img_params.abg_remover_enable)
    {
        add_pair_into_json(request_body_json, "script_name", "abg remover");
        picojson::array args_array
        {
            picojson::value{ false },
            picojson::value{ false },
            picojson::value{ false },
            picojson::value{ "#000000" },
            picojson::value{ false }
        };
        add_pair_into_json(request_body_json, "script_args", args_array);
    }

    add_pair_into_json(request_body_json, "send_images", config.sd_txt2img_params.send_images);
    add_pair_into_json(request_body_json, "save_images", config.sd_txt2img_params.save_images);

    picojson::object alwayson_scripts;
    if (config.sd_txt2img_params.alwayson_scripts.adetailer_parametesrs.ad_enable)
    {
        picojson::object adetailer;
        picojson::array args_array;
        picojson::object args;
        picojson::object object;
        add_pair_into_json(object, "ad_model", config.sd_txt2img_params.alwayson_scripts.adetailer_parametesrs.args1.ad_model);
        if (!config.sd_txt2img_params.alwayson_scripts.adetailer_parametesrs.args1.ad_prompt.empty())
        {
            const std::string ad_prompt{ expand_macro(config.sd_txt2img_params.alwayson_scripts.adetailer_parametesrs.args1.ad_prompt, config) };
            add_pair_into_json(object, "ad_prompt", ad_prompt);
        }
        if (!config.sd_txt2img_params.alwayson_scripts.adetailer_parametesrs.args1.ad_negative_prompt.empty())
        {
            const std::string ad_negative_prompt{ expand_macro(config.sd_txt2img_params.alwayson_scripts.adetailer_parametesrs.args1.ad_negative_prompt, config) };
            add_pair_into_json(object, "ad_negative_prompt", ad_negative_prompt);
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
    add_pair_into_json(request_body_json, "alwayson_scripts", alwayson_scripts);

    if (!config.sd_txt2img_params.infotext.empty())
    {
        add_pair_into_json(request_body_json, "infotext", config.sd_txt2img_params.infotext);
    }

    const std::string request_body{ picojson::value{ request_body_json }.serialize() };
    BOOST_LOG_TRIVIAL(info) << "Send JSON\n```\n" << request_body << "\n```";

    http::request<http::string_body> request{ http::verb::post, config.sd_txt2img_params.target, 11 }; // HTTP/1.1
    request.set(http::field::host, config.sd_txt2img_params.host);
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

    boost::nowide::ofstream ofs{ path, std::ios::binary };
    if (!ofs.is_open())
    {
        throw file_open_exception{} << error_info::path{ path };
    }
    ofs.write(decoded_image.data(), decoded_image.size());
    BOOST_LOG_TRIVIAL(info) << "Save image to " << path;
}

void send_style_bert_voice_request(
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

    const auto results = resolver.resolve(config.sb_generation_params.host, config.sb_generation_params.port);
    tcp_stream.expires_after(std::chrono::seconds{ config.expires_after });
    tcp_stream.connect(results, error_code);
    if_error_throw<connect_exception>(error_code);

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

    const std::string macro_expanded_string{ expand_macro(config.sb_generation_params.output_file, config) };
    const std::filesystem::path output_file_path{ string_to_path_by_config(macro_expanded_string, config) };
    boost::nowide::ofstream ofs{ output_file_path, std::ios::binary };
    if (!ofs.is_open())
    {
        throw file_open_exception{} << error_info::path{ output_file_path };
    }
    ofs.write(response.body().data(), response.body().size());
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

    const auto results = resolver.resolve(config.cu_generation_params.host, config.cu_generation_params.port);
    tcp_stream.expires_after(std::chrono::seconds{ config.expires_after });
    tcp_stream.connect(results, error_code);
    if_error_throw<connect_exception>(error_code);

    http::request<http::string_body> request{ http::verb::post, "/upload/image", 11 };
    request.set(http::field::host, config.cu_generation_params.host);
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
    for (const std::string& key_value_pair : config.cu_generation_params.upload_images)
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
                context.variables[variable_name] = server_path;
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

    const auto results = resolver.resolve(config.cu_generation_params.host, config.cu_generation_params.port);
    tcp_stream.expires_after(std::chrono::seconds{ config.expires_after });
    tcp_stream.connect(results, error_code);
    if_error_throw<connect_exception>(error_code);

    picojson::object request_body_json;
    picojson::value prompt_json;
    picojson::parse(prompt_json, std::string{ prompt });
    add_pair_into_json(request_body_json, "prompt", prompt_json);

    const std::string request_body{ picojson::value{ request_body_json }.serialize() };
    BOOST_LOG_TRIVIAL(info) << "Send JSON\n```\n" << request_body << "\n```";

    http::request<http::string_body> request{ http::verb::post, config.cu_generation_params.target, 11 }; // HTTP/1.1
    request.set(http::field::host, config.cu_generation_params.host);
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
                config.cu_generation_params.host,
                config.cu_generation_params.port,
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
        std::filesystem::path relative_file_path{ config.cu_generation_params.output_directory };
        if (config.cu_generation_params.preserve_subdirectories)
        {
            relative_file_path /= file_info.subfolder;
        }
        relative_file_path /= file_info.filename;

        const std::string view_target
            = "/view?filename=" + file_info.filename
            + "&subfolder=" + file_info.subfolder
            + "&type=" + file_info.type;

        const http::response<http::string_body> view_response{ send_http_get(
            config.cu_generation_params.host,
            config.cu_generation_params.port,
            view_target,
            config.expires_after
        ) };

        {
            const std::filesystem::path output_file_path{ string_to_path_by_config(relative_file_path.string(), config) };
            create_parent_directories(output_file_path);
            boost::nowide::ofstream ofs{ output_file_path, std::ios::binary };
            if (!ofs.is_open())
            {
                throw file_open_exception{} << error_info::path{ output_file_path };
            }
            ofs.write(view_response.body().data(), view_response.body().size());
            BOOST_LOG_TRIVIAL(info) << "Saved output to " << output_file_path;
        }
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

    auto const results = resolver.resolve(config.llm_prompt_params.host, config.llm_prompt_params.port);
    tcp_stream.expires_after(std::chrono::seconds{ config.expires_after });
    tcp_stream.connect(results, error_code);
    if_error_throw<connect_exception>(error_code);

    const std::string request_body{ params.get_request_body_for_text_completions(prompt, max_tokens) };
    BOOST_LOG_TRIVIAL(info) << "Send JSON\n```\n" << request_body << "\n```";

    http::request<http::string_body> request{ http::verb::post, config.llm_prompt_params.completions_target, 11 }; // HTTP/1.1
    request.set(http::field::host, config.llm_prompt_params.host);
    request.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
    request.set(http::field::content_type, "application/json; charset=UTF-8");
    request.body() = request_body;
    request.prepare_payload();

    if (!config.llm_prompt_params.api_key.empty())
    {
        request.set(http::field::authorization, ("Bearer ") + config.llm_prompt_params.api_key);
    }

    http::write(tcp_stream, request, error_code);
    if_error_throw<http_send_exception>(error_code);

    beast::flat_buffer buffer;
    http::response<http::string_body> response;
    http::read(tcp_stream, buffer, response, error_code);
    if_error_throw<http_receive_exception>(error_code);

    tcp_stream.socket().shutdown(tcp::socket::shutdown_both);

    if (response.result() == http::status::ok)
    {
        return params.parse_response_for_text_completions(response);
    }
    else
    {
        throw http_status_exception{}
            << error_info::http::response::status{ response.result() }
            << error_info::http::response::reason{ std::to_string(response.result_int()) }
        ;
    }

    return {};
}

std::string tg_completions_parameters::get_request_body_for_text_completions(std::string_view prompt, int max_tokens) const
{
    picojson::object request_body_json;
    add_pair_into_json(request_body_json, "prompt", prompt);
    add_pair_into_json(request_body_json, "model", model);
    add_pair_into_json(request_body_json, "best_of", best_of);
    add_pair_into_json(request_body_json, "echo", echo);
    add_pair_into_json(request_body_json, "frequency_penalty", frequency_penalty);
    //add_pair_into_json(request_body_json, "logit_bias", logit_bias);
    add_pair_into_json(request_body_json, "logprobs", logprobs);
    add_pair_into_json(request_body_json, "max_tokens", max_tokens);
    add_pair_into_json(request_body_json, "n", n);
    add_pair_into_json(request_body_json, "presence_penalty", presence_penalty);
    add_pair_into_json_from_vector(request_body_json, "stop", stop);
    add_pair_into_json(request_body_json, "stream", stream);
    add_pair_into_json(request_body_json, "suffix", suffix);
    add_pair_into_json(request_body_json, "temperature", temperature);
    add_pair_into_json(request_body_json, "top_p", top_p);

    if (seed != -1)
    {
        add_pair_into_json(request_body_json, "seed", seed);
    }

    add_pair_into_json(request_body_json, "user", user);
    add_pair_into_json(request_body_json, "preset", preset);
    add_pair_into_json(request_body_json, "dynatemp_low", dynatemp_low);
    add_pair_into_json(request_body_json, "dynatemp_high", dynatemp_high);
    add_pair_into_json(request_body_json, "dynatemp_exponent", dynatemp_exponent);
    add_pair_into_json(request_body_json, "smoothing_factor", smoothing_factor);
    add_pair_into_json(request_body_json, "smoothing_curve", smoothing_curve);
    add_pair_into_json(request_body_json, "min_p", min_p);
    add_pair_into_json(request_body_json, "top_k", top_k);
    add_pair_into_json(request_body_json, "typical_p", typical_p);
    add_pair_into_json(request_body_json, "xtc_threshold", xtc_threshold);
    add_pair_into_json(request_body_json, "xtc_probability", xtc_probability);
    add_pair_into_json(request_body_json, "epsilon_cutoff", epsilon_cutoff);
    add_pair_into_json(request_body_json, "eta_cutoff", eta_cutoff);
    add_pair_into_json(request_body_json, "tfs", tfs);
    add_pair_into_json(request_body_json, "top_a", top_a);
    add_pair_into_json(request_body_json, "top_n_sigma", top_n_sigma);
    add_pair_into_json(request_body_json, "dry_multiplier", dry_multiplier);
    add_pair_into_json(request_body_json, "dry_allowed_length", dry_allowed_length);
    add_pair_into_json(request_body_json, "dry_base", dry_base);
    add_pair_into_json(request_body_json, "repetition_penalty", repetition_penalty);
    add_pair_into_json(request_body_json, "encoder_repetition_penalty", encoder_repetition_penalty);
    add_pair_into_json(request_body_json, "no_repeat_ngram_size", no_repeat_ngram_size);
    add_pair_into_json(request_body_json, "repetition_penalty_range", repetition_penalty_range);
    add_pair_into_json(request_body_json, "penalty_alpha", penalty_alpha);
    add_pair_into_json(request_body_json, "guidance_scale", guidance_scale);
    add_pair_into_json(request_body_json, "mirostat_mode", mirostat_mode);
    add_pair_into_json(request_body_json, "mirostat_tau", mirostat_tau);
    add_pair_into_json(request_body_json, "mirostat_eta", mirostat_eta);
    add_pair_into_json(request_body_json, "prompt_lookup_num_tokens", prompt_lookup_num_tokens);
    add_pair_into_json(request_body_json, "max_tokens_second", max_tokens_second);
    add_pair_into_json(request_body_json, "do_sample", do_sample);
    add_pair_into_json(request_body_json, "dynamic_temperature", max_tokens_second);
    add_pair_into_json(request_body_json, "temperature_last", temperature_last);
    add_pair_into_json(request_body_json, "auto_max_new_tokens", auto_max_new_tokens);
    add_pair_into_json(request_body_json, "ban_eos_token", ban_eos_token);
    add_pair_into_json(request_body_json, "add_bos_token", add_bos_token);
    add_pair_into_json(request_body_json, "skip_special_tokens", skip_special_tokens);
    add_pair_into_json(request_body_json, "static_cache", static_cache);
    add_pair_into_json(request_body_json, "truncation_length", truncation_length);
    add_pair_into_json_from_vector(request_body_json, "sampler_priority", sampler_priority);
    add_pair_into_json(request_body_json, "custom_token_bans", custom_token_bans);
    add_pair_into_json(request_body_json, "negative_prompt", negative_prompt);
    add_pair_into_json(request_body_json, "dry_sequence_breakers", dry_sequence_breakers);
    add_pair_into_json(request_body_json, "grammar_string", grammar_string);

    return picojson::value{ request_body_json }.serialize();
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
    picojson::object request_body_json;
    add_pair_into_json(request_body_json, "text", prompt);
    return picojson::value{ request_body_json }.serialize();
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
    picojson::object request_body_json;

    add_pair_into_json(request_body_json, "max_context_length", max_context_length);
    add_pair_into_json(request_body_json, "max_length", max_tokens);
    add_pair_into_json(request_body_json, "prompt", std::string{ prompt });
    add_pair_into_json(request_body_json, "rep_pen", rep_pen);
    add_pair_into_json(request_body_json, "rep_pen_range", rep_pen_range);
    add_pair_into_json_from_vector(request_body_json, "sampler_order", sampler_order);

    if (sampler_seed != -1)
    {
        add_pair_into_json(request_body_json, "sampler_seed", sampler_seed);
    }

    add_pair_into_json_from_vector(request_body_json, "stop_sequence", stop_sequence);
    add_pair_into_json(request_body_json, "temperature", temperature);
    add_pair_into_json(request_body_json, "tfs", tfs);
    add_pair_into_json(request_body_json, "top_a", top_a);
    add_pair_into_json(request_body_json, "top_k", top_k);
    add_pair_into_json(request_body_json, "top_p", top_p);
    add_pair_into_json(request_body_json, "min_p", min_p);
    add_pair_into_json(request_body_json, "typical", typical);
    add_pair_into_json(request_body_json, "use_default_badwordsids", use_default_badwordsids);
    add_pair_into_json(request_body_json, "dynatemp_range", dynatemp_range);
    add_pair_into_json(request_body_json, "smoothing_factor", smoothing_factor);
    add_pair_into_json(request_body_json, "dynatemp_exponent", dynatemp_exponent);
    add_pair_into_json(request_body_json, "mirostat", mirostat);
    add_pair_into_json(request_body_json, "mirostat_tau", mirostat_tau);
    add_pair_into_json(request_body_json, "mirostat_eta", mirostat_eta);
    add_pair_into_json(request_body_json, "genkey", genkey);
    add_pair_into_json(request_body_json, "grammar", grammar);
    add_pair_into_json(request_body_json, "grammar_retain_state", grammar_retain_state);
    add_pair_into_json(request_body_json, "memory", memory);
    add_pair_into_json_from_vector(request_body_json, "images", images);
    add_pair_into_json(request_body_json, "trim_stop", trim_stop);
    add_pair_into_json(request_body_json, "render_special", render_special);
    add_pair_into_json(request_body_json, "bypass_eos", bypass_eos);
    add_pair_into_json_from_vector(request_body_json, "banned_tokens", banned_tokens);
    add_pair_into_json(request_body_json, "dry_multiplier", dry_multiplier);
    add_pair_into_json(request_body_json, "dry_base", dry_base);
    add_pair_into_json(request_body_json, "dry_allowed_length", dry_allowed_length);
    add_pair_into_json(request_body_json, "dry_penalty_last_n", dry_penalty_last_n);
    add_pair_into_json_from_vector(request_body_json, "dry_sequence_breakers", dry_sequence_breakers);
    add_pair_into_json(request_body_json, "xtc_probability", xtc_probability);
    add_pair_into_json(request_body_json, "nsigma", nsigma);
    add_pair_into_json(request_body_json, "logprobs", logprobs);
    add_pair_into_json(request_body_json, "replace_instruct_placeholders", replace_instruct_placeholders);

    return picojson::value{ request_body_json }.serialize();
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
    picojson::object request_body_json;
    add_pair_into_json(request_body_json, "prompt", prompt);
    return picojson::value{ request_body_json }.serialize();
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

    const auto results = resolver.resolve(config.llm_prompt_params.host, config.llm_prompt_params.port);
    tcp_stream.expires_after(std::chrono::seconds{ config.expires_after });
    tcp_stream.connect(results, error_code);
    if_error_throw<connect_exception>(error_code);

    const std::string request_body{ config.llm_backend_params->get_request_body_for_token_count(prompt) };
    BOOST_LOG_TRIVIAL(trace) << "Send JSON\n```\n" << request_body << "\n```";

    http::request<http::string_body> request{ http::verb::post, config.llm_prompt_params.token_count_target, 11 }; // HTTP/1.1
    request.set(http::field::host, config.llm_prompt_params.host);
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

    return config.llm_backend_params->parse_response_for_token_count(response);
}

int get_tokens_from_cache(const config& config, std::string_view str)
{
    constexpr std::size_t capacity{ 1000 };
    int tokens{};

    auto iter = config.lru_cache.get<key_tag>().find(std::string{ str });
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
    std::string_view prompts,
    std::string_view prefix
)
{
    std::string expanded_prompt{ expand_macro(prompts, config) };
    const std::string expanded_prefix{ expand_macro(prefix, config) };
    const std::size_t initial_prompts_size{
        config.llm_prompt_params.skip_generation_prefix
        ? expanded_prompt.size() + expanded_prefix.size() : expanded_prompt.size()
    };
    expanded_prompt += expanded_prefix;

    const int initial_tokens{ send_token_count_request(config, expanded_prompt) };

    BOOST_LOG_TRIVIAL(info) << "Prompt created.\n```\n" << expanded_prompt << "\n```";

    std::string current_prompt{ expanded_prompt };
    int current_tokens = initial_tokens;
    for (int completion_iterations{}; completion_iterations < config.max_completion_iterations; ++completion_iterations)
    {
        BOOST_LOG_TRIVIAL(trace) << "completion_iterations: " << completion_iterations;

        if (current_tokens - initial_tokens >= config.min_completion_tokens)
        {
            break;
        }

        const int remaining_context{ config.llm_backend_params->get_truncation_length() - current_tokens };
        if (remaining_context <= 0)
        {
            BOOST_LOG_TRIVIAL(warning) << "Context window full. Cannot generate more tokens.";
            break;
        }

        int tokens_to_generate = std::min(config.llm_backend_params->get_max_tokens(), remaining_context);
        if (tokens_to_generate <= 0)
        {
            BOOST_LOG_TRIVIAL(warning) << "No tokens left to generate. Aborting.";
            break;
        }

        const int max_tokens{ tokens_to_generate };
        const std::string response{ send_completions_request(
            config, current_prompt, *config.llm_backend_params, max_tokens
        ) };

        if (response.empty())
        {
            break;
        }

        current_prompt += response;
        current_tokens = send_token_count_request(config, current_prompt);
    }

    return current_prompt.substr(initial_prompts_size);
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
                context.variables[key] = value;
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
    if (config.llm_prompt_params.generation_prefix.empty())
    {
        config.llm_prompt_params.generation_prefix = "\\n{{phase}}: ";
    }
}

void set_phase_variables(
    const std::vector<std::string>& phases,
    std::size_t phase_index,
    std::unordered_map<std::string, std::string>& variables
)
{
    if (phase_index >= phases.size())
    {
        throw array_index_out_of_bounds_exception{};
    }

    if (phase_index > 0)
    {
        variables["prev_phase"] = phases[phase_index - 1];
    }
    else
    {
        variables.erase("prev_phase");
    }

    variables["phase"] = phases[phase_index];

    if (phase_index < phases.size() - 1)
    {
        variables["next_phase"] = phases[phase_index + 1];
    }
    else
    {
        variables.erase("next_phase");
    }
}

void set_builtin_variables(
    config& config
)
{
    config.context.variables["stdin"] = builtin::stdin_(config);
}

void set_builtin_variables_each_iteration(
    config& config
)
{
    config.context.variables["date"] = builtin::date();
    config.context.variables["time"] = builtin::time();
    config.context.variables["datetime"] = builtin::datetime();
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
    if (!config.llm_prompt_params.paragraphs_file.empty())
    {
        config.phases.clear();
        const std::filesystem::path plot_file_path{ string_to_path_by_config(config.llm_prompt_params.paragraphs_file, config) };
        const std::string content{ read_file_to_string(plot_file_path) };
        std::vector<item> paragraphs{ parse_item_list(content) };
        set_paragraphs_to_phases(paragraphs, config.phases);
    }

    if (config.mode == "tg")
    {
        config.llm_backend_params = &config.tg_completions_params;
        if (config.llm_prompt_params.completions_target.empty())
        {
            config.llm_prompt_params.completions_target = "/v1/completions";
        }
        if (config.llm_prompt_params.token_count_target.empty())
        {
            config.llm_prompt_params.token_count_target = "/v1/internal/token-count";
        }
    }
    else if (config.mode == "kc")
    {
        config.llm_backend_params = &config.kc_generation_params;
        if (config.llm_prompt_params.completions_target.empty())
        {
            config.llm_prompt_params.completions_target = "/api/v1/generate";
        }
        if (config.llm_prompt_params.token_count_target.empty())
        {
            config.llm_prompt_params.token_count_target = "/api/extra/tokencount";
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
    const std::regex code_block_regex(R"(```(\S+)\s*\n([\s\S]*?)```)");

    for (std::cregex_iterator iter = std::cregex_iterator(markdown_content.data(), markdown_content.data() + markdown_content.size(), code_block_regex); iter != std::cregex_iterator{}; ++iter)
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
    auto results = resolver.resolve(host, port, error_code);
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
    HANDLE snapshot{ CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) };
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

            ("llm-system-prompts-file", po::value<std::string>(&config.llm_prompt_params.system_prompts_file)->default_value("system_prompts.txt"), "LLM system prompt file path")
            ("llm-output-file", po::value<std::string>(&config.llm_prompt_params.output_file)->default_value("output.txt"), "LLM output file path")
            ("llm-generation-prefix", po::value<std::string>(&config.llm_prompt_params.generation_prefix)->default_value(""), "LLM generation prefix")
            ("llm-generation-suffix", po::value<std::string>(&config.llm_prompt_params.generation_suffix)->default_value(""), "LLM generation suffix")
            ("llm-skip-generation-prefix", po::bool_switch(&config.llm_prompt_params.skip_generation_prefix)->default_value(false), "LLM skip generation prefix")
            ("llm-retry-generation-prefix", po::value<std::string>(&config.llm_prompt_params.retry_generation_prefix)->default_value(""), "LLM prefix to be used after a failed text generation")
            ("llm-paragraphs-file", po::value<std::string>(&config.llm_prompt_params.paragraphs_file)->default_value(""), "LLM paragraphs file")
            ("llm-host", po::value<std::string>(&config.llm_prompt_params.host)->default_value("localhost"), "LLM host")
            ("llm-port", po::value<std::string>(&config.llm_prompt_params.port)->default_value("5000"), "LLM port")
            ("llm-api-key", po::value<std::string>(&config.llm_prompt_params.api_key)->default_value(""), "LLM API key")
            ("llm-completions-target", po::value<std::string>(&config.llm_prompt_params.completions_target)->default_value(""), "LLM completions target")
            ("llm-token-count-target", po::value<std::string>(&config.llm_prompt_params.token_count_target)->default_value(""), "LLM token count target")
            ("llm-reasoning-prefix", po::value<std::string>(&config.llm_prompt_params.reasoning_prefix)->default_value(""), "LLM reasoning prefix")
            ("llm-reasoning-suffix", po::value<std::string>(&config.llm_prompt_params.reasoning_suffix)->default_value(""), "LLM reasoning suffix")
            ("llm-code-block-extract", po::bool_switch(&config.llm_prompt_params.code_block_extract)->default_value(false), "code block extract switch")

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

            ("kc-max-context-length", po::value<int>(&config.kc_generation_params.max_context_length)->default_value(4096), "Maximum number of tokens to send to the model. (minimum: 1)")
            ("kc-max-length", po::value<int>(&config.kc_generation_params.max_length)->default_value(512), "Number of tokens to generate. (minimum: 1)")
            ("kc-rep-pen", po::value<double>(&config.kc_generation_params.rep_pen)->default_value(1.0), "Base repetition penalty value. (minimum: 1.0)")
            ("kc-rep-pen-range", po::value<int>(&config.kc_generation_params.rep_pen_range)->default_value(0), "Repetition penalty range. (minimum: 0)")
            ("kc-sampler-order", po::value<std::vector<int>>(&config.kc_generation_params.sampler_order)->multitoken(), "Sampler order to be used. If N is the length of this array, then N must be greater than or equal to 6 and the array must be a permutation of the first N non-negative integers.")
            ("kc-sampler-seed", po::value<int>(&config.kc_generation_params.sampler_seed)->default_value(1), "RNG seed to use for sampling. If not specified, the global RNG will be used. (minimum: 1, maximum: 999999)")
            ("kc-stop-sequence", po::value<std::vector<std::string>>(&config.kc_generation_params.stop_sequence)->multitoken(), "An array of string sequences where the API will stop generating further tokens. The returned text WILL contain the stop sequence if trim_stop is false.")
            ("kc-temperature", po::value<double>(&config.kc_generation_params.temperature)->default_value(1.0), "Temperature value.")
            ("kc-tfs", po::value<double>(&config.kc_generation_params.tfs)->default_value(1.0), "Tail free sampling value. (minimum: 0.0, maximum: 1.0)")
            ("kc-top-a", po::value<double>(&config.kc_generation_params.top_a)->default_value(1.0), "Top-a sampling value. (minimum: 0.0)")
            ("kc-top-k", po::value<double>(&config.kc_generation_params.top_k)->default_value(0.0), "Top-k sampling value. (minimum: 0.0)")
            ("kc-top-p", po::value<double>(&config.kc_generation_params.top_p)->default_value(1.0), "Top-p sampling value. (minimum: 0.0, maximum: 1.0)")
            ("kc-min-p", po::value<double>(&config.kc_generation_params.min_p)->default_value(0.1), "Min-p sampling value. (minimum: 0.0, maximum: 1.0)")
            ("kc-typical", po::value<double>(&config.kc_generation_params.typical)->default_value(1.0), "Typical sampling value. (minimum: 0.0, maximum: 1.0)")
            ("kc-use-default-badwordsids", po::bool_switch(&config.kc_generation_params.use_default_badwordsids)->default_value(false), "If true, prevents the EOS token from being generated (Ban EOS).")
            ("kc-dynatemp_range", po::value<double>(&config.kc_generation_params.dynatemp_range)->default_value(0.0), "If not equal to 0, uses dynamic temperature. Dynamic temperature range will be between Temp+Range and Temp-Range. If equal to 0 , uses static temperature. (default: 0, minimum: -5.0, maximum: 5.0)")
            ("kc-smoothing-factor", po::value<double>(&config.kc_generation_params.smoothing_factor)->default_value(0.0), "Modifies temperature behavior. If greater than 0 uses smoothing factor. (default: 0.0, minimum: 0.0)")
            ("kc-dynatemp-exponent", po::value<double>(&config.kc_generation_params.dynatemp_exponent)->default_value(1.0), "Exponent used in dynatemp. (default: 0.0)")
            ("kc-mirostat", po::value<int>(&config.kc_generation_params.mirostat)->default_value(0), "KoboldCpp ONLY. Sets the mirostat mode, 0=disabled, 1=mirostat_v1, 2=mirostat_v2. (minimum: 0, maximum: 2)")
            ("kc-mirostat-tau", po::value<double>(&config.kc_generation_params.mirostat_tau)->default_value(0.0), "KoboldCpp ONLY. Mirostat tau value. (minimum: 0.0)")
            ("kc-mirostat-eta", po::value<double>(&config.kc_generation_params.mirostat_eta)->default_value(0.0), "KoboldCpp ONLY. Mirostat eta value. (minimum: 0.0)")
            ("kc-genkey", po::value<std::string>(&config.kc_generation_params.genkey)->default_value(""), "KoboldCpp ONLY. A unique genkey set by the user. When checking a polled-streaming request, use this key to be able to fetch pending text even if multiuser is enabled.")
            ("kc-grammar", po::value<std::string>(&config.kc_generation_params.grammar)->default_value(""), "KoboldCpp ONLY. A string containing the GBNF grammar to use.")
            ("kc-grammar-retain-state", po::bool_switch(&config.kc_generation_params.grammar_retain_state)->default_value(false), "KoboldCpp ONLY. If true, retains the previous generation's grammar state, otherwise it is reset on new generation.")
            ("kc-memory", po::value<std::string>(&config.kc_generation_params.memory)->default_value(""), "KoboldCpp ONLY. If set, forcefully appends this string to the beginning of any submitted prompt text. If resulting context exceeds the limit, forcefully overwrites text from the beginning of the main prompt until it can fit. Useful to guarantee full memory insertion even when you cannot determine exact token count.")
            ("kc-images", po::value<std::vector<std::string>>(&config.kc_generation_params.images)->multitoken(), "KoboldCpp ONLY. If set, takes an array of base64 encoded strings, each one representing an image to be processed.")
            ("kc-trim-stop", po::bool_switch(&config.kc_generation_params.trim_stop)->default_value(true), "KoboldCpp ONLY. If true, also removes detected stop_sequences from the output and truncates all text after them. If false, output will also include stop sequence and potentially a few additional characters.")
            ("kc-render-special", po::bool_switch(&config.kc_generation_params.render_special)->default_value(false), "KoboldCpp ONLY. If true, prints special tokens as text for GGUF models")
            ("kc-bypass-eos", po::bool_switch(&config.kc_generation_params.trim_stop)->default_value(false), "KoboldCpp ONLY. If true, allows EOS token to be generated, but does not stop generation. Not recommended unless you know what you are doing.")
            ("kc-banned-tokens", po::value<std::vector<std::string>>(&config.kc_generation_params.banned_tokens)->multitoken(), "An array of string sequences, each entry represents a word or phrase prevented from being generated, either modifying model vocab or by backtracking and regenerating when they appear.")
            ("kc-dry-multiplier", po::value<double>(&config.kc_generation_params.dry_multiplier)->default_value(0.0), "KoboldCpp ONLY. DRY multiplier value, 0 to disable. (minimum: 0)")
            ("kc-dry-base", po::value<double>(&config.kc_generation_params.dry_base)->default_value(1.75), "KoboldCpp ONLY. DRY base value. (minimum: 0)")
            ("kc-dry-allowed-length", po::value<int>(&config.kc_generation_params.dry_allowed_length)->default_value(2), "KoboldCpp ONLY. DRY allowed length value. (minimum: 0)")
            ("kc-dry-penalty-last-n", po::value<int>(&config.kc_generation_params.dry_penalty_last_n)->default_value(0), "KoboldCpp ONLY. DRY last n tokens penalized value. (minimum: 0)")
            ("kc-dry-sequence-breakers", po::value<std::vector<std::string>>(&config.kc_generation_params.dry_sequence_breakers)->multitoken(), "An array of string sequence breakers for DRY.")
            ("kc-xtc-threshold", po::value<double>(&config.kc_generation_params.xtc_threshold)->default_value(0.1), "KoboldCpp ONLY. XTC threshold. (minimum: 0)")
            ("kc-xtc-probability", po::value<double>(&config.kc_generation_params.xtc_probability)->default_value(0.0), "KoboldCpp ONLY. XTC probability. Set to above 0 to enable XTC. (minimum: 0)")
            ("kc-nsigma", po::value<double>(&config.kc_generation_params.nsigma)->default_value(0.0), "KoboldCpp ONLY. Top N-Sigma value. Set to above 0 to enable nsigma. (minimum: 0)")
            ("kc-logprobs", po::bool_switch(&config.kc_generation_params.logprobs)->default_value(false), "If true, return up to 5 top logprobs for generated tokens. Incurs performance overhead.")
            ("kc-replace-instruct-placeholders", po::bool_switch(&config.kc_generation_params.use_default_badwordsids)->default_value(false), "If true, replaces instruct placeholders {{[INPUT]}} and {{[OUTPUT]}} with backend selected instruct tags.")

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

            ("cu-host", po::value<std::string>(&config.cu_generation_params.host)->default_value("localhost"), "Comfy UI host")
            ("cu-port", po::value<std::string>(&config.cu_generation_params.port)->default_value("8188"), "Comfy UI port")
            ("cu-target", po::value<std::string>(&config.cu_generation_params.target)->default_value("/prompt"), "Comfy UI prompt target")
            ("cu-prompt-file", po::value<std::string>(&config.cu_generation_params.prompt_file)->default_value("prompt.json"), "Comfy UI prompt file")
            ("cu-output-directory", po::value<std::string>(&config.cu_generation_params.output_directory)->default_value("output"), "Comfy UI output directory")
            ("cu-upload-images", po::value<std::vector<std::string>>(&config.cu_generation_params.upload_images)->multitoken(), "Comfy UI upload images (macro_name=local_path)")
            ("cu-preserve-subdirectories", po::bool_switch(&config.cu_generation_params.preserve_subdirectories)->default_value(false), "Comfy UI preserve server side sub-directories")
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

        if (vm.count("help"))
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

        std::transform(config.tg_completions_params.stop.begin(), config.tg_completions_params.stop.end(), config.tg_completions_params.stop.begin(), unescape_string);
        std::transform(config.user_defined_variables.begin(), config.user_defined_variables.end(), config.user_defined_variables.begin(), unescape_string);
        std::transform(config.kc_generation_params.stop_sequence.begin(), config.kc_generation_params.stop_sequence.end(), config.kc_generation_params.stop_sequence.begin(), unescape_string);
        config.tg_completions_params.dry_sequence_breakers = unescape_string(config.tg_completions_params.dry_sequence_breakers);
        config.llm_prompt_params.generation_prefix = unescape_string(config.llm_prompt_params.generation_prefix);
        config.llm_prompt_params.generation_suffix = unescape_string(config.llm_prompt_params.generation_suffix);
        config.llm_prompt_params.retry_generation_prefix = unescape_string(config.llm_prompt_params.retry_generation_prefix);
        config.llm_prompt_params.reasoning_prefix = unescape_string(config.llm_prompt_params.reasoning_prefix);
        config.llm_prompt_params.reasoning_suffix = unescape_string(config.llm_prompt_params.reasoning_suffix);
        config.sd_txt2img_params.prompt = unescape_string(config.sd_txt2img_params.prompt);
        config.sd_txt2img_params.negative_prompt = unescape_string(config.sd_txt2img_params.negative_prompt);
        config.sb_generation_params.text = unescape_string(config.sb_generation_params.text);

        parse_user_defined_variables(config.user_defined_variables, config.context);
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

    int remaining_tokens{ config.tg_completions_params.truncation_length - config.tg_completions_params.max_tokens };

    const std::string expanded_system_prompts{ expand_macro(concatenate(system_prompts), config) };
    try_append(expanded_system_prompts, config, false, result, remaining_tokens);

    return result;
}

void read_prompts(const config& config, prompts& prompts)
{
    if (config.mode == "tg" || config.mode == "kc")
    {
        const std::filesystem::path system_prompts_path{ string_to_path_by_config(config.llm_prompt_params.system_prompts_file, config) };
        read_file_to_container(system_prompts_path, prompts.system_prompts);
    }
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

void write_response(const config& config, std::string_view response, std::string_view filepath, std::ios_base::openmode mode)
{
    const std::filesystem::path file_path{ string_to_path_by_config(filepath, config) };
    create_parent_directories(file_path);
    boost::nowide::ofstream ofs{ file_path, mode };
    if (!ofs.is_open())
    {
        throw file_open_exception{} << error_info::path{ file_path };
    }
    ofs << response;
    BOOST_LOG_TRIVIAL(info) << "Write response to " << file_path;
}

void llm_write_code_block(const config& config, std::string_view markdown)
{
    if (config.llm_prompt_params.code_block_extract)
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
                write_response(config, code, complement_extension(name, ".txt"), 0);
            }
        }
    }
}

void llm_append_mode(const config& config, prompts& prompts)
{
    const std::string prompts_string{ expand_macro(prompts.to_string(config), config) };

    try
    {
        std::string response{ generate_and_complete_text(config, prompts_string, config.llm_prompt_params.generation_prefix) };
        response = remove_reasoning(response, config.llm_prompt_params.reasoning_prefix, config.llm_prompt_params.reasoning_suffix);
        response += config.llm_prompt_params.generation_suffix;

        write_response(config, response, config.llm_prompt_params.output_file, std::ios_base::app);

        if (!config.verbose)
        {
            boost::nowide::cout << response << std::flush;
        }

        llm_write_code_block(config, response);

    }
    catch (const text_generation_exception& exception)
    {
        BOOST_LOG_TRIVIAL(warning) << boost::diagnostic_information(exception);
        if (!config.llm_prompt_params.retry_generation_prefix.empty())
        {
            BOOST_LOG_TRIVIAL(info) << "Start to retry text generation with retry-generation-prefix.";
            std::string response{ generate_and_complete_text(config, prompts_string, config.llm_prompt_params.retry_generation_prefix) };
            response = remove_reasoning(response, config.llm_prompt_params.reasoning_prefix, config.llm_prompt_params.reasoning_suffix);
            response += config.llm_prompt_params.generation_suffix;
            write_response(config, response, config.llm_prompt_params.output_file, std::ios_base::out | std::ios_base::app);
            if (!config.verbose)
            {
                boost::nowide::cout << response << std::flush;
            }
            llm_write_code_block(config, response);
        }
    }
}

std::string prompt_from_string_or_file_path(
    std::string_view string,
    std::string_view file_path,
    const config& config
)
{
    return expand_macro(string.empty() ? read_file_to_string(string_to_path_by_config(file_path, config)) : std::string{ string }, config);
}

void generate_and_output(const config& config, prompts& prompts)
{
    if (config.mode == "tg" || config.mode == "kc")
    {
        llm_append_mode(config, prompts);
    }
    else if (config.mode == "sd")
    {
        const std::filesystem::path output_file_path{ string_to_path_by_config(config.sd_txt2img_params.output_file, config) };
        create_parent_directories(output_file_path);

        const std::string prompt_string{ prompt_from_string_or_file_path(config.sd_txt2img_params.prompt, config.sd_txt2img_params.prompt_file, config) };
        const std::string negative_prompt_string{ prompt_from_string_or_file_path(config.sd_txt2img_params.negative_prompt, config.sd_txt2img_params.negative_prompt_file, config) };

        send_automatic1111_txt2img_request(config, prompt_string, negative_prompt_string, output_file_path);
    }
    else if (config.mode == "sb")
    {
        const std::string text{ prompt_from_string_or_file_path(config.sb_generation_params.text, config.sb_generation_params.text_file, config) };

        send_style_bert_voice_request(config, text);
    }
    else if (config.mode == "cu")
    {
        const std::filesystem::path prompt_file{ string_to_path_by_config(config.cu_generation_params.prompt_file, config) };
        const std::string prompt{ expand_macro(read_file_to_string(prompt_file), config) };

        send_comfy_ui_prompt(config, prompt);
    }
}

void set_seed(config& config)
{
    if (config.seed == -1)
    {
        config.tg_completions_params.seed = random<int>(0);
        config.kc_generation_params.sampler_seed = random<int>(0, 999999);
        config.sd_txt2img_params.seed = random<int>(0);
    }
    else
    {
        config.tg_completions_params.seed = config.seed;
        config.kc_generation_params.sampler_seed = config.seed;
        config.sd_txt2img_params.seed = config.seed;
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
        prompts prompts;
        read_prompts(config, prompts);

        set_seed(config);

        set_builtin_variables_each_iteration(config);
        config.context.variables["N"] = std::to_string(iteration_count + 1);

        for (std::size_t phase_index{}; phase_index < config.phases.size(); ++phase_index)
        {
            set_phase_variables(config.phases, phase_index, config.context.variables);
            generate_and_output(config, prompts);
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
        }
        else
        {
            set_builtin_variables(config);

            if (config.mode == "cu")
            {
                upload_images_to_comfy_ui(config, config.context);
            }

            iterate(config);
        }
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