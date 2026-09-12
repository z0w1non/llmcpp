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
#include <concepts>

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/readable_pipe.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/error.hpp>
#include <boost/program_options.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
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
#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/binary_from_base64.hpp>
#include <boost/archive/iterators/transform_width.hpp>
#include <boost/url.hpp>
#include <boost/process/v2/process.hpp>
#include <boost/process/v2/environment.hpp>
#include <boost/process/v2/execute.hpp>
#include <boost/process/v2/stdio.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/noncopyable.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/variant.hpp>
#include <boost/optional.hpp>
#include <boost/spirit/include/qi.hpp>

#include "json.hpp"

#ifdef _WIN32
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

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace llmcpp
{
    struct base_exception
        : virtual boost::exception
        , virtual std::exception
    {
    public:
        base_exception();
    };

    struct runtime_exception : base_exception {};
    struct logic_error : base_exception {};
    struct io_exception : runtime_exception {};
    struct file_open_exception : io_exception {};
    struct socket_exception : runtime_exception {};
    struct text_generation_exception : runtime_exception {};
    struct image_generation_exception : runtime_exception {};
    struct comfy_ui_generation_exception : runtime_exception {};
    struct syntax_exception : runtime_exception {};
    struct json_parse_exception : runtime_exception {};
    struct macro_exception : runtime_exception {};
    struct command_line_exception : runtime_exception {};
    struct array_index_out_of_bounds_exception : runtime_exception {};
    struct dns_resolve_exception : runtime_exception {};
    struct connect_exception : runtime_exception {};
    struct http_send_exception : runtime_exception {};
    struct http_receive_exception : runtime_exception {};
    struct http_status_exception : runtime_exception {};
    struct png_exception : runtime_exception {};

    namespace error_info
    {
        using stacktrace = boost::error_info<struct tag_stacktrace, boost::stacktrace::stacktrace>;
        using description = boost::error_info<struct tag_description, std::string>;
        using nested_exception = boost::error_info<struct tag_nested_exception, boost::exception_ptr>;
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

        namespace system
        {
            using error_code = boost::error_info<struct tag_error_code, boost::system::error_code>;
        }
    }

    base_exception::base_exception()
    {
        *this << error_info::stacktrace{ boost::stacktrace::stacktrace() };
    }

    template<typename Exception>
    [[noreturn]] void throw_exception(const Exception& e, const boost::source_location& location = BOOST_CURRENT_LOCATION)
    {
        boost::throw_exception(e, location);
    }

    template<typename Exception>
    [[noreturn]] void throw_nested_exception(const Exception& e, const boost::source_location& location = BOOST_CURRENT_LOCATION)
    {
        if (boost::exception_ptr ptr{ boost::current_exception() }; ptr)
        {
            boost::throw_exception(boost::enable_error_info(e) << error_info::nested_exception{ ptr }, location);
        }
        boost::throw_exception(logic_error{} << error_info::description{ "throw_nested_exception must be called within catch block" }, location);
    }

    template<typename Key, typename T, typename Compare = std::less<>, typename Allocator = std::allocator<std::pair<const Key, T> >>
    using transparent_map = std::map<Key, T, Compare>;

    template<typename T>
    using string_map = std::map<std::string, T, std::less<>>;

    template<typename Key, typename T, typename Allocator = std::allocator<std::pair<const Key, T> >>
    using transparent_unordered_map = std::unordered_map<Key, T, std::hash<Key>, std::equal_to<>, Allocator>;

    // for <= C++ 17
    struct string_hash
    {
        using is_transparent = void;

        std::size_t operator()(std::string_view sv) const noexcept
        {
            return std::hash<std::string_view>{}(sv);
        }

        std::size_t operator()(const std::string& s) const noexcept
        {
            return std::hash<std::string>{}(s);
        }

        std::size_t operator()(const char* s) const noexcept
        {
            return std::hash<std::string_view>{}(s);
        }
    };

    template<typename T>
    using string_unordered_map = std::unordered_map<std::string, T, string_hash, std::equal_to<>>;

    struct config;

    struct text_generation_parameters
    {
        virtual ~text_generation_parameters() {}
        virtual std::string get_request_body_for_text_completions(std::string_view prompt, int max_tokens) const = 0;
        virtual std::string parse_response_for_text_completions(const std::string& response) const = 0;
        virtual std::string get_request_body_for_token_count(std::string_view prompt) const = 0;
        virtual int parse_response_for_token_count(const std::string& response) const = 0;
        virtual std::string get_request_body_for_vision(std::string_view prompt, int max_tokens, std::string_view base64_image, std::string_view mime_type) const = 0;
        virtual std::string parse_response_for_vision(const std::string& response) const = 0;
        virtual int get_max_tokens() const = 0;
        virtual int get_truncation_length() const = 0;
    };

    enum class llm_mode
    {
        completions, vision
    };

    llm_mode string_to_llm_mode(std::string_view str);

    std::string llm_mode_to_target(llm_mode mode, const config& cfg);

    struct llm_prompt_parameters
    {
        std::string prompt;
        std::string prompt_file;
        std::string output_file;
        std::string generation_prefix;
        std::string generation_suffix;
        std::string paragraphs_file;
        std::string image_file;

        std::string host;
        std::string port;
        std::string api_key;
        std::string completions_target;
        std::string token_count_target;
        std::string vision_target;

        int min_completion_tokens{};
        int max_completion_iterations{};

        std::string reasoning_prefix;
        std::string reasoning_suffix;

        bool code_block_extract{};

        text_generation_parameters* backend{};
        llm_mode mode;
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
        std::string parse_response_for_text_completions(const std::string& response) const override;
        std::string get_request_body_for_token_count(std::string_view prompt) const override;
        int parse_response_for_token_count(const std::string& response) const override;
        std::string get_request_body_for_vision(std::string_view prompt, int max_tokens, std::string_view base64_image, std::string_view mime_type) const override;
        std::string parse_response_for_vision(const std::string& response) const override;

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
        std::string parse_response_for_text_completions(const std::string& response) const override;
        std::string get_request_body_for_token_count(std::string_view prompt) const override;
        int parse_response_for_token_count(const std::string& response) const override;
        std::string get_request_body_for_vision(std::string_view prompt, int max_tokens, std::string_view base64_image, std::string_view mime_type) const override;
        std::string parse_response_for_vision(const std::string& response) const override;

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
            std::string ad_controlnet_weight;
            double ad_controlnet_guidance_start{};
            double ad_controlnet_guidance_end{};
        };

        arg args1;
    };

    struct alwayson_scripts
    {
        adetailer_parametesrs adetailer_parametesrs;
    };

    enum class sd_mode
    {
        txt2img, img2img
    };

    sd_mode string_to_sd_mode(std::string_view str);

    std::string sd_mode_to_target(sd_mode mode, const config& cfg);

    struct sd_txt2img_parameters
    {
        std::string target;

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
    };

    struct sd_img2img_parameters
    {
        std::string target;

        std::vector<std::string> init_images;

        int seed_resize_from_h{};
        int seed_resize_from_w{};

        int resize_mode{};
        double image_cfg_scale{};
        std::string mask;
        int mask_blur_x{};
        int mask_blur_y{};
        int mask_blur{};
        bool mask_round{};
        int inpainting_fill{};
        bool inpaint_full_res{};
        int inpaint_full_res_padding{};
        int inpainting_mask_invert{};

        double initial_noise_multiplier{};
        std::string latent_mask;
    };

    struct sd_parameters
    {
        std::string host;
        std::string port;

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
        std::string force_task_id;
        std::string sampler_index;
        std::string script_name;
        std::vector<std::string> script_args;
        bool send_images{};
        bool save_images{};
        alwayson_scripts alwayson_scripts;
        std::string infotext;

        bool abg_remover_enable{};

        sd_mode mode;
        sd_txt2img_parameters txt2img;
        sd_img2img_parameters img2img;
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
        std::string reference_audio_path;
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


    using primitive_type = boost::variant<int, bool, char, double, std::string>;

    struct undefined_variable_type
    {
        std::string name;
    };

    template<typename ... Args>
    using value_and_reference_variant = boost::variant<Args ..., std::reference_wrapper<Args> ..., undefined_variable_type>;

    using vr_primitive_type = value_and_reference_variant<int, bool, char, double, std::string>;

    struct unwrap_impl
    {
        template<typename T>
        decltype(auto) operator()(T&& value)
        {
            return std::forward<T>(value);
        }

        template<typename T>
        T& operator()(std::reference_wrapper<T> ref)
        {
            return ref.get();
        }

        template<typename T>
        const T& operator()(std::reference_wrapper<const T> ref)
        {
            return ref.get();
        }
    };

    template<typename T>
    decltype(auto) unwrap(T&& arg)
    {
        return unwrap_impl{}(std::forward<T>(arg));
    }

    template<typename T>
    struct unwrap_type_impl
    {
        using type = T;
    };

    template<typename T>
    struct unwrap_type_impl<std::reference_wrapper<T>>
    {
        using type = T;
    };

    template <typename T>
    using unwrap_type_t = typename unwrap_type_impl<std::decay_t<T>>::type;

    struct x_primitive_to_string_visitor
    {
        template<typename T>
        std::string operator ()(const T& value) const
        {
            if constexpr (std::is_same_v<unwrap_type_t<T>, std::string>)
            {
                return unwrap(value);
            }
            return boost::lexical_cast<std::string>(unwrap(value));
        }

        [[noreturn]] std::string operator ()(const undefined_variable_type& undefined_variable) const
        {
            BOOST_LOG_TRIVIAL(warning) << "Failed to convert undefined variable to string (" << undefined_variable.name << ")";
            llmcpp::throw_exception(macro_exception{});
        }
    };

    std::string primitive_to_string(const primitive_type& primitive)
    {
        return boost::apply_visitor(x_primitive_to_string_visitor{}, primitive);
    }

    std::string vr_primitive_to_string(const vr_primitive_type& primitive)
    {
        return boost::apply_visitor(x_primitive_to_string_visitor{}, primitive);
    }

    template<typename Result>
    const Result& get_or_throw(const primitive_type& value)
    {
        if (const Result* ptr{ boost::get<Result>(&value) }; ptr)
        {
            return *ptr;
        }
        llmcpp::throw_exception(macro_exception{});
    }

    template<typename Result>
    std::optional<Result> get_optional(const primitive_type& value)
    {
        if (const Result* ptr{ boost::get<Result>(&value) }; ptr)
        {
            return *ptr;
        }
        return std::nullopt;
    }

    class context
        : private boost::noncopyable
    {
    public:
        using variable_map_type = string_unordered_map<primitive_type>;

        context();
        context make_pushed() const;
        void set(std::string_view key, const primitive_type& value);
        const primitive_type* get(std::string_view key) const;
        primitive_type* get(std::string_view key);

    private:
        context(const context& ctx);
        variable_map_type variable_map;
        const context* base{};
    };

    struct token_count_string
    {
        std::string str;
        int tokens{};
    };

    struct by_key {};
    struct by_lru {};

    using lru_cache = boost::multi_index::multi_index_container<
        token_count_string,
        boost::multi_index::indexed_by<
        boost::multi_index::hashed_unique<
        boost::multi_index::tag<by_key>,
        boost::multi_index::member<token_count_string, std::string, &token_count_string::str>
        >,
        boost::multi_index::sequenced<
        boost::multi_index::tag<by_lru>
        >
        >
    >;

    struct item
    {
        std::string head;
        std::vector<std::string> descriptions;
    };

    enum class command_mode
    {
        none, tg, kc, sd, sb, cu, extract_png_parameters
    };

    command_mode string_to_command_mode(std::string_view str);

    struct config
    {
        command_mode command_mode;
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

        std::string png_file;

        std::string server_executable_file;
        std::string server_arguments;
        std::string server_host;
        std::string server_port;
        int server_max_retries;
        int server_wait_ms;

        llm_prompt_parameters llm;
        tg_completions_parameters tg;
        kc_generation_parameters kc;
        sd_parameters sd;
        sb_generation_parameters sb;
        cu_generation_parameters cu;

        mutable lru_cache lru_cache;
        context ctx;
    };

    std::string truncate_prompt_by_config(std::string_view prompt, const config& cfg);

    std::string base64_encode(std::string_view encoded_string);

    std::string base64_decode(std::string_view encoded_string);

    std::string trim(std::string_view str);

    std::string console_string_to_u8string(std::string_view input);

    void truncate_by_tokens(std::string_view string, int max_tokens, const config& cfg, bool reverse, std::string& result, int& tokens);

    void truncate_prompt(std::string_view string, const config& cfg, bool reverse, std::string& result, int& remaining_tokens);

    void create_parent_directories(const std::filesystem::path& path);

    std::string read_file_to_string(const std::filesystem::path& file, std::ios::openmode openmode = {});

    std::string read_binary_file_to_string(std::string_view file, const config& cfg);

    std::string read_text_file_to_string(std::string_view path, const config& cfg, std::string_view extension = ".txt");

    std::string image_path_to_base64_encoded_string(std::string_view image_path, const config& cfg);

    std::vector<std::string> image_paths_to_base64_encoded_strings(const std::vector<std::string>& paths, const config& cfg);

    template<typename Integer>
    Integer random(Integer min = std::numeric_limits<Integer>::min(), Integer max = std::numeric_limits<Integer>::max());

    std::string extension_to_mime_type(std::string_view extension);

    std::string complement_extension(std::string_view filepath, std::string_view extension);

    std::filesystem::path string_to_path_by_config(std::string_view path, const config& cfg);

    boost::beast::http::request<boost::beast::http::string_body> make_request(
        boost::beast::http::verb method,
        std::string_view host,
        std::string_view target,
        std::optional<std::string_view> content_type = std::nullopt,
        std::optional<std::string_view> body = std::nullopt
    );

    boost::beast::http::request<boost::beast::http::string_body> make_post_json_request(
        std::string_view host,
        std::string_view target,
        std::string_view body
    );

    boost::beast::http::response<boost::beast::http::string_body> send_http_get(
        std::string_view host,
        std::string_view port,
        std::string_view target,
        unsigned int expires_after
    );

    std::string make_automatic1111_png_parameters(const sd_parameters& parameters, std::string_view prompt, std::string_view negative_prompt);

    std::string send_automatic1111_txt2img_request(
        const config& cfg,
        std::string_view prompt,
        std::string_view negative_prompt
    );

    std::string send_style_bert_voice_request(
        const config& cfg,
        std::string_view text
    );

    std::string generate_boundary();

    std::string upload_image_to_comfy_ui(
        const config& cfg,
        std::string_view image_path,
        bool overwrite = true
    );

    void send_comfy_ui_prompt(
        const config& cfg,
        std::string_view workflow
    );

    std::vector<item> parse_item_list(std::string_view str);

    void write_item_list(const config& cfg, std::string_view task);

    int send_token_count_request(const config& cfg, std::string_view prompt);
    int get_tokens_from_cache(const config& cfg, std::string_view str);
    void write_cache(const config& cfg);
    void read_cache(const config& cfg);

    std::string generate_text(const config& cfg, std::string_view prompt, const context& ctx);

    std::string vision_image(const config& cfg, std::string_view prompt, const context& ctx);

    std::string unescape_string(std::string_view str);

    std::string json_escape_string(std::string_view str);

    void unescape_parameters(config& cfg);

    void parse_user_defined_variables(const std::vector<std::string>& predefined_macros, context& ctx);

    void init_logging_with_nowide_cout();
    void init_logging_with_nowide_file_log(const std::filesystem::path& log);
    void init_logging(const config& cfg);

    class log_stream
    {
    public:
        log_stream(boost::log::trivial::severity_level level)
            : level{ level }
        {
        }

        ~log_stream()
        {
            namespace logging = boost::log::trivial;
            switch (level)
            {
            case logging::trace:
                BOOST_LOG_TRIVIAL(trace) << oss.str();
                break;
            case logging::debug:
                BOOST_LOG_TRIVIAL(debug) << oss.str();
                break;
            case logging::info:
                BOOST_LOG_TRIVIAL(info) << oss.str();
                break;
            case logging::warning:
                BOOST_LOG_TRIVIAL(warning) << oss.str();
                break;
            case logging::error:
                BOOST_LOG_TRIVIAL(error) << oss.str();
                break;
            case logging::fatal:
                BOOST_LOG_TRIVIAL(fatal) << oss.str();
                break;
            }
        }

        template<typename T>
        log_stream& operator<<(const T& value)
        {
            oss << value;
            return *this;
        }

    private:
        boost::log::trivial::severity_level level;
        std::ostringstream oss;
    };

    void init_chat_mode(config& cfg);

    void set_phase_variables(
        const std::vector<std::string>& phases,
        std::size_t phase_index,
        const context& ctx
    );

    void set_static_builtin_variables(
        config& cfg
    );

    void set_dynamic_builtin_variables(
        config& cfg
    );

    void set_paragraphs_to_phases(
        const std::vector<item>& paragraphs,
        std::vector<std::string>& phases
    );

    void init_llm_mode(config& cfg);

    std::string sanitize_as_filename(std::string_view name);

    using code_blocks = string_unordered_map<std::string>;

    code_blocks extract_code_block_from_markdown(std::string_view markdown_content);

    bool wait_for_port(const std::string& host, const std::string& port, unsigned int max_retries, unsigned int wait_ms);

    void create_process_async(std::string_view excutable_file, const std::vector<std::string>& arguments);

    std::size_t terminate_process_by_path(const std::filesystem::path& executable_file_path);

    std::vector<std::string> parse_command_line_args(std::string_view args);

    int parse_command_line(
        int argc,
        char** argv,
        config& cfg
    );

    std::string remove_reasoning(std::string_view response, std::string_view prefix, std::string_view suffix);

    void write_file(const config& cfg, const char* data, std::size_t size, std::string_view filepath, std::ios_base::openmode mode = 0);

    void write_file(const config& cfg, std::string_view data, std::string_view filepath, std::ios_base::openmode mode = 0);

    void write_code_block(const config& cfg, std::string_view markdown);

    void generate_text_and_write(const config& cfg, std::string_view prompt, const context& ctx);

    std::string prompt_from_string_or_file_path(
        std::string_view string,
        std::string_view file_path,
        const config& cfg
    );

    void generate_and_output(const config& cfg);

    void set_seed(config& cfg);

    void create_process(const config& cfg);

    void terminate_process(const config& cfg);

    void create_process_or_terminate(const config& cfg);

    void iterate(config& cfg);

    int exception_safe_main(int argc, char** argv);

    command_mode string_to_command_mode(std::string_view str)
    {
        static const std::unordered_map<std::string_view, command_mode> map
        {
            { "tg", command_mode::tg },
            { "kc", command_mode::kc },
            { "sd", command_mode::sd },
            { "sb", command_mode::sb },
            { "cu", command_mode::cu },
            { "extract-png-parameters", command_mode::extract_png_parameters }
        };
        if (const auto iter{ map.find(str) }; iter != map.end())
        {
            return iter->second;
        }
        llmcpp::throw_exception(command_line_exception{} << error_info::description{ "Unknown mode string " + std::string{ str } });
    }

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

    void context::set(std::string_view key, const primitive_type& value)
    {
        variable_map[std::string{ key }] = value;
    }

    const primitive_type* context::get(std::string_view key) const
    {
        const std::string key_string{ key };
        const context* current{ this };

        while (current != nullptr)
        {
            const context::variable_map_type::const_iterator map_iterator{ current->variable_map.find(key_string) };
            if (map_iterator != current->variable_map.end())
            {
                return &map_iterator->second;
            }
            current = current->base;
        }
        return nullptr;
    }

    primitive_type* context::get(std::string_view key)
    {
        return const_cast<primitive_type*>(static_cast<const context&>(*this).get(key));
    }

    sd_mode string_to_sd_mode(std::string_view str)
    {
        static const std::unordered_map<std::string_view, sd_mode> map
        {
            { "txt2img", sd_mode::txt2img },
            { "img2img", sd_mode::img2img }
        };
        if (const auto iter{ map.find(str) }; iter != map.end())
        {
            return iter->second;
        }
        llmcpp::throw_exception(command_line_exception{} << error_info::description{ "Unknown sd-mode string " + std::string{ str } });
    }

    std::string sd_mode_to_target(sd_mode mode, const config& cfg)
    {
        if (mode == sd_mode::txt2img)
        {
            return cfg.sd.txt2img.target;
        }
        else if (mode == sd_mode::img2img)
        {
            return cfg.sd.img2img.target;
        }
        llmcpp::throw_exception(logic_error{} << error_info::description{ "Unknown sd-mode" });
    }

    llm_mode string_to_llm_mode(std::string_view str)
    {
        static const std::unordered_map<std::string_view, llm_mode> map
        {
            { "completions", llm_mode::completions },
            { "vision", llm_mode::vision }
        };
        if (const auto iter{ map.find(str) }; iter != map.end())
        {
            return iter->second;
        }
        llmcpp::throw_exception(command_line_exception{} << error_info::description{ "Unknown llm-mode string " + std::string{ str } });
    }

    std::string llm_mode_to_target(llm_mode mode, const config& cfg)
    {
        if (mode == llm_mode::completions)
        {
            return cfg.llm.completions_target;
        }
        else if (mode == llm_mode::vision)
        {
            return cfg.llm.vision_target;
        }
        llmcpp::throw_exception(logic_error{} << error_info::description{ "Unknown sd-mode" });
    }

    std::string base64_encode(std::string_view input)
    {
        using namespace boost::archive::iterators;
        using iterator = base64_from_binary<transform_width<std::string::const_iterator, 6, 8>>;

        const std::size_t missing_size{ (3 - input.size() % 3) % 3 };
        std::string padded_input{ input };
        padded_input.append(missing_size, '\0');

        std::string encoded{ iterator{ padded_input.begin() }, iterator{ padded_input.end() } };
        if (missing_size > 0)
        {
            encoded.replace(encoded.size() - missing_size, missing_size, missing_size, '=');
        }
        return encoded;
    }

    std::string base64_decode(std::string_view input)
    {
        using namespace boost::archive::iterators;
        using iterator = transform_width<binary_from_base64<std::string_view::const_iterator>, 8, 6>;

        std::size_t padding_count{};
        while (padding_count < input.size() && input[input.size() - 1 - padding_count] == '=')
        {
            ++padding_count;
        }

        const std::string_view trimed_input{ input.substr(0, input.size() - padding_count) };
        std::string decoded{ iterator{ input.begin() }, iterator{ input.end() } };

        std::size_t expected_size{ input.size() / 4 * 3 - padding_count };
        if (decoded.size() > expected_size)
        {
            decoded.resize(expected_size);
        }

        return decoded;
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

    std::string console_string_to_u8string(std::string_view input)
    {
#ifdef _WIN32
        if (input.empty())
        {
            return std::string{};
        }

        const UINT cp{ GetConsoleOutputCP() };
        if (cp == CP_UTF8)
        {
            return std::string{ input };
        }

        const int wstring_length{ MultiByteToWideChar(cp, 0, input.data(), static_cast<int>(input.size()), nullptr, 0) };
        if (wstring_length <= 0)
        {
            return std::string{ input };
        }

        std::wstring wstring(wstring_length, L'\0');
        MultiByteToWideChar(cp, 0, input.data(), static_cast<int>(input.size()), wstring.data(), wstring_length);

        const int u8string_length{ WideCharToMultiByte(CP_UTF8, 0, wstring.data(), wstring_length, nullptr, 0, nullptr, nullptr) };
        if (u8string_length <= 0)
        {
            return std::string{ input };
        }

        std::string u8_string(u8string_length, '\0');
        WideCharToMultiByte(CP_UTF8, 0, wstring.data(), wstring_length, u8_string.data(), u8string_length, nullptr, nullptr);

        return u8_string;
#else
        return std::string{ input };
#endif
    }

    template<typename Integer>
    Integer random(Integer min, Integer max)
    {
        static std::random_device seed_gen;
        static std::default_random_engine random_engine{ seed_gen() };
        static std::uniform_int_distribution<Integer> distribution{ min, max };
        return distribution(random_engine);
    }

    namespace builtin
    {
        using macro_type = std::function<primitive_type(const std::vector<primitive_type>&, const config&, context&)>;
        std::optional<macro_type> get_macro(std::string_view name);

        primitive_type int_(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx);
        primitive_type double_(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx);
        primitive_type char_(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx);
        primitive_type string_(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx);

        primitive_type file(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx);
        primitive_type head(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx);
        primitive_type tail(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx);
        primitive_type head_tail(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx);
        primitive_type json_literal(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx);
        primitive_type env(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx);
        primitive_type generated(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx);
        primitive_type random(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx);
        primitive_type choice(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx);
        primitive_type exec(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx);
        primitive_type code_block(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx);
        primitive_type summary(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx);

        std::string date();
        std::string time();
        std::string datetime();
        std::string stdin_(const config& cfg);
    } // namespace builtin

    namespace parser
    {
        enum class assignment_operator
        {
            assign,             // =
            plus_assign,        // +=
            minus_assign,       // -=
            multiplies_assign,  // *=
            divides_assign,     // /=
            modulus_assign,     // %=
            shift_left_assign,  // <<=
            shift_right_assign, // >>=
            and_assign,         // &=
            xor_assign,         // ^=
            or_assign           // |=
        };

        enum class equality_operator
        {
            equal,    // ==
            not_equal // !=
        };

        enum class relational_operator
        {
            less,         // <
            greater,      // >
            less_equal,   // <=
            greater_equal // >=
        };

        enum class shift_operator
        {
            shift_left, // <<
            shift_right // >>
        };

        enum class additive_operator
        {
            plus, // +
            minus // -
        };

        enum class multiplicative_operator
        {
            multiplies, // *
            divides,    // /
            modulus     // %
        };

        enum class prefix_operator
        {
            prefix_increment, // ++a
            prefix_decrement, // --a
            prefix_plus,      // +a
            prefix_minus,     // -a
            logical_not,      // !
            bitwise_not       // ~
        };

        enum class suffix_operator
        {
            suffix_increment, // ++
            suffix_decrement  // --
        };

        struct variable_type
        {
            std::string name;
        };
        using primary_type = boost::variant<primitive_type, variable_type>;

        template<typename Operand, typename Operator>
        struct operator_operand_pair
        {
            using operand_type = Operand;
            using operator_type = Operator;
            operator_type operator_;
            operand_type operand;
        };

        template<typename LowerExpression, typename Operator>
        struct basic_variadic_binary_expression
        {
            using lower_expression_type = LowerExpression;
            using operator_type = Operator;
            lower_expression_type first;
            std::vector<operator_operand_pair<lower_expression_type, operator_type>> rest;
        };

        struct macro_expression_node_type;
        using macro_expression_type = boost::variant<primary_type, boost::recursive_wrapper<macro_expression_node_type>>;
        struct expression_type;
        using parentheses_expression_type = boost::variant<macro_expression_type, boost::recursive_wrapper<expression_type>>;

        struct suffix_expression_type
        {
            parentheses_expression_type operand;
            std::vector<suffix_operator> operators;
        };

        struct prefix_expression_node_type;
        using prefix_expression_type = boost::make_recursive_variant<suffix_expression_type, prefix_expression_node_type, boost::recursive_variant_>::type;
        struct prefix_expression_node_type
        {
            prefix_operator operator_;
            boost::recursive_wrapper<prefix_expression_type> operand;
        };

        using multiplicative_expression_type = basic_variadic_binary_expression<prefix_expression_type, multiplicative_operator>;
        using additive_expression_type = basic_variadic_binary_expression<multiplicative_expression_type, additive_operator>;
        using shift_expression_type = basic_variadic_binary_expression<additive_expression_type, shift_operator>;
        using relational_expression_type = basic_variadic_binary_expression<shift_expression_type, relational_operator>;
        using equality_expression_type = basic_variadic_binary_expression<relational_expression_type, equality_operator>;
        using and_expression_type = std::vector<equality_expression_type>;
        using xor_expression_type = std::vector<and_expression_type>;
        using or_expression_type = std::vector<xor_expression_type>;
        using logical_and_expression_type = std::vector<or_expression_type>;
        using logical_or_expression_type = std::vector<logical_and_expression_type>;

        struct conditional_expression_node_type;
        using conditional_expression_type = boost::variant<logical_or_expression_type, boost::recursive_wrapper<conditional_expression_node_type>>;
        struct conditional_expression_node_type
        {
            logical_or_expression_type condition;
            boost::recursive_wrapper<expression_type> then_expr;
            conditional_expression_type else_expr;
        };

        struct assignment_expression_node_type;
        using assignment_expression_type = boost::variant<conditional_expression_type, boost::recursive_wrapper<assignment_expression_node_type>>;
        struct assignment_expression_node_type
        {
            conditional_expression_type lhs;
            assignment_operator operator_;
            assignment_expression_type rhs;
        };

        struct macro_expression_node_type
        {
            std::string name;
            std::vector<assignment_expression_type> arguments;
        };

        struct expression_type
        {
            std::vector<assignment_expression_type> expressions;
            bool terminated{};
        };

        struct statement_type
            : std::vector<expression_type>
        {
            using std::vector<expression_type>::vector;
        };

        struct placeholder_type
        {
            expression_type expression;
        };

        using node_type = boost::variant<std::string, placeholder_type>;

        struct assignment_symbols
            : boost::spirit::qi::symbols<char, assignment_operator>
        {
            assignment_symbols()
            {
                add
                ("=", assignment_operator::assign)
                    ("+=", assignment_operator::plus_assign)
                    ("-=", assignment_operator::minus_assign)
                    ("*=", assignment_operator::multiplies_assign)
                    ("/=", assignment_operator::divides_assign)
                    ("%=", assignment_operator::modulus_assign)
                    ("<<=", assignment_operator::shift_left_assign)
                    (">>=", assignment_operator::shift_right_assign)
                    ("&=", assignment_operator::and_assign)
                    ("^=", assignment_operator::xor_assign)
                    ("|=", assignment_operator::or_assign)
                    ;
            }
        } assignment_operator_;

        struct equality_symbols
            : boost::spirit::qi::symbols<char, equality_operator>
        {
            equality_symbols()
            {
                add
                ("==", equality_operator::equal)
                    ("!=", equality_operator::not_equal)
                    ;
            }
        } equality_operator_;

        struct relational_symbols
            : boost::spirit::qi::symbols<char, relational_operator>
        {
            relational_symbols()
            {
                add
                ("<", relational_operator::less)
                    (">", relational_operator::greater)
                    ("<=", relational_operator::less_equal)
                    (">=", relational_operator::greater_equal)
                    ;
            }
        } relational_operator_;

        struct shift_symbols
            : boost::spirit::qi::symbols<char, shift_operator>
        {
            shift_symbols()
            {
                add
                ("<<", shift_operator::shift_left)
                    (">>", shift_operator::shift_right)
                    ;
            }
        } shift_operator_;

        struct additive_symbols
            : boost::spirit::qi::symbols<char, additive_operator>
        {
            additive_symbols()
            {
                add
                ("+", additive_operator::plus)
                    ("-", additive_operator::minus)
                    ;
            }
        } additive_operator_;

        struct multiplicative_symbols
            : boost::spirit::qi::symbols<char, multiplicative_operator>
        {
            multiplicative_symbols()
            {
                add
                ("*", multiplicative_operator::multiplies)
                    ("/", multiplicative_operator::divides)
                    ("%", multiplicative_operator::modulus)
                    ;
            }
        } multiplicative_operator_;

        struct prefix_symbols
            : boost::spirit::qi::symbols<char, prefix_operator>
        {
            prefix_symbols()
            {
                add
                ("++", prefix_operator::prefix_increment)
                    ("--", prefix_operator::prefix_decrement)
                    ("+", prefix_operator::prefix_plus)
                    ("-", prefix_operator::prefix_minus)
                    ("!", prefix_operator::logical_not)
                    ("~", prefix_operator::bitwise_not)
                    ;
            }
        } prefix_operator_;

        struct suffix_symbols
            : boost::spirit::qi::symbols<char, suffix_operator>
        {
            suffix_symbols()
            {
                add
                ("++", suffix_operator::suffix_increment)
                    ("--", suffix_operator::suffix_decrement)
                    ;
            }
        } suffix_operator_;

        struct escaped_chars
            : boost::spirit::qi::symbols<char, char>
        {
            escaped_chars()
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
            : boost::spirit::qi::grammar<Iterator, std::vector<node_type>()>
        {
            document_grammar()
                : document_grammar::base_type(document)
            {
                namespace qi = boost::spirit::qi;

                using qi::double_;
                using qi::int_;
                using qi::char_;
                using qi::bool_;
                using qi::lexeme;
                using qi::lit;
                using qi::skip;
                using qi::space;
                using qi::matches;

                document = *node;
                node = placeholder | plain_text;
                plain_text = +(!lit("{{") >> char_);
                placeholder = lit("{{") >> skip(space)[expression] >> lit("}}");

                expression = (assignment_expression % lit(';')) >> matches[lit(';')];
                assignment_expression = assignment_expression_node | conditional_expression;
                assignment_expression_node = conditional_expression >> assignment_operator_ >> assignment_expression;
                conditional_expression = conditional_expression_node | logical_or_expression;
                conditional_expression_node = logical_or_expression >> lit('?') >> expression >> lit(':') >> conditional_expression;
                logical_or_expression = logical_and_expression % lit("||");
                logical_and_expression = or_expression % lit("&&");
                or_expression = xor_expression % lit('|');
                xor_expression = and_expression % lit('^');
                and_expression = equality_expression % lit('&');
                equality_expression = relational_expression >> *(equality_operator_ >> relational_expression);
                relational_expression = shift_expression >> *(relational_operator_ >> shift_expression);
                shift_expression = additive_expression >> *(shift_operator_ >> additive_expression);
                additive_expression = multiplicative_expression >> *(additive_operator_ >> multiplicative_expression);
                multiplicative_expression = prefix_expression >> *(multiplicative_operator_ >> prefix_expression);
                prefix_expression = prefix_expression_node | suffix_expression;
                prefix_expression_node = prefix_operator_ >> prefix_expression;
                suffix_expression = parentheses_expression >> *suffix_operator_;
                parentheses_expression = (lit('(') >> expression >> lit(')')) | macro_expression;

                macro_expression = macro_expression_node | primary;
                macro_expression_node = name >> arguments;
                arguments = lit('(') >> -(assignment_expression % ',') >> lit(')');

                primary = variable | primitive;
                variable = name;
                primitive = bool_ | character | int_ | double_ | string;
                name = lexeme[char_("a-zA-Z_") >> *(char_("a-zA-Z0-9_"))];
                character = lexeme['\'' >> (('\\' >> escaped_char) | (char_ - '\'' - '\\')) >> '\''];
                string = lexeme['"' >> *(('\\' >> escaped_char) | (char_ - '"' - '\\')) >> '"'];
            }

            template<typename ... Args>
            using rule = boost::spirit::qi::rule<Iterator, Args ...>;

            template<typename ... Args>
            using skipped_rule = boost::spirit::qi::rule<Iterator, boost::spirit::qi::space_type, Args ...>;

            rule<std::vector<node_type>()> document;
            rule<node_type()> node;
            rule<std::string()> plain_text;
            rule<placeholder_type()> placeholder;

            skipped_rule<expression_type()> expression;
            skipped_rule<assignment_expression_type()> assignment_expression;
            skipped_rule<assignment_expression_node_type()> assignment_expression_node;
            skipped_rule<conditional_expression_type()> conditional_expression;
            skipped_rule<conditional_expression_node_type()> conditional_expression_node;
            skipped_rule<logical_or_expression_type()> logical_or_expression;
            skipped_rule<logical_and_expression_type()> logical_and_expression;
            skipped_rule<or_expression_type()> or_expression;
            skipped_rule<xor_expression_type()> xor_expression;
            skipped_rule<and_expression_type()> and_expression;
            skipped_rule<equality_expression_type()> equality_expression;
            skipped_rule<relational_expression_type()> relational_expression;
            skipped_rule<shift_expression_type()> shift_expression;
            skipped_rule<additive_expression_type()> additive_expression;
            skipped_rule<multiplicative_expression_type()> multiplicative_expression;
            skipped_rule<prefix_expression_type()> prefix_expression;
            skipped_rule<prefix_expression_node_type()> prefix_expression_node;
            skipped_rule<suffix_expression_type()> suffix_expression;
            skipped_rule<parentheses_expression_type()> parentheses_expression;
            skipped_rule<macro_expression_type()> macro_expression;
            skipped_rule<macro_expression_node_type()> macro_expression_node;

            skipped_rule<primary_type()> primary;
            skipped_rule<variable_type()> variable;
            skipped_rule<primitive_type()> primitive;
            skipped_rule<vr_primitive_type()> vr_primitive;
            skipped_rule<std::string()> name;
            skipped_rule<std::vector<assignment_expression_type>()> arguments;
            skipped_rule<char()> character;
            skipped_rule<std::string()> string;
        };

        using grammar = document_grammar<std::string_view::const_iterator>;

        std::string evaluate_document_recursive(std::string input, const config& cfg, unsigned int max_depth, context& ctx);
        std::string evaluate_document(std::string_view document, const config& cfg, const grammar& grammar, context& ctx);
        std::string evaluate_node(const std::vector<node_type>& ast, const config& cfg, const grammar& grammar, context& ctx);

        vr_primitive_type evaluate_expression(const expression_type& expr, const config& cfg, context& ctx);
        vr_primitive_type evaluate_assignment_expression(const assignment_expression_type& expr, const config& cfg, context& ctx);
        vr_primitive_type evaluate_assignment_expression_node(const assignment_expression_node_type& expr, const config& cfg, context& ctx);
        vr_primitive_type evaluate_conditional_expression(const conditional_expression_type& expr, const config& cfg, context& ctx);
        vr_primitive_type evaluate_logical_or_expression(const logical_or_expression_type& expr, const config& cfg, context& ctx);
        vr_primitive_type evaluate_logical_and_expression(const logical_and_expression_type& expr, const config& cfg, context& ctx);
        vr_primitive_type evaluate_or_expression(const or_expression_type& expr, const config& cfg, context& ctx);
        vr_primitive_type evaluate_xor_expression(const xor_expression_type& expr, const config& cfg, context& ctx);
        vr_primitive_type evaluate_and_expression(const and_expression_type& expr, const config& cfg, context& ctx);
        vr_primitive_type evaluate_equality_expression(const equality_expression_type& expr, const config& cfg, context& ctx);
        vr_primitive_type evaluate_relational_expression(const relational_expression_type& expr, const config& cfg, context& ctx);
        vr_primitive_type evaluate_shift_expression(const shift_expression_type& expr, const config& cfg, context& ctx);
        vr_primitive_type evaluate_additive_expression(const additive_expression_type& expr, const config& cfg, context& ctx);
        vr_primitive_type evaluate_multiplicative_expression(const multiplicative_expression_type& expr, const config& cfg, context& ctx);
        vr_primitive_type evaluate_prefix_expression(const prefix_expression_type& expr, const config& cfg, context& ctx);
        vr_primitive_type evaluate_prefix_expression_node(const prefix_expression_node_type& expr, const config& cfg, context& ctx);
        vr_primitive_type evaluate_suffix_expression(const suffix_expression_type& expr, const config& cfg, context& ctx);
        vr_primitive_type evaluate_parentheses_expression(const parentheses_expression_type& expr, const config& cfg, context& ctx);
        vr_primitive_type evaluate_macro_expression(const macro_expression_type& expr, const config& cfg, context& ctx);
        vr_primitive_type evaluate_macro_expression_node(const macro_expression_node_type& expr, const config& cfg, context& ctx);
        vr_primitive_type evaluate_primary(const primary_type& primary, const config& cfg, context& ctx);
        vr_primitive_type evaluate_variable(const variable_type& symbol, const config& cfg, context& ctx);
        vr_primitive_type primitive_ref_to_vr_primitive(primitive_type& primitive);
        vr_primitive_type primitive_val_to_vr_primitive(const primitive_type& primitive);
        primitive_type vr_primitive_to_primitive(const vr_primitive_type& primitive);

        namespace detail
        {
            struct unary_fallback_visitor
            {
                template<typename A>
                primitive_type operator()(A&&) const
                {
                    llmcpp::throw_exception(macro_exception{});
                }
            };

            struct binary_fallback_visitor
            {
                template<typename A, typename B>
                primitive_type operator()(A&&, B&&) const
                {
                    llmcpp::throw_exception(macro_exception{});
                }
            };

            template<typename Result>
            struct static_cast_impl
            {
                template<typename A>
                    requires requires(const A& a) { static_cast<Result>(a); }
                Result operator ()(const A& a) const
                {
                    return static_cast<Result>(a);
                }
            };

            template<typename Result>
            struct static_cast_
                : static_cast_impl<Result>
            {
            };

            template<>
            struct static_cast_<bool>
                : static_cast_impl<bool>
            {
                using static_cast_impl<bool>::operator();

                bool operator()(const std::string& s) const
                {
                    return !s.empty();
                }

                [[noreturn]] bool operator()(const undefined_variable_type&) const
                {
                    llmcpp::throw_exception(macro_exception{} << error_info::description{ "Boolean cast of an undefined variable" });
                }
            };
        } // namespace detail

        template<typename T>
        concept bitwise_operable = !std::same_as<T, bool>&& requires(T a, T b, int shift)
        {
            { ~a } -> std::same_as<T>;
            { a& b } -> std::same_as<T>;
            { a | b } -> std::same_as<T>;
            { a^ b } -> std::same_as<T>;
            { a << shift } -> std::same_as<T>;
            { a >> shift } -> std::same_as<T>;
        };

        template<typename A, typename B>
        concept safe_equality_comparable_with = requires(const A & a, const B & b)
        {
            { a == b } -> std::convertible_to<bool>;
        } && !(std::same_as<std::decay_t<A>, bool>^ std::same_as<std::decay_t<B>, bool>);

        template<typename T>
        concept safe_equality_comparable = requires(const T & a, const T & b)
        {
            { a == b } -> std::convertible_to<bool>;
        };

        template<typename A, typename B>
        concept safe_totally_ordered_with
            = std::totally_ordered_with<A, B>
            && !(std::same_as<std::decay_t<A>, bool>^ std::same_as<std::decay_t<B>, bool>);

        template<typename A>
        concept has_safe_unary_plus_minus = requires(const A & a)
        {
            { +a };
            { -a };
        } && !std::same_as<std::decay_t<A>, bool>;

        template<typename A>
        concept has_safe_logical_not = requires(const A & a)
        {
            { !a };
        };

        template<typename A>
        concept has_safe_bitwise_not = requires(const A & a)
        {
            { ~a };
        } && !std::same_as<std::decay_t<A>, bool>;

#define LLMCPP_DEFINE_FUNCTION(opecode)                                                       \
        struct opecode                                                                        \
        {                                                                                     \
            template<typename ... Args>                                                       \
            decltype(auto) operator()(Args&& ... args) const                                  \
            {                                                                                 \
                return boost::apply_visitor(detail::opecode{}, std::forward<Args>(args) ...); \
            }                                                                                 \
        };

        namespace detail
        {
            struct assign
            {
                context& ctx;
                assign(context& ctx)
                    :ctx{ ctx }
                {
                }

                template<typename A, typename B>
                vr_primitive_type operator ()(A& a, const B& b) const
                {
                    using A_ = std::decay_t<unwrap_type_t<A>>;
                    using B_ = std::decay_t<unwrap_type_t<B>>;
                    if constexpr (std::is_same_v<A_, undefined_variable_type> && !std::is_same_v<B_, undefined_variable_type>)
                    {
                        const primitive_type value{ unwrap(b) };
                        ctx.set(a.name, value);
                        if (primitive_type* ptr{ ctx.get(a.name) }; ptr)
                        {
                            return primitive_ref_to_vr_primitive(*ptr);
                        }
                    }
                    else if constexpr (std::is_same_v<A_, B_> && !std::is_same_v<A_, bool>)
                    {
                        if constexpr (requires { unwrap(a) = unwrap(b); })
                        {
                            return unwrap(a) = unwrap(b);
                        }
                    }
                    else if constexpr (
                        std::is_arithmetic_v<A_>
                        && std::is_arithmetic_v<B_>
                        && !std::is_same_v<A_, bool>
                        && !std::is_same_v<B_, bool>)
                    {
                        if constexpr (requires { unwrap(a) = static_cast<A_>(unwrap(b)); })
                        {
                            return unwrap(a) = static_cast<A_>(unwrap(b));
                        }
                    }
                    llmcpp::throw_exception(macro_exception{});
                }
            };
        }

        struct assign
        {
            context& ctx;
            assign(context& ctx)
                :ctx{ ctx }
            {
            }

            vr_primitive_type operator()(vr_primitive_type& a, const vr_primitive_type& b) const
            {
                return boost::apply_visitor(detail::assign{ ctx }, a, b);
            }
        };

#define LLMCPP_DEFINE_FUNCTION_OBJECT(operator_, opecode, zero_check)                                   \
            namespace detail                                                                            \
            {                                                                                           \
                struct opecode                                                                          \
                {                                                                                       \
                    template<typename A, typename B>                                                    \
                    vr_primitive_type operator ()(A& a, const B& b) const                               \
                    {                                                                                   \
                        using A_ = std::decay_t<unwrap_type_t<A>>;                                      \
                        using B_ = std::decay_t<unwrap_type_t<B>>;                                      \
                        if constexpr (std::is_same_v<A_, B_> && !std::is_same_v<A_, bool>)              \
                        {                                                                               \
                            if constexpr (requires { unwrap(a) operator_ unwrap(b); })                  \
                            {                                                                           \
                                if constexpr (zero_check)                                               \
                                {                                                                       \
                                    if (unwrap(b) == B_{})                                              \
                                    {                                                                   \
                                        llmcpp::throw_exception(macro_exception{});                     \
                                    }                                                                   \
                                }                                                                       \
                                return unwrap(a) operator_ unwrap(b);                                   \
                            }                                                                           \
                        }                                                                               \
                        else if constexpr (                                                             \
                            std::is_arithmetic_v<A_>                                                    \
                            && std::is_arithmetic_v<B_>                                                 \
                            && !std::is_same_v<A_, bool>                                                \
                            && !std::is_same_v<B_, bool>)                                               \
                        {                                                                               \
                            if constexpr (requires { unwrap(a) operator_ static_cast<A_>(unwrap(b)); }) \
                            {                                                                           \
                                return unwrap(a) operator_ static_cast<A_>(unwrap(b));                  \
                            }                                                                           \
                        }                                                                               \
                        llmcpp::throw_exception(macro_exception{});                                     \
                    }                                                                                   \
                };                                                                                      \
            }                                                                                           \
                                                                                                        \
            LLMCPP_DEFINE_FUNCTION(opecode);

        LLMCPP_DEFINE_FUNCTION_OBJECT(+=, plus_assign, false);
        LLMCPP_DEFINE_FUNCTION_OBJECT(-=, minus_assign, false);
        LLMCPP_DEFINE_FUNCTION_OBJECT(*=, multiplies_assign, false);
        LLMCPP_DEFINE_FUNCTION_OBJECT(<<=, shift_left_assign, false);
        LLMCPP_DEFINE_FUNCTION_OBJECT(>>=, shift_right_assign, false);
        LLMCPP_DEFINE_FUNCTION_OBJECT(&=, and_assign, false);
        LLMCPP_DEFINE_FUNCTION_OBJECT(^=, xor_assign, false);
        LLMCPP_DEFINE_FUNCTION_OBJECT(|=, or_assign, false);
        LLMCPP_DEFINE_FUNCTION_OBJECT(/=, divides_assign, true);
        LLMCPP_DEFINE_FUNCTION_OBJECT(%=, modulus_assign, true);

#undef LLMCPP_DEFINE_FUNCTION_OBJECT

#define LLMCPP_DEFINE_FUNCTION_OBJECT(operator_, opecode)                                                   \
        namespace detail                                                                                    \
        {                                                                                                   \
            struct opecode                                                                                  \
            {                                                                                               \
                template<typename A, typename B>                                                            \
                    requires (bitwise_operable<unwrap_type_t<A>> && bitwise_operable<unwrap_type_t<B>>)     \
                vr_primitive_type operator ()(const A& a, const B& b) const                                 \
                {                                                                                           \
                    return unwrap(a) operator_ unwrap(b);                                                   \
                }                                                                                           \
                template<typename A, typename B>                                                            \
                    requires (!(bitwise_operable<unwrap_type_t<A>> && bitwise_operable<unwrap_type_t<B>>))  \
                [[noreturn]] vr_primitive_type operator ()(const A& a, const B& b) const                    \
                {                                                                                           \
                    llmcpp::throw_exception(macro_exception{});                                             \
                }                                                                                           \
            };                                                                                              \
        }                                                                                                   \
        LLMCPP_DEFINE_FUNCTION(opecode);

        //LLMCPP_DEFINE_FUNCTION_OBJECT(|| , logical_or);
        //LLMCPP_DEFINE_FUNCTION_OBJECT(&&, logical_and);
        LLMCPP_DEFINE_FUNCTION_OBJECT(| , or_);
        LLMCPP_DEFINE_FUNCTION_OBJECT(^, xor_);
        LLMCPP_DEFINE_FUNCTION_OBJECT(&, and_);
        LLMCPP_DEFINE_FUNCTION_OBJECT(<< , shift_left);
        LLMCPP_DEFINE_FUNCTION_OBJECT(>> , shift_right);
#undef LLMCPP_DEFINE_FUNCTION_OBJECT

#define LLMCPP_DEFINE_FUNCTION_OBJECT(operator_, opecode)                                           \
        namespace detail                                                                            \
        {                                                                                           \
            struct opecode                                                                          \
            {                                                                                       \
                template<typename A, typename B>                                                    \
                    requires (safe_equality_comparable_with<unwrap_type_t<A>, unwrap_type_t<B>>)    \
                vr_primitive_type operator ()(const A& a, const B& b) const                         \
                {                                                                                   \
                    return unwrap(a) operator_ unwrap(b);                                           \
                }                                                                                   \
                template<typename A, typename B>                                                    \
                    requires (!(safe_equality_comparable_with<unwrap_type_t<A>, unwrap_type_t<B>>)) \
                [[noreturn]] vr_primitive_type operator ()(const A& a, const B& b) const            \
                {                                                                                   \
                    llmcpp::throw_exception(macro_exception{});                                     \
                }                                                                                   \
            };                                                                                      \
        }                                                                                           \
        LLMCPP_DEFINE_FUNCTION(opecode);

        LLMCPP_DEFINE_FUNCTION_OBJECT(== , equal);
        LLMCPP_DEFINE_FUNCTION_OBJECT(!= , not_equal);
#undef LLMCPP_DEFINE_FUNCTION_OBJECT

#define LLMCPP_DEFINE_FUNCTION_OBJECT(operator_, opecode)                                       \
        namespace detail                                                                        \
        {                                                                                       \
            struct opecode                                                                      \
            {                                                                                   \
                template<typename A, typename B>                                                \
                    requires (safe_totally_ordered_with<unwrap_type_t<A>, unwrap_type_t<B>>)    \
                vr_primitive_type operator ()(const A& a, const B& b) const                     \
                {                                                                               \
                    return unwrap(a) operator_ unwrap(b);                                       \
                }                                                                               \
                template<typename A, typename B>                                                \
                    requires (!(safe_totally_ordered_with<unwrap_type_t<A>, unwrap_type_t<B>>)) \
                vr_primitive_type operator ()(const A& a, const B& b) const                     \
                {                                                                               \
                    llmcpp::throw_exception(macro_exception{});                                 \
                }                                                                               \
            };                                                                                  \
        }                                                                                       \
        LLMCPP_DEFINE_FUNCTION(opecode);

        LLMCPP_DEFINE_FUNCTION_OBJECT(< , less);
        LLMCPP_DEFINE_FUNCTION_OBJECT(> , greater);
        LLMCPP_DEFINE_FUNCTION_OBJECT(<= , less_equal);
        LLMCPP_DEFINE_FUNCTION_OBJECT(>= , greater_equal);
#undef LLMCPP_DEFINE_FUNCTION_OBJECT

#define LLMCPP_DEFINE_FUNCTION_OBJECT(operator_, opecode, zero_check)                                   \
        namespace detail                                                                                \
        {                                                                                               \
            struct opecode                                                                              \
            {                                                                                           \
                template<typename A, typename B>                                                        \
                static constexpr bool operable =                                                        \
                    requires(const A& a, const B& b) { unwrap(a) operator_ unwrap(b); }                 \
                    && !std::same_as<std::decay_t<unwrap_type_t<A>>, bool>                              \
                    && !std::same_as<std::decay_t<unwrap_type_t<B>>, bool>;                             \
                template<typename A, typename B>                                                        \
                    requires (operable<A, B>)                                                           \
                vr_primitive_type operator ()(const A& a, const B& b) const                             \
                {                                                                                       \
                    if constexpr (zero_check)                                                           \
                    {                                                                                   \
                        using B_ = std::decay_t<unwrap_type_t<B>>;                                      \
                        if constexpr (safe_equality_comparable<B_>)                                     \
                        {                                                                               \
                            if (unwrap(b) == B_{})                                                      \
                            {                                                                           \
                                llmcpp::throw_exception(macro_exception{});                             \
                            }                                                                           \
                        }                                                                               \
                    }                                                                                   \
                    return unwrap(a) operator_ unwrap(b);                                               \
                }                                                                                       \
                template<typename A, typename B>                                                        \
                    requires (!operable<A, B>)                                                          \
                [[noreturn]] vr_primitive_type operator ()(const A& a, const B& b) const                \
                {                                                                                       \
                    llmcpp::throw_exception(macro_exception{});                                         \
                }                                                                                       \
            };                                                                                          \
        }                                                                                               \
        LLMCPP_DEFINE_FUNCTION(opecode);

        LLMCPP_DEFINE_FUNCTION_OBJECT(+, plus, false);
        LLMCPP_DEFINE_FUNCTION_OBJECT(-, minus, false);
        LLMCPP_DEFINE_FUNCTION_OBJECT(*, multiplies, false);
        LLMCPP_DEFINE_FUNCTION_OBJECT(/ , divides, true);
        LLMCPP_DEFINE_FUNCTION_OBJECT(%, modulus, true);
#undef LLMCPP_DEFINE_FUNCTION_OBJECT

#define LLMCPP_DEFINE_FUNCTION_OBJECT(operator_, opecode)                       \
        namespace detail                                                        \
        {                                                                       \
            struct opecode                                                      \
            {                                                                   \
                template<typename A>                                            \
                    requires requires(A& a) { operator_ unwrap(a); }            \
                vr_primitive_type operator ()(A& a) const                       \
                {                                                               \
                    return operator_ unwrap(a);                                 \
                }                                                               \
                template<typename A>                                            \
                    requires (!requires(A& a) { operator_ unwrap(a); })         \
                [[noreturn]] vr_primitive_type operator ()(A& a) const          \
                {                                                               \
                    llmcpp::throw_exception(macro_exception{});                 \
                }                                                               \
            };                                                                  \
        }                                                                       \
                                                                                \
        LLMCPP_DEFINE_FUNCTION(opecode);

        LLMCPP_DEFINE_FUNCTION_OBJECT(++, prefix_increment);
        LLMCPP_DEFINE_FUNCTION_OBJECT(--, prefix_decrement);
#undef LLMCPP_DEFINE_FUNCTION_OBJECT

#define LLMCPP_DEFINE_FUNCTION_OBJECT(operator_, opecode, concept_name)         \
        namespace detail                                                        \
        {                                                                       \
            struct opecode                                                      \
            {                                                                   \
                template<typename A>                                            \
                    requires (concept_name<unwrap_type_t<A>>)                   \
                vr_primitive_type operator ()(const A& a) const                 \
                {                                                               \
                    return operator_ unwrap(a);                                 \
                }                                                               \
                template<typename A>                                            \
                    requires (!(concept_name<unwrap_type_t<A>>))                \
                [[noreturn]] vr_primitive_type operator ()(const A& a) const    \
                {                                                               \
                    llmcpp::throw_exception(macro_exception{});                 \
                }                                                               \
            };                                                                  \
        }                                                                       \
                                                                                \
        LLMCPP_DEFINE_FUNCTION(opecode);

        LLMCPP_DEFINE_FUNCTION_OBJECT(+, prefix_plus, has_safe_unary_plus_minus);
        LLMCPP_DEFINE_FUNCTION_OBJECT(-, prefix_minus, has_safe_unary_plus_minus);
        LLMCPP_DEFINE_FUNCTION_OBJECT(!, logical_not, has_safe_logical_not);
        LLMCPP_DEFINE_FUNCTION_OBJECT(~, bitwise_not, has_safe_bitwise_not);
#undef LLMCPP_DEFINE_FUNCTION_OBJECT

#define LLMCPP_DEFINE_FUNCTION_OBJECT(operator_, opecode)                       \
        namespace detail                                                        \
        {                                                                       \
            struct opecode                                                      \
            {                                                                   \
                template<typename A>                                            \
                    requires requires(A & a) { unwrap(a) operator_; }           \
                vr_primitive_type operator ()(A& a) const                       \
                {                                                               \
                    return unwrap(a) operator_;                                 \
                }                                                               \
                template<typename A>                                            \
                    requires (!requires(A & a) { unwrap(a) operator_; })        \
                [[noreturn]] vr_primitive_type operator ()(A& a) const          \
                {                                                               \
                    llmcpp::throw_exception(macro_exception{});                 \
                }                                                               \
            };                                                                  \
        }                                                                       \
                                                                                \
        LLMCPP_DEFINE_FUNCTION(opecode);                                        \

        LLMCPP_DEFINE_FUNCTION_OBJECT(++, suffix_increment);
        LLMCPP_DEFINE_FUNCTION_OBJECT(--, suffix_decrement);

#undef LLMCPP_DEFINE_FUNCTION_OBJECT
#undef LLMCPP_DEFINE_FUNCTION

        template<typename Result>
        struct static_cast_
        {
            template<typename A>
            Result operator ()(const A& a) const
            {
                return boost::apply_visitor(detail::static_cast_<Result>{}, a);
            }
        };

        template<typename ForwardIterator, typename Evaluator, typename BinaryOperator, typename ... Args>
        decltype(auto) accumulate_expression(ForwardIterator first, ForwardIterator last, Evaluator evaluator, BinaryOperator binary_operator, Args && ... args)
        {
            if (first == last)
            {
                llmcpp::throw_exception(logic_error{});
            }
            auto accumulated{ evaluator(*first, std::forward<Args>(args) ...) };
            ++first;
            if (first == last)
            {
                return accumulated;
            }
            for (; first != last; ++first)
            {
                accumulated = binary_operator(accumulated, evaluator(*first, std::forward<Args>(args) ...));
            }
            return accumulated;
        }

        struct evaluation_visitor
        {
            evaluation_visitor(const config& cfg, context& ctx)
                : cfg{ cfg }
                , ctx{ ctx }
            {
            }

            evaluation_visitor(const evaluation_visitor&) = default;
            evaluation_visitor(evaluation_visitor&&) = default;

            const config& cfg;
            context& ctx;
        };

        struct node_visitor
            : evaluation_visitor
            , boost::static_visitor<std::string>
        {
            node_visitor(const config& cfg, context& ctx)
                : evaluation_visitor{ cfg, ctx }
                , boost::static_visitor<std::string>{}
            {
            }

            std::string operator()(const std::string& str) const
            {
                return str;
            }

            std::string operator()(const placeholder_type& value) const
            {
                try
                {
                    const std::string evaluated{ vr_primitive_to_string(evaluate_expression(value.expression, cfg, ctx)) };
                    BOOST_LOG_TRIVIAL(trace) << "Placeholder evaluated (" << evaluated << ")";
                    return evaluated;
                }
                catch (const macro_exception&)
                {
                    BOOST_LOG_TRIVIAL(warning) << "Placeholder evaluation failed";
                    return std::string{};
                }
            }
        };

        struct assignment_expression_visitor
            : boost::static_visitor<vr_primitive_type>
        {
            const config& cfg; context& ctx;
            assignment_expression_visitor(const config& cfg, context& ctx) : cfg{ cfg }, ctx{ ctx } {}

            vr_primitive_type operator()(const assignment_expression_node_type& expr) const
            {
                return evaluate_assignment_expression_node(expr, cfg, ctx);
            }

            vr_primitive_type operator()(const conditional_expression_type& expr) const
            {
                return evaluate_conditional_expression(expr, cfg, ctx);
            }

            vr_primitive_type operator()(const assignment_expression_type& expr) const
            {
                return evaluate_assignment_expression(expr, cfg, ctx);
            }
        };

        struct conditional_expression_visitor
            : evaluation_visitor
            , boost::static_visitor<vr_primitive_type>
        {
            conditional_expression_visitor(const config& cfg, context& ctx)
                : evaluation_visitor{ cfg, ctx }
                , boost::static_visitor<vr_primitive_type>{}
            {
            }

            vr_primitive_type operator()(const conditional_expression_node_type& value) const
            {
                const vr_primitive_type evaluated_condition{ evaluate_logical_or_expression(value.condition, cfg, ctx) };
                if (static_cast_<bool>{}(evaluated_condition))
                {
                    return evaluate_expression(value.then_expr.get(), cfg, ctx);
                }
                return evaluate_conditional_expression(value.else_expr, cfg, ctx);
            }

            vr_primitive_type operator()(const logical_or_expression_type& value) const
            {
                return evaluate_logical_or_expression(value, cfg, ctx);
            }
        };

        struct prefix_expression_visitor
            : evaluation_visitor
            , boost::static_visitor<vr_primitive_type>
        {
            prefix_expression_visitor(const config& cfg, context& ctx)
                : evaluation_visitor{ cfg, ctx }
                , boost::static_visitor<vr_primitive_type>{}
            {
            }

            vr_primitive_type operator()(const prefix_expression_node_type& expr) const
            {
                return evaluate_prefix_expression_node(expr, cfg, ctx);
            }

            vr_primitive_type operator()(const suffix_expression_type& expr) const
            {
                return evaluate_suffix_expression(expr, cfg, ctx);
            }

            vr_primitive_type operator()(const prefix_expression_type& expr) const
            {
                return evaluate_prefix_expression(expr, cfg, ctx);
            }
        };

        struct parentheses_expression_visitor
            : evaluation_visitor
            , boost::static_visitor<vr_primitive_type>
        {
            parentheses_expression_visitor(const config& cfg, context& ctx)
                : evaluation_visitor{ cfg, ctx }
                , boost::static_visitor<vr_primitive_type>{}
            {
            }

            vr_primitive_type operator()(const macro_expression_type& expr) const
            {
                return evaluate_macro_expression(expr, cfg, ctx);
            }

            vr_primitive_type operator()(const expression_type& expr) const
            {
                return evaluate_expression(expr, cfg, ctx);
            }
        };

        struct primary_visitor
            : evaluation_visitor
            , boost::static_visitor<vr_primitive_type>
        {
            primary_visitor(const config& cfg, context& ctx)
                : evaluation_visitor{ cfg, ctx }
                , boost::static_visitor<vr_primitive_type>{}
            {
            }

            vr_primitive_type operator()(const variable_type& variable) const
            {
                return evaluate_variable(variable, cfg, ctx);
            }

            vr_primitive_type operator()(const primitive_type& primitive) const
            {
                return primitive;
            }
        };

        struct macro_expression_visitor
            : evaluation_visitor
            , boost::static_visitor<vr_primitive_type>
        {
            macro_expression_visitor(const config& cfg, context& ctx)
                : evaluation_visitor{ cfg, ctx }
                , boost::static_visitor<vr_primitive_type>{}
            {
            }

            vr_primitive_type operator()(const macro_expression_node_type& expr) const
            {
                return evaluate_macro_expression_node(expr, cfg, ctx);
            }

            vr_primitive_type operator()(const primary_type& primary) const
            {
                return evaluate_primary(primary, cfg, ctx);
            }
        };

        struct primitive_ref_to_vr_primitive_visitor
            : boost::static_visitor<vr_primitive_type>
        {
            primitive_ref_to_vr_primitive_visitor() {}

            template<typename T>
            vr_primitive_type operator()(T& value) const
            {
                return std::ref(value);
            }
        };

        struct primitive_val_to_vr_primitive_visitor
            : boost::static_visitor<vr_primitive_type>
        {
            primitive_val_to_vr_primitive_visitor() {}

            template<typename T>
            vr_primitive_type operator()(const T& value) const
            {
                return value;
            }
        };

        struct vr_primitive_to_primitive_visitor
            : boost::static_visitor<primitive_type>
        {
            template<typename T>
            primitive_type operator()(const T& value) const
            {
                return unwrap(value);
            }

            [[noreturn]] primitive_type operator()(const undefined_variable_type& undefined_variable) const
            {
                llmcpp::throw_exception(macro_exception{} << error_info::description{ "Undefined variable" });
            }
        };
    } // namespace paraser
} // namespace llmcpp

BOOST_FUSION_ADAPT_STRUCT(
    llmcpp::parser::suffix_expression_type,
    operand, operators
)

BOOST_FUSION_ADAPT_STRUCT(
    llmcpp::parser::prefix_expression_node_type,
    operator_, operand
)

BOOST_FUSION_ADAPT_TPL_STRUCT(
    (Operand)(Operator),
    (llmcpp::parser::operator_operand_pair)(Operand)(Operator),
    operator_, operand
)

BOOST_FUSION_ADAPT_TPL_STRUCT(
    (LowerExpression)(Operator),
    (llmcpp::parser::basic_variadic_binary_expression)(LowerExpression)(Operator),
    first, rest
)

BOOST_FUSION_ADAPT_STRUCT(
    llmcpp::parser::conditional_expression_node_type,
    condition, then_expr, else_expr
);

BOOST_FUSION_ADAPT_STRUCT(
    llmcpp::parser::macro_expression_node_type,
    name, arguments
);

BOOST_FUSION_ADAPT_STRUCT(
    llmcpp::parser::variable_type,
    name
);

BOOST_FUSION_ADAPT_STRUCT(
    llmcpp::parser::assignment_expression_node_type,
    lhs, operator_, rhs
)

BOOST_FUSION_ADAPT_STRUCT(
    llmcpp::parser::expression_type,
    expressions, terminated
)

namespace llmcpp
{
    std::string expand_macro(std::string_view input, const config& cfg, const context& ctx);

    namespace parser
    {
        std::string evaluate_document_recursive(std::string input, const config& cfg, unsigned int max_depth, context& ctx)
        {
            grammar grammar;

            unsigned int depth{};
            for (; depth < max_depth; ++depth)
            {
                if (input.find("{{") == std::string_view::npos)
                {
                    return input;
                }

                std::string evaluated{ evaluate_document(input, cfg, grammar, ctx) };

                if (evaluated == input)
                {
                    break;
                }

                input = std::move(evaluated);
            }

            if (depth >= max_depth)
            {
                llmcpp::throw_exception(macro_exception{} << error_info::description{ "Maximum recursion depth reached" });
            }

            return input;
        }

        std::string evaluate_document(std::string_view document, const config& cfg, const grammar& grammar, context& ctx)
        {
            namespace qi = boost::spirit::qi;

            std::vector<node_type> ast;

            grammar::iterator_type iter{ document.begin() };
            grammar::iterator_type end{ document.end() };

            if (qi::parse(iter, end, grammar, ast) && iter == end)
            {
                return evaluate_node(ast, cfg, grammar, ctx);
            }
            else
            {
                std::ostringstream description;
                description << "Parse failed at: " << std::string{ iter, end };
                llmcpp::throw_exception(macro_exception{} << error_info::description{ description.str() });
            }
        }

        std::string evaluate_node(const std::vector<node_type>& ast, const config& cfg, const grammar& grammar, context& ctx)
        {
            std::string result;
            for (const node_type& node : ast)
            {
                result += boost::apply_visitor(node_visitor{ cfg, ctx }, node);
            }
            return result;
        }

        vr_primitive_type evaluate_expression(const expression_type& expr, const config& cfg, context& ctx)
        {
            if (expr.expressions.empty())
            {
                llmcpp::throw_exception(macro_exception{});
            }
            vr_primitive_type last{};
            for (const auto& assignment_expression : expr.expressions)
            {
                last = evaluate_assignment_expression(assignment_expression, cfg, ctx);
            }
            if (expr.terminated)
            {
                return std::string{};
            }
            return last;
        }

        vr_primitive_type evaluate_assignment_expression(const assignment_expression_type& expr, const config& cfg, context& ctx)
        {
            return boost::apply_visitor(assignment_expression_visitor{ cfg, ctx }, expr);
        }

        vr_primitive_type evaluate_assignment_expression_node(const assignment_expression_node_type& expr, const config& cfg, context& ctx)
        {
            vr_primitive_type lhs{ evaluate_conditional_expression(expr.lhs, cfg, ctx) };
            vr_primitive_type rhs{ evaluate_assignment_expression(expr.rhs, cfg, ctx) };

            switch (expr.operator_)
            {
            case assignment_operator::assign:
                assign{ ctx }(lhs, rhs);
                break;
            case assignment_operator::plus_assign:
                plus_assign{}(lhs, rhs);
                break;
            case assignment_operator::minus_assign:
                minus_assign{}(lhs, rhs);
                break;
            case assignment_operator::multiplies_assign:
                multiplies_assign{}(lhs, rhs);
                break;
            case assignment_operator::divides_assign:
                divides_assign{}(lhs, rhs);
                break;
            case assignment_operator::modulus_assign:
                modulus_assign{}(lhs, rhs);
                break;
            case assignment_operator::shift_left_assign:
                shift_left_assign{}(lhs, rhs);
                break;
            case assignment_operator::shift_right_assign:
                shift_right_assign{}(lhs, rhs);
                break;
            case assignment_operator::and_assign:
                and_assign{}(lhs, rhs);
                break;
            case assignment_operator::xor_assign:
                xor_assign{}(lhs, rhs);
                break;
            case assignment_operator::or_assign:
                or_assign{}(lhs, rhs);
                break;
            default:
                llmcpp::throw_exception(logic_error{});
            }

            return lhs;
        }

        vr_primitive_type evaluate_conditional_expression(const conditional_expression_type& expr, const config& cfg, context& ctx)
        {
            return boost::apply_visitor(conditional_expression_visitor{ cfg, ctx }, expr);
        }

        vr_primitive_type evaluate_logical_or_expression(const logical_or_expression_type& expr, const config& cfg, context& ctx)
        {
            if (expr.empty())
            {
                llmcpp::throw_exception(logic_error{});
            }
            vr_primitive_type lhs{ evaluate_logical_and_expression(expr.front(), cfg, ctx) };
            if (expr.size() == 1)
            {
                return lhs;
            }
            if (static_cast_<bool>{}(lhs))
            {
                return true;
            }
            for (auto iter{ expr.begin() + 1 }; iter != expr.end(); ++iter)
            {
                const auto& rhs = *iter;
                lhs = evaluate_logical_and_expression(rhs, cfg, ctx);
                if (static_cast_<bool>{}(lhs))
                {
                    return true;
                }
            }
            return false;
        }

        vr_primitive_type evaluate_logical_and_expression(const logical_and_expression_type& expr, const config& cfg, context& ctx)
        {
            if (expr.empty())
            {
                llmcpp::throw_exception(logic_error{});
            }
            vr_primitive_type lhs{ evaluate_or_expression(expr.front(), cfg, ctx) };
            if (expr.size() == 1)
            {
                return lhs;
            }
            if (!static_cast_<bool>{}(lhs))
            {
                return false;
            }
            for (auto iter{ expr.begin() + 1 }; iter != expr.end(); ++iter)
            {
                const auto& rhs = *iter;
                lhs = evaluate_or_expression(rhs, cfg, ctx);
                if (!static_cast_<bool>{}(lhs))
                {
                    return false;
                }
            }
            return true;
        }

        vr_primitive_type evaluate_or_expression(const or_expression_type& expr, const config& cfg, context& ctx)
        {
            return accumulate_expression(expr.begin(), expr.end(), evaluate_xor_expression, or_{}, cfg, ctx);
        }

        vr_primitive_type evaluate_xor_expression(const xor_expression_type& expr, const config& cfg, context& ctx)
        {
            return accumulate_expression(expr.begin(), expr.end(), evaluate_and_expression, xor_{}, cfg, ctx);
        }

        vr_primitive_type evaluate_and_expression(const and_expression_type& expr, const config& cfg, context& ctx)
        {
            return accumulate_expression(expr.begin(), expr.end(), evaluate_equality_expression, and_{}, cfg, ctx);
        }

        vr_primitive_type evaluate_equality_expression(const equality_expression_type& expr, const config& cfg, context& ctx)
        {
            vr_primitive_type lhs{ evaluate_relational_expression(expr.first, cfg, ctx) };
            for (const auto& [operator_, operand] : expr.rest)
            {
                const vr_primitive_type rhs{ evaluate_relational_expression(operand, cfg, ctx) };
                switch (operator_)
                {
                case equality_operator::equal:
                    lhs = equal{}(lhs, rhs);
                    break;
                case equality_operator::not_equal:
                    lhs = not_equal{}(lhs, rhs);
                    break;
                default:
                    llmcpp::throw_exception(logic_error{});
                }
            }
            return lhs;
        }

        vr_primitive_type evaluate_relational_expression(const relational_expression_type& expr, const config& cfg, context& ctx)
        {
            vr_primitive_type lhs{ evaluate_shift_expression(expr.first, cfg, ctx) };
            for (const auto& [operator_, operand] : expr.rest)
            {
                const vr_primitive_type rhs{ evaluate_shift_expression(operand, cfg, ctx) };
                switch (operator_)
                {
                case relational_operator::less:
                    lhs = less{}(lhs, rhs);
                    break;
                case relational_operator::greater:
                    lhs = greater{}(lhs, rhs);
                    break;
                case relational_operator::less_equal:
                    lhs = less_equal{}(lhs, rhs);
                    break;
                case relational_operator::greater_equal:
                    lhs = greater_equal{}(lhs, rhs);
                    break;
                default:
                    llmcpp::throw_exception(logic_error{});
                }
            }
            return lhs;
        }

        vr_primitive_type evaluate_shift_expression(const shift_expression_type& expr, const config& cfg, context& ctx)
        {
            vr_primitive_type lhs{ evaluate_additive_expression(expr.first, cfg, ctx) };
            for (const auto& [operator_, operand] : expr.rest)
            {
                const vr_primitive_type rhs{ evaluate_additive_expression(operand, cfg, ctx) };
                switch (operator_)
                {
                case shift_operator::shift_left:
                    lhs = shift_left{}(lhs, rhs);
                    break;
                case shift_operator::shift_right:
                    lhs = shift_right{}(lhs, rhs);
                    break;
                default:
                    llmcpp::throw_exception(logic_error{});
                }
            }
            return lhs;
        }

        vr_primitive_type evaluate_additive_expression(const additive_expression_type& expr, const config& cfg, context& ctx)
        {
            vr_primitive_type lhs{ evaluate_multiplicative_expression(expr.first, cfg, ctx) };
            for (const auto& [operator_, operand] : expr.rest)
            {
                const vr_primitive_type rhs{ evaluate_multiplicative_expression(operand, cfg, ctx) };
                switch (operator_)
                {
                case additive_operator::plus:
                    lhs = plus{}(lhs, rhs);
                    break;
                case additive_operator::minus:
                    lhs = minus{}(lhs, rhs);
                    break;
                default:
                    llmcpp::throw_exception(logic_error{});
                }
            }
            return lhs;
        }

        vr_primitive_type evaluate_multiplicative_expression(const multiplicative_expression_type& expr, const config& cfg, context& ctx)
        {
            vr_primitive_type lhs{ evaluate_prefix_expression(expr.first, cfg, ctx) };
            for (const auto& [operator_, operand] : expr.rest)
            {
                const vr_primitive_type rhs{ evaluate_prefix_expression(operand, cfg, ctx) };
                switch (operator_)
                {
                case multiplicative_operator::multiplies:
                    lhs = multiplies{}(lhs, rhs);
                    break;
                case multiplicative_operator::divides:
                    lhs = divides{}(lhs, rhs);
                    break;
                case multiplicative_operator::modulus:
                    lhs = modulus{}(lhs, rhs);
                    break;
                default:
                    llmcpp::throw_exception(logic_error{});
                }
            }
            return lhs;
        }

        vr_primitive_type evaluate_prefix_expression(const prefix_expression_type& expr, const config& cfg, context& ctx)
        {
            return boost::apply_visitor(prefix_expression_visitor{ cfg, ctx }, expr);
        }

        vr_primitive_type evaluate_prefix_expression_node(const prefix_expression_node_type& expr, const config& cfg, context& ctx)
        {
            vr_primitive_type operand{ evaluate_prefix_expression(expr.operand.get(), cfg, ctx) };
            switch (expr.operator_)
            {
            case prefix_operator::prefix_increment:
                return prefix_increment{}(operand);
            case prefix_operator::prefix_decrement:
                return prefix_decrement{}(operand);
            case prefix_operator::prefix_plus:
                return prefix_plus{}(operand);
            case prefix_operator::prefix_minus:
                return prefix_minus{}(operand);
            case prefix_operator::logical_not:
                return logical_not{}(operand);
            case prefix_operator::bitwise_not:
                return bitwise_not{}(operand);
            default:
                llmcpp::throw_exception(logic_error{});
            }
        }

        vr_primitive_type evaluate_suffix_expression(const suffix_expression_type& expr, const config& cfg, context& ctx)
        {
            vr_primitive_type operand{ evaluate_parentheses_expression(expr.operand, cfg, ctx) };
            for (const auto& operator_ : expr.operators)
            {
                switch (operator_)
                {
                case suffix_operator::suffix_increment:
                    return prefix_increment{}(operand);
                case suffix_operator::suffix_decrement:
                    return prefix_decrement{}(operand);
                default:
                    llmcpp::throw_exception(logic_error{});
                }
            }
            return operand;
        }

        vr_primitive_type evaluate_parentheses_expression(const parentheses_expression_type& expr, const config& cfg, context& ctx)
        {
            return boost::apply_visitor(parentheses_expression_visitor{ cfg, ctx }, expr);
        }

        vr_primitive_type evaluate_macro_expression(const macro_expression_type& expr, const config& cfg, context& ctx)
        {
            return boost::apply_visitor(macro_expression_visitor{ cfg, ctx }, expr);
        }

        vr_primitive_type evaluate_macro_expression_node(const macro_expression_node_type& expr, const config& cfg, context& ctx)
        {
            std::vector<primitive_type> evaluated_args;
            for (const assignment_expression_type& arg : expr.arguments)
            {
                const vr_primitive_type evaluated_arg{ evaluate_assignment_expression(arg, cfg, ctx) };
                evaluated_args.push_back(vr_primitive_to_primitive(evaluated_arg));
            }

            if (const std::optional<builtin::macro_type> macro{ builtin::get_macro(expr.name) }; macro)
            {
                try
                {
                    primitive_type evaluated{ (*macro)(evaluated_args, cfg, ctx) };
                    BOOST_LOG_TRIVIAL(trace) << "Macro evaluated (" << expr.name << " => " << primitive_to_string(evaluated) << ")";
                    return primitive_val_to_vr_primitive(evaluated);
                }
                catch (const boost::exception&)
                {
                    BOOST_LOG_TRIVIAL(warning) << "Evaluation failed (" << expr.name << ")";
                    throw_nested_exception(macro_exception{});
                }
            }

            BOOST_LOG_TRIVIAL(warning) << "Macro not found (" << expr.name << ")";
            llmcpp::throw_exception(macro_exception{});
        }

        vr_primitive_type evaluate_primary(const primary_type& primary, const config& cfg, context& ctx)
        {
            return boost::apply_visitor(primary_visitor{ cfg, ctx }, primary);
        }

        vr_primitive_type evaluate_variable(const variable_type& variable, const config& cfg, context& ctx)
        {
            if (primitive_type* variable_value_ptr{ ctx.get(variable.name) }; variable_value_ptr)
            {
                BOOST_LOG_TRIVIAL(trace) << "Variable found (" << variable.name << "=" << primitive_to_string(*variable_value_ptr) << ")";
                return primitive_ref_to_vr_primitive(*variable_value_ptr);
            }
            BOOST_LOG_TRIVIAL(trace) << "Variable not found (" << variable.name << ")";
            return undefined_variable_type{ variable.name };
        }

        vr_primitive_type primitive_ref_to_vr_primitive(primitive_type& primitive)
        {
            return boost::apply_visitor(primitive_ref_to_vr_primitive_visitor{}, primitive);
        }

        vr_primitive_type primitive_val_to_vr_primitive(const primitive_type& primitive)
        {
            return boost::apply_visitor(primitive_val_to_vr_primitive_visitor{}, primitive);
        }

        primitive_type vr_primitive_to_primitive(const vr_primitive_type& primitive)
        {
            return boost::apply_visitor(vr_primitive_to_primitive_visitor(), primitive);
        }
    }

    std::optional<builtin::macro_type> builtin::get_macro(std::string_view name)
    {
        static const string_unordered_map<macro_type> macros
        {
            {"int", int_},
            {"double", double_},
            {"char", char_},
            {"string", string_},
            {"file", file},
            {"head", head},
            {"tail", tail},
            {"head_tail", head_tail},
            {"json_literal", json_literal},
            {"env", env},
            {"generated", generated},
            {"random", random},
            {"choice", choice},
            {"exec", exec},
            {"code_block", code_block},
            {"summary", summary}
        };

        if (const auto iter{ macros.find(name) }; iter != macros.end())
        {
            return iter->second;
        }

        return std::nullopt;
    }

    template<typename T>
    primitive_type cast_to(const primitive_type argument)
    {
        try
        {
            return boost::apply_visitor([&](const auto& value) { return boost::lexical_cast<T>(value); }, argument);
        }
        catch (const boost::exception&)
        {
            throw_nested_exception(macro_exception{});
        }
    }

    template<typename T>
    primitive_type cast_to(const std::vector<primitive_type>& arguments)
    {
        if (arguments.empty())
        {
            llmcpp::throw_exception(macro_exception{});
        }

        return cast_to<T>(arguments[0]);
    }

    primitive_type builtin::int_(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx)
    {
        return cast_to<int>(arguments);
    }

    primitive_type builtin::double_(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx)
    {
        return cast_to<double>(arguments);
    }

    primitive_type builtin::char_(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx)
    {
        return cast_to<char>(arguments);
    }

    primitive_type builtin::string_(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx)
    {
        return cast_to<std::string>(arguments);
    }

    primitive_type builtin::file(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx)
    {
        if (arguments.empty())
        {
            llmcpp::throw_exception(macro_exception{});
        }

        const std::string_view filename{ get_or_throw<std::string>(arguments[0]) };

        return read_text_file_to_string(filename, cfg);
    }

    primitive_type head_tail_impl(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx, bool reverse)
    {
        if (arguments.size() < 2)
        {
            llmcpp::throw_exception(macro_exception{});
        }

        const std::string_view str{ get_or_throw<std::string>(arguments[0]) };
        const int max_tokens{ get_or_throw<int>(arguments[1]) };

        std::string result;
        int tokens{};
        truncate_by_tokens(str, max_tokens, cfg, reverse, result, tokens);

        return result;
    }

    primitive_type builtin::head(const std::vector< primitive_type>& arguments, const config& cfg, context& ctx)
    {
        return head_tail_impl(arguments, cfg, ctx, false);
    }

    primitive_type builtin::tail(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx)
    {
        return head_tail_impl(arguments, cfg, ctx, true);
    }

    primitive_type builtin::head_tail(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx)
    {
        if (arguments.size() < 3)
        {
            llmcpp::throw_exception(macro_exception{});
        }

        const std::string_view str{ get_or_throw<std::string>(arguments[0]) };
        const int head_max_tokens{ get_or_throw<int>(arguments[1]) };
        const int tail_max_tokens{ get_or_throw<int>(arguments[2]) };

        const std::string_view ellipsis{ "..." };
        const int ellipsis_tokens{ get_tokens_from_cache(cfg, ellipsis) };

        const int total_tokens{ get_tokens_from_cache(cfg, str) };

        if (head_max_tokens + ellipsis_tokens + tail_max_tokens >= total_tokens)
        {
            return std::string{ str };
        }

        std::string result;
        int tokens{};
        truncate_by_tokens(str, head_max_tokens, cfg, false, result, tokens);
        result.append(ellipsis);
        truncate_by_tokens(str, tail_max_tokens, cfg, true, result, tokens);

        return result;
    }

    primitive_type builtin::json_literal(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx)
    {
        if (arguments.size() < 1)
        {
            llmcpp::throw_exception(macro_exception{});
        }

        return json_escape_string(get_or_throw<std::string>(arguments[0]));
    }

    primitive_type builtin::env(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx)
    {
        if (arguments.size() < 1)
        {
            llmcpp::throw_exception(macro_exception{});
        }

        const std::string& name{ get_or_throw<std::string>(arguments[0]) };

        if (const char* env{ boost::nowide::getenv(name.c_str()) }; env)
        {
            return std::string{ env };
        }

        return std::string{};
    }

    primitive_type builtin::generated(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx)
    {
        if (arguments.size() < 1)
        {
            llmcpp::throw_exception(macro_exception{});
        }

        const std::string_view prompt{ get_or_throw<std::string>(arguments[0]) };

        std::string result;
        {
            context pushed{ ctx.make_pushed() };
            result = generate_text(cfg, prompt, pushed);
        }
        return result;
    }

    primitive_type builtin::random(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx)
    {
        const std::optional<int> optional_min{ arguments.size() > 0 ? get_optional<int>(arguments[0]) : std::nullopt };
        const std::optional<int> optional_max{ arguments.size() > 1 ? get_optional<int>(arguments[1]) : std::nullopt };

        const std::int64_t min{ optional_min ? *optional_min : 0 };
        const std::int64_t max{ optional_max ? *optional_max : static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max()) };

        return std::to_string(llmcpp::random<std::int64_t>(min, max));
    }

    primitive_type builtin::choice(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx)
    {
        if (arguments.empty())
        {
            llmcpp::throw_exception(macro_exception{});
        }

        return arguments[llmcpp::random<std::size_t>(0, arguments.size() - 1)];
    }

    primitive_type builtin::exec(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx)
    {
        namespace process = boost::process::v2;
        namespace asio = boost::asio;

        if (arguments.empty())
        {
            llmcpp::throw_exception(macro_exception{});
        }

        const std::string_view exe_name{ get_or_throw<std::string>(arguments[0]) };
        std::vector<std::string> args;
        args.reserve(arguments.size());

        for (std::size_t i{ 1 }; i < arguments.size(); ++i)
        {
            args.push_back(get_or_throw<std::string>(arguments[i]));
        }

        const auto exe_path{ process::environment::find_executable(exe_name) };
        if (exe_path.empty())
        {
            llmcpp::throw_exception(macro_exception{});
        }

        asio::io_context ioctx;
        asio::readable_pipe pipe{ ioctx };
        int exit_code{};

        {
            process::process_stdio pstdio{ nullptr, pipe, {} };
            process::process child{ ioctx, exe_path, args,  pstdio };
            exit_code = child.wait();
        }

        std::string output;
        boost::system::error_code error_code;
        asio::read(pipe, asio::dynamic_buffer(output), error_code);

        if (error_code && error_code != asio::error::eof && error_code != asio::error::broken_pipe)
        {
            llmcpp::throw_exception(macro_exception{} << error_info::system::error_code{ error_code });
        }

        ctx.set("exit_code", exit_code);

        return console_string_to_u8string(output);
    }

    primitive_type builtin::code_block(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx)
    {
        if (arguments.size() < 2)
        {
            llmcpp::throw_exception(macro_exception{});
        }

        const std::string_view markdown{ get_or_throw<std::string>(arguments[0]) };
        const std::string_view code_block{ get_or_throw<std::string>(arguments[1]) };

        const code_blocks map{ extract_code_block_from_markdown(markdown) };
        if (const code_blocks::const_iterator iter{ map.find(code_block) }; iter != map.end())
        {
            return iter->second;
        }

        return std::string{};
    }

    primitive_type builtin::summary(const std::vector<primitive_type>& arguments, const config& cfg, context& ctx)
    {
        if (arguments.size() < 3)
        {
            llmcpp::throw_exception(macro_exception{});
        }

        const std::string_view prompt{ get_or_throw<std::string>(arguments[0]) };
        const std::string& target{ get_or_throw<std::string>(arguments[1]) };
        const int max_token{ get_or_throw<int>(arguments[2]) };

        std::string output;

        {
            context pushed{ ctx.make_pushed() };
            pushed.set("target", target);
            pushed.set("max_token", std::to_string(max_token));
            output = generate_text(cfg, prompt, pushed);
            output = remove_reasoning(output, cfg.llm.reasoning_prefix, cfg.llm.reasoning_suffix);
        }

        std::string truncated;
        int tokens{};
        truncate_by_tokens(output, max_token, cfg, false, truncated, tokens);

        return truncated;
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

    std::string builtin::stdin_(const config& cfg)
    {
#ifdef _WIN32
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

    std::string expand_macro(std::string_view input, const config& cfg, const context& ctx)
    {
        constexpr unsigned int max_depth{ 32 };
        context pushed{ ctx.make_pushed() };
        return parser::evaluate_document_recursive(std::string{ input }, cfg, max_depth, pushed);
    }

    void truncate_by_tokens(std::string_view string, int max_tokens, const config& cfg, bool reverse, std::string& result, int& tokens)
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
                    const int next_tokens{ get_tokens_from_cache(cfg, *first) };
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

    void truncate_prompt(std::string_view string, const config& cfg, bool reverse, std::string& result, int& remaining_tokens)
    {
        std::string truncated;
        int tokens{};
        truncate_by_tokens(string, remaining_tokens, cfg, reverse, truncated, tokens);
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
            llmcpp::throw_exception(file_open_exception{} << error_info::path{ file });
        }
        boost::nowide::ifstream ifs{ file, openmode };
        if (!ifs.is_open())
        {
            llmcpp::throw_exception(file_open_exception{} << error_info::path{ file });
        }
        const std::string file_content{ (std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>() };
        result = file_content;
        return result;
    }

    std::string read_binary_file_to_string(std::string_view file, const config& cfg)
    {
        return read_file_to_string(string_to_path_by_config(file, cfg), std::ios::binary);
    }

    std::string read_text_file_to_string(std::string_view path, const config& cfg, std::string_view extension)
    {
        return read_file_to_string(string_to_path_by_config(complement_extension(path, extension), cfg));
    };

    std::string image_path_to_base64_encoded_string(std::string_view image_path, const config& cfg)
    {
        return base64_encode(read_binary_file_to_string(image_path, cfg));
    }

    std::vector<std::string> image_paths_to_base64_encoded_strings(const std::vector<std::string>& paths, const config& cfg)
    {
        std::vector<std::string> encoded_images;
        encoded_images.reserve(paths.size());
        const auto unary_operator = [&cfg](std::string_view image_path) { return image_path_to_base64_encoded_string(image_path, cfg); };
        boost::transform(paths, std::back_inserter(encoded_images), unary_operator);
        return encoded_images;
    }

    std::string extension_to_mime_type(std::string_view extension)
    {
        static const string_unordered_map<std::string> map
        {
            { ".jpg", "jpg" },
            { ".jpeg", "jpg" },
            { ".png", "png" },
            { ".webp", "webp" },
            { ".gif", "gif" },
            { ".bmp", "bmp" },
            { ".svg", "svg+xml" },
            { ".avif", "avif" },
            { ".tif", "tiff" },
            { ".tiff", "tiff" },
            { ".ico", "x-icon" }
        };
        if (const auto iter{ map.find(extension) }; iter != map.end())
        {
            return iter->second;
        }
        llmcpp::throw_exception(logic_error{});
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

    std::filesystem::path string_to_path_by_config(std::string_view path, const config& cfg)
    {
        const std::filesystem::path file_path{ expand_macro(path, cfg, cfg.ctx) };
        if (file_path.is_relative())
        {
            const std::filesystem::path base_path{ expand_macro(cfg.base_path, cfg, cfg.ctx) };
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
                llmcpp::throw_exception(png_exception{});
            }

            const std::vector<unsigned char> text_chunk{ create_tEXt_chunk(key, metadata) };

            std::string result;
            result.reserve(image.size() + text_chunk.size());

            result.append(image.substr(0, ihdr_end_offset));
            result.append(reinterpret_cast<const char*>(text_chunk.data()), text_chunk.size());
            result.append(image.substr(ihdr_end_offset));

            return result;
        }

        std::uint32_t read_uint32_be(const unsigned char* p)
        {
            return (static_cast<std::uint32_t>(p[0]) << 24)
                | (static_cast<std::uint32_t>(p[1]) << 16)
                | (static_cast<std::uint32_t>(p[2]) << 8)
                | static_cast<std::uint32_t>(p[3]);
        }

        std::string extract_parameters(std::string_view image, std::string_view target_key = "parameters")
        {
            const unsigned char* data{ reinterpret_cast<const unsigned char*>(image.data()) };
            const std::size_t size{ image.size() };

            constexpr std::size_t png_header_size = 8;
            if (size < png_header_size || data[0] != 0x89 || data[1] != 'P' || data[2] != 'N' || data[3] != 'G')
            {
                llmcpp::throw_exception(png_exception{});
            }

            std::size_t offset = png_header_size;

            while (offset + 12 <= size)
            {
                const std::uint32_t length{ read_uint32_be(data + offset) };
                const std::string_view type{ reinterpret_cast<const char*>(data + offset + 4), 4 };

                const std::size_t data_offset = offset + 8;
                if (data_offset + length + 4 > size)
                {
                    break;
                }

                if (type == "tEXt")
                {
                    const unsigned char* chunk_data{ data + data_offset };

                    std::size_t key_length{};
                    while (key_length < length && chunk_data[key_length] != '\0')
                    {
                        ++key_length;
                    }

                    if (key_length < length)
                    {
                        const std::string_view key(reinterpret_cast<const char*>(chunk_data), key_length);

                        if (key == target_key)
                        {
                            const std::size_t text_offset{ key_length + 1 };
                            const std::size_t text_length{ length - text_offset };
                            return std::string(reinterpret_cast<const char*>(chunk_data + text_offset), text_length);
                        }
                    }
                }
                else if (type == "IEND")
                {
                    break;
                }

                offset += 12 + length;
            }

            llmcpp::throw_exception(png_exception{});
        }
    }

    template<typename BoostException>
    void if_error_throw(const boost::beast::error_code& error_code)
    {
        if (error_code)
        {
            llmcpp::throw_exception(BoostException{} << error_info::beast::error_code{ error_code });
        }
    }

    struct tcp
    {
        tcp()
        {
        }

        ~tcp()
        {
            close();
        }

        tcp(const tcp&) = delete;
        tcp& operator=(const tcp&) = delete;
        tcp(tcp&&) = default;
        tcp& operator=(tcp&&) = default;

        void connect(std::string_view host, std::string_view port)
        {
            const boost::asio::ip::tcp::resolver::results_type results{ resolver.resolve(host, port) };
            tcp_stream.connect(results, error_code);
            if_error_throw<connect_exception>(error_code);
            connected = true;
        }

        void close() noexcept
        {
            if (connected)
            {
                tcp_stream.socket().shutdown(boost::asio::ip::tcp::socket::shutdown_both, error_code);
                tcp_stream.socket().close(error_code);
                connected = false;
            }
        }

        void send(boost::beast::http::request<boost::beast::http::string_body>& request)
        {
            request.prepare_payload();
            boost::beast::http::write(tcp_stream, request, error_code);
            if_error_throw<http_send_exception>(error_code);
        }

        boost::beast::http::response<boost::beast::http::string_body> recieve()
        {
            boost::beast::flat_buffer buffer;
            boost::beast::http::response_parser<boost::beast::http::string_body> parser;
            parser.body_limit(boost::none);
            boost::beast::http::read(tcp_stream, buffer, parser, error_code);
            if_error_throw<http_receive_exception>(error_code);
            boost::beast::http::response<boost::beast::http::string_body> response{ parser.release() };

            if (response.result() != boost::beast::http::status::ok)
            {
                llmcpp::throw_exception(http_status_exception{}
                    << error_info::http::response::status{ response.result() }
                    << error_info::http::response::reason{ std::to_string(response.result_int()) }
                );
            }

            return response;
        }

        boost::beast::error_code error_code;
        boost::asio::io_context ioc;
        boost::asio::ip::tcp::resolver resolver{ ioc };
        boost::beast::tcp_stream tcp_stream{ ioc };
        bool connected{};
    };

    boost::beast::http::request<boost::beast::http::string_body> make_request(
        boost::beast::http::verb method,
        std::string_view host,
        std::string_view target,
        std::optional<std::string_view> content_type,
        std::optional<std::string_view> body
    )
    {
        boost::beast::http::request<boost::beast::http::string_body> request{ method, target, 11 };
        request.set(boost::beast::http::field::host, host);
        request.set(boost::beast::http::field::user_agent, BOOST_BEAST_VERSION_STRING);
        if (content_type)
        {
            request.set(boost::beast::http::field::content_type, *content_type);
        }
        if (body)
        {
            request.body() = *body;
        }
        return request;
    }

    boost::beast::http::request<boost::beast::http::string_body> make_post_json_request(
        std::string_view host,
        std::string_view target,
        std::string_view body
    )
    {
        return make_request(boost::beast::http::verb::post, host, target, "application/json; charset=UTF-8", body);
    }

    boost::beast::http::request<boost::beast::http::string_body> make_get_json_request(
        std::string_view host,
        std::string_view target
    )
    {
        return make_request(boost::beast::http::verb::get, host, target, "application/json; charset=UTF-8", std::nullopt);
    }

    boost::beast::http::response<boost::beast::http::string_body> send_http_get(
        std::string_view host,
        std::string_view port,
        std::string_view target,
        unsigned int expires_after)
    {
        tcp tcp;
        tcp.tcp_stream.expires_after(std::chrono::seconds{ expires_after });

        tcp.connect(host, port);

        boost::beast::http::request<boost::beast::http::string_body> request{ make_request(boost::beast::http::verb::get, host, target) };

        tcp.send(request);

        return tcp.recieve();
    };

    // unused
    std::string make_automatic1111_png_parameters(const sd_parameters& parameters, std::string_view prompt, std::string_view negative_prompt)
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
            << "Hires upscale: " << parameters.txt2img.hr_scale << ", "
            << "Hires steps: " << parameters.txt2img.hr_second_pass_steps << ", "
            << "Hires upscaler: " << parameters.txt2img.hr_upscaler
            << std::flush;
        return oss.str();
    }

    std::string send_automatic1111_txt2img_request(
        const config& cfg,
        std::string_view prompt,
        std::string_view negative_prompt
    )
    {
        const std::string_view host{ cfg.sd.host };
        const std::string_view port{ cfg.sd.port };

        tcp tcp;
        tcp.connect(host, port);

        nlohmann::json json;

        json["prompt"] = prompt;
        if (!negative_prompt.empty())
        {
            json["negative_prompt"] = negative_prompt;
        }
        //json["styles"] = cfg.sd_txt2img_params.styles;
        json["seed"] = cfg.sd.seed;
        json["subseed"] = cfg.sd.subseed;
        json["subseed_strength"] = cfg.sd.subseed_strength;
        json["seed_resize_from_h"] = cfg.sd.seed_resize_from_h;
        json["seed_resize_from_w"] = cfg.sd.seed_resize_from_w;
        json["sampler_name"] = cfg.sd.sampler_name;
        json["scheduler"] = cfg.sd.scheduler;
        json["batch_size"] = cfg.sd.batch_size;
        json["n_iter"] = cfg.sd.n_iter;
        json["steps"] = cfg.sd.steps;
        json["cfg_scale"] = cfg.sd.cfg_scale;
        json["width"] = cfg.sd.width;
        json["height"] = cfg.sd.height;
        json["restore_faces"] = cfg.sd.restore_faces;
        json["tiling"] = cfg.sd.tiling;
        json["do_not_save_samples"] = cfg.sd.do_not_save_samples;
        json["do_not_save_grid"] = cfg.sd.do_not_save_grid;
        json["eta"] = cfg.sd.eta;
        json["denoising_strength"] = cfg.sd.denoising_strength;
        json["s_min_uncond"] = cfg.sd.s_min_uncond;
        json["s_churn"] = cfg.sd.s_churn;
        json["s_tmax"] = cfg.sd.s_tmax;
        json["s_tmin"] = cfg.sd.s_tmin;
        json["s_noise"] = cfg.sd.s_noise;
        if (!cfg.sd.override_settings.empty())
        {
            json["override_settings"] = nlohmann::json::parse(cfg.sd.override_settings);
        }
        json["override_settings_restore_afterwards"] = cfg.sd.override_settings_restore_afterwards;
        json["refiner_checkpoint"] = cfg.sd.refiner_checkpoint;
        json["refiner_switch_at"] = cfg.sd.refiner_switch_at;
        json["disable_extra_networks"] = cfg.sd.disable_extra_networks;
        if (!cfg.sd.firstpass_image.empty())
        {
            json["firstpass_image"] = image_path_to_base64_encoded_string(cfg.sd.firstpass_image, cfg);;
        }
        if (!cfg.sd.comments.empty())
        {
            json["comments"] = cfg.sd.comments;
        }
        if (cfg.sd.mode == sd_mode::txt2img)
        {
            json["enable_hr"] = cfg.sd.txt2img.enable_hr;
            json["firstphase_width"] = cfg.sd.txt2img.firstphase_width;
            json["firstphase_height"] = cfg.sd.txt2img.firstphase_height;
            json["hr_scale"] = cfg.sd.txt2img.hr_scale;
            json["hr_upscaler"] = cfg.sd.txt2img.hr_upscaler;
            json["hr_second_pass_steps"] = cfg.sd.txt2img.hr_second_pass_steps;
            json["hr_resize_x"] = cfg.sd.txt2img.hr_resize_x;
            json["hr_resize_y"] = cfg.sd.txt2img.hr_resize_y;
            if (!cfg.sd.txt2img.hr_checkpoint_name.empty())
            {
            }
            //json["hr_prompt"] = prompt;
            //json["hr_negative_prompt"] = negative_prompt;
        }
        else if (cfg.sd.mode == sd_mode::img2img)
        {
            json["sd_init_images"] = image_paths_to_base64_encoded_strings(cfg.sd.img2img.init_images, cfg);
            json["sd_seed_resize_from_h"] = cfg.sd.img2img.seed_resize_from_h;
            json["sd_seed_resize_from_w"] = cfg.sd.img2img.seed_resize_from_w;
            json["sd_resize_mode"] = cfg.sd.img2img.resize_mode;
            json["sd_image_cfg_scale"] = cfg.sd.img2img.image_cfg_scale;
            json["sd_mask"] = image_path_to_base64_encoded_string(cfg.sd.img2img.mask, cfg);
            json["sd_mask_blur_x"] = cfg.sd.img2img.mask_blur_x;
            json["sd_mask_blur_y"] = cfg.sd.img2img.mask_blur_y;
            json["sd_mask_blur"] = cfg.sd.img2img.mask_blur;
            json["sd_mask_round"] = cfg.sd.img2img.mask_round;
            json["sd_inpainting_fill"] = cfg.sd.img2img.inpainting_fill;
            json["sd_inpaint_full_res"] = cfg.sd.img2img.inpaint_full_res;
            json["sd_inpaint_full_res_padding"] = cfg.sd.img2img.inpaint_full_res_padding;
            json["sd_inpainting_mask_invert"] = cfg.sd.img2img.inpainting_mask_invert;
            json["sd_initial_noise_multiplier"] = cfg.sd.img2img.initial_noise_multiplier;
            json["sd_latent_mask"] = image_path_to_base64_encoded_string(cfg.sd.img2img.latent_mask, cfg);
        }

        json["force_task_id"] = cfg.sd.force_task_id;

        if (!cfg.sd.sampler_index.empty() && cfg.sd.sampler_name.empty())
        {
            json["sampler_index"] = cfg.sd.sampler_index;
        }

        if (cfg.sd.abg_remover_enable)
        {
            json["script_name"] = "abg remover";
            json["script_args"] = {
                false,
                false,
                false,
                "#000000",
                false
            };
        }

        json["send_images"] = cfg.sd.send_images;
        json["save_images"] = cfg.sd.save_images;

        nlohmann::json alwayson_scripts{ nlohmann::json::object() };
        if (cfg.sd.alwayson_scripts.adetailer_parametesrs.ad_enable)
        {
            nlohmann::json adetailer{ nlohmann::json::object() };
            nlohmann::json object{ nlohmann::json::object() };
            object["ad_model"] = cfg.sd.alwayson_scripts.adetailer_parametesrs.args1.ad_model;
            if (!cfg.sd.alwayson_scripts.adetailer_parametesrs.args1.ad_prompt.empty())
            {
                object["ad_prompt"] = cfg.sd.alwayson_scripts.adetailer_parametesrs.args1.ad_prompt;
            }
            if (!cfg.sd.alwayson_scripts.adetailer_parametesrs.args1.ad_negative_prompt.empty())
            {
                object["ad_negative_prompt"] = cfg.sd.alwayson_scripts.adetailer_parametesrs.args1.ad_negative_prompt;
            }
            adetailer["args"] = { true, false, object };
            alwayson_scripts["ADetailer"] = adetailer;
        }
        //{
        //    nlohmann::json sampler{ nlohmann::json::object() };
        //    sampler["args"] =
        //    {
        //        cfg.sd.steps,
        //        cfg.sd.sampler_name,
        //        cfg.sd.scheduler
        //    };
        //    alwayson_scripts["Sampler"] = sampler;
        //}
        //{
        //    nlohmann::json seed{ nlohmann::json::object() };
        //    seed["args"] = 
        //    {
        //        cfg.sd.seed,
        //        false,
        //        cfg.sd.subseed,
        //        0,
        //        0,
        //        0
        //    };
        //    alwayson_scripts["Seed"] = seed;
        //}
        json["alwayson_scripts"] = alwayson_scripts;

        if (!cfg.sd.infotext.empty())
        {
            json["infotext"] = cfg.sd.infotext;
        }

        const std::string request_body{ json.dump() };
        BOOST_LOG_TRIVIAL(info) << "Send JSON\n```\n" << request_body << "\n```";

        const std::string target{ sd_mode_to_target(cfg.sd.mode, cfg) };
        boost::beast::http::request<boost::beast::http::string_body> request{ make_post_json_request(host, target, request_body) };

        tcp.send(request);
        boost::beast::http::response<boost::beast::http::string_body> response{ tcp.recieve() };

        BOOST_LOG_TRIVIAL(trace) << "Receive JSON\n```\n" << response.body() << "\n```";

        nlohmann::json response_json{ nlohmann::json::parse(response.body()) };

        const std::string base64_image_data{ response_json.at("images").at(0).get<std::string>() };

        if (base64_image_data.empty())
        {
            llmcpp::throw_exception(image_generation_exception{} << error_info::description{ "No image data found in the response" });
        }

        const std::string decoded_image{ base64_decode(base64_image_data) };

        return decoded_image;
    }

    std::string send_style_bert_voice_request(
        const config& cfg,
        std::string_view text
    )
    {
        const std::string_view host{ cfg.sb.host };
        const std::string_view port{ cfg.sb.port };

        tcp tcp;
        tcp.connect(host, port);

        boost::url target{ cfg.sb.target };
        target.params().set("text", text);
        //target.params().set("encoding", "utf-8");

        if (!cfg.sb.model_name.empty())
        {
            target.params().set("model_name", cfg.sb.model_name);
        }
        else
        {
            target.params().set("model_id", std::to_string(cfg.sb.model_id));
        }

        if (!cfg.sb.speaker_name.empty())
        {
            target.params().set("speaker_name", cfg.sb.speaker_name);
        }
        else
        {
            target.params().set("speaker_id", std::to_string(cfg.sb.speaker_id));
        }

        target.params().set("sdp_ratio", std::to_string(cfg.sb.sdp_ratio));
        target.params().set("noise", std::to_string(cfg.sb.noise));
        target.params().set("noisew", std::to_string(cfg.sb.noisew));
        target.params().set("length", std::to_string(cfg.sb.length));
        target.params().set("language", cfg.sb.language);
        target.params().set("auto_split", cfg.sb.auto_split ? "true" : "false");
        target.params().set("split_interval", std::to_string(cfg.sb.split_interval));

        if (!cfg.sb.assist_text.empty())
        {
            target.params().set("assist_text", cfg.sb.assist_text);
            target.params().set("assist_text_weight", std::to_string(cfg.sb.assist_text_weight));
        }

        if (!cfg.sb.style.empty())
        {
            target.params().set("style", cfg.sb.style);
            target.params().set("style_weight", std::to_string(cfg.sb.style_weight));
        }

        if (!cfg.sb.reference_audio_path.empty())
        {
            target.params().set("reference_audio_path", cfg.sb.reference_audio_path);
        }

        BOOST_LOG_TRIVIAL(info) << "Send target\n```\n" << target.c_str() << "\n```";

        boost::beast::http::request<boost::beast::http::string_body> request{ make_get_json_request(host, target.buffer()) };

        tcp.send(request);
        return tcp.recieve().body();
    }

    std::string generate_boundary()
    {
        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        oss << std::setw(sizeof(std::uint64_t) * 2) << random<std::uint64_t>()
            << std::setw(sizeof(std::uint64_t) * 2) << random<std::uint64_t>()
            << std::setw(sizeof(std::uint64_t) * 2) << random<std::uint64_t>()
            << std::setw(sizeof(std::uint64_t) * 2) << random<std::uint64_t>();
        return oss.str();
    }

    std::string upload_image_to_comfy_ui(
        const config& cfg,
        std::string_view image_path,
        bool overwrite
    )
    {
        const std::string image_data{ read_binary_file_to_string(image_path, cfg) };
        const std::string boundary{ generate_boundary() };
        const std::string filename{ std::filesystem::path{ image_path }.filename().string() };

        std::string body;
        body.reserve(512 + image_data.size());

        body += "--";
        body += boundary;
        body += "\r\n";
        body += "Content-Disposition: form-data; name=\"image\"; filename=\"";
        body += filename;
        body += "\"\r\n";
        body += "Content-Type: image/png\r\n\r\n";

        body += image_data;
        body += "\r\n";

        if (overwrite)
        {
            body += "--";
            body += boundary;
            body += "\r\n";
            body += "Content-Disposition: form-data; name=\"overwrite\"\r\n\r\n";
            body += "true\r\n";
        }

        body += "--";
        body += boundary;
        body += "--\r\n";

        const std::string_view host{ cfg.cu.host };
        const std::string_view port{ cfg.cu.port };
        const std::string_view target{ cfg.cu.upload_image_target };

        tcp tcp;
        tcp.tcp_stream.expires_after(std::chrono::seconds{ cfg.expires_after });
        tcp.connect(host, port);

        std::string content_type;
        content_type.reserve(30 + boundary.size());
        content_type += "multipart/form-data; boundary=";
        content_type += boundary;
        boost::beast::http::request<boost::beast::http::string_body> request{ make_post_json_request(host, target, body) };
        request.set(boost::beast::http::field::content_type, content_type);

        tcp.send(request);
        boost::beast::http::response<boost::beast::http::string_body> response{ tcp.recieve() };

        nlohmann::json response_json{ nlohmann::json::parse(response.body()) };

        return response_json.at("name").get<std::string>();
    }

    void upload_images_to_comfy_ui(
        const config& cfg,
        context& ctx
    )
    {
        for (const std::string& key_value_pair : cfg.cu.upload_images)
        {
            const std::size_t separator_position{ key_value_pair.find('=') };
            if (separator_position != std::string::npos)
            {
                const std::string variable_name{ key_value_pair.substr(0, separator_position) };
                const std::string local_relative_path{ key_value_pair.substr(separator_position + 1) };
                if (!variable_name.empty())
                {
                    const std::string server_path{ upload_image_to_comfy_ui(cfg, local_relative_path) };
                    ctx.set(variable_name, server_path);
                    BOOST_LOG_TRIVIAL(info) << "Successfully uploaded. (" << variable_name << "=" << server_path << ")";
                }
            }
            else
            {
                BOOST_LOG_TRIVIAL(warning) << "Invalid upload images format: " << key_value_pair << ". Expected variable_name=local_path";
            }
        }
    }

    void send_comfy_ui_prompt(
        const config& cfg,
        std::string_view prompt
    )
    {
        struct generated_file_info
        {
            std::string filename;
            std::string subfolder;
            std::string type;
        };

        const std::string_view host{ cfg.cu.host };
        const std::string_view port{ cfg.cu.port };
        const std::string_view target{ cfg.cu.prompt_target };

        tcp tcp;
        tcp.connect(host, port);

        nlohmann::json json;
        nlohmann::json prompt_json{ nlohmann::json::parse(prompt) };
        json["prompt"] = prompt_json;

        const std::string request_body{ json.dump() };
        BOOST_LOG_TRIVIAL(info) << "Send JSON\n```\n" << request_body << "\n```";

        boost::beast::http::request<boost::beast::http::string_body> request{ make_post_json_request(host, target, request_body) };

        tcp.send(request);
        boost::beast::http::response<boost::beast::http::string_body> response{ tcp.recieve() };

        nlohmann::json response_json{ nlohmann::json::parse(response.body()) };

        BOOST_LOG_TRIVIAL(info) << "Response: " << response.body();

        const std::string prompt_id{ response_json.at("prompt_id").get<std::string>() };
        BOOST_LOG_TRIVIAL(info) << "Queued successfully. Prompt ID: " << prompt_id;

        std::vector<generated_file_info> target_files;

        while (true)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));

            boost::beast::http::response<boost::beast::http::string_body> history_response{
                send_http_get(
                    cfg.cu.host,
                    cfg.cu.port,
                    "/history/" + prompt_id,
                    cfg.expires_after
                )
            };

            nlohmann::json history_json{ nlohmann::json::parse(history_response.body()) };

            if (!history_json.is_object())
            {
                continue;
            }

            try
            {
                const nlohmann::json& prompt_response_obj{ history_json.at(prompt_id) };

                try
                {
                    const nlohmann::json& status_object{ prompt_response_obj.at("status") };
                    const std::string status_str{ status_object.at("status_str").get<std::string>() };
                    if (status_str == "error")
                    {
                        llmcpp::throw_exception(comfy_ui_generation_exception{} << error_info::description{ "ComfyUI generation failed on server" });
                    }
                }
                catch (const nlohmann::json::out_of_range&) {
                    ;
                }

                target_files.clear();

                const nlohmann::json& outputs_object{ prompt_response_obj.at("outputs") };
                for (const auto& [key, value] : outputs_object.items())
                {
                    if (!value.is_object())
                    {
                        continue;
                    }

                    for (const auto& [prop_key, file_list] : value.items())
                    {
                        if (!file_list.is_array())
                        {
                            continue;
                        }

                        for (const nlohmann::json& file_item : file_list)
                        {
                            if (!file_item.is_object())
                            {
                                continue;
                            }

                            target_files.emplace_back
                            (
                                file_item.at("filename").get<std::string>(),
                                file_item.value("subfolder", ""),
                                file_item.at("type").get<std::string>()
                            );
                        }
                    }
                }

                if (!target_files.empty())
                {
                    break;
                }
            }
            catch (const nlohmann::json::out_of_range&)
            {
                continue;
            }
        }

        BOOST_LOG_TRIVIAL(info) << "Generation complete";

        for (const generated_file_info& file_info : target_files)
        {
            std::filesystem::path relative_file_path{ cfg.cu.output_directory };
            if (cfg.cu.preserve_subdirectories)
            {
                relative_file_path /= file_info.subfolder;
            }
            relative_file_path /= file_info.filename;

            std::string view_target;
            view_target += "/view?filename=";
            view_target += file_info.filename;
            view_target += "&subfolder=";
            view_target += file_info.subfolder;
            view_target += "&type=";
            view_target += file_info.type;

            const boost::beast::http::response<boost::beast::http::string_body> view_response{ send_http_get(
                cfg.cu.host,
                cfg.cu.port,
                view_target,
                cfg.expires_after
            ) };

            write_file(cfg, view_response.body(), relative_file_path.string(), std::ios::binary);
        }
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

    void write_item_list(const config& cfg, std::string_view task)
    {
        const std::vector<item> items{ parse_item_list(task) };

        for (const item& item : items)
        {
            std::string descriptions;
            for (const std::string& description : item.descriptions)
            {
                descriptions += description;
            }
            write_file(cfg, descriptions, item.head, std::ios::binary);
        }
    }

    std::string send_completions_request(
        const config& cfg,
        std::string_view prompt,
        const text_generation_parameters& params,
        int max_tokens
    )
    {
        const std::string_view host{ cfg.llm.host };
        const std::string_view port{ cfg.llm.port };
        const std::string target{ llm_mode_to_target(cfg.llm.mode, cfg) };

        tcp tcp;
        tcp.tcp_stream.expires_after(std::chrono::seconds{ cfg.expires_after });
        tcp.connect(host, port);

        std::string request_body;
        if (cfg.llm.mode == llm_mode::completions)
        {
            request_body = params.get_request_body_for_text_completions(prompt, max_tokens);
            BOOST_LOG_TRIVIAL(info) << "Send JSON\n```\n" << request_body << "\n```";
        }
        else if (cfg.llm.mode == llm_mode::vision)
        {
            const std::string base64_image{ image_path_to_base64_encoded_string(cfg.llm.image_file, cfg) };
            const std::string mime_type{ extension_to_mime_type(std::filesystem::path{ cfg.llm.image_file }.extension().string()) };
            request_body = params.get_request_body_for_vision(prompt, max_tokens, base64_image, mime_type);
            BOOST_LOG_TRIVIAL(info) << "Send JSON";
        }
        else
        {
            llmcpp::throw_exception(logic_error{});
        }

        boost::beast::http::request<boost::beast::http::string_body> request{ make_post_json_request(host, target, request_body) };

        if (!cfg.llm.api_key.empty())
        {
            request.set(boost::beast::http::field::authorization, ("Bearer ") + cfg.llm.api_key);
        }

        tcp.send(request);
        boost::beast::http::response<boost::beast::http::string_body> response{ tcp.recieve() };
        BOOST_LOG_TRIVIAL(trace) << "Receive JSON\n```\n" << response.body() << "\n```";

        if (cfg.llm.mode == llm_mode::completions)
        {
            return params.parse_response_for_text_completions(response.body());
        }
        else if (cfg.llm.mode == llm_mode::vision)
        {
            return params.parse_response_for_vision(response.body());
        }
        llmcpp::throw_exception(logic_error{});
    }

    std::string tg_completions_parameters::get_request_body_for_text_completions(std::string_view prompt, int max_tokens) const
    {
        nlohmann::json json{ nlohmann::json::object() };

        json["prompt"] = prompt;
        json["model"] = model;
        json["best_of"] = best_of;
        json["echo"] = echo;
        json["frequency_penalty"] = frequency_penalty;
        //json["logit_bias"] = logit_bias;
        json["logprobs"] = logprobs;
        json["max_tokens"] = max_tokens;
        json["n"] = n;
        json["presence_penalty"] = presence_penalty;
        json["stop"] = stop;
        json["stream"] = stream;
        json["suffix"] = suffix;
        json["temperature"] = temperature;
        json["top_p"] = top_p;

        if (seed != -1)
        {
            json["seed"] = seed;
        }

        json["user"] = user;
        json["preset"] = preset;
        json["dynatemp_low"] = dynatemp_low;
        json["dynatemp_high"] = dynatemp_high;
        json["dynatemp_exponent"] = dynatemp_exponent;
        json["smoothing_factor"] = smoothing_factor;
        json["smoothing_curve"] = smoothing_curve;
        json["min_p"] = min_p;
        json["top_k"] = top_k;
        json["typical_p"] = typical_p;
        json["xtc_threshold"] = xtc_threshold;
        json["xtc_probability"] = xtc_probability;
        json["epsilon_cutoff"] = epsilon_cutoff;
        json["eta_cutoff"] = eta_cutoff;
        json["tfs"] = tfs;
        json["top_a"] = top_a;
        json["top_n_sigma"] = top_n_sigma;
        json["dry_multiplier"] = dry_multiplier;
        json["dry_allowed_length"] = dry_allowed_length;
        json["dry_base"] = dry_base;
        json["repetition_penalty"] = repetition_penalty;
        json["encoder_repetition_penalty"] = encoder_repetition_penalty;
        json["no_repeat_ngram_size"] = no_repeat_ngram_size;
        json["repetition_penalty_range"] = repetition_penalty_range;
        json["penalty_alpha"] = penalty_alpha;
        json["guidance_scale"] = guidance_scale;
        json["mirostat_mode"] = mirostat_mode;
        json["mirostat_tau"] = mirostat_tau;
        json["mirostat_eta"] = mirostat_eta;
        json["prompt_lookup_num_tokens"] = prompt_lookup_num_tokens;
        json["max_tokens_second"] = max_tokens_second;
        json["do_sample"] = do_sample;
        json["dynamic_temperature"] = max_tokens_second;
        json["temperature_last"] = temperature_last;
        json["auto_max_new_tokens"] = auto_max_new_tokens;
        json["ban_eos_token"] = ban_eos_token;
        json["add_bos_token"] = add_bos_token;
        json["skip_special_tokens"] = skip_special_tokens;
        json["static_cache"] = static_cache;
        json["truncation_length"] = truncation_length;
        json["sampler_priority"] = sampler_priority;
        json["custom_token_bans"] = custom_token_bans;
        json["negative_prompt"] = negative_prompt;
        json["dry_sequence_breakers"] = dry_sequence_breakers;
        json["grammar_string"] = grammar_string;

        return json.dump();
    }

    std::string tg_completions_parameters::parse_response_for_text_completions(const std::string& response) const
    {
        const nlohmann::json response_json{ nlohmann::json::parse(response) };
        return response_json.at("choices").at(0).at("text").get<std::string>();
    }

    std::string tg_completions_parameters::get_request_body_for_token_count(std::string_view prompt) const
    {
        nlohmann::json json{ nlohmann::json::object() };
        json["text"] = prompt;
        return json.dump();
    }

    int tg_completions_parameters::parse_response_for_token_count(const std::string& response) const
    {
        const nlohmann::json response_json{ nlohmann::json::parse(response) };
        return response_json.at("length").get<int>();
    }

    std::string tg_completions_parameters::get_request_body_for_vision(std::string_view prompt, int max_tokens, std::string_view base64_image, std::string_view mime_type) const
    {
        return {};
    }

    std::string tg_completions_parameters::parse_response_for_vision(const std::string& response) const
    {
        return {};
    }

    std::string kc_generation_parameters::get_request_body_for_text_completions(std::string_view prompt, int max_tokens) const
    {
        nlohmann::json json{ nlohmann::json::object() };

        json["max_context_length"] = max_context_length;
        json["max_length"] = max_tokens;
        json["prompt"] = prompt;
        json["rep_pen"] = rep_pen;
        json["rep_pen_range"] = rep_pen_range;
        json["sampler_order"] = sampler_order;

        if (sampler_seed != -1)
        {
            json["sampler_seed"] = sampler_seed;
        }

        json["stop_sequence"] = stop_sequence;
        json["temperature"] = temperature;
        json["tfs"] = tfs;
        json["top_a"] = top_a;
        json["top_k"] = top_k;
        json["top_p"] = top_p;
        json["min_p"] = min_p;
        json["typical"] = typical;
        json["use_default_badwordsids"] = use_default_badwordsids;
        json["dynatemp_range"] = dynatemp_range;
        json["smoothing_factor"] = smoothing_factor;
        json["dynatemp_exponent"] = dynatemp_exponent;
        json["mirostat"] = mirostat;
        json["mirostat_tau"] = mirostat_tau;
        json["mirostat_eta"] = mirostat_eta;
        json["genkey"] = genkey;
        json["grammar"] = grammar;
        json["grammar_retain_state"] = grammar_retain_state;
        json["memory"] = memory;
        json["images"] = images;
        json["trim_stop"] = trim_stop;
        json["render_special"] = render_special;
        json["bypass_eos"] = bypass_eos;
        json["banned_tokens"] = banned_tokens;
        json["dry_multiplier"] = dry_multiplier;
        json["dry_base"] = dry_base;
        json["dry_allowed_length"] = dry_allowed_length;
        json["dry_penalty_last_n"] = dry_penalty_last_n;
        json["dry_sequence_breakers"] = dry_sequence_breakers;
        json["xtc_probability"] = xtc_probability;
        json["nsigma"] = nsigma;
        json["logprobs"] = logprobs;
        json["replace_instruct_placeholders"] = replace_instruct_placeholders;

        return json.dump();
    }

    std::string kc_generation_parameters::parse_response_for_text_completions(const std::string& response) const
    {
        const nlohmann::json response_json{ nlohmann::json::parse(response) };
        return response_json.at("results").at(0).at("text").get<std::string>();
    }

    std::string kc_generation_parameters::get_request_body_for_token_count(std::string_view prompt) const
    {
        nlohmann::json json{ nlohmann::json::object() };
        json["prompt"] = prompt;
        return json.dump();
    }

    int kc_generation_parameters::parse_response_for_token_count(const std::string& response) const
    {
        const nlohmann::json response_json{ nlohmann::json::parse(response) };
        return response_json.at("value").get<int>();
    }

    std::string kc_generation_parameters::get_request_body_for_vision(std::string_view prompt, int max_tokens, std::string_view base64_image, std::string_view mime_type) const
    {
        nlohmann::json json{ nlohmann::json::object() };

        //json["max_length"] = max_tokens;
        json["max_tokens"] = max_tokens;
        json["rep_pen"] = rep_pen;
        json["rep_pen_range"] = rep_pen_range;
        json["sampler_order"] = sampler_order;

        if (sampler_seed != -1)
        {
            json["sampler_seed"] = sampler_seed;
        }

        const std::vector<std::string> stop_sequence
        {
            "{{[INPUT]}}",
            "{{[OUTPUT]}}"
        };

        json["stop_sequence"] = stop_sequence;
        json["temperature"] = temperature;
        json["tfs"] = tfs;
        json["top_a"] = top_a;
        json["top_k"] = top_k;
        json["top_p"] = top_p;
        json["min_p"] = min_p;
        json["typical"] = typical;
        json["use_default_badwordsids"] = use_default_badwordsids;
        json["dynatemp_range"] = dynatemp_range;
        json["smoothing_factor"] = smoothing_factor;
        json["dynatemp_exponent"] = dynatemp_exponent;
        json["mirostat"] = mirostat;
        json["genkey"] = genkey;
        json["trim_stop"] = trim_stop;
        json["render_special"] = render_special;
        json["bypass_eos"] = bypass_eos;
        json["banned_tokens"] = banned_tokens;
        json["logprobs"] = logprobs;

        std::string url{ "data:image/" };
        url.reserve(19 + mime_type.size() + base64_image.size());
        url += mime_type;
        url += ";base64,";
        url += base64_image;

        nlohmann::json message{
            { "role", "user" },
            { "content", nlohmann::json::array({
                {
                    { "type", "text" },
                    { "text", prompt }
                },
                {
                    { "type", "image_url" },
                    { "image_url", { { "url", url } } }
                }
            })}
        };

        json["messages"] = nlohmann::json::array({ message });

        return json.dump();
    }

    std::string kc_generation_parameters::parse_response_for_vision(const std::string& response) const
    {
        const nlohmann::json response_json{ nlohmann::json::parse(response) };
        return response_json.at("choices").at(0).at("message").at("content").get<std::string>();
    }

    int send_token_count_request(const config& cfg, std::string_view prompt)
    {
        const std::string_view host{ cfg.llm.host };
        const std::string_view port{ cfg.llm.port };
        const std::string_view target{ cfg.llm.token_count_target };

        tcp tcp;
        tcp.connect(host, port);

        const std::string request_body{ cfg.llm.backend->get_request_body_for_token_count(prompt) };
        BOOST_LOG_TRIVIAL(trace) << "Send JSON\n```\n" << request_body << "\n```";

        boost::beast::http::request<boost::beast::http::string_body> request{ make_post_json_request(host, target, request_body) };

        tcp.send(request);
        boost::beast::http::response<boost::beast::http::string_body> response{ tcp.recieve() };

        return cfg.llm.backend->parse_response_for_token_count(response.body());
    }

    int get_tokens_from_cache(const config& cfg, std::string_view str)
    {
        constexpr std::size_t capacity{ 1000 };
        int tokens{};

        lru_cache::const_iterator iter{ cfg.lru_cache.get<by_key>().find(std::string{ str }) };
        if (iter != cfg.lru_cache.get<by_key>().end())
        {
            tokens = iter->tokens;
            cfg.lru_cache.get<by_lru>().relocate(
                cfg.lru_cache.get<by_lru>().end(),
                cfg.lru_cache.get<by_lru>().iterator_to(*iter));
        }
        else
        {
            tokens = send_token_count_request(cfg, str);
            cfg.lru_cache.insert({ std::string{ str }, tokens });
        }

        if (cfg.lru_cache.size() > capacity)
        {
            cfg.lru_cache.get<by_lru>().pop_front();
        }

        return tokens;
    }

    void write_cache(const config& cfg)
    {
        if (cfg.command_mode != command_mode::tg && cfg.command_mode != command_mode::kc)
        {
            return;
        }

        nlohmann::json cache{ nlohmann::json::array() };
        for (const token_count_string& element : cfg.lru_cache.get<by_lru>())
        {
            cache.push_back({
                { "string", element.str },
                { "tokens", element.tokens }
                });
        }
        nlohmann::json json{ { "cache", std::move(cache) } };
        const std::vector<std::uint8_t> cbor{ nlohmann::json::to_cbor(json) };
        write_file(cfg, reinterpret_cast<const char*>(cbor.data()), cbor.size(), ".token_cache.bin", std::ios::binary);
    }

    void read_cache(const config& cfg)
    {
        if (cfg.command_mode != command_mode::tg && cfg.command_mode != command_mode::kc)
        {
            return;
        }

        const std::filesystem::path cache_path{ string_to_path_by_config(".token_cache.bin", cfg) };

        if (!std::filesystem::exists(cache_path))
        {
            return;
        }

        try
        {
            boost::nowide::ifstream ifs{ cache_path, std::ios::binary };
            if (!ifs.is_open())
            {
                return;
            }

            const std::vector<std::uint8_t> cbor{ std::istreambuf_iterator<char>{ ifs }, std::istreambuf_iterator<char>{} };

            nlohmann::json json{ nlohmann::json::from_cbor(cbor) };
            if (!json.is_object() || !json.contains("cache") || !json["cache"].is_array())
            {
                return;
            }

            lru_cache temp_lru_cache;
            const nlohmann::json caches{ json["cache"] };
            for (const nlohmann::json& cache : caches)
            {
                if (cache.is_object() && cache.contains("string") && cache.contains("tokens"))
                {
                    temp_lru_cache.insert({
                        cache["string"].get<std::string>(),
                        cache["tokens"].get<int>()
                        });
                }
            }
            cfg.lru_cache = std::move(temp_lru_cache);
        }
        catch (const nlohmann::json::exception& e)
        {
            cfg.lru_cache.clear();
            BOOST_LOG_TRIVIAL(warning) << boost::diagnostic_information(e);
        }
    }

    std::string generate_text(
        const config& cfg,
        std::string_view prompt,
        const context& ctx
    )
    {
        std::string expanded_prompt{ expand_macro(prompt, cfg, ctx) };
        const std::string expanded_prefix{ expand_macro(cfg.llm.generation_prefix, cfg, ctx) };
        const std::size_t initial_prompt_size{ expanded_prompt.size() };
        expanded_prompt += expanded_prefix;

        const int initial_tokens{ send_token_count_request(cfg, expanded_prompt) };

        BOOST_LOG_TRIVIAL(info) << "Prompt created.\n```\n" << expanded_prompt << "\n```";

        std::string current_prompt{ expanded_prompt };
        int current_tokens = initial_tokens;
        for (int completion_iterations{}; completion_iterations < cfg.llm.max_completion_iterations; ++completion_iterations)
        {
            BOOST_LOG_TRIVIAL(trace) << "completion_iterations: " << completion_iterations;

            if (current_tokens - initial_tokens >= cfg.llm.min_completion_tokens)
            {
                break;
            }

            const int remaining_tokens{ cfg.llm.backend->get_truncation_length() - current_tokens };
            if (remaining_tokens <= 0)
            {
                BOOST_LOG_TRIVIAL(warning) << "Context window full. Cannot generate more tokens";
                break;
            }

            int tokens_to_generate = std::min(cfg.llm.backend->get_max_tokens(), remaining_tokens);
            if (tokens_to_generate <= 0)
            {
                BOOST_LOG_TRIVIAL(warning) << "No tokens left to generate. Aborting";
                break;
            }

            const int max_tokens{ tokens_to_generate };
            const std::string response{ send_completions_request(
                cfg, current_prompt, *cfg.llm.backend, max_tokens
            ) };

            if (response.empty())
            {
                break;
            }

            current_prompt += response;
            current_tokens = send_token_count_request(cfg, current_prompt);
        }

        std::string generated{ current_prompt.substr(initial_prompt_size) };
        generated += cfg.llm.generation_suffix;

        return generated;
    }

    std::string vision_image(
        const config& cfg,
        std::string_view prompt,
        const context& ctx
    )
    {
        const std::string expanded_prompt{ expand_macro(prompt, cfg, ctx) };

        const int prompt_tokens{ send_token_count_request(cfg, expanded_prompt) };

        BOOST_LOG_TRIVIAL(info) << "Prompt created.\n```\n" << expanded_prompt << "\n```";

        const int remaining_tokens{ cfg.llm.backend->get_truncation_length() - prompt_tokens };
        if (remaining_tokens <= 0)
        {
            BOOST_LOG_TRIVIAL(warning) << "Context window full. Cannot generate more tokens";
            return std::string{};
        }

        const int max_tokens{ std::min(cfg.llm.backend->get_max_tokens(), remaining_tokens) };
        if (max_tokens <= 0)
        {
            BOOST_LOG_TRIVIAL(warning) << "No tokens left to generate. Aborting";
            return std::string{};
        }

        return send_completions_request(cfg, expanded_prompt, *cfg.llm.backend, max_tokens);
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
            case '"':   result += "\\\""; break;
            case '\\':  result += "\\\\"; break;
            case '\b':  result += "\\b"; break;
            case '\f':  result += "\\f"; break;
            case '\n':  result += "\\n"; break;
            case '\r':  result += "\\r"; break;
            case '\t':  result += "\\t"; break;
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

    void unescape_parameters(config& cfg)
    {
        boost::transform(cfg.user_defined_variables, cfg.user_defined_variables.begin(), unescape_string);
        boost::transform(cfg.phases, cfg.phases.begin(), unescape_string);
        cfg.llm.prompt = unescape_string(cfg.llm.prompt);
        cfg.llm.generation_prefix = unescape_string(cfg.llm.generation_prefix);
        cfg.llm.generation_suffix = unescape_string(cfg.llm.generation_suffix);
        cfg.llm.reasoning_prefix = unescape_string(cfg.llm.reasoning_prefix);
        cfg.llm.reasoning_suffix = unescape_string(cfg.llm.reasoning_suffix);
        boost::transform(cfg.tg.stop, cfg.tg.stop.begin(), unescape_string);
        cfg.tg.dry_sequence_breakers = unescape_string(cfg.tg.dry_sequence_breakers);
        boost::transform(cfg.kc.stop_sequence, cfg.kc.stop_sequence.begin(), unescape_string);
        boost::transform(cfg.kc.banned_tokens, cfg.kc.banned_tokens.begin(), unescape_string);
        boost::transform(cfg.kc.dry_sequence_breakers, cfg.kc.dry_sequence_breakers.begin(), unescape_string);
        cfg.sd.prompt = unescape_string(cfg.sd.prompt);
        cfg.sd.negative_prompt = unescape_string(cfg.sd.negative_prompt);
        cfg.sb.text = unescape_string(cfg.sb.text);
        cfg.cu.prompt = unescape_string(cfg.cu.prompt);
    }

    void parse_user_defined_variables(const std::vector<std::string>& user_defined_variables, context& ctx)
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
                    ctx.set(key, value);
                    BOOST_LOG_TRIVIAL(info) << "Variable set " << key << " = " << value;
                }
            }
            else
            {
                BOOST_LOG_TRIVIAL(warning) << "Invalid define format: " << key_value_pair << ". Expected key=value";
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
            llmcpp::throw_exception(file_open_exception{} << error_info::path{ log });
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

    void init_logging(const config& cfg)
    {
        boost::log::trivial::severity_level level = boost::log::trivial::info;
        if (cfg.log_level == "trace")
        {
            level = boost::log::trivial::trace;
        }
        else if (cfg.log_level == "debug")
        {
            level = boost::log::trivial::debug;
        }
        else if (cfg.log_level == "info")
        {
            level = boost::log::trivial::info;
        }
        else if (cfg.log_level == "warning")
        {
            level = boost::log::trivial::warning;
        }
        else if (cfg.log_level == "error")
        {
            level = boost::log::trivial::error;
        }
        else if (cfg.log_level == "fatal")
        {
            level = boost::log::trivial::fatal;
        }
        else
        {
            BOOST_LOG_TRIVIAL(warning) << "Unkown log level: \"" << cfg.log_level << "\"";
        }

        if (cfg.verbose)
        {
            init_logging_with_nowide_cout();
        }

        if (!cfg.log_file.empty())
        {
            const std::filesystem::path log_file_path{ string_to_path_by_config(complement_extension(cfg.log_file, ".txt"), cfg) };
            init_logging_with_nowide_file_log(log_file_path);
        }

        boost::log::core::get()->set_filter(boost::log::trivial::severity >= level);
    }

    void init_chat_mode(config& cfg)
    {
        if (cfg.phases.empty())
        {
            cfg.phases = { "{{user}}", "{{char}}" };
        }
        if (cfg.llm.generation_prefix.empty())
        {
            cfg.llm.generation_prefix = "\\n{{phase}}: ";
        }
    }

    void set_phase_variables(
        const std::vector<std::string>& phases,
        std::size_t phase_index,
        context& ctx
    )
    {
        if (phase_index >= phases.size())
        {
            llmcpp::throw_exception(array_index_out_of_bounds_exception{});
        }

        if (phase_index > 0)
        {
            ctx.set("prev_phase", phases[phase_index - 1]);
        }

        ctx.set("phase", phases[phase_index]);

        if (phase_index < phases.size() - 1)
        {
            ctx.set("next_phase", phases[phase_index + 1]);
        }
    }

    void set_static_builtin_variables(
        config& cfg
    )
    {
        cfg.ctx.set("stdin", builtin::stdin_(cfg));
    }

    void set_dynamic_builtin_variables(
        config& cfg
    )
    {
        cfg.ctx.set("date", builtin::date());
        cfg.ctx.set("time", builtin::time());
        cfg.ctx.set("datetime", builtin::datetime());
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

    void init_llm_mode(config& cfg)
    {
        if (!cfg.llm.paragraphs_file.empty())
        {
            cfg.phases.clear();
            const std::string content{ read_text_file_to_string(cfg.llm.paragraphs_file, cfg) };
            std::vector<item> paragraphs{ parse_item_list(content) };
            set_paragraphs_to_phases(paragraphs, cfg.phases);
        }

        if (cfg.command_mode == command_mode::tg)
        {
            cfg.llm.backend = &cfg.tg;
            if (cfg.llm.completions_target.empty())
            {
                cfg.llm.completions_target = "/v1/completions";
            }
            if (cfg.llm.token_count_target.empty())
            {
                cfg.llm.token_count_target = "/v1/internal/token-count";
            }
        }
        else if (cfg.command_mode == command_mode::kc)
        {
            cfg.llm.backend = &cfg.kc;
            if (cfg.llm.completions_target.empty())
            {
                cfg.llm.completions_target = "/api/v1/generate";
            }
            if (cfg.llm.token_count_target.empty())
            {
                cfg.llm.token_count_target = "/api/extra/tokencount";
            }
            if (cfg.llm.vision_target.empty())
            {
                cfg.llm.vision_target = "/api/v1/chat/completions";
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

    code_blocks extract_code_block_from_markdown(std::string_view markdown_content)
    {
        code_blocks result;
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
            llmcpp::throw_exception(dns_resolve_exception{} << error_info::asio::error_code{ error_code });
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
        //    BOOST_LOG_TRIVIAL(warning) << "exe not found";
        //    return;
        //}
        process::process proc{ ctx, excutable_file, arguments, process::windows::create_new_console };
        proc.detach();
    }

    std::size_t terminate_process_by_path(const std::filesystem::path& executable_file_path)
    {
        std::size_t terminated_count{};

#ifdef _WIN32
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
        config& cfg
    )
    {
        namespace po = boost::program_options;

        try
        {
            cfg.tg.stop = { "\\n\\n", ":", "***" };
            cfg.tg.sampler_priority =
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
            cfg.tg.dry_sequence_breakers = "(\"\\n\", \":\", \"\\\"\", \"*\")";

            std::string command_mode_string;
            std::string llm_mode_string;
            std::string sd_mode_string;

            po::options_description allowed_options("Allowed options");
            allowed_options.add_options()
                ("help,h", "produce help message")
                ("mode", po::value<std::string>(&command_mode_string)->default_value(""), "mode (tg | kc | sd | sb | cu | extract-png-parameters)")
                ("base-path", po::value<std::string>(&cfg.base_path)->default_value("."), "base path")
                ("log-level", po::value<std::string>(&cfg.log_level)->default_value("info"), "log level (trace|debug|info|warning|error|fatal)")
                ("log-file", po::value<std::string>(&cfg.log_file)->default_value("log"), "log file path")
                ("config-file,c", po::value<std::string>(&cfg.config_file)->default_value("config.ini"), "config file path")
                ("verbose,v", po::bool_switch(&cfg.verbose)->default_value(false), "enable verbose output")
                ("expires-after", po::value<unsigned int>(&cfg.expires_after)->default_value(30), "connection timeout")
                ("number-iterations,N", po::value<int>(&cfg.number_iterations)->default_value(1), "number of iterations (-1 means infinity)")
                ("define,D", po::value<std::vector<std::string>>(&cfg.user_defined_variables)->multitoken(), "define variables (key=value)")
                ("phases", po::value<std::vector<std::string>>(&cfg.phases)->multitoken(), "phases name list")
                ("seed", po::value<int>(&cfg.seed)->default_value(-1), "seed value")

                ("create-process", po::bool_switch(&cfg.create_process)->default_value(false), "create process switch")
                ("terminate-process", po::bool_switch(&cfg.terminate_process)->default_value(false), "terminate process switch")
                ("png-file", po::value<std::string>(&cfg.png_file)->default_value(""), "for extract-png-parametesrs")
                ("server-executable-file", po::value<std::string>(&cfg.server_executable_file)->default_value(""), "server executable file")
                ("server-arguments", po::value<std::string>(&cfg.server_arguments), "server arguments")
                ("server-host", po::value<std::string>(&cfg.server_host)->default_value("localhost"), "server ip")
                ("server-port", po::value<std::string>(&cfg.server_port)->default_value("5000"), "server port")
                ("server-max-retries", po::value<int>(&cfg.server_max_retries)->default_value(60), "server max retries")
                ("server-wait-ms", po::value<int>(&cfg.server_wait_ms)->default_value(1000), "server wait ms")

                ("llm-prompt", po::value<std::string>(&cfg.llm.prompt)->default_value(""), "LLM prompt")
                ("llm-prompt-file", po::value<std::string>(&cfg.llm.prompt_file)->default_value("prompt"), "LLM prompt file path")
                ("llm-output-file", po::value<std::string>(&cfg.llm.output_file)->default_value("output"), "LLM output file path")
                ("llm-generation-prefix", po::value<std::string>(&cfg.llm.generation_prefix)->default_value(""), "LLM generation prefix")
                ("llm-generation-suffix", po::value<std::string>(&cfg.llm.generation_suffix)->default_value(""), "LLM generation suffix")
                ("llm-paragraphs-file", po::value<std::string>(&cfg.llm.paragraphs_file)->default_value(""), "LLM paragraphs file")
                ("llm-image-file", po::value<std::string>(&cfg.llm.image_file)->default_value(""), "LLM image file")
                ("llm-host", po::value<std::string>(&cfg.llm.host)->default_value("localhost"), "LLM host")
                ("llm-port", po::value<std::string>(&cfg.llm.port)->default_value("5000"), "LLM port")
                ("llm-api-key", po::value<std::string>(&cfg.llm.api_key)->default_value(""), "LLM API key")
                ("llm-completions-target", po::value<std::string>(&cfg.llm.completions_target)->default_value(""), "LLM completions target")
                ("llm-token-count-target", po::value<std::string>(&cfg.llm.token_count_target)->default_value(""), "LLM token count target")
                ("llm-vision-target", po::value<std::string>(&cfg.llm.vision_target)->default_value(""), "LLM vision target")
                ("llm-min-completion-tokens", po::value<int>(&cfg.llm.min_completion_tokens)->default_value(256), "LLM min completion tokens")
                ("llm-max-completion-iterations", po::value<int>(&cfg.llm.max_completion_iterations)->default_value(5), "LLM max completion iterations")
                ("llm-reasoning-prefix", po::value<std::string>(&cfg.llm.reasoning_prefix)->default_value(""), "LLM reasoning prefix")
                ("llm-reasoning-suffix", po::value<std::string>(&cfg.llm.reasoning_suffix)->default_value(""), "LLM reasoning suffix")
                ("llm-code-block-extract", po::bool_switch(&cfg.llm.code_block_extract)->default_value(false), "LLM code block extract switch")

                ("llm-mode", po::value<std::string>(&llm_mode_string)->default_value("completions"), "LLM mode (completions | vision)")

                ("tg-model", po::value<std::string>(&cfg.tg.model)->default_value("", "TG model"))
                ("tg-num-best-of", po::value<int>(&cfg.tg.best_of)->default_value(1), "TG best of")
                ("tg-echo", po::bool_switch(&cfg.tg.echo)->default_value(false), "TG echo")
                ("tg-frequency-penalty", po::value<double>(&cfg.tg.frequency_penalty)->default_value(0.0), "TG frequency penalty")
                //std::map<int, double> logit_bias;
                ("tg-logprobs", po::value<double>(&cfg.tg.logprobs)->default_value(0.0), "TG presence penalty")
                ("tg-max-tokens", po::value<int>(&cfg.tg.max_tokens)->default_value(512), "TG max tokens")
                ("tg-n", po::value<int>(&cfg.tg.n)->default_value(1), "TG number of responses generated for the same prompt")
                ("tg-presence-penalty", po::value<double>(&cfg.tg.presence_penalty)->default_value(0.0), "TG presence penalty")
                ("tg-stop", po::value<std::vector<std::string>>(&cfg.tg.stop)->multitoken(), "TG stop sequences")
                ("tg-stream", po::bool_switch(&cfg.tg.stream)->default_value(false), "TG stream")
                ("tg-suffix", po::value<std::string>(&cfg.tg.suffix)->default_value(""), "TG suffix")
                ("tg-temperature", po::value<double>(&cfg.tg.temperature)->default_value(1.0), "TG temperature")
                ("tg-top-p", po::value<double>(&cfg.tg.top_p)->default_value(1.0), "TG top p")
                ("tg-dynatemp-low", po::value<double>(&cfg.tg.dynatemp_low)->default_value(0.75, "0.75"), "TG dynatemp low")
                ("tg-dynatemp-high", po::value<double>(&cfg.tg.dynatemp_high)->default_value(1.25, "1.25"), "TG dynatemp high")
                ("tg-dynatemp-exponent", po::value<double>(&cfg.tg.dynatemp_exponent)->default_value(1.0), "TG dynatemp exponent")
                ("tg-smoothing-factor", po::value<double>(&cfg.tg.smoothing_factor)->default_value(0.0), "TG smoothing factor")
                ("tg-smoothing-curve", po::value<double>(&cfg.tg.smoothing_curve)->default_value(1.0), "TG smoothing curve")
                ("tg-min-p", po::value<double>(&cfg.tg.min_p)->default_value(0.1, "0.1"), "TG min p")
                ("tg-top-k", po::value<int>(&cfg.tg.top_k)->default_value(0), "TG top k")
                ("tg-typical-p", po::value<double>(&cfg.tg.typical_p)->default_value(1.0), "TG typical p")
                ("tg-xtc-threshold", po::value<double>(&cfg.tg.xtc_threshold)->default_value(0.1, "0.1"), "TG Exclude Top Choices (XTC) threshold")
                ("tg-xtc-probability", po::value<double>(&cfg.tg.xtc_probability)->default_value(0.0), "TG Exclude Top Choices (XTC) probability")
                ("tg-epsilon-cutoff", po::value<double>(&cfg.tg.epsilon_cutoff)->default_value(0), "TG epsilon cutoff")
                ("tg-eta-cutoff", po::value<double>(&cfg.tg.eta_cutoff)->default_value(0), "TG eta cutoff")
                ("tg-tfs", po::value<double>(&cfg.tg.tfs)->default_value(1.0), "TG tfs")
                ("tg-top-a", po::value<double>(&cfg.tg.top_a)->default_value(0.0), "TG top a")
                ("tg-top-n-sigma", po::value<double>(&cfg.tg.top_n_sigma)->default_value(1.0), "TG top n sigma")
                ("tg-dry-multiplier", po::value<double>(&cfg.tg.dry_multiplier)->default_value(0.0), "TG DRY multiplier")
                ("tg-dry-allowed-length", po::value<int>(&cfg.tg.dry_allowed_length)->default_value(2), "TG DRY allowed length")
                ("tg-dry-base", po::value<double>(&cfg.tg.dry_base)->default_value(1.75), "TG DRY base")
                ("tg-repetition-penalty", po::value<double>(&cfg.tg.repetition_penalty)->default_value(1.2), "TG repetition penalty")
                ("tg-encoder-repetition-penalty", po::value<double>(&cfg.tg.encoder_repetition_penalty)->default_value(1.0), "TG encoder repetition penalty")
                ("tg-no-repeat-ngram-size", po::value<int>(&cfg.tg.no_repeat_ngram_size)->default_value(0), "TG no repeat ngram size")
                ("tg-repetition-penalty-range", po::value<int>(&cfg.tg.repetition_penalty_range)->default_value(0), "TG repetition penalty range")
                ("tg-penalty-alpha", po::value<double>(&cfg.tg.penalty_alpha)->default_value(0.9, "0.9"), "TG penalty alpha")
                ("tg-guidance-scale", po::value<double>(&cfg.tg.guidance_scale)->default_value(1.0), "TG guidance scale")
                ("tg-mirostat-mode", po::value<int>(&cfg.tg.mirostat_mode)->default_value(0), "TG mirostat mode")
                ("tg-mirostat-tau", po::value<double>(&cfg.tg.mirostat_tau)->default_value(5), "TG mirostat tau")
                ("tg-mirostat-eta", po::value<double>(&cfg.tg.mirostat_eta)->default_value(0.1, "0.1"), "TG mirostat eta")
                ("tg-prompt-lookup-num-tokens", po::value<int>(&cfg.tg.prompt_lookup_num_tokens)->default_value(0), "TG prompt lookup num tokens")
                ("tg-max-tokens-second", po::value<int>(&cfg.tg.max_tokens_second)->default_value(0), "TG max tokens second")
                ("tg-do-sample", po::bool_switch(&cfg.tg.do_sample)->default_value(true), "TG do sample")
                ("tg-dynamic-temperature", po::bool_switch(&cfg.tg.dynamic_temperature)->default_value(false), "TG dynamic temperature")
                ("tg-temperature-last", po::bool_switch(&cfg.tg.temperature_last)->default_value(false), "TG temperature last")
                ("tg-auto-max-new-tokens", po::bool_switch(&cfg.tg.auto_max_new_tokens)->default_value(false), "TG auto max_new tokens")
                ("tg-ban-eos-token", po::bool_switch(&cfg.tg.ban_eos_token)->default_value(false), "TG ban eos token")
                ("tg-add-bos-token", po::bool_switch(&cfg.tg.add_bos_token)->default_value(true), "TG add Beginning of Sequence Token (BOS) token")
                ("tg-skip-special-tokens", po::bool_switch(&cfg.tg.skip_special_tokens)->default_value(true), "TG skip special tokens (bos_token, eos_token, unk_token, pad_token, etc.)")
                ("tg-static-cache", po::bool_switch(&cfg.tg.static_cache)->default_value(false), "TG static cache")
                ("tg-truncation-length", po::value<int>(&cfg.tg.truncation_length)->default_value(4096), "TG truncation length")
                ("tg-sampler-priority", po::value<std::vector<std::string>>(&cfg.tg.sampler_priority)->multitoken(), "TG sampler priority")
                ("tg-custom-token-bans", po::value<std::string>(&cfg.tg.custom_token_bans)->default_value(""), "TG custom token bans")
                ("tg-negative-prompt", po::value<std::string>(&cfg.tg.negative_prompt)->default_value(""), "TG negative prompt")
                ("tg-dry-sequence-breakers", po::value<std::string>(&cfg.tg.dry_sequence_breakers)->default_value(""), "TG dry sequence breakers")
                ("tg-grammar-string", po::value<std::string>(&cfg.tg.grammar_string)->default_value(""), "TG grammar-string")

                ("kc-max-context-length", po::value<int>(&cfg.kc.max_context_length)->default_value(4096), "Maximum number of tokens to send to the model. (minimum: 1)")
                ("kc-max-length", po::value<int>(&cfg.kc.max_length)->default_value(512), "Number of tokens to generate. (minimum: 1)")
                ("kc-rep-pen", po::value<double>(&cfg.kc.rep_pen)->default_value(1.0), "Base repetition penalty value. (minimum: 1.0)")
                ("kc-rep-pen-range", po::value<int>(&cfg.kc.rep_pen_range)->default_value(0), "Repetition penalty range. (minimum: 0)")
                ("kc-sampler-order", po::value<std::vector<int>>(&cfg.kc.sampler_order)->multitoken(), "Sampler order to be used. If N is the length of this array, then N must be greater than or equal to 6 and the array must be a permutation of the first N non-negative integers.")
                ("kc-sampler-seed", po::value<int>(&cfg.kc.sampler_seed)->default_value(1), "RNG seed to use for sampling. If not specified, the global RNG will be used. (minimum: 1, maximum: 999999)")
                ("kc-stop-sequence", po::value<std::vector<std::string>>(&cfg.kc.stop_sequence)->multitoken(), "An array of string sequences where the API will stop generating further tokens. The returned text WILL contain the stop sequence if trim_stop is false.")
                ("kc-temperature", po::value<double>(&cfg.kc.temperature)->default_value(1.0), "Temperature value.")
                ("kc-tfs", po::value<double>(&cfg.kc.tfs)->default_value(1.0), "Tail free sampling value. (minimum: 0.0, maximum: 1.0)")
                ("kc-top-a", po::value<double>(&cfg.kc.top_a)->default_value(1.0), "Top-a sampling value. (minimum: 0.0)")
                ("kc-top-k", po::value<double>(&cfg.kc.top_k)->default_value(0.0), "Top-k sampling value. (minimum: 0.0)")
                ("kc-top-p", po::value<double>(&cfg.kc.top_p)->default_value(1.0), "Top-p sampling value. (minimum: 0.0, maximum: 1.0)")
                ("kc-min-p", po::value<double>(&cfg.kc.min_p)->default_value(0.1), "Min-p sampling value. (minimum: 0.0, maximum: 1.0)")
                ("kc-typical", po::value<double>(&cfg.kc.typical)->default_value(1.0), "Typical sampling value. (minimum: 0.0, maximum: 1.0)")
                ("kc-use-default-badwordsids", po::bool_switch(&cfg.kc.use_default_badwordsids)->default_value(false), "If true, prevents the EOS token from being generated (Ban EOS).")
                ("kc-dynatemp_range", po::value<double>(&cfg.kc.dynatemp_range)->default_value(0.0), "If not equal to 0, uses dynamic temperature. Dynamic temperature range will be between Temp+Range and Temp-Range. If equal to 0 , uses static temperature. (default: 0, minimum: -5.0, maximum: 5.0)")
                ("kc-smoothing-factor", po::value<double>(&cfg.kc.smoothing_factor)->default_value(0.0), "Modifies temperature behavior. If greater than 0 uses smoothing factor. (default: 0.0, minimum: 0.0)")
                ("kc-dynatemp-exponent", po::value<double>(&cfg.kc.dynatemp_exponent)->default_value(1.0), "Exponent used in dynatemp. (default: 0.0)")
                ("kc-mirostat", po::value<int>(&cfg.kc.mirostat)->default_value(0), "KoboldCpp ONLY. Sets the mirostat mode, 0=disabled, 1=mirostat_v1, 2=mirostat_v2. (minimum: 0, maximum: 2)")
                ("kc-mirostat-tau", po::value<double>(&cfg.kc.mirostat_tau)->default_value(0.0), "KoboldCpp ONLY. Mirostat tau value. (minimum: 0.0)")
                ("kc-mirostat-eta", po::value<double>(&cfg.kc.mirostat_eta)->default_value(0.0), "KoboldCpp ONLY. Mirostat eta value. (minimum: 0.0)")
                ("kc-genkey", po::value<std::string>(&cfg.kc.genkey)->default_value(""), "KoboldCpp ONLY. A unique genkey set by the user. When checking a polled-streaming request, use this key to be able to fetch pending text even if multiuser is enabled.")
                ("kc-grammar", po::value<std::string>(&cfg.kc.grammar)->default_value(""), "KoboldCpp ONLY. A string containing the GBNF grammar to use.")
                ("kc-grammar-retain-state", po::bool_switch(&cfg.kc.grammar_retain_state)->default_value(false), "KoboldCpp ONLY. If true, retains the previous generation's grammar state, otherwise it is reset on new generation.")
                ("kc-memory", po::value<std::string>(&cfg.kc.memory)->default_value(""), "KoboldCpp ONLY. If set, forcefully appends this string to the beginning of any submitted prompt text. If resulting context exceeds the limit, forcefully overwrites text from the beginning of the main prompt until it can fit. Useful to guarantee full memory insertion even when you cannot determine exact token count.")
                ("kc-images", po::value<std::vector<std::string>>(&cfg.kc.images)->multitoken(), "KoboldCpp ONLY. If set, takes an array of base64 encoded strings, each one representing an image to be processed.")
                ("kc-trim-stop", po::bool_switch(&cfg.kc.trim_stop)->default_value(true), "KoboldCpp ONLY. If true, also removes detected stop_sequences from the output and truncates all text after them. If false, output will also include stop sequence and potentially a few additional characters.")
                ("kc-render-special", po::bool_switch(&cfg.kc.render_special)->default_value(false), "KoboldCpp ONLY. If true, prints special tokens as text for GGUF models")
                ("kc-bypass-eos", po::bool_switch(&cfg.kc.trim_stop)->default_value(false), "KoboldCpp ONLY. If true, allows EOS token to be generated, but does not stop generation. Not recommended unless you know what you are doing.")
                ("kc-banned-tokens", po::value<std::vector<std::string>>(&cfg.kc.banned_tokens)->multitoken(), "An array of string sequences, each entry represents a word or phrase prevented from being generated, either modifying model vocab or by backtracking and regenerating when they appear.")
                ("kc-dry-multiplier", po::value<double>(&cfg.kc.dry_multiplier)->default_value(0.0), "KoboldCpp ONLY. DRY multiplier value, 0 to disable. (minimum: 0)")
                ("kc-dry-base", po::value<double>(&cfg.kc.dry_base)->default_value(1.75), "KoboldCpp ONLY. DRY base value. (minimum: 0)")
                ("kc-dry-allowed-length", po::value<int>(&cfg.kc.dry_allowed_length)->default_value(2), "KoboldCpp ONLY. DRY allowed length value. (minimum: 0)")
                ("kc-dry-penalty-last-n", po::value<int>(&cfg.kc.dry_penalty_last_n)->default_value(0), "KoboldCpp ONLY. DRY last n tokens penalized value. (minimum: 0)")
                ("kc-dry-sequence-breakers", po::value<std::vector<std::string>>(&cfg.kc.dry_sequence_breakers)->multitoken(), "An array of string sequence breakers for DRY.")
                ("kc-xtc-threshold", po::value<double>(&cfg.kc.xtc_threshold)->default_value(0.1), "KoboldCpp ONLY. XTC threshold. (minimum: 0)")
                ("kc-xtc-probability", po::value<double>(&cfg.kc.xtc_probability)->default_value(0.0), "KoboldCpp ONLY. XTC probability. Set to above 0 to enable XTC. (minimum: 0)")
                ("kc-nsigma", po::value<double>(&cfg.kc.nsigma)->default_value(0.0), "KoboldCpp ONLY. Top N-Sigma value. Set to above 0 to enable nsigma. (minimum: 0)")
                ("kc-logprobs", po::bool_switch(&cfg.kc.logprobs)->default_value(false), "If true, return up to 5 top logprobs for generated tokens. Incurs performance overhead.")
                ("kc-replace-instruct-placeholders", po::bool_switch(&cfg.kc.use_default_badwordsids)->default_value(false), "If true, replaces instruct placeholders {{[INPUT]}} and {{[OUTPUT]}} with backend selected instruct tags.")

                ("sd-host", po::value<std::string>(&cfg.sd.host)->default_value("localhost"), "SD host")
                ("sd-port", po::value<std::string>(&cfg.sd.port)->default_value("7860"), "SD port")
                ("sd-prompt-file", po::value<std::string>(&cfg.sd.prompt_file)->default_value("prompt"), "SD prompt file")
                ("sd-negative-prompt-file", po::value<std::string>(&cfg.sd.negative_prompt_file)->default_value("negative_prompt"), "SD negative prompt file")
                ("sd-output-file", po::value<std::string>(&cfg.sd.output_file)->default_value("{{datetime}}.png"), "SD output PNG file")
                ("sd-prompt", po::value<std::string>(&cfg.sd.prompt)->default_value(""), "SD prompt")
                ("sd-negative-prompt", po::value<std::string>(&cfg.sd.negative_prompt)->default_value(""), "SD negative prompt")
                ("sd-styles", po::value<std::vector<std::string>>(&cfg.sd.styles), "SD styles")
                ("sd-seed", po::value<int>(&cfg.sd.seed)->default_value(-1), "SD seed")
                ("sd-subseed", po::value<int>(&cfg.sd.subseed)->default_value(-1), "SD subseed")
                ("sd-subseed-strength", po::value<double>(&cfg.sd.subseed_strength)->default_value(0), "SD subseed strength")
                ("sd-seed-resize-from-h", po::value<int>(&cfg.sd.seed_resize_from_h)->default_value(-1), "SD seed resize from height")
                ("sd-seed-resize-from-w", po::value<int>(&cfg.sd.seed_resize_from_w)->default_value(-1), "SD seed resize from width")
                ("sd-sampler-name", po::value<std::string>(&cfg.sd.sampler_name)->default_value("Euler a"), "SD sampler name")
                ("sd-scheduler", po::value<std::string>(&cfg.sd.scheduler)->default_value("Automatic"), "SD scheduler")
                ("sd-batch_size", po::value<int>(&cfg.sd.batch_size)->default_value(1), "SD batch size")
                ("sd-n-iter", po::value<int>(&cfg.sd.n_iter)->default_value(1), "SD n iter")
                ("sd-steps", po::value<int>(&cfg.sd.steps)->default_value(30), "SD steps")
                ("sd-cfg-scale", po::value<double>(&cfg.sd.cfg_scale)->default_value(7), "SD cfg scale")
                ("sd-width", po::value<int>(&cfg.sd.width)->default_value(1024), "SD image width")
                ("sd-height", po::value<int>(&cfg.sd.height)->default_value(1024), "SD image height")
                ("sd-restore-faces", po::bool_switch(&cfg.sd.restore_faces)->default_value(false), "SD restore faces")
                ("sd-tiling", po::bool_switch(&cfg.sd.tiling)->default_value(false), "SD tiling")
                ("sd-do-not-save-samples", po::bool_switch(&cfg.sd.do_not_save_samples)->default_value(false), "SD do not save samples")
                ("sd-do-not-save-grid", po::bool_switch(&cfg.sd.do_not_save_grid)->default_value(false), "SD do not save grid")
                ("sd-eta", po::value<int>(&cfg.sd.eta)->default_value(0), "SD eta")
                ("sd-denoising-strength", po::value<double>(&cfg.sd.denoising_strength)->default_value(0.75, "0.75"), "SD denoising strength")
                ("sd-s-min-uncond", po::value<int>(&cfg.sd.s_min_uncond)->default_value(0), "SD s min uncond")
                ("sd-s-churn", po::value<int>(&cfg.sd.s_churn)->default_value(0), "SD s churn")
                ("sd-s-tmax", po::value<int>(&cfg.sd.s_tmax)->default_value(0), "SD s tmax")
                ("sd-s-tmin", po::value<int>(&cfg.sd.s_tmin)->default_value(0), "SD s tmin")
                ("sd-s-noise", po::value<int>(&cfg.sd.s_noise)->default_value(1), "SD s noise")
                ("sd-override-settings", po::value<std::string>(&cfg.sd.override_settings)->default_value(""), "SD override settings")
                ("sd-override-settings-restore-afterwards", po::bool_switch(&cfg.sd.override_settings_restore_afterwards)->default_value(true), "SD override settings restore afterwards")
                ("sd-refiner-checkpoint", po::value<std::string>(&cfg.sd.refiner_checkpoint)->default_value(""), "SD refiner checkpoint")
                ("sd-refiner-switch-at", po::value<double>(&cfg.sd.refiner_switch_at)->default_value(0.8, "0.8"), "SD refiner switch at")
                ("sd-disable-extra-networks", po::bool_switch(&cfg.sd.disable_extra_networks)->default_value(false), "SD disable extra networks")
                ("sd-firstpass-image", po::value<std::string>(&cfg.sd.firstpass_image)->default_value(""), "SD firstpass image")
                ("sd-comments", po::value<std::string>(&cfg.sd.comments)->default_value(""), "SD comments")
                ("sd-force-task-id", po::value<std::string>(&cfg.sd.force_task_id)->default_value(""), "SD force task id")
                ("sd-sampler-index", po::value<std::string>(&cfg.sd.sampler_index)->default_value(""), "SD sampler index")
                ("sd-script-name", po::value<std::string>(&cfg.sd.script_name)->default_value(""), "SD script name")
                ("sd-script-args", po::value<std::vector<std::string>>(&cfg.sd.script_args), "SD script_args")
                ("sd-send-images", po::bool_switch(&cfg.sd.send_images)->default_value(true), "SD send images")
                ("sd-save-images", po::bool_switch(&cfg.sd.save_images)->default_value(false), "SD save images")
                ("sd-ad-enable", po::bool_switch(&cfg.sd.alwayson_scripts.adetailer_parametesrs.ad_enable)->default_value(false), "SD ADetailer enable")
                ("sd-ad-model", po::value<std::string>(&cfg.sd.alwayson_scripts.adetailer_parametesrs.args1.ad_model)->default_value("face_yolov8n.pt"), "SD ADetailer model")
                ("sd-ad-prompt", po::value<std::string>(&cfg.sd.alwayson_scripts.adetailer_parametesrs.args1.ad_prompt)->default_value(""), "SD ADetailer prompt")
                ("sd-ad-negative-prompt", po::value<std::string>(&cfg.sd.alwayson_scripts.adetailer_parametesrs.args1.ad_negative_prompt)->default_value(""), "SD ADetailer negative prompt")
                ("sd-infotext", po::value<std::string>(&cfg.sd.infotext)->default_value(""), "SD infotext")
                ("sd-abg-remover-enable", po::bool_switch(&cfg.sd.abg_remover_enable)->default_value(false), "SD ABG Remover enable")

                ("sd-mode", po::value<std::string>(&sd_mode_string)->default_value("txt2img"), "SD mode (txt2img | img2img)")

                ("sd-txt2img-target", po::value<std::string>(&cfg.sd.txt2img.target)->default_value("/sdapi/v1/txt2img"), "SD txt2img target")
                ("sd-enable-hr", po::bool_switch(&cfg.sd.txt2img.enable_hr)->default_value(false), "SD enable hr")
                ("sd-firstphase-width", po::value<int>(&cfg.sd.txt2img.firstphase_width)->default_value(0), "SD firstphase width")
                ("sd-firstphase-height", po::value<int>(&cfg.sd.txt2img.firstphase_height)->default_value(0), "SD firstphase height")
                ("sd-hr-scale", po::value<double>(&cfg.sd.txt2img.hr_scale)->default_value(0), "SD hr scale")
                ("sd-hr-upscaler", po::value<std::string>(&cfg.sd.txt2img.hr_upscaler)->default_value("SwinIR_4x"), "SD hr upscaler")
                ("sd-hr-second-pass-steps", po::value<int>(&cfg.sd.txt2img.hr_second_pass_steps)->default_value(0), "SD hr second pass steps")
                ("sd-hr-resize-x", po::value<int>(&cfg.sd.txt2img.hr_resize_x)->default_value(0), "SD hr resize x")
                ("sd-hr-resize-y", po::value<int>(&cfg.sd.txt2img.hr_resize_y)->default_value(0), "SD hr resize y")
                ("sd-hr-checkpoint-name", po::value<std::string>(&cfg.sd.txt2img.hr_checkpoint_name)->default_value(""), "SD hr checkpoint name")
                //("sd-hr-prompt", po::value<std::string>(&cfg.sd_txt2img_params.hr_prompt)->default_value(""), "SD hr prompt")
                //("sd-hr-negative-prompt", po::value<std::string>(&cfg.sd_txt2img_params.hr_negative_prompt)->default_value(""), "SD hr negative prompt")

                ("sd-img2img-target", po::value<std::string>(&cfg.sd.img2img.target)->default_value("/sdapi/v1/img2img"), "SD img2img target")
                ("sd-init-images", po::value<std::vector<std::string>>(&cfg.sd.img2img.init_images)->multitoken(), "SD img2img init_images (Base64 encoded images)")
                ("sd-seed-resize-from-h", po::value<int>(&cfg.sd.img2img.seed_resize_from_h)->default_value(-1), "SD img2img seed_resize_from_h")
                ("sd-seed-resize-from-w", po::value<int>(&cfg.sd.img2img.seed_resize_from_w)->default_value(-1), "SD img2img seed_resize_from_w")
                ("sd-resize-mode", po::value<int>(&cfg.sd.img2img.resize_mode)->default_value(0), "SD img2img resize_mode [0 - 3] (0: Just resize, 1: Crop and resize, 2: Resize and fill, 3: Just resize)")
                ("sd-image-cfg-scale", po::value<double>(&cfg.sd.img2img.image_cfg_scale)->default_value(1.0), "SD img2img image_cfg_scale")
                ("sd-mask", po::value<std::string>(&cfg.sd.img2img.mask)->default_value(""), "SD img2img mask (Base64 encoded image)")
                ("sd-mask-blur-x", po::value<int>(&cfg.sd.img2img.mask_blur_x)->default_value(4), "SD img2img mask_blur_x")
                ("sd-mask-blur-y", po::value<int>(&cfg.sd.img2img.mask_blur_y)->default_value(4), "SD img2img mask_blur_y")
                ("sd-mask-blur", po::value<int>(&cfg.sd.img2img.mask_blur)->default_value(4), "SD img2img mask_blur")
                ("sd-mask-round", po::bool_switch(&cfg.sd.img2img.mask_round)->default_value(true), "SD img2img mask_round")
                ("sd-inpainting-fill", po::value<int>(&cfg.sd.img2img.inpainting_fill)->default_value(0), "SD img2img inpainting_fill")
                ("sd-inpaint-full-res", po::bool_switch(&cfg.sd.img2img.inpaint_full_res)->default_value(true), "SD img2img inpaint_full_res")
                ("sd-inpaint-full-res-padding", po::value<int>(&cfg.sd.img2img.inpaint_full_res_padding)->default_value(0), "SD img2img inpaint_full_res_padding")
                ("sd-inpainting-mask-invert", po::value<int>(&cfg.sd.img2img.inpainting_mask_invert)->default_value(0), "SD img2img inpainting_mask_invert")
                ("sd-initial-noise-multiplier", po::value<double>(&cfg.sd.img2img.initial_noise_multiplier)->default_value(1.0), "SD img2img initial_noise_multiplier")
                ("sd-latent-mask", po::value<std::string>(&cfg.sd.img2img.latent_mask)->default_value(""), "SD img2img latent_mask (Base64 encoded image)")

                ("sb-host", po::value<std::string>(&cfg.sb.host)->default_value("localhost"), "SB host")
                ("sb-port", po::value<std::string>(&cfg.sb.port)->default_value("5001"), "SB port")
                ("sb-target", po::value<std::string>(&cfg.sb.target)->default_value("/voice"), "SB voide target")
                ("sb-text-file", po::value<std::string>(&cfg.sb.text_file)->default_value("text"), "SB text file")
                ("sb-output-file", po::value<std::string>(&cfg.sb.output_file)->default_value("{{datetime}}.wav"), "SB output WAV")
                ("sb-text", po::value<std::string>(&cfg.sb.text)->default_value(""), "SB text")
                ("sb-model-name", po::value<std::string>(&cfg.sb.model_name)->default_value(""), "SB model name")
                ("sb-model-id", po::value<int>(&cfg.sb.model_id)->default_value(0), "SB model id")
                ("sb-speaker-name", po::value<std::string>(&cfg.sb.speaker_name)->default_value(""), "SB speaker name")
                ("sb-speaker-id", po::value<int>(&cfg.sb.speaker_id)->default_value(0), "SB speaker id")
                ("sb-sdp-ratio", po::value<double>(&cfg.sb.sdp_ratio)->default_value(0.2, "0.2"), "SB sdp ratio")
                ("sb-noise", po::value<double>(&cfg.sb.noise)->default_value(0.6, "0.6"), "SB noise")
                ("sb-noisew", po::value<double>(&cfg.sb.noisew)->default_value(0.8, "0.8"), "SB noisew")
                ("sb-length", po::value<double>(&cfg.sb.length)->default_value(1), "SB length")
                ("sb-language", po::value<std::string>(&cfg.sb.language)->default_value(""), "SB language")
                ("sb-auto-split", po::bool_switch(&cfg.sb.auto_split)->default_value(true), "SB auto split")
                ("sb-split-interval", po::value<double>(&cfg.sb.split_interval)->default_value(0.5, "0.5"), "SB split interval")
                ("sb-assist-text", po::value<std::string>(&cfg.sb.assist_text)->default_value(""), "SB assist text")
                ("sb-assist-text-weight", po::value<double>(&cfg.sb.assist_text_weight)->default_value(1), "SB assist text weight")
                ("sb-style", po::value<std::string>(&cfg.sb.style)->default_value(""), "SB style")
                ("sb-style-weight", po::value<double>(&cfg.sb.style_weight)->default_value(1), "SB style weight")
                ("sb-reference-audio-path", po::value<std::string>(&cfg.sb.reference_audio_path)->default_value(""), "SB reference audio path")

                ("cu-host", po::value<std::string>(&cfg.cu.host)->default_value("localhost"), "Comfy UI host")
                ("cu-port", po::value<std::string>(&cfg.cu.port)->default_value("8188"), "Comfy UI port")
                ("cu-prompt-target", po::value<std::string>(&cfg.cu.prompt_target)->default_value("/prompt"), "Comfy UI prompt target")
                ("cu-upload-image-target", po::value<std::string>(&cfg.cu.upload_image_target)->default_value("/upload/image"), "Comfy UI upload image target")
                ("cu-prompt", po::value<std::string>(&cfg.cu.prompt)->default_value(""), "Comfy UI prompt")
                ("cu-prompt-file", po::value<std::string>(&cfg.cu.prompt_file)->default_value("prompt.json"), "Comfy UI prompt file")
                ("cu-output-directory", po::value<std::string>(&cfg.cu.output_directory)->default_value("output"), "Comfy UI output directory")
                ("cu-upload-images", po::value<std::vector<std::string>>(&cfg.cu.upload_images)->multitoken(), "Comfy UI upload images (macro_name=local_path)")
                ("cu-preserve-subdirectories", po::bool_switch(&cfg.cu.preserve_subdirectories)->default_value(false), "Comfy UI preserve server side sub-directories")
                ;

            po::options_description config_file_options;
            config_file_options.add(allowed_options);

            po::variables_map vm;
            po::store(po::parse_command_line(argc, argv, allowed_options), vm, true);
            po::notify(vm);

            try
            {
                cfg.command_mode = string_to_command_mode(command_mode_string);
            }
            catch (const command_line_exception&)
            {
                BOOST_LOG_TRIVIAL(error) << "mode options must be (tg | kc | sd | sb | cu | extract-png-parameters).";
                return 1;
            }

            cfg.llm.mode = string_to_llm_mode(llm_mode_string);
            cfg.sd.mode = string_to_sd_mode(sd_mode_string);

            if (!cfg.config_file.empty())
            {
                std::istringstream config_file{ read_text_file_to_string(cfg.config_file, cfg, ".ini") };
                po::store(po::parse_config_file(config_file, allowed_options), vm, true);
                po::notify(vm);
            }

            if (vm.find("help") != vm.end())
            {
                boost::nowide::cout << allowed_options << std::endl;
                return 1;
            }

            init_logging(cfg);

            if (cfg.command_mode == command_mode::tg || cfg.command_mode == command_mode::kc)
            {
                init_llm_mode(cfg);
            }

            if (cfg.phases.empty())
            {
                cfg.phases = { "" };
            }

            unescape_parameters(cfg);
            parse_user_defined_variables(cfg.user_defined_variables, cfg.ctx);
        }
        catch (const po::error& e)
        {
            llmcpp::throw_exception(command_line_exception{} << error_info::description{ std::string{ "boost::program_options::error: " } + e.what() });
        }

        return 0;
    }

    std::string truncate_prompt_by_config(std::string_view prompt, const config& cfg)
    {
        std::string result;
        int remaining_tokens{ cfg.tg.truncation_length - cfg.tg.max_tokens };
        truncate_prompt(prompt, cfg, false, result, remaining_tokens);
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

    void write_file(const config& cfg, const char* data, std::size_t size, std::string_view filepath, std::ios_base::openmode mode)
    {
        const bool is_binary{ (mode & std::ios::binary) != 0 };
        const std::string complemented{ is_binary ? filepath : complement_extension(filepath, ".txt") };
        const std::filesystem::path file_path{ string_to_path_by_config(complemented, cfg) };
        create_parent_directories(file_path);
        boost::nowide::ofstream ofs{ file_path, mode };
        if (!ofs.is_open())
        {
            llmcpp::throw_exception(file_open_exception{} << error_info::path{ file_path });
        }
        ofs.write(data, size);
        const std::string_view file_type{ is_binary ? "binary" : "text" };
        BOOST_LOG_TRIVIAL(info) << "Write " << file_type << " to " << file_path;
    }

    void write_file(const config& cfg, std::string_view data, std::string_view filepath, std::ios_base::openmode mode)
    {
        return write_file(cfg, data.data(), data.size(), filepath, mode);
    }

    void write_code_block(const config& cfg, std::string_view markdown)
    {
        if (cfg.llm.code_block_extract)
        {
            const code_blocks blocks{ extract_code_block_from_markdown(markdown) };
            for (const auto& [name, code] : blocks)
            {
                if (name == "stdout")
                {
                    boost::nowide::cout << code << std::flush;
                }
                else
                {
                    write_file(cfg, code, name, 0);
                }
            }
        }
    }

    void generate_text_and_write(const config& cfg, std::string_view prompt, const context& ctx)
    {
        const std::string truncated_prompt{ truncate_prompt_by_config(prompt, cfg) };

        std::string response{ generate_text(cfg, truncated_prompt, ctx) };
        response = remove_reasoning(response, cfg.llm.reasoning_prefix, cfg.llm.reasoning_suffix);
        response += cfg.llm.generation_suffix;

        write_file(cfg, response, cfg.llm.output_file, std::ios_base::app);

        if (!cfg.verbose)
        {
            boost::nowide::cout << response << std::flush;
        }

        write_code_block(cfg, response);
    }

    void vision_image_and_write(const config& cfg, std::string_view prompt, const context& ctx)
    {
        const std::string truncated_prompt{ truncate_prompt_by_config(prompt, cfg) };

        std::string response{ vision_image(cfg, truncated_prompt, ctx) };

        write_file(cfg, response, cfg.llm.output_file, std::ios_base::app);

        if (!cfg.verbose)
        {
            boost::nowide::cout << response << std::flush;
        }

        write_code_block(cfg, response);
    }

    std::string prompt_from_string_or_file_path(
        std::string_view string,
        std::string_view file_path,
        const config& cfg
    )
    {
        return string.empty() ? read_text_file_to_string(file_path, cfg) : std::string{ string };
    }

    void generate_and_output(const config& cfg)
    {
        if (cfg.command_mode == command_mode::tg || cfg.command_mode == command_mode::kc)
        {
            if (cfg.llm.mode == llm_mode::completions)
            {
                const std::string prompt{ prompt_from_string_or_file_path(cfg.llm.prompt, cfg.llm.prompt_file, cfg) };
                generate_text_and_write(cfg, prompt, cfg.ctx);
            }
            else if (cfg.llm.mode == llm_mode::vision)
            {
                const std::string prompt{ prompt_from_string_or_file_path(cfg.llm.prompt, cfg.llm.prompt_file, cfg) };
                vision_image_and_write(cfg, prompt, cfg.ctx);
            }
        }
        else if (cfg.command_mode == command_mode::sd)
        {
            const std::string prompt_string{ expand_macro(prompt_from_string_or_file_path(cfg.sd.prompt, cfg.sd.prompt_file, cfg), cfg, cfg.ctx) };
            const std::string negative_prompt_string{ expand_macro(prompt_from_string_or_file_path(cfg.sd.negative_prompt, cfg.sd.negative_prompt_file, cfg), cfg, cfg.ctx) };
            const std::string image{ send_automatic1111_txt2img_request(cfg, prompt_string, negative_prompt_string) };
            write_file(cfg, image, cfg.sd.output_file, std::ios::binary);
        }
        else if (cfg.command_mode == command_mode::sb)
        {
            const std::string text{ expand_macro(prompt_from_string_or_file_path(cfg.sb.text, cfg.sb.text_file, cfg), cfg, cfg.ctx) };
            const std::string voice{ send_style_bert_voice_request(cfg, text) };
            write_file(cfg, voice, cfg.sb.output_file, std::ios::binary);
        }
        else if (cfg.command_mode == command_mode::cu)
        {
            const std::string prompt{ expand_macro(prompt_from_string_or_file_path(cfg.cu.prompt, cfg.cu.prompt_file, cfg), cfg, cfg.ctx) };
            send_comfy_ui_prompt(cfg, prompt);
        }
    }

    void set_seed(config& cfg)
    {
        if (cfg.seed == -1)
        {
            cfg.tg.seed = random<int>(0);
            cfg.kc.sampler_seed = random<int>(0, 999999);
            cfg.sd.seed = random<int>(0);
        }
        else
        {
            cfg.tg.seed = cfg.seed;
            cfg.kc.sampler_seed = cfg.seed;
            cfg.sd.seed = cfg.seed;
        }
    }

    void create_process(const config& cfg)
    {
        if (!cfg.server_executable_file.empty())
        {
            const std::vector<std::string> arguments{ parse_command_line_args(cfg.server_arguments) };
            create_process_async(cfg.server_executable_file, arguments);
            if (!wait_for_port(cfg.server_host, cfg.server_port, cfg.server_max_retries, cfg.server_wait_ms))
            {
                BOOST_LOG_TRIVIAL(warning) << "Connection timed out waiting for server response.";
            }
        }
    }

    void terminate_process(const config& cfg)
    {
        if (!cfg.server_executable_file.empty())
        {
            if (terminate_process_by_path(cfg.server_executable_file) == 0)
            {
                BOOST_LOG_TRIVIAL(warning) << "Failed to terminate process by executable file path (" << cfg.server_executable_file << ").";
            }
        }
    }

    void create_process_or_terminate(const config& cfg)
    {
        if (cfg.create_process)
        {
            create_process(cfg);
        }
        else if (cfg.terminate_process)
        {
            terminate_process(cfg);
        }
    }

    void iterate(config& cfg)
    {
        read_cache(cfg);

        int iteration_count{};
        while (cfg.number_iterations == -1 || iteration_count < cfg.number_iterations)
        {
            set_seed(cfg);

            set_dynamic_builtin_variables(cfg);
            cfg.ctx.set("N", std::to_string(iteration_count + 1));

            for (std::size_t phase_index{}; phase_index < cfg.phases.size(); ++phase_index)
            {
                set_phase_variables(cfg.phases, phase_index, cfg.ctx);
                generate_and_output(cfg);
            }

            write_cache(cfg);

            iteration_count += 1;
        }
    }

    int exception_safe_main(int argc, char** argv)
    {
        try
        {
            config cfg;

            if (parse_command_line(argc, argv, cfg))
            {
                return 0;
            }

            if (cfg.create_process || cfg.terminate_process)
            {
                create_process_or_terminate(cfg);
                return 0;
            }

            if (cfg.command_mode == command_mode::extract_png_parameters)
            {
                const std::string parameters{ tEXt::extract_parameters(read_binary_file_to_string(cfg.png_file, cfg)) };
                boost::nowide::cout << parameters << std::flush;
            }

            set_static_builtin_variables(cfg);

            if (cfg.command_mode == command_mode::cu)
            {
                upload_images_to_comfy_ui(cfg, cfg.ctx);
            }

            iterate(cfg);
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
}

int main(int argc, char** argv)
{
    boost::nowide::args a(argc, argv);
    return llmcpp::exception_safe_main(argc, argv);
}